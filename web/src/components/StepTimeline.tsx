import type { EventEnvelope } from "./LiveTranscript";
import type { ExecutionEventDTO } from "../lib/api";

/** What the agent DID, in order — read off the transcript rather than off a
 *  separate marker stream.
 *
 *  The stages used to describe themselves with `__PHASE__` markers, which meant
 *  a stage only appeared to do something if its script remembered to say so,
 *  and a marker read out of somebody else's log showed up as this stage's work.
 *  The transcript is the record of what actually ran, so derive the summary from
 *  it and there is nothing to keep in sync.
 */

type Step = {
  key: string;
  label: string;
  detail: string;
  /** The agent's own remark, when it was folded into the call it announced. */
  note?: string;
  /** The untrimmed argument, when `detail` is a reduction of it. Hover recovers
   *  what the line could not hold; absent when nothing was dropped. */
  full?: string;
  kind: "act" | "say";
  /** How far a long call got, when progress arrived while it was running.
   *  A remote training run is ONE tool call that lasts twenty-five minutes; the
   *  readings that came back during it belong to that line, not to 37 lines of
   *  their own. */
  progress?: { step: number; total: number; loss?: number; n: number };
  ts: string;
};

/** How much of a primary argument belongs on one step line. Narration is never
 *  cut — a sentence carries a finding and half of one is worse than none. An
 *  argument is different: it is an address, and past a certain width it stops
 *  being read and starts pushing the step off the line. The full text stays
 *  reachable on hover. */
const ARG_MAX = 120;

/** A shell command reduced to the step it represents.
 *
 *  A stage doing its real work on another machine issues one `ssh <flags> <<EOF
 *  …script… EOF` per action, and the whole thing — key path, four `-o` options,
 *  an embedded program — arrived here as the step's description, so the Overview
 *  read as a wall of shell with the actual action buried mid-line.
 *
 *  Two structural cuts, no knowledge of any particular command:
 *   * SETUP is not the step. A script opens by arranging its shell — `set -e`,
 *     `cd`, `export`, sourcing an env file — and only then does the thing it was
 *     run for. Taking the first segment yields "set" for every such call, which
 *     is the same answer for every step and so distinguishes none of them; take
 *     the first segment that is not setup.
 *   * OPTION FLAGS are plumbing. They repeat identically on every call to the
 *     same tool, so they are the part that cannot distinguish two steps. A short
 *     flag takes the following word as its value unless that word is itself a
 *     flag.
 *
 *  What survives is the command word and its operands — the host, the path, the
 *  program being run. Both cuts respect quoting: an `ssh host "…; …"` carries its
 *  whole remote program in one quoted word, and splitting inside it would tear
 *  the one informative part of the line in half.
 */
const FLAG = /^--?[A-Za-z]/;
const SETUP = new Set(["set", "cd", "export", "source", ".", "unset", "umask", "shopt"]);

/** Split on unquoted separators, longest first so `&&` wins over `&`. Matching
 *  whole separators rather than characters is what keeps a redirection like
 *  `2>&1` in one piece while still breaking `a && b` into two. */
function splitOutsideQuotes(s: string, seps: string[]): string[] {
  const out: string[] = [];
  const ordered = [...seps].sort((a, b) => b.length - a.length);
  let buf = "";
  let quote = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (quote) {
      buf += c;
      if (c === quote && s[i - 1] !== "\\") quote = "";
      continue;
    }
    if (c === "'" || c === '"') {
      quote = c;
      buf += c;
      continue;
    }
    const hit = ordered.find((sep) => s.startsWith(sep, i));
    if (hit) {
      out.push(buf);
      buf = "";
      i += hit.length - 1;
      continue;
    }
    buf += c;
  }
  out.push(buf);
  return out;
}

const words = (s: string) => splitOutsideQuotes(s, [" ", "\t"]).filter(Boolean);

/** Drop the flags. `greedy` also drops the word after a short flag, on the
 *  guess that it is that flag's value — right for `-o Opt=x`, wrong for
 *  `mkdir -p dir`, and there is no way to tell without knowing the command. */
function stripFlags(ws: string[], greedy: boolean): string[] {
  const kept: string[] = [];
  for (let i = 0; i < ws.length; i++) {
    const w = ws[i];
    if (FLAG.test(w)) {
      if (greedy && /^-[A-Za-z]$/.test(w) && ws[i + 1] && !FLAG.test(ws[i + 1])) i++;
      continue;
    }
    kept.push(w);
  }
  return kept;
}

function summarizeCommand(cmd: string): string {
  const segments = splitOutsideQuotes(cmd, ["\n", ";", "&&", "||"])
    .map((s) => s.trim())
    .filter(Boolean);
  const isSetup = (seg: string) => {
    const head = words(seg)[0] || "";
    return SETUP.has(head) || /^[A-Za-z_][A-Za-z0-9_]*=/.test(head);
  };
  const segment = segments.find((s) => !isSetup(s)) || segments[0] || cmd.trim();

  const ws = words(segment);
  // Guess that short flags take a value, then check the result: if the line came
  // out as nothing but the command word, the guess ate the operand that WAS the
  // point of the step (`mkdir -p <dir>`, `curl -X POST <url>`), so take the
  // conservative reading instead. Self-correcting, and it needs to know nothing
  // about which command this is.
  let kept = stripFlags(ws, true);
  if (kept.length <= 1) kept = stripFlags(ws, false);
  // Falling back to the segment itself: a command that is nothing but flags is
  // rare, but showing "" for it would be worse than showing the flags.
  return kept.join(" ") || segment;
}

function clamp(s: string): string {
  return s.length > ARG_MAX ? s.slice(0, ARG_MAX).trimEnd() + "…" : s;
}

/** Display-only safety net for events produced without backend enrichment.
 * Keep this semantic and non-sensitive: the transcript below still shows the
 * untouched command, while Overview says what kind of action it represents. */
function fallbackBashDescription(command: string): string {
  const raw = command.trim();
  const lower = raw.toLowerCase();
  if (!raw) return "Run a shell command";
  if (/^\s*post_message\s+(?!\()/m.test(raw) || /\/tickets\/[^\s'\"]+\/messages/.test(lower)) {
    return "Post a ticket update";
  }
  if (/(^|[/\s])train\.py(\s|$)/.test(lower)) return "Run training on the assigned GPU";
  if (/(^|[/\s])predict\.py(\s|$)/.test(lower)) return "Run inference on the assigned GPU";
  if (/(^|[/\s])eval(?:uate)?\.py(\s|$)/.test(lower)) return "Run evaluation";
  if (/\b(?:ssh|srun)\b/.test(lower)) {
    if (lower.includes("nvidia-smi")) return "Check remote GPU availability";
    return lower.includes("<<")
      ? "Run a script on the assigned remote host"
      : "Run a command on the assigned remote host";
  }
  if (raw.includes("<<")) {
    return /\bpython3?\b/.test(lower) ? "Run an inline Python script" : "Run an inline shell script";
  }
  // Codex reports its shell transport as part of the command. Peel it once so
  // `bash -lc 'rg …'` describes the inspection, not the bash executable.
  const wrapped = raw.match(/^\s*(?:bash|sh|zsh)\s+-l?c\s+['\"]?([\s\S]*?)['\"]?\s*$/i);
  if (wrapped?.[1] && wrapped[1] !== raw) return fallbackBashDescription(wrapped[1]);
  const head = raw.match(/^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*([^\s;]+)/)?.[1]
    ?.split("/").pop()?.toLowerCase() || "";
  if (["rg", "grep", "find", "ls"].includes(head)) return "Inspect files and state";
  if (["cat", "head", "tail", "sed", "jq"].includes(head)) return "Inspect command output";
  if (["scp", "rsync"].includes(head)) return "Transfer files";
  if (head === "mkdir") return "Prepare working directories";
  if (head === "chmod") return "Prepare an executable script";
  if (head === "curl") return /\s-x\s+(?:post|put|patch|delete)\b/i.test(raw)
    ? "Update Zevo API state" : "Read Zevo API state";
  if (["python", "python3"].includes(head)) return raw.includes(" -c ")
    ? "Run a Python command" : "Run a Python script";
  return head ? `Run ${head} command` : "Run a shell command";
}

/** The one line that says what a tool call was for. Canonical Bash events carry
 *  a top-level description (agent-authored or deterministic fallback) while
 *  the untouched raw tool input remains under `input`. Older/un-normalized
 *  events may still carry the agent's description only inside that input. */
function describeTool(payload: Record<string, unknown>): { label: string; detail: string; full?: string } {
  const tool = String(payload.tool || payload.name || "tool");
  const input = (payload.input || {}) as Record<string, unknown>;
  const canonical = typeof payload.description === "string" ? payload.description.trim() : "";
  const authored = typeof input.description === "string" ? input.description.trim() : "";
  const desc = canonical || authored;
  if (desc) return { label: tool, detail: desc };
  if (["bash", "run_bash", "exec_command", "shell"].includes(tool.toLowerCase())) {
    const command = String(input.command || payload.command || "");
    if (command.trim()) return { label: tool, detail: fallbackBashDescription(command) };
  }
  // Whichever field carries the point of the call. `skill` first: WHICH skill
  // an agent invoked is the decision it made — a bare "SKILL" says only that it
  // consulted one, which is true of every stage and distinguishes nothing.
  for (const k of ["skill", "file_path", "path", "query", "pattern", "command", "url", "prompt"]) {
    const v = input[k];
    if (typeof v === "string" && v.trim()) {
      const s = v.trim();
      // Paths repeat the run directory on every line; the basename is the part
      // that differs. A command is reduced to the action it performs. Everything
      // else is already the short form.
      const short =
        k === "file_path" || k === "path"
          ? s.split("/").pop() || s
          : k === "command"
          ? summarizeCommand(s)
          : s;
      // `full` only when something was actually dropped, so the tooltip appears
      // exactly where there is more to see.
      const detail = clamp(short);
      return { label: tool, detail, full: detail === s ? undefined : s };
    }
  }
  return { label: tool, detail: "" };
}

/** Attach each progress reading to the call that was running when it arrived.
 *
 *  The readings live in `execution_events`, not in the transcript: they are
 *  written by the training script over the API while the agent is blocked
 *  inside one long ssh call, so they belong to no transcript event and have no
 *  place in its sequence. Mirroring them in was tried and was wrong — the
 *  stream numbers its own events, so a second writer picking `max(seq)+1`
 *  collides, and the client's `seq > lastSeen` de-dup then drops one of each
 *  colliding pair. The rows were in the database and never on the screen.
 *
 *  Matched by TIME instead, which is what actually relates them: a reading
 *  belongs to whichever call had started and not yet finished. */
function foldProgress(steps: Step[], executionEvents: ExecutionEventDTO[]): Step[] {
  const readings = executionEvents
    .filter((ph) => ph.current_step > 0 && ph.ts)
    .sort((a, b) => a.ts.localeCompare(b.ts));
  if (!readings.length) return steps;
  const acts = steps.filter((s) => s.kind === "act" && s.ts);
  for (const r of readings) {
    // The last call that had begun by the time this reading arrived.
    let owner: Step | undefined;
    for (const a of acts) {
      if (a.ts <= r.ts) owner = a;
      else break;
    }
    if (!owner) continue;
    owner.progress = {
      step: r.current_step,
      total: r.total_steps,
      loss: r.loss > 0 ? r.loss : owner.progress?.loss,
      n: (owner.progress?.n ?? 0) + 1,
    };
  }
  return steps;
}

export function buildSteps(events: EventEnvelope[], executionEvents: ExecutionEventDTO[] = []): Step[] {
  const all: Step[] = [];
  for (const e of events) {
    const payload = (e.payload || {}) as Record<string, unknown>;
    if (e.type === "tool_call") {
      const { label, detail, full } = describeTool(payload);
      all.push({ key: `${e.seq}`, label, detail, full, kind: "act", ts: e.ts });
    } else if (e.type === "agent_message") {
      // The agent's own narration between actions. This IS the summary — it is
      // what the agent chose to say about what it had just found — so it leads
      // the timeline rather than being buried under the calls that produced it.
      const msg = String(payload.message || "").trim();
      const first = msg.split("\n").find((l) => l.trim())?.trim();
      if (!first) continue;
      // The final message is usually the result payload itself. It is a machine
      // record, not something the agent said about its work, and it is rendered
      // properly elsewhere on the page. Judged by shape, not by wording.
      if (first.startsWith("{") || first.startsWith("[")) continue;
      all.push({ key: `${e.seq}`, label: "", detail: first, kind: "say", ts: e.ts });
    }
  }
  // A run of identical calls (three greps in a row) is one step, not three.
  const deduped = all.filter((s, i) => {
    const prev = all[i - 1];
    return !prev || prev.label !== s.label || prev.detail !== s.detail;
  });

  // Every step stays. Filtering to "what it said plus what it wrote" seemed like
  // the summary, but it is only the summary for an agent that narrates: a data
  // stage that inspects a dataset, writes one script, runs it and verifies the
  // output has done four things and said one, and the filtered view showed the
  // one. What each call was FOR is the answer to "what did this agent do", so
  // the calls stay and the condensing happens below, on structure.
  const kept = deduped;

  // Two things that are one thing:
  //
  //  * A line ending in a colon introduces something — a table, a listing — that
  //    lives in the feed below, not here. Without its body it is a stub.
  //  * "…Writing `predict.py`." followed by `WRITE predict.py` is one event said
  //    twice; fold the sentence into the call it announced.
  //
  // Nothing stronger than this: a run of consecutive narration LOOKS like one
  // thought, but collapsing it to its last line deletes findings — the sentence
  // that says a result is degenerate is not superseded by the one that measures
  // how degenerate.
  // Narration that comes before the first action is a preamble. Not because of
  // how it is worded — because of where it sits: nothing has happened yet, so
  // it cannot be reporting anything, only announcing. Every agent opens this
  // way, and the steps that follow are the announcement carried out.
  const firstAct = kept.findIndex((s) => s.kind === "act");
  const body = firstAct === -1 ? kept : kept.slice(firstAct);

  const out: Step[] = [];
  for (const s of body) {
    if (s.kind === "say" && s.detail.endsWith(":")) continue;
    const prev = out[out.length - 1];
    if (s.kind === "act" && prev?.kind === "say") {
      out[out.length - 1] = { ...s, note: prev.detail };
      continue;
    }
    out.push(s);
  }
  return foldProgress(out, executionEvents);
}

export function StepTimeline({
  events,
  executionEvents = [],
  empty = "Nothing recorded yet.",
  height = "max-h-[18rem]",
  onSeek,
}: {
  events: EventEnvelope[];
  /** The Ticket's `execution_events`. A long call shows how far it got. */
  executionEvents?: ExecutionEventDTO[];
  empty?: string;
  /** Tailwind max-height for the scroll box. */
  height?: string;
  /** Given, each step becomes a button that jumps the feed below to it. */
  onSeek?: (ts: string) => void;
}) {
  const steps = buildSteps(events, executionEvents);
  if (steps.length === 0) {
    return <div className="text-xs text-dim">{empty}</div>;
  }
  return (
    // Fixed height, scrolled. The number of steps swings from a handful to a few
    // dozen depending on how much trouble the stage ran into, and the sections
    // under this one should not move down the page because of it.
    // Scrolling on an OUTER box, not on the list. A marker sits on the rail,
    // which means it hangs a few pixels outside the list's own border — and CSS
    // will not clip one axis while leaving the other free, so `overflow-y` on
    // the list itself sliced every dot in half.
    <div className={`overflow-y-auto pl-1.5 pr-1 ${height}`}>
    <ol className="relative space-y-1.5 border-l border-hair pl-4">
      {steps.map((s, i) => {
        const body = (
          <>
            {/* Marker on the rail, then the count. The dot is what the eye
                follows down the thread; the number is what you cite. */}
            {/* One marker for every step. The rail is the thread; making its
                nodes differ split it into two threads. What kind of step it is
                is carried by the text, which is where you are reading anyway. */}
            <span className="absolute -left-[1.2rem] mt-[0.4rem] h-1.5 w-1.5 shrink-0 rounded-full bg-brass-500" />
            <span className="w-5 shrink-0 text-right font-mono text-2xs text-brass-500 tabular-nums">
              {i + 1}
            </span>
            {s.label && (
              <span className="shrink-0 font-mono text-2xs uppercase tracking-wider text-brass-300">
                {s.label}
              </span>
            )}
            {/* Two voices. What the agent DID is plain; what it THOUGHT — a
                reading, a conclusion, a change of plan — is set apart, because
                a list where every line looks like a command hides the handful
                of lines that explain why the commands changed. */}
            <span className={`min-w-0 ${s.kind === "say" ? "italic text-skyx-300" : "text-ink"}`}>
              {s.detail}
              {s.note && <span className="italic text-skyx-300"> · {s.note}</span>}
              {/* How far this call got. The readings arrive every 30s while it
                  runs, so this line is the difference between a stage that is
                  working and one that is hung. */}
              {s.progress && (
                <span className="font-mono text-phosphor-300/80">
                  {" · "}step {s.progress.step}
                  {s.progress.total > 0 && `/${s.progress.total}`}
                  {typeof s.progress.loss === "number" && ` · loss ${s.progress.loss.toFixed(4)}`}
                  <span className="text-dim">
                    {` · ${s.progress.n} reading${s.progress.n === 1 ? "" : "s"}`}
                  </span>
                </span>
              )}
            </span>
          </>
        );
        const cls = "relative flex items-baseline gap-2 text-xs leading-relaxed";
        return onSeek ? (
          <li key={s.key}>
            <button
              onClick={() => onSeek(s.ts)}
              title={s.full || "Jump to this point in the feed below"}
              className={`${cls} w-full text-left transition hover:text-brass-200`}
            >
              {body}
            </button>
          </li>
        ) : (
          <li key={s.key} className={cls} title={s.full}>
            {body}
          </li>
        );
      })}
    </ol>
    </div>
  );
}
