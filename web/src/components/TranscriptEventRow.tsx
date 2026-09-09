/**
 * Shared per-event renderer used by LiveTranscript (per-heartbeat feed) and
 * the run Timeline's StageDetail. Rendered as tight, mono "oscilloscope" rows.
 */
import { Link } from "react-router-dom";
import { scrubText } from "../lib/format";
import { statusLabelFor, waitingDetailFor } from "./StatusBadge";

export type TranscriptEvent = {
  seq?: number;
  ts: string;
  type: string;
  payload: Record<string, unknown>;
  // Run-timeline only; LiveTranscript leaves these empty:
  ticket_id?: string;
  agent_id?: string;
};

export function fmtTs(iso: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    const h = d.getHours().toString().padStart(2, "0");
    const m = d.getMinutes().toString().padStart(2, "0");
    const s = d.getSeconds().toString().padStart(2, "0");
    return `${h}:${m}:${s}`;
  } catch {
    return iso;
  }
}

// Stable colour per agent — small palette, keyed by name. Observatory tokens.
const AGENT_COLORS: Record<string, string> = {
  "orchestrator":    "bg-lilac-500/15 text-lilac-300 border-lilac-500/30",
  "data":            "bg-brass-500/15 text-brass-300 border-brass-500/30",
  "infrastructure":  "bg-coral-500/15 text-coral-300 border-coral-500/30",
  "train":           "bg-phosphor-500/15 text-phosphor-300 border-phosphor-500/30",
  "inference":       "bg-skyx-500/15 text-skyx-300 border-skyx-500/30",
  "evaluation":      "bg-lilac-500/15 text-lilac-300 border-lilac-500/30",
  "registry":        "bg-phosphor-500/15 text-phosphor-300 border-phosphor-500/30",
};
function agentClass(agentId: string): string {
  return AGENT_COLORS[agentId] || "bg-raised text-dim border-hair";
}

/* ---------- per-type bodies ---------- */

function AgentMessage({ ev }: { ev: TranscriptEvent }) {
  const text =
    (ev.payload.message as string) ||
    (ev.payload.text as string) ||
    (ev.payload.content as string) ||
    "";
  if (!text.trim()) return null;
  return <div className="whitespace-pre-wrap text-sm text-slate-200">{scrubText(text)}</div>;
}

function Reasoning({ ev }: { ev: TranscriptEvent }) {
  const text = (ev.payload.text as string) || (ev.payload.message as string) || "";
  if (!text.trim()) return null;
  return (
    <div className="rounded-md border border-lilac-500/20 bg-lilac-500/5 p-2">
      <div className="mb-1 font-mono text-[13px] uppercase tracking-[0.18em] text-lilac-300/70">
        thinking
      </div>
      <div className="whitespace-pre-wrap text-[16px] italic text-lilac-100/80">
        {scrubText(text)}
      </div>
    </div>
  );
}

function ToolCall({ ev }: { ev: TranscriptEvent }) {
  const tool = (ev.payload.tool as string) || (ev.payload.name as string) || "tool";
  const cmd = ev.payload.command || ev.payload.args || ev.payload.input;
  const cmdStr =
    typeof cmd === "string" ? cmd : cmd ? JSON.stringify(cmd, null, 2) : "";
  return (
    <div className="rounded-md border border-hair bg-raised/60 p-2">
      <div className="font-mono text-[13px] uppercase tracking-[0.18em] text-brass-300/80">
        used {tool}
      </div>
      {cmdStr && (
        <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-[15px] text-slate-300">
          {scrubText(cmdStr)}
        </pre>
      )}
    </div>
  );
}

/** Drop the __PROGRESS__/__PHASE__/__CONFIG__ marker lines and tqdm progress
 *  bars from a tool's stdout — they're captured in the train/infer monitor, so
 *  in the transcript they're just noise. */
function stripProgressNoise(s: unknown): string {
  if (typeof s !== "string") return "";
  return s
    .split("\n")
    .filter((ln) => {
      if (ln.includes("__PROGRESS__") || ln.includes("__PHASE__") || ln.includes("__CONFIG__")) return false;
      if (/\d+%\|/.test(ln)) return false;                    // tqdm bar
      if (/\d+\/\d+\s*\[\d{2}:\d{2}/.test(ln)) return false;  // tqdm "1/3 [00:06<…"
      return true;
    })
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function ToolResult({ ev }: { ev: TranscriptEvent }) {
  // output/stdout can be a string (bash) OR an object (read/write tools) — only
  // treat strings as raw text; otherwise fall back to a JSON dump below.
  const rawVal = ev.payload.output ?? ev.payload.stdout;
  const raw = typeof rawVal === "string" ? rawVal : "";
  const out = scrubText(raw ? stripProgressNoise(raw) : JSON.stringify(ev.payload, null, 2));
  const exitCode = ev.payload.exit_code;
  return (
    <details className="rounded-md border border-hair bg-raised/40 p-2">
      <summary className="cursor-pointer font-mono text-[13px] uppercase tracking-[0.18em] text-dim">
        ↳ tool result
        {typeof exitCode === "number" && (
          <span
            className={`ml-2 rounded px-1.5 py-0.5 font-mono text-[13px] ${
              exitCode === 0
                ? "bg-phosphor-500/15 text-phosphor-300"
                : "bg-coral-500/15 text-coral-300"
            }`}
          >
            exit {exitCode}
          </span>
        )}
      </summary>
      <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words font-mono text-[15px] text-slate-300">
        {out.slice(0, 4000)}
        {out.length > 4000 && "\n...(truncated)"}
      </pre>
    </details>
  );
}

function Phase({ ev }: { ev: TranscriptEvent }) {
  const name = (ev.payload.phase as string) || "phase";
  return (
    <div className="inline-flex items-center gap-1 rounded-md bg-lilac-500/15 px-2 py-0.5 font-mono text-[14px] uppercase tracking-wider text-lilac-300">
      ▸ {name}
    </div>
  );
}

function Progress({ ev }: { ev: TranscriptEvent }) {
  const step = (ev.payload.step as number) ?? 0;
  const total = (ev.payload.total as number) ?? 0;
  const loss = ev.payload.loss as number | undefined;
  const pct = total > 0 ? Math.round((100 * step) / total) : 0;
  return (
    <div className="flex items-center gap-3 font-mono text-[15px] text-dim">
      <span>
        step {step}
        {total > 0 ? ` / ${total}` : ""}
      </span>
      {total > 0 && (
        <div className="h-1.5 w-32 overflow-hidden rounded bg-raised">
          <div className="h-full bg-phosphor-400/70 transition-all" style={{ width: `${pct}%` }} />
        </div>
      )}
      {loss !== undefined && loss >= 0 && (
        <span className="text-brass-300">loss {loss.toFixed(4)}</span>
      )}
    </div>
  );
}

function Stderr({ ev }: { ev: TranscriptEvent }) {
  const t = (ev.payload.text as string) || "";
  if (!t.trim()) return null;
  return (
    <div className="rounded-md border border-coral-500/30 bg-coral-500/5 p-2 font-mono text-[15px] text-coral-300">
      {t}
    </div>
  );
}

function HeartbeatStarted({ ev }: { ev: TranscriptEvent }) {
  return (
    <div className="rounded-md border border-hair bg-raised/70 px-3 py-1 font-mono text-[15px] text-dim">
      <span className="text-phosphor-400">●</span> heartbeat started · agent{" "}
      <span className="text-slate-200">{String(ev.payload.agent_id || "")}</span>{" "}
      · driver {String(ev.payload.driver || "")} ·
      model {String(ev.payload.model || "")}
    </div>
  );
}

function HeartbeatFinished({ ev }: { ev: TranscriptEvent }) {
  const ec = (ev.payload.exit_code as number) ?? -1;
  const err = (ev.payload.error_message as string) || "";
  return (
    <div
      className={`rounded-md border px-3 py-1 font-mono text-[15px] ${
        ec === 0
          ? "border-phosphor-500/30 bg-phosphor-500/5 text-phosphor-300"
          : "border-coral-500/30 bg-coral-500/5 text-coral-300"
      }`}
    >
      ■ heartbeat finished · exit {ec}
      {err && <span className="ml-2 italic text-dim">{err}</span>}
    </div>
  );
}

function Finished({ ev }: { ev: TranscriptEvent }) {
  const exit = (ev.payload.exit_code as number) ?? -1;
  const err = (ev.payload.error_message as string) || "";
  return (
    <div
      className={`rounded-md border px-3 py-1.5 font-mono text-[15px] ${
        exit === 0
          ? "border-phosphor-500/30 bg-phosphor-500/5 text-phosphor-300"
          : "border-coral-500/30 bg-coral-500/5 text-coral-300"
      }`}
    >
      ■ heartbeat finished · exit {exit}
      {err && <span className="ml-2 italic text-dim">{err}</span>}
    </div>
  );
}

function RunFinished({ ev }: { ev: TranscriptEvent }) {
  const status = String(ev.payload.status || "");
  return (
    <div className="rounded-md border border-lilac-500/30 bg-lilac-500/10 px-3 py-1.5 font-mono text-[15px] text-lilac-200">
      ◆ run {status}
      {ev.payload.halted_reason ? (
        <span className="ml-2 italic text-dim">
          {String(ev.payload.halted_reason)}
        </span>
      ) : null}
    </div>
  );
}

function TicketCreated({ ev }: { ev: TranscriptEvent }) {
  const agent = String(ev.agent_id || "");
  const lane = String(ev.payload.lane || "");
  return (
    <div className="font-mono text-[15px] text-dim">
      ▤ created ticket{" "}
      <Link to={`/tickets/${ev.ticket_id}`} className="text-slate-200 hover:text-brass-300">
        {ev.ticket_id}
      </Link>{" "}
      · {agent}{lane ? ` · ${lane}` : ""}
      {ev.payload.summary ? (
        <span className="ml-2 text-dim">: {String(ev.payload.summary)}</span>
      ) : null}
    </div>
  );
}

function TicketStatus({ ev }: { ev: TranscriptEvent }) {
  const status = String(ev.payload.status || "");
  const label = statusLabelFor(status);
  const waitingDetail = waitingDetailFor(status);
  const color =
    status === "succeeded" ? "text-phosphor-300"
      : status === "failed" ? "text-coral-300"
      : status === "running" ? "text-brass-300"
      : status === "repairing" ? "text-lilac-300"
      : status === "degraded" ? "text-lilac-300"
      : "text-slate-300";
  return (
    <div className="font-mono text-[15px] text-dim">
      →{" "}
      <Link to={`/tickets/${ev.ticket_id}`} className="text-slate-200 hover:text-brass-300">
        {ev.ticket_id}
      </Link>{" "}
      <span className={color}>{label}</span>
      {waitingDetail ? <span className="ml-2 text-dim">· {waitingDetail}</span> : null}
    </div>
  );
}

function Message({ ev }: { ev: TranscriptEvent }) {
  const author = String(ev.payload.author || "");
  const body = String(ev.payload.body || "");
  return (
    <div className="rounded-md border border-hair bg-raised/60 p-2">
      <div className="mb-1 font-mono text-[13px] uppercase tracking-[0.18em] text-dim">
        {author}
      </div>
      <pre className="whitespace-pre-wrap break-words font-mono text-[15px] text-slate-300">
        {body.slice(0, 2000)}
      </pre>
    </div>
  );
}

function HeartbeatAlive({ ev }: { ev: TranscriptEvent }) {
  const msg = (ev.payload.message as string) || "Agent is still running…";
  const secs = (ev.payload.silent_seconds as number) || 0;
  return (
    <div className="flex items-center gap-2 font-mono text-[15px] text-dim">
      <span className="lamp lamp-live" />
      <span>{msg}</span>
      {secs > 0 && (
        <span className="text-slate-600">({Math.round(secs)}s since last event)</span>
      )}
    </div>
  );
}

function Raw({ ev }: { ev: TranscriptEvent }) {
  const t = (ev.payload.text as string) || "";
  if (!t.trim()) return null;
  return (
    <pre className="whitespace-pre-wrap break-words font-mono text-[15px] text-dim">
      {scrubText(t)}
    </pre>
  );
}

function CodexEvent({ ev }: { ev: TranscriptEvent }) {
  return (
    <details className="rounded-md border border-hair bg-raised/40 p-2">
      <summary className="cursor-pointer font-mono text-[13px] uppercase tracking-[0.18em] text-dim">
        ⓘ {ev.type}
      </summary>
      <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-[15px] text-dim">
        {scrubText(JSON.stringify(ev.payload, null, 2))}
      </pre>
    </details>
  );
}

/* ---------- the row itself ---------- */

export function TranscriptEventRow({
  ev,
  showAgentPill = false,
  showTicketPill = false,
}: {
  ev: TranscriptEvent;
  showAgentPill?: boolean;
  showTicketPill?: boolean;
}) {
  let body: JSX.Element | null = null;
  switch (ev.type) {
    case "heartbeat_started": body = <HeartbeatStarted ev={ev} />; break;
    case "heartbeat_finished": body = <HeartbeatFinished ev={ev} />; break;
    case "ticket_created": body = <TicketCreated ev={ev} />; break;
    case "ticket_status": body = <TicketStatus ev={ev} />; break;
    case "message": body = <Message ev={ev} />; break;
    case "agent_message": body = <AgentMessage ev={ev} />; break;
    case "agent_reasoning":
    case "reasoning": body = <Reasoning ev={ev} />; break;
    case "tool_call": body = <ToolCall ev={ev} />; break;
    case "tool_result": body = <ToolResult ev={ev} />; break;
    case "phase": body = <Phase ev={ev} />; break;
    case "progress": body = <Progress ev={ev} />; break;
    case "stderr": body = <Stderr ev={ev} />; break;
    case "finished": body = <Finished ev={ev} />; break;
    case "run_finished": body = <RunFinished ev={ev} />; break;
    case "heartbeat_alive": body = <HeartbeatAlive ev={ev} />; break;
    case "cancelled":
      body = (
        <div className="rounded-md border border-coral-500/30 bg-coral-500/10 px-3 py-1.5 font-mono text-[15px] text-coral-300">
          ⊘ {(ev.payload.message as string) || "Cancelled by user"}
        </div>
      ); break;
    case "raw": body = <Raw ev={ev} />; break;
    default: body = <CodexEvent ev={ev} />;
  }
  if (body === null) return null;

  const agent = ev.agent_id || "";
  const ticket = ev.ticket_id || "";

  return (
    <div className="flex items-start gap-3 px-3 py-1.5 transition hover:bg-raised/40">
      <div className="w-14 shrink-0 pt-0.5 font-mono text-[14px] tabular-nums text-slate-600">
        {fmtTs(ev.ts)}
      </div>
      {(showAgentPill || showTicketPill) && (
        <div className="flex w-44 shrink-0 flex-col gap-0.5 pt-0.5">
          {showAgentPill && agent && (
            agent === "evaluation" ? (
              <span
                title="Deterministic Evaluation runner; no Agent or LLM call"
                className={`inline-flex w-fit items-center rounded border px-1.5 py-0.5 font-mono text-[13px] uppercase tracking-wider ${agentClass(agent)}`}
              >
                {agent}
              </span>
            ) : (
              <Link
                to={`/agents/${agent}`}
                className={`inline-flex w-fit items-center rounded border px-1.5 py-0.5 font-mono text-[13px] uppercase tracking-wider ${agentClass(agent)}`}
              >
                {agent}
              </Link>
            )
          )}
          {showTicketPill && ticket && (
            <Link
              to={`/tickets/${ticket}`}
              className="inline-flex w-fit items-center rounded border border-hair bg-raised/40 px-1.5 py-0.5 font-mono text-[13px] text-dim hover:text-slate-200"
            >
              {ticket}
            </Link>
          )}
        </div>
      )}
      <div className="min-w-0 flex-1">{body}</div>
    </div>
  );
}
