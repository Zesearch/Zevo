import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import useSWR from "swr";
import { MessageSquare } from "lucide-react";

import { api, type RunInstructionDTO } from "../lib/api";
import { fmtDate } from "../lib/format";
import { Bezel, Kicker } from "./zevo/primitives";

const LABELS: Record<RunInstructionDTO["status"], string> = {
  queued: "Waiting for agent",
  delivered: "Delivered to agent",
  scheduled: "Planned",
  applied: "Applied",
  needs_input: "Needs your reply",
  declined: "Cannot apply",
};

const TERMINAL_RUN_STATUSES = new Set(["success", "degraded", "failed", "halted", "cancelled"]);

export function RunInstructionPanel({
  runId, runStatus, cancelling = false,
}: {
  runId: string;
  runStatus: string;
  cancelling?: boolean;
}) {
  const { data = [], error, mutate } = useSWR<RunInstructionDTO[]>(
    runId ? `/api/runs/${encodeURIComponent(runId)}/instructions` : null,
    { refreshInterval: 3000 },
  );
  const [body, setBody] = useState("");
  const [busy, setBusy] = useState(false);
  const [postError, setPostError] = useState("");
  const closed = cancelling || TERMINAL_RUN_STATUSES.has(runStatus);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!body.trim() || busy || closed) return;
    setBusy(true);
    setPostError("");
    try {
      const created = await api<RunInstructionDTO>(
        `/runs/${encodeURIComponent(runId)}/instructions`,
        {
          method: "POST",
          body: JSON.stringify({
            body: body.trim(),
          }),
        },
      );
      setBody("");
      await mutate((current) => [...(current ?? []), created], false);
      void mutate();
    } catch (err) {
      setPostError(String((err as Error).message || err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Bezel id="instructions" className="mt-5 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <Kicker strong className="!text-sm">User instructions</Kicker>
      </div>
      <p className="mt-1 text-sm text-slate-400">
        In case you want to change this run or guide a future iteration, tell the system here. The agent will decide when to apply it and report back.
      </p>
      {!closed ? (
        <form onSubmit={submit} className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-end">
          <textarea
            value={body}
            onChange={(event) => setBody(event.target.value)}
            maxLength={4000}
            rows={3}
            placeholder="What should the agent consider or change?"
            className="min-w-0 flex-1 resize-y rounded-lg border border-hair bg-canvas px-3 py-2 text-sm text-ink placeholder:text-slate-600 focus:border-brass-500/50"
          />
          <button type="submit" disabled={busy || !body.trim()} className="btn btn-brass shrink-0 disabled:opacity-50">
            <MessageSquare size={14} /> {busy ? "Posting…" : "Post instruction"}
          </button>
        </form>
      ) : (
        <p className="mt-3 text-xs text-slate-500">This Run is closing or has ended. Start a new Run to give new instructions.</p>
      )}
      {postError && <p className="mt-2 text-sm text-coral-300">{postError}</p>}
      {error && <p className="mt-2 text-sm text-coral-300">Could not load instructions.</p>}
      {data.length > 0 && (
        <div className="mt-4 space-y-2 border-t border-hair pt-4">
          {[...data].reverse().map((item) => {
            const unresolvedAtEnd = closed && ["queued", "delivered", "scheduled", "needs_input"].includes(item.status);
            return (
              <div key={item.id} className="rounded-lg border border-hair bg-canvas/60 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
                  <span>{fmtDate(item.created_at)}{item.source_ticket_id && (
                    <> · <Link className="text-brass-300 hover:underline" to={`/tickets/${item.source_ticket_id}`}>
                      {item.source_ticket_id}
                    </Link></>
                  )}</span>
                  <span className={item.status === "applied" ? "text-phosphor-300" : "text-brass-300"}>
                    {unresolvedAtEnd ? "Run ended before completion" : LABELS[item.status]}
                  </span>
                </div>
                <p className="mt-2 whitespace-pre-wrap text-sm text-ink">{item.body}</p>
                {item.agent_response && (
                  <div className="mt-3 border-l-2 border-brass-500/50 pl-3">
                    <div className="font-mono text-2xs uppercase tracking-wider text-brass-300">Agent decision</div>
                    <p className="mt-1 whitespace-pre-wrap text-sm text-slate-300">{item.agent_response}</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Bezel>
  );
}
