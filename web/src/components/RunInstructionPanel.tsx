import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import useSWR from "swr";
import { MessageSquare } from "lucide-react";

import { api, type RunInstructionDTO } from "../lib/api";
import { fmtDate } from "../lib/format";
import { Bezel, Kicker } from "./zevo/primitives";

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
    const instruction = body.trim();
    const optimisticId = `local-${Date.now()}`;
    const optimistic: RunInstructionDTO = {
      id: optimisticId,
      run_id: runId,
      source_ticket_id: null,
      body: instruction,
      status: "queued",
      agent_response: "",
      activity_status: "reviewing",
      activity_agent: "",
      target_ticket_id: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    setBody("");
    await mutate((current) => [...(current ?? []), optimistic], false);
    try {
      const created = await api<RunInstructionDTO>(
        `/runs/${encodeURIComponent(runId)}/instructions`,
        {
          method: "POST",
          body: JSON.stringify({
            body: instruction,
          }),
        },
      );
      await mutate(
        (current) => (current ?? []).map((item) => item.id === optimisticId ? created : item),
        false,
      );
      void mutate();
    } catch (err) {
      await mutate(
        (current) => (current ?? []).filter((item) => item.id !== optimisticId),
        false,
      );
      setBody(instruction);
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
        In case you want to change this run or guide a future iteration, tell the system here. New actions pause while the orchestrator decides when to apply it.
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
        <div className="mt-4 space-y-4 border-t border-hair pt-4">
          {data.map((item) => {
            const unresolvedAtEnd = closed && ["queued", "delivered", "scheduled", "needs_input"].includes(item.status);
            const actor = item.activity_agent
              ? item.activity_agent.charAt(0).toUpperCase() + item.activity_agent.slice(1)
              : "System";
            const activityText: Record<RunInstructionDTO["activity_status"], string> = {
              reviewing: "System is reviewing your instruction…",
              waiting: `${actor} will apply your instruction after the current work stops…`,
              applying: `${actor} is applying your instruction…`,
              scheduled: "The system scheduled this for a later safe point.",
              applied: "Instruction applied.",
              needs_input: "The system needs more information from you.",
              declined: "The system could not apply this instruction.",
              needs_attention: `${actor} could not finish applying this instruction.`,
            };
            const active = ["reviewing", "waiting", "applying"].includes(item.activity_status);
            return (
              <div key={item.id} className="space-y-2">
                <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm border border-brass-500/30 bg-brass-500/10 px-4 py-3">
                  <p className="whitespace-pre-wrap text-sm text-ink">{item.body}</p>
                  <div className="mt-2 text-right text-2xs text-slate-500">
                    You · {fmtDate(item.created_at)}{item.source_ticket_id && (
                      <> · <Link className="text-brass-300 hover:underline" to={`/tickets/${item.source_ticket_id}`}>{item.source_ticket_id}</Link></>
                    )}
                  </div>
                </div>
                <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-hair bg-canvas/70 px-4 py-3">
                  <div className="flex items-center gap-2 text-xs text-brass-300">
                    {active && <span className="lamp lamp-running" />}
                    <span>{unresolvedAtEnd ? "Run ended before this instruction was resolved." : activityText[item.activity_status]}</span>
                  </div>
                  {item.agent_response && (
                    <p className="mt-2 whitespace-pre-wrap text-sm text-slate-300">{item.agent_response}</p>
                  )}
                  {item.target_ticket_id && (
                    <div className="mt-2 text-2xs text-slate-500">
                      {actor} · <Link className="text-brass-300 hover:underline" to={`/tickets/${item.target_ticket_id}`}>{item.target_ticket_id}</Link>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Bezel>
  );
}
