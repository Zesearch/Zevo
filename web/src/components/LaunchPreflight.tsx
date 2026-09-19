import { useState } from "react";
import { api } from "../lib/api";
import type { RunInputValues } from "./RunInputs";

type Result = { status: "ready" | "risky" | "blocked"; summary: string; items: Array<{ code: string; severity: string; message: string; hint: string }> };
type Draft = { mode?: string; user_request?: unknown; gpu_provider?: string; cloud_backend?: string; ssh_host_id?: string; num_gpus?: number; generation_backend?: string; customizations?: unknown };

export function useLaunchPreflight() {
  const [review, setReview] = useState<{ key: string; result: Result } | null>(null);
  async function check(body: Draft) {
    const payload = {
      mode: body.mode ?? "full_pipeline", user_request: body.user_request,
      gpu_provider: body.gpu_provider, cloud_backend: body.cloud_backend,
      ssh_host_id: body.ssh_host_id, num_gpus: body.num_gpus,
      generation_backend: body.generation_backend, customizations: body.customizations,
    };
    const key = JSON.stringify(payload);
    const result = await api<Result>("/preflight", { method: "POST", body: key });
    const acknowledged = review?.key === key && JSON.stringify(review.result.items) === JSON.stringify(result.items);
    setReview({ key, result });
    return result.status === "ready" || (result.status === "risky" && acknowledged);
  }
  return { check, panel: review ? <section aria-label="Launch checks" role="status" className="my-4 rounded border border-hair bg-raised p-4 text-sm">
    <h3 className="font-semibold">{review.result.status === "blocked" ? "Resolve these launch checks" : review.result.status === "risky" ? "Review before launching" : "Configuration checks passed"}</h3>
    <p className="mt-1 text-xs text-slate-400">Checks verify configuration and local assets. Provider connectivity, available capacity, and final Auto evaluation are verified during setup.</p>
    <ul className="mt-3 space-y-2">{review.result.items.filter((item) => item.severity !== "info").map((item) => <li key={item.code}><strong>{item.severity === "blocker" ? "Required: " : "Review: "}</strong>{item.message} {item.hint}</li>)}</ul>
    {review.result.status === "risky" && <p className="mt-3">Click start again to acknowledge these notes and launch. Changing configuration requires a fresh review.</p>}
  </section> : null };
}

export function LaunchLimitsSummary({ inputs }: { inputs: RunInputValues }) {
  const limit = (value: string, unit = "") => Number(value) > 0 ? `${value}${unit}` : "Unlimited";
  return <aside aria-label="Execution limits" className="rounded border border-hair bg-raised p-3 text-xs leading-relaxed">
    <strong>Execution limits:</strong> {limit(inputs.iterations)} rounds · {Number(inputs.budget) > 0 ? `$${inputs.budget}` : "Unlimited spend"} · {limit(inputs.timeLimitHours, " hours")} · {limit(inputs.numGpus)} GPUs · {inputs.queueWaitHours.trim() || "48"} hours queue wait.
    <p className="mt-1 text-slate-400">Rounds, spend, active runtime, and GPU count start without a user cap. Slurm queue wait defaults to 48 hours. Budgets include estimated agent and rented GPU costs.</p>
  </aside>;
}
