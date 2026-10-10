import { useState } from "react";
import { Cpu, Clock3, Wallet, Repeat2, HardDrive, CloudUpload } from "lucide-react";
import { modelSaveReady, type ModelSavePolicy } from "./ModelSaveFields";
import { api } from "../lib/api";
import type { RunInputValues } from "./RunInputs";

type Result = { status: "ready" | "risky" | "blocked"; summary: string; items: Array<{ code: string; severity: string; message: string; hint: string }> };
type Draft = { mode?: string; user_request?: unknown; gpu_provider?: string; gpu_allocation_mode?: string; cloud_backend?: string; ssh_host_id?: string; num_gpus?: number; generation_backend?: string; customizations?: unknown };

export function useLaunchPreflight() {
  const [review, setReview] = useState<{ key: string; result: Result } | null>(null);
  async function check(body: Draft) {
    const payload = {
      mode: body.mode ?? "full_pipeline", user_request: body.user_request,
      gpu_allocation_mode: body.gpu_allocation_mode,
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

export function LaunchLimitsSummary({ inputs, modelSave }: { inputs: RunInputValues; modelSave: ModelSavePolicy }) {
  const limit = (value: string, unit = "") => Number(value) > 0 ? `${value}${unit}` : "Unlimited";
  const limits = [
    { label: "Rounds", value: limit(inputs.iterations), icon: Repeat2 },
    { label: "Spend", value: Number(inputs.budget) > 0 ? `$${inputs.budget}` : "Unlimited", icon: Wallet },
    { label: "Active runtime", value: limit(inputs.timeLimitHours, " hours"), icon: Clock3 },
    { label: "GPUs", value: limit(inputs.numGpus), icon: Cpu },
  ];
  const isHf = modelSave.weights === "hf";
  const DestinationIcon = isHf ? CloudUpload : HardDrive;
  const destination = (isHf ? modelSave.hf_repo_id : modelSave.remote_dir).trim();
  const ready = modelSaveReady(modelSave);
  return <aside aria-label="Run summary" className="rounded-xl bg-gradient-to-br from-raised to-canvas px-5 py-4 ring-1 ring-inset ring-white/[0.06]">
    <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-5">
      {limits.map(({ label, value, icon: Icon }) => <div key={label} className="min-w-0">
        <dt className="mb-2 flex items-center gap-2 text-sm font-medium text-brass-300"><Icon size={14} />{label}</dt>
        <dd className="text-sm font-medium tabular-nums text-slate-100">{value}</dd>
      </div>)}
      <div className="col-span-2 min-w-0 sm:col-span-1">
        <dt className="mb-2 flex items-center gap-2 text-sm font-medium text-brass-300"><DestinationIcon size={14} />Final model</dt>
        <dd>
          <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-slate-100">
            {isHf ? "Hugging Face" : "GPU machine"}
            {isHf && <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] font-normal text-slate-400">{modelSave.hf_private ? "Private" : "Public"}</span>}
          </div>
          {destination && <p title={destination} className={`mt-1 truncate font-mono text-[11px] ${ready ? "text-slate-400" : "text-brass-300"}`}>
            {destination}
          </p>}
          {destination && !ready && <p className="mt-1 text-[11px] text-brass-300">Complete destination in Others</p>}
        </dd>
      </div>
    </dl>
  </aside>;
}
