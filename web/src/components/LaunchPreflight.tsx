import { useState } from "react";
import { Cpu, Clock3, Wallet, Repeat2, HardDrive, CloudUpload, AlertTriangle, CheckCircle2 } from "lucide-react";
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
  const blocked = review?.result.status === "blocked";
  const ready = review?.result.status === "ready";
  const Icon = ready ? CheckCircle2 : AlertTriangle;
  return { check, panel: review ? <section aria-label="Launch checks" role={blocked ? "alert" : "status"} className={`rounded-lg border p-3 text-sm ${blocked ? "border-coral-500/40 bg-coral-500/10 text-coral-300" : ready ? "border-hair bg-raised text-slate-300" : "border-brass-500/40 bg-brass-500/10 text-brass-200"}`}>
    <h3 className="flex items-center gap-2 font-semibold"><Icon size={16} className="shrink-0" />{blocked ? "Run not started — check configuration" : ready ? "Configuration checks passed" : "Review before starting your run"}</h3>
    <div className="max-h-32 overflow-y-auto">
      <ul className="mt-2 space-y-2 text-xs leading-relaxed">{review.result.items.filter((item) => item.severity !== "info").map((item) => <li key={item.code}><strong>{item.severity === "blocker" ? "Required: " : "Review: "}</strong>{item.message} {item.hint}</li>)}</ul>
    </div>
    {review.result.status === "risky" && <p className="mt-2 text-xs font-medium">Run not started. Click start run again to accept these notes and launch.</p>}
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
  return <div className="sticky top-0 z-30 -mx-2 bg-panel px-2 pb-4" data-launch-summary-bar>
    <aside aria-label="Run summary" className="rounded-xl border border-hair bg-raised px-5 py-4 shadow-lg">
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
          <p title={destination || "Complete destination in Others"} className={`mt-1 truncate font-mono text-[11px] ${ready ? "text-slate-400" : "text-brass-300"}`}>
            {destination || "Not set"}
          </p>
          {destination && !ready && <p className="mt-1 text-[11px] text-brass-300">Complete destination in Others</p>}
        </dd>
      </div>
    </dl>
    </aside>
  </div>;
}
