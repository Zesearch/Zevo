import { ChoiceField, requiredFieldStateCls, type RunSummaryValue } from "./RunInputs";

export type ModelSavePolicy = {
  weights: "hf" | "remote";
  hf_repo_id: string;
  hf_private: boolean;
  remote_dir: string;
};
export const defaultModelSavePolicy: ModelSavePolicy = {
  weights: "hf", hf_repo_id: "", hf_private: true, remote_dir: "",
};
export function modelSaveReady(value: ModelSavePolicy): boolean {
  return value.weights === "hf"
    ? /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+$/.test(value.hf_repo_id.trim())
    : value.remote_dir.trim().startsWith("/") && value.remote_dir.trim() !== "/"
      && !value.remote_dir.split("/").includes("..");
}
export function modelSaveSummaryValues(value: ModelSavePolicy): RunSummaryValue[] {
  const hf = value.weights === "hf";
  return [
    { name: "Final model destination", value: hf ? "Hugging Face" : "GPU machine", complete: true },
    { name: hf ? "Hugging Face repository" : "Directory on the GPU machine", value: (hf ? value.hf_repo_id : value.remote_dir).trim() || "Not set", complete: modelSaveReady(value) },
    ...(hf ? [{ name: "Repository visibility", value: value.hf_private ? "Private" : "Public", complete: true }] : []),
  ];
}

export function ModelSaveFields({ value, onChange }: {
  value: ModelSavePolicy; onChange: (value: ModelSavePolicy) => void;
}) {
  const fieldClass = "h-11 w-full rounded-md border border-hair bg-canvas p-2.5 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-brass-500/50 focus:outline-none";
  return <section className="space-y-3">
    <div className="grid grid-cols-1 items-start gap-3 sm:grid-cols-2">
      <ChoiceField label="Final model destination" value={value.weights}
        onChange={weights => onChange({ ...value, weights: weights as ModelSavePolicy["weights"] })}
        options={[["hf", "Upload to Hugging Face"], ["remote", "Save on the GPU machine"]]} />
      {value.weights === "hf" ? <label className="block">
        <span className="field-label mb-1.5 block">Hugging Face repository</span>
        <input aria-label="Hugging Face repository" value={value.hf_repo_id}
          onChange={e => onChange({ ...value, hf_repo_id: e.target.value })} placeholder="owner/model-name"
          spellCheck={false} className={`${fieldClass} ${requiredFieldStateCls(Boolean(value.hf_repo_id.trim()))}`} />
      </label> : <label className="block">
        <span className="field-label mb-1.5 block">Directory on the GPU machine</span>
        <input aria-label="Directory on the GPU machine" value={value.remote_dir}
          onChange={e => onChange({ ...value, remote_dir: e.target.value })} placeholder="/lustre/my-project/models"
          spellCheck={false} className={`${fieldClass} ${requiredFieldStateCls(Boolean(value.remote_dir.trim()))}`} />
      </label>}
    </div>
    {value.weights === "hf" ? <>
      <label className="flex items-center gap-2 text-xs text-slate-400">
        <input type="checkbox" checked={value.hf_private} onChange={e => onChange({ ...value, hf_private: e.target.checked })} /> Private repository
      </label>
      <p className="text-xs text-slate-500">Uses your Hugging Face write token from Settings. The repository is created if needed; existing repository visibility must match.</p>
    </> : <p className="text-xs text-slate-500">Saves in a run-specific subfolder. Use persistent storage on a cluster. A rented cloud machine is kept running to preserve the model and continues billing until you release it.</p>}
  </section>;
}
