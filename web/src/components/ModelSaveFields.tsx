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
export function ModelSaveFields({ value, onChange }: {
  value: ModelSavePolicy; onChange: (value: ModelSavePolicy) => void;
}) {
  return <section className="space-y-3 rounded-lg border border-hair p-3">
    <label className="block">
      <span className="field-label">Final model destination</span>
      <select aria-label="Final model destination" value={value.weights}
        onChange={e => onChange({ ...value, weights: e.target.value as ModelSavePolicy["weights"] })}
        className="mt-1 w-full rounded-lg border border-hair bg-raised px-3 py-2 text-sm text-ink">
        <option value="hf">Upload to Hugging Face</option>
        <option value="remote">Save on the GPU machine</option>
      </select>
    </label>
    {value.weights === "hf" ? <>
      <label className="block"><span className="field-label">Hugging Face repository</span>
        <input aria-label="Hugging Face repository" value={value.hf_repo_id}
          onChange={e => onChange({ ...value, hf_repo_id: e.target.value })} placeholder="owner/model-name"
          className="mt-1 w-full rounded-lg border border-hair bg-raised px-3 py-2 text-sm text-ink" />
      </label>
      <label className="flex items-center gap-2 text-sm text-slate-300">
        <input type="checkbox" checked={value.hf_private} onChange={e => onChange({ ...value, hf_private: e.target.checked })} /> Private repository
      </label>
      <p className="text-xs text-slate-400">Uses your Hugging Face write token from Settings. The repository is created if needed; existing repository visibility must match.</p>
    </> : <>
      <label className="block"><span className="field-label">Directory on the GPU machine</span>
        <input aria-label="Directory on the GPU machine" value={value.remote_dir}
          onChange={e => onChange({ ...value, remote_dir: e.target.value })} placeholder="/lustre/my-project/models"
          className="mt-1 w-full rounded-lg border border-hair bg-raised px-3 py-2 text-sm text-ink" />
      </label>
      <p className="text-xs text-slate-400">Saves in a run-specific subfolder. Use persistent storage on a cluster. A rented cloud machine is kept running to preserve the model and continues billing until you release it.</p>
    </>}
  </section>;
}
