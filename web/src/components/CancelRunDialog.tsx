import { useState } from "react";
import { createPortal } from "react-dom";
import { StopCircle, X } from "lucide-react";
import { api } from "../lib/api";
import { toast } from "../lib/toast";

/**
 * Cancel a run and say what happens to its trained weights.
 *
 * The three choices are the server's CancelWeightsPolicy. `download` and `hf`
 * leave the run open while the daemon copies the champion checkpoint off the
 * GPU box, then close it and release the box; `discard` tears everything down
 * at once. The fields under each choice are exactly what that policy needs.
 */

type Weights = "download" | "hf" | "discard";

const CHOICES: { id: Weights; title: string; body: string }[] = [
  { id: "download", title: "Download to local",
    body: "Copy the champion checkpoint off the box into a folder on the Zevo server, register it in Saved models, then release the GPU." },
  { id: "hf", title: "Upload to Hugging Face",
    body: "Copy the checkpoint off the box, push it to a Hub repo with the HF token from Settings, then release the GPU." },
  { id: "discard", title: "Discard weights",
    body: "Kill everything and release the GPU now. Checkpoints that only exist on a rented box are lost." },
];

export function CancelRunDialog({ runId, defaultDir, onClose, onCancelled }: {
  runId: string;
  /** Where `download` lands when the folder is left empty. */
  defaultDir: string;
  onClose: () => void;
  onCancelled: () => void;
}) {
  const [weights, setWeights] = useState<Weights>("download");
  const [localDir, setLocalDir] = useState("");
  const [repoId, setRepoId] = useState("");
  const [isPrivate, setIsPrivate] = useState(true);
  const [busy, setBusy] = useState(false);

  const repoOk = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+$/.test(repoId.trim());
  const dirOk = !localDir.trim() || localDir.trim().startsWith("/");
  const ready = weights === "discard" || (dirOk && (weights !== "hf" || repoOk));

  async function submit() {
    setBusy(true);
    try {
      const body = weights === "discard"
        ? { weights }
        : { weights, local_dir: localDir.trim(), hf_repo_id: repoId.trim(), hf_private: isPrivate };
      const out = await api<{ status: string; note?: string }>(
        `/runs/${encodeURIComponent(runId)}/cancel`,
        { method: "POST", body: JSON.stringify(body) },
      );
      toast(out.status === "cancelling"
        ? "Cancel requested; keeping the weights before the GPU is released."
        : "Run cancelled.", "info");
      onCancelled();
      onClose();
    } catch (e) {
      toast(e instanceof Error ? e.message : "cancel failed", "error");
    } finally {
      setBusy(false);
    }
  }

  // Portaled to <body>: mounted where the cancel button lives (the page
  // header's `right` slot) the fixed overlay is trapped in that ancestor's
  // stacking context and the hero panel below intercepts the clicks.
  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center px-4" onMouseDown={onClose}>
      <div className="absolute inset-0 bg-canvas/80 backdrop-blur-sm" />
      <div className="bezel relative w-full max-w-lg p-6 animate-zevo-in" onMouseDown={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-ink">Cancel this run</h2>
            <p className="mt-1 text-sm text-slate-400">
              Every in-flight ticket is killed and the orchestrator stops. What should happen to the trained weights?
            </p>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-ink" aria-label="close"><X size={16} /></button>
        </div>

        <div className="mt-5 flex flex-col gap-2">
          {CHOICES.map((c) => (
            <label key={c.id} className={`flex cursor-pointer gap-3 rounded-bezel border p-3 transition ${
              weights === c.id ? "border-brass-500/50 bg-brass-500/10" : "bezel-flat hover:border-brass-500/30"
            }`}>
              <input type="radio" name="weights" className="mt-1 accent-brass-500" checked={weights === c.id}
                onChange={() => setWeights(c.id)} />
              <span>
                <span className="block text-sm font-semibold text-ink">{c.title}</span>
                <span className="mt-0.5 block text-xs leading-snug text-slate-400">{c.body}</span>
              </span>
            </label>
          ))}
        </div>

        {weights !== "discard" && (
          <div className="mt-4 flex flex-col gap-3">
            <label className="block">
              <span className="field-label">Folder on the Zevo server</span>
              <input value={localDir} onChange={(e) => setLocalDir(e.target.value)} placeholder={defaultDir}
                className="mt-1 w-full rounded-lg border border-hair bg-raised px-3 py-2 font-mono text-sm text-ink placeholder:text-slate-600 focus:outline-none" />
              <span className="mt-1 block text-xs text-slate-500">
                {dirOk ? "Absolute path. Empty uses the run's own models folder shown above." : "Must be an absolute path."}
              </span>
            </label>
            {weights === "hf" && (
              <>
                <label className="block">
                  <span className="field-label">Hub repo</span>
                  <input value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="owner/model-name"
                    className="mt-1 w-full rounded-lg border border-hair bg-raised px-3 py-2 font-mono text-sm text-ink placeholder:text-slate-600 focus:outline-none" />
                  <span className="mt-1 block text-xs text-slate-500">
                    Created if missing, using the HF token saved in Settings.
                  </span>
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input type="checkbox" className="accent-brass-500" checked={isPrivate} onChange={(e) => setIsPrivate(e.target.checked)} />
                  Private repo
                </label>
              </>
            )}
          </div>
        )}

        <div className="mt-6 flex justify-end gap-2">
          <button onClick={onClose} className="btn">keep running</button>
          <button onClick={submit} disabled={busy || !ready}
            className="btn border-coral-500/40 text-coral-300 hover:border-coral-500/70 hover:text-coral-200 disabled:opacity-50">
            <StopCircle size={14} /> {busy ? "cancelling…" : "cancel run"}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
