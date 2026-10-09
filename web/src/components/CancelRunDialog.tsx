import { ModelSaveFields, defaultModelSavePolicy, modelSaveReady, type ModelSavePolicy } from "./ModelSaveFields";
import { useState } from "react";
import { createPortal } from "react-dom";
import { StopCircle, X } from "lucide-react";
import { DialogOwner, useDialogFocus } from "../lib/dialog";
import { api } from "../lib/api";
import { toast } from "../lib/toast";

/** Save the champion to the same destinations as Registry, or cancel anyway. */

export function CancelRunDialog({ runId, modelSavePolicy, onClose, onCancelled }: {
  runId: string;
  modelSavePolicy?: ModelSavePolicy;
  onClose: () => void;
  onCancelled: () => void;
}) {
  const { id, ref } = useDialogFocus(true, onClose, 100);
  const [save, setSave] = useState<ModelSavePolicy>(modelSavePolicy || { ...defaultModelSavePolicy });
  const [discard, setDiscard] = useState(false);
  const [busy, setBusy] = useState(false);
  const ready = discard || modelSaveReady(save);

  async function submit() {
    setBusy(true);
    try {
      const body = discard ? { weights: "discard" } : { ...save, rescue_timeout_seconds: 3600, attempt_timeout_seconds: 3600 };
      const out = await api<{ status: string; note?: string }>(
        `/runs/${encodeURIComponent(runId)}/cancel`,
        { method: "POST", body: JSON.stringify(body) },
      );
      toast(out.status === "cancelling"
        ? "Cancel requested; saving the model to your selected destination."
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
    <DialogOwner.Provider value={id}><div className="fixed inset-0 z-[100] flex items-center justify-center px-4" onMouseDown={onClose}>
      <div className="absolute inset-0 bg-canvas/80 backdrop-blur-sm" />
      <div ref={ref} role="dialog" aria-modal="true" aria-label="Checkpoint recovery options" tabIndex={-1} className="bezel relative max-h-[90vh] overflow-y-auto w-full max-w-lg p-6 animate-zevo-in" onMouseDown={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-ink">Cancel this run</h2>
            <p className="mt-1 text-sm text-slate-400">
              Every in-flight ticket is killed and the orchestrator stops. What should happen to the trained weights?
            </p>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-ink" aria-label="close"><X size={16} /></button>
        </div>

        <div className="mt-5 space-y-3">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={discard} onChange={e => setDiscard(e.target.checked)} />
            Cancel anyway — skip model saving
          </label>
          {discard ? <p className="text-sm text-coral-300">Stops work and releases compute immediately. Weights stored only on a rented machine may be lost.</p>
            : <ModelSaveFields value={save} onChange={setSave} />}
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <button onClick={onClose} className="btn">keep running</button>
          <button onClick={submit} disabled={busy || !ready}
            className="btn border-coral-500/40 text-coral-300 hover:border-coral-500/70 hover:text-coral-200 disabled:opacity-50">
            <StopCircle size={14} /> {busy ? "cancelling…" : discard ? "cancel anyway" : "save and cancel"}
          </button>
        </div>
      </div>
    </div></DialogOwner.Provider>,
    document.body,
  );
}
