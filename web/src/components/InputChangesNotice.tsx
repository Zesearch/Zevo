import { useState } from "react";
import { createPortal } from "react-dom";
import useSWR from "swr";
import { X } from "lucide-react";
import { api, type RunDetail } from "../lib/api";
import { DialogOwner, useDialogFocus } from "../lib/dialog";

type DiffLine = { text: string; changed: boolean };
const displayName = (change: { kind: string; name: string }) => change.kind === "file" ? change.name.split("/").at(-1) : change.name;
type ChangeDetails = {
  kind: "task" | "file";
  name: string;
  status: "modified" | "deleted";
  note: string;
  fields: Array<{ field: string; before: unknown; after: unknown; before_lines?: DiffLine[]; after_lines?: DiffLine[] }>;
};

function Value({ value, side, lines }: { value: unknown; side: "before" | "after"; lines?: DiffLine[] }) {
  const text = value == null ? "Not present" : typeof value === "string" ? value || "(empty)" : JSON.stringify(value, null, 2);
  return <div className={`min-w-0 p-3 ${side === "before" ? "bg-coral-500/5" : "bg-emerald-500/5"}`}>
    <div className="mb-2 text-xs font-semibold text-slate-400">{side === "before" ? "At Run launch" : "Current version"}</div>
    <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-slate-200">{lines?.length ? lines.map((line, index) => <span key={index} className={line.changed ? (side === "before" ? "bg-coral-500/20 text-coral-200" : "bg-emerald-500/20 text-emerald-200") : "text-slate-400"}>{line.text}</span>) : text}</pre>
  </div>;
}

function ChangesDialog({ runId, onClose }: { runId: string; onClose: () => void }) {
  const { id, ref } = useDialogFocus(true, onClose, 100);
  const { data, error, mutate } = useSWR<{ changes: ChangeDetails[] }>(
    `/runs/${encodeURIComponent(runId)}/input-changes`, api,
    { revalidateOnFocus: false },
  );
  return createPortal(<DialogOwner.Provider value={id}>
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" onMouseDown={onClose}>
      <div className="absolute inset-0 bg-canvas/80 backdrop-blur-sm" />
      <div ref={ref} role="dialog" aria-modal="true" aria-labelledby="input-changes-title" tabIndex={-1}
        className="bezel relative max-h-[90vh] w-full max-w-6xl overflow-y-auto p-6" onMouseDown={(event) => event.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div><h2 id="input-changes-title" className="text-lg font-semibold text-ink">Input changes</h2>
            <p className="mt-1 text-sm text-slate-400">Only changed fields are shown. This Run continues to use its launch snapshot.</p></div>
          <button className="btn" onClick={onClose} aria-label="Close input changes"><X size={18} /></button>
        </div>
        {!data && !error && <p role="status" className="mt-6 text-sm">Loading comparison…</p>}
        {error && <div role="alert" className="mt-6 text-sm text-coral-300">Could not load the comparison. <button className="underline" onClick={() => void mutate()}>Try again</button></div>}
        {data?.changes.length === 0 && <p className="mt-6 text-sm">The current inputs match this Run’s snapshot.</p>}
        {data?.changes.map((change) => <section key={`${change.kind}:${change.name}`} className="mt-6">
          <h3 className="font-semibold text-ink">{change.kind === "task" ? "Task" : "File"}: {displayName(change)} <span className="text-sm font-normal text-slate-400">({change.status})</span></h3>
          {change.note && <p className="mt-2 text-sm text-slate-400">{change.note}</p>}
          {change.fields.map((field) => <div key={field.field} className="mt-3 overflow-hidden rounded-bezel border border-hair">
            <h4 className="break-words border-b border-hair bg-raised px-3 py-2 text-sm font-medium">{field.field.replaceAll("task_objective", "Task objective").replaceAll("test_sets", "Test sets").replaceAll("inference_query", "Inference query").replaceAll("sample_submission", "Sample submission")}</h4>
            <div className="grid grid-cols-1 divide-y divide-hair md:grid-cols-2 md:divide-x md:divide-y-0">
              <Value value={field.before} side="before" lines={field.before_lines} /><Value value={field.after} side="after" lines={field.after_lines} />
            </div>
          </div>)}
        </section>)}
      </div>
    </div>
  </DialogOwner.Provider>, document.body);
}

export function InputChangesNotice({ run }: { run: RunDetail }) {
  const [open, setOpen] = useState(false);
  if (!run.input_changes?.length) return null;
  return <>
    <div role="status" className="mb-5 rounded-bezel border border-brass-500/30 bg-brass-500/10 px-4 py-3 text-sm text-brass-200">
      <strong>Inputs changed after this Run started.</strong>{" "}
      {run.input_changes.map((change) => `${change.kind === "task" ? "Task" : "File"} ${displayName(change)} was ${change.status}`).join("; ")}.{" "}
      This Run continues to use its launch snapshot; the current versions apply to future Runs.{" "}
      <button className="font-semibold underline underline-offset-4 hover:text-ink" onClick={() => setOpen(true)}>See details</button>
    </div>
    {open && <ChangesDialog runId={run.id} onClose={() => setOpen(false)} />}
  </>;
}
