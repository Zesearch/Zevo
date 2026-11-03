import { ClearableInput } from "./ClearableField";
export function SaveRunPrompt({ newTask, askTaskName, save, onSave, taskName, onTaskName, settingName, onSettingName, nameTaken, busy, onBack, onLaunch, feedback }: {
  newTask: boolean; askTaskName: boolean; save: boolean; onSave: (save: boolean) => void; taskName: string; onTaskName: (name: string) => void;
  settingName: string; onSettingName: (name: string) => void; nameTaken: boolean; busy: boolean; onBack: () => void; onLaunch: () => void; feedback?: React.ReactNode;
}) {
  return <section aria-label="Save configuration" className="my-auto w-full max-w-3xl self-center py-10 motion-safe:animate-zevo-in">
    <h3 className="font-display text-3xl font-semibold tracking-tight text-ink md:text-4xl">Keep this for next time?</h3>
    <p className="mb-8 mt-4 text-sm text-slate-400">{newTask ? "Save your Task and Setting to reuse them in future runs." : "Save this Setting to reuse it in future runs."}</p>
    <label className="flex items-center gap-3 text-sm text-slate-200"><input type="checkbox" checked={save} disabled={busy} onChange={event => onSave(event.target.checked)} />{newTask ? "Save Task and Setting" : "Save Setting"}</label>
    {save && <div className="mt-6 grid gap-5 sm:grid-cols-2">
      {askTaskName && <label><span className="field-label mb-2 block">Task name</span><ClearableInput aria-label="Task name to save" value={taskName} onChange={event => onTaskName(event.target.value)} placeholder="Task name" disabled={busy} className={`w-full border border-hair ${!taskName.trim() ? "launch-field-missing" : ""}`} /></label>}
      <label><span className="field-label mb-2 block">Setting name</span><ClearableInput aria-label="Setting name to save" value={settingName} onChange={event => onSettingName(event.target.value)} placeholder="Setting name" disabled={busy} className={`w-full border border-hair ${!settingName.trim() || nameTaken ? "launch-field-missing" : ""}`} />{nameTaken && <span className="mt-2 block text-xs text-coral-300">That name is already taken.</span>}</label>
    </div>}
    {feedback}
    <div className="mt-10 flex items-center justify-between gap-6">
      <button type="button" className="launch-back" disabled={busy} onClick={onBack}>← Back to checklist</button>
      <button type="button" className="launch-next" disabled={busy || (save && (nameTaken || !settingName.trim() || (askTaskName && !taskName.trim())))} onClick={onLaunch}>{busy ? "Starting…" : save ? "Save & launch →" : "Launch without saving →"}</button>
    </div>
  </section>;
}
