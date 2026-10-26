import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { DialogOwner, useDialogFocus } from "../lib/dialog";
import type { ExecutionEventDTO, MidTrainingValidation } from "../lib/api";

type Measurement = MidTrainingValidation["measurements"][number];
const decisionLabel = (row: Measurement) => row.plan_completed ? "Plan completed"
  : row.decision?.action === "stop" ? "Stop"
  : row.decision?.action === "continue" ? `Continue → step ${row.decision.next_step?.toLocaleString()}`
  : "Awaiting decision";

function Reason({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const [overflow, setOverflow] = useState(false);
  const ref = useRef<HTMLParagraphElement>(null);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || expanded) return;
    const measure = () => setOverflow(element.scrollHeight > element.clientHeight + 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [text, expanded]);
  return <section><h3 className="mb-2 font-medium">Reason</h3>
    <p ref={ref} className={`whitespace-pre-wrap break-words text-slate-300 leading-6 ${expanded ? "" : "line-clamp-4"}`}>{text}</p>
    {(overflow || expanded) && <button type="button" aria-expanded={expanded} className="mt-2 text-xs text-sky-300 hover:underline"
      onClick={() => setExpanded(!expanded)}>{expanded ? "Show less" : "Show more"}</button>}
  </section>;
}

function ValidationDetails({ data, step, onStep, onClose }: {
  data: MidTrainingValidation; step?: number; onStep: (step: number) => void; onClose: () => void;
}) {
  const { id, ref } = useDialogFocus(true, onClose, 100);
  const selected = data.measurements.find((row) => row.step === step) ?? data.measurements.at(-1);
  return createPortal(<DialogOwner.Provider value={id}>
    <div className="fixed inset-0 z-[100] flex justify-end" onMouseDown={onClose}>
      <div className="absolute inset-0 bg-canvas/80 backdrop-blur-sm" />
      <div ref={ref} role="dialog" aria-modal="true" aria-label="Mid-training Validation details" tabIndex={-1}
        className="relative h-full w-full max-w-4xl overflow-y-auto border-l border-hair bg-panel p-6 shadow-2xl animate-validation-drawer"
        onMouseDown={(event) => event.stopPropagation()}>
        <div className="mb-5 flex items-center justify-between gap-3">
          <h2 className="text-base font-semibold">Mid-training Validation</h2>
          <button type="button" aria-label="Close details" onClick={onClose}><X size={18} /></button>
        </div>
        <div className="mb-5 flex flex-wrap gap-2" aria-label="Validation checkpoints">
          {data.measurements.map((row, index) => <button key={row.step} type="button"
            aria-pressed={row.step === selected?.step} onClick={() => onStep(row.step)}
            className={`rounded border px-3 py-2 text-xs ${row.step === selected?.step ? "border-sky-400 text-sky-300" : "border-hair text-dim"}`}>
            Check {index + 1} · Step {row.step.toLocaleString()}
          </button>)}
        </div>
        {selected && <div className="space-y-5 text-sm">
          <div className="flex flex-wrap gap-3 text-sky-300"><span>{decisionLabel(selected)}</span>
            <span>Suite average: {selected.score.toFixed(4)}</span>
            {selected.step === data.best_step && <span>Best checkpoint</span>}
          </div>
          <section><h3 className="mb-2 font-medium">Benchmarks · {Object.keys(selected.metrics).length}</h3>
            <table className="w-full text-left text-xs"><thead><tr className="border-b border-hair text-dim">
              <th className="py-2 font-medium">Benchmark</th><th className="py-2 text-right font-medium">Score</th>
            </tr></thead><tbody>{Object.entries(selected.metrics).map(([name, score]) => <tr key={name} className="border-b border-hair">
              <td className="py-2 pr-4 break-words">{name}</td><td className="py-2 text-right font-mono">{score.toFixed(4)}</td>
            </tr>)}</tbody></table>
          </section>
          <Reason key={selected.step} text={selected.decision?.reason || (selected.plan_completed
            ? "Full training plan completed; selected the best measured checkpoint." : "Decision not recorded yet.")} />
        </div>}
      </div>
    </div>
  </DialogOwner.Provider>, document.body);
}

export function MidTrainingValidationPanel({ data, events = [], activity, showExecutionProgress = true }: {
  showExecutionProgress?: boolean;
  data?: MidTrainingValidation | null; events?: ExecutionEventDTO[]; activity?: string | null;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const newestStep = data?.measurements.at(-1)?.step;
  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [newestStep]);
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState<number>();
  const progress = [...events].filter((event) => event.extras?.execution_purpose === "workload"
    && event.extras?.benchmark_name).sort((a, b) => a.ts.localeCompare(b.ts)).at(-1);
  const generating = activity === "Mid-training Validation · Inference";
  return <div>
    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">Mid-training Validation</h3>
      {!!data?.measurements.length && <button type="button" className="text-xs text-sky-300 hover:underline"
        onClick={() => { setStep(undefined); setOpen(true); }}>See more details</button>}
    </div>
    {showExecutionProgress && generating && progress && <div className="mb-3 text-xs text-sky-300">{String(progress.extras?.benchmark_name)} · {progress.current_step.toLocaleString()} / {progress.total_steps.toLocaleString()} responses</div>}
    {data && <>
      <div className="mb-3 flex flex-wrap gap-4 text-xs text-slate-300">
        <span>{data.complete ? "Selected checkpoint" : "Best checkpoint so far"}: step {data.best_step.toLocaleString()}</span>
        {data.next_step != null && <span>Next validation: step {data.next_step.toLocaleString()}</span>}
        <span>{data.direction === "max" ? "Higher" : "Lower"} is better</span>
      </div>
      <div ref={scrollRef} className="max-h-[27.5rem] overflow-auto" role="region" aria-label="Validation history; scroll up for earlier checks" tabIndex={0}><table className="w-full text-left text-xs">
        <thead className="sticky top-0 z-10 h-10 border-b border-slate-700 bg-panel text-slate-400"><tr>
          {["Step", "Progress", "Suite average", "Decision"].map((title) => <th key={title} className="px-3 py-2 font-medium whitespace-nowrap">{title}</th>)}
        </tr></thead><tbody>{data.measurements.map((row) => <tr key={row.step} className="h-10 border-b border-slate-800 hover:bg-sky-500/5 cursor-pointer"
          onClick={() => { setStep(row.step); setOpen(true); }}>
          <td className="px-3 py-2 whitespace-nowrap"><button type="button" className="text-sky-300 hover:underline" aria-label={`View validation at step ${row.step}`}
            onClick={(event) => { event.stopPropagation(); setStep(row.step); setOpen(true); }}>{row.step.toLocaleString()}</button>
            {row.step === data.best_step && <span className="ml-2 text-sky-300">Best</span>}</td>
          <td className="px-3 py-2">{(100 * row.step / data.planned_steps).toFixed(1)}%</td>
          <td className="px-3 py-2 font-mono">{row.score.toFixed(4)}</td>
          <td className="px-3 py-2 whitespace-nowrap">{decisionLabel(row)}</td>
        </tr>)}</tbody>
      </table></div>
      {open && <ValidationDetails data={data} step={step} onStep={setStep} onClose={() => setOpen(false)} />}
    </>}
  </div>;
}
