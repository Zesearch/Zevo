import { useEffect, useRef } from "react";
import { Check, ChevronRight } from "lucide-react";

export type LaunchStep = { label: string; complete: boolean; group?: string };

export function LaunchContents({ steps, active, onSelect }: { steps: LaunchStep[]; active: number; onSelect: (index: number) => void }) {
  const current = useRef<HTMLButtonElement>(null);
  useEffect(() => { current.current?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" }); }, [active]);
  return <nav aria-label="Run setup steps" className="min-h-0 md:-ml-6 md:overflow-y-auto md:pr-1">
    <label className="flex items-center gap-3 md:hidden">
      <span className="shrink-0 text-xs text-brass-300">Step {active + 1} / {steps.length}</span>
      <select aria-label="Current setup step" className="min-w-0 flex-1 rounded-lg border border-hair bg-canvas px-3 py-2 text-sm text-ink" value={active} onChange={event => onSelect(Number(event.target.value))}>
        {steps.map((step, index) => <option key={step.label} value={index}>{step.complete && step.group !== "Review" ? "✓ " : ""}{step.label}</option>)}
      </select>
    </label>
    <div className="relative hidden md:block">
      <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 top-0 h-[76px] rounded-xl bg-gradient-to-r from-brass-500/[0.06] to-transparent transition-transform duration-500 ease-out motion-reduce:transition-none" style={{ transform: `translateY(${active * 76}px)` }} />
      <ol>{steps.map((step, index) => <li key={step.label}>
        <button ref={active === index ? current : undefined} type="button" aria-current={active === index ? "step" : undefined} aria-label={step.group === "Review" ? step.label : `${index + 1}. ${step.label}${step.complete ? " — complete" : ""}`} onClick={() => onSelect(index)}
          className={`relative flex h-[76px] w-full items-center gap-2 rounded-xl px-2 text-left transition-colors duration-300 focus-visible:outline focus-visible:outline-brass-400 ${active === index ? "text-[#d9c5a3]" : "text-slate-500 hover:text-slate-200"}`}>
          <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] transition-all duration-300 motion-reduce:transition-none ${step.complete ? "text-emerald-300" : active === index ? "text-brass-200" : "text-slate-600"}`}>
            {step.group === "Review" ? null : step.complete ? <Check size={12} /> : String(index + 1).padStart(2, "0")}
          </span>
          <span className="min-w-0 flex-1"><span className="block text-[10px] uppercase tracking-[0.16em] opacity-60">{step.group}</span><span className="mt-1 block whitespace-nowrap text-[18px]">{step.label}</span></span>
          {active === index && <ChevronRight size={12} className="text-brass-300 motion-safe:animate-zevo-in" />}
        </button>
      </li>)}</ol>
    </div>
  </nav>;
}

const prompts: Record<string, [string, string]> = {
  "Basic info": ["Let’s start with the essentials.", "Give this run a name and tell us what you want to achieve."],
  "Test setup": ["How will you measure success?", "Define the benchmarks for this run."],
  "Training": ["How would you like to train?", "Add your training preferences, or continue with the defaults."],
  "Validation setup": ["Set up your validation.", "By default, constructed from Test suite: 20% per eligible set, minimum 200 rows. Or add your own validation sets."],
  "Others": ["A few finishing touches.", "Adjust resources and limits if you need to."],
  "Checklist": ["Ready for a final look?", "Review your configuration before starting the run."],
  "Evaluation": ["What should evaluation focus on?", "Add guidance to help choose the right benchmarks."],
  "Agent": ["Who should take this on?", "Choose the agent for this task."],
  "Attachments": ["Anything else to include?", "Attach supporting files, or continue to review."],
};
export function LaunchStepHeading({ label }: { label: string }) {
  const [title, description] = prompts[label] || [label, ""];
  return <header className="mb-8 motion-safe:animate-zevo-in">
    <h3 className="font-display text-2xl font-semibold tracking-tight text-ink lg:text-3xl">{title}</h3>
    <p className="mt-3 text-sm leading-relaxed text-slate-400">{description}</p>
  </header>;
}
