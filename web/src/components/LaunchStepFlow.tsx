import { LaunchNavigation } from "./LaunchNavigation";
import { LaunchContents, LaunchStepHeading } from "./LaunchContents";
import { useEffect, useRef, useState, type ReactNode } from "react";

type Step = { label: string; group: "Required" | "Optional" | "Review"; content: ReactNode; complete: boolean; ready?: boolean };

/** Navigation changes only the visible panel; draft controls remain mounted. */
export function LaunchStepFlow({ steps, onReviewChange, reviewAction }: { steps: Step[]; onReviewChange?: (review: boolean) => void; reviewAction?: ReactNode }) {
  const [active, setActive] = useState(0);
  useEffect(() => { onReviewChange?.(active === steps.length - 1); }, [active, steps.length, onReviewChange]);
  const [visited, setVisited] = useState<number[]>([0]);
  const ready = (index: number) => steps[index].ready ?? (steps[index].group === "Optional" || steps[index].complete);
  const checklistReady = steps[steps.length - 1].complete && steps.slice(0, -1).every((_, index) => visited.includes(index) && ready(index));
  useEffect(() => { if (active === steps.length - 1 && !checklistReady) setActive(steps.length - 2); }, [active, checklistReady, steps.length]);
  const panel = useRef<HTMLDivElement>(null);
  const go = (index: number) => {
    if (index === steps.length - 1 && !checklistReady) return;
    setVisited(value => value.includes(index) ? value : [...value, index]);
    setActive(index);
    if (panel.current) panel.current.scrollTop = 0;
  };
  return <div className="grid min-h-0 flex-1 grid-rows-[auto_minmax(0,1fr)] gap-5 md:grid-cols-[210px_minmax(0,1fr)] md:grid-rows-[minmax(0,1fr)] md:gap-x-6 lg:gap-x-10">
    <LaunchContents steps={steps.filter((_, index) => index !== steps.length - 1 || checklistReady).map((step, index) => ({ ...step, complete: visited.includes(index) && ready(index) }))} active={active} onSelect={go} />
    <div ref={panel} className="min-h-0 overflow-y-auto overscroll-contain pr-2">
      <LaunchStepHeading key={active} label={steps[active].label} />
      {steps.map((step, index) => <section key={step.label} hidden={active !== index} aria-label={step.label} className="space-y-4 motion-safe:animate-zevo-in">{step.content}</section>)}
      <LaunchNavigation index={active} total={steps.length} onBack={() => go(active - 1)} onNext={() => go(active + 1)} canNext={active === steps.length - 2 ? checklistReady : ready(active)} reviewAction={reviewAction} />
    </div>
  </div>;
}
