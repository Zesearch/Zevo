import { useLayoutEffect, useRef } from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";
export function LaunchNavigation({ index, total, onBack, onNext, canNext = true, reviewAction }: {
  index: number; total: number; onBack: () => void; onNext: () => void; canNext?: boolean; reviewAction?: React.ReactNode;
}) {
  const footer = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const element = footer.current;
    if (!element) return;
    // Step changes establish a new baseline; entering Review must stay at the top.
    let height = element.getBoundingClientRect().height;
    const observer = new ResizeObserver(() => {
      const next = element.getBoundingClientRect().height;
      if (next > height + 1) element.scrollIntoView({ block: "end", behavior: "auto" });
      height = next;
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [index]);
  return <div ref={footer} className={`mt-8 w-full pb-6 ${index === total - 1 ? "launch-navigation-review" : "flex items-center justify-between gap-5"}`}>
    <button type="button" className="launch-back" disabled={index === 0} onClick={onBack}><ArrowLeft size={15} />Back</button>
    {index === total - 1 ? reviewAction : <button type="button" className="launch-next" disabled={!canNext} onClick={onNext}>{index === total - 2 ? "Review checklist" : "Continue"}<ArrowRight size={15} /></button>}
  </div>;
}
