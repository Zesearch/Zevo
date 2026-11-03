import { ArrowLeft, ArrowRight } from "lucide-react";
export function LaunchNavigation({ index, total, onBack, onNext, canNext = true, reviewAction }: {
  index: number; total: number; onBack: () => void; onNext: () => void; canNext?: boolean; reviewAction?: React.ReactNode;
}) {
  return <div className="mt-8 flex w-full items-center justify-between gap-5 pb-6">
    <button type="button" className="launch-back" disabled={index === 0} onClick={onBack}><ArrowLeft size={15} />Back</button>
    {index === total - 1 ? reviewAction : <button type="button" className="launch-next" disabled={!canNext} onClick={onNext}>{index === total - 2 ? "Review checklist" : "Continue"}<ArrowRight size={15} /></button>}
  </div>;
}
