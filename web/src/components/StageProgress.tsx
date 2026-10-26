import type { ReactNode } from "react";

/** Shared progress card for inference and evaluation. */
export function StageProgress({ children, completed, total, label }: {
  children: ReactNode; completed: number; total: number; label: string;
}) {
  const maximum = Math.max(1, total);
  const value = Math.min(maximum, Math.max(0, completed));
  return <div className="mb-3 rounded-bezel border border-brass-500/25 bg-brass-500/[0.05] px-3 py-2.5">
    {children}
    <div className="mt-2 h-1 overflow-hidden rounded-full bg-slate-800" role="progressbar"
      aria-label={label} aria-valuemin={0} aria-valuemax={maximum} aria-valuenow={value}>
      <div className="h-full rounded-full bg-brass-400 transition-[width] duration-500" style={{ width: `${value / maximum * 100}%` }} />
    </div>
  </div>;
}
