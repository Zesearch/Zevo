import { useCallback, useEffect, useRef, useState } from "react";


export type RunSetupProgress = {
  status: "waiting" | "active" | "complete" | "failed";
  phase: string;
  completed: number;
  total: number;
  label: string;
  run_id?: string;
  error?: string;
};


const INITIAL_PROGRESS: RunSetupProgress = {
  status: "waiting",
  phase: "checking",
  completed: 0,
  total: 0,
  label: "Configuration checks",
};


export function useRunSetupProgress() {
  const [progress, setProgress] = useState<RunSetupProgress | null>(null);
  const activeId = useRef("");

  const stop = useCallback(() => {
    activeId.current = "";
  }, []);

  useEffect(() => stop, [stop]);

  const begin = useCallback((phase = "preparing") => {
    stop();
    const setupId = crypto.randomUUID();
    activeId.current = setupId;
    setProgress({ ...INITIAL_PROGRESS, phase, label: phase === "checking" ? "Configuration checks" : phase === "finalizing" ? "Creating Run" : "Preparing run" });
    return setupId;
  }, [stop]);

  const wait = useCallback(async (setupId: string): Promise<RunSetupProgress> => {
    while (activeId.current === setupId) {
      let next: RunSetupProgress | null = null;
      try {
        const response = await fetch(`/api/run-setups/${encodeURIComponent(setupId)}`);
        if (response.ok && activeId.current === setupId) {
          next = await response.json() as RunSetupProgress;
        }
      } catch {
        // A transient progress-poll failure must not cancel server-side setup.
      }
      if (next && activeId.current === setupId) {
        setProgress(next);
        if (next.status === "complete") {
          if (!next.run_id) throw new Error("Run setup completed without a Run id.");
          return next;
        }
        if (next.status === "failed") {
          throw new Error(next.error || next.label || "Run setup failed.");
        }
      }
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    throw new Error("Run setup was cancelled.");
  }, []);

  return { progress, begin, wait, stop };
}


export function RunSetupProgressView({ progress }: { progress: RunSetupProgress }) {
  const total = Math.max(0, progress.total || 0);
  const completed = Math.min(total, Math.max(0, progress.completed || 0));
  const percent = progress.status === "complete" ? 100 : total > 0 ? Math.round((completed / total) * 100) : 30;
  const current = total > 0 ? Math.min(total, completed + 1) : 0;

  let title = progress.phase === "checking" ? "Configuration checks" : "Preparing run";
  if (progress.phase === "benchmarks") {
    title = completed >= total
      ? `Test data ready · ${total}/${total}`
      : `Preparing Test data · ${current}/${total}`;
  } else if (progress.phase === "validation") {
    title = completed >= total
      ? `Validation ready · ${total}/${total}`
      : `Preparing Validation · ${current}/${total}`;
  } else if (progress.phase === "finalizing") {
    title = "Creating Run";
  } else if (progress.phase === "complete") {
    title = "Run started";
  } else if (progress.phase === "failed" || progress.status === "failed") {
    title = "Run setup stopped";
  }

  return (
    <div className="w-full min-w-0" aria-live="polite" title={progress.label}>
      <div className="mb-3 flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className={`text-sm font-medium ${progress.status === "failed" ? "text-coral-300" : "text-ink"}`}>{title}</p>
        </div>
        {total > 0 && <span className="shrink-0 text-sm tabular-nums text-slate-400">{percent}%</span>}
      </div>
      <div role="progressbar" aria-label={title} aria-valuemin={0} aria-valuemax={100} aria-valuenow={total > 0 || progress.status === "complete" ? percent : undefined} aria-valuetext={progress.label || title} className="h-1.5 overflow-hidden rounded-full bg-raised">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${
            progress.status === "failed" ? "bg-coral-400" : "bg-brass-400"
          } ${
            total === 0 && progress.status !== "complete" && progress.status !== "failed" ? "motion-safe:animate-pulse" : ""
          }`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
