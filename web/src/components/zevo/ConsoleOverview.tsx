// ZEVO's system-level outcome summary. Runs stay on the Runs page; this panel
// groups completed test outcomes by task because two tasks may use different
// metrics or opposite target directions.
import { useMemo, useState } from "react";
import useSWR from "swr";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { RunSummary } from "../../lib/api";
import { fmtMetric, isPercentageMetric } from "../../lib/format";
import { Bezel, Kicker, Note, Readout } from "./primitives";

type Point = {
  taskName: string;
  baseline: number;
  evolved: number;
  improvement: number;
  metric: string;
  metricDirection: "max" | "min";
};

type TaskSummary = {
  key: string;
  taskName: string;
  metric: string;
  metricDirection: "max" | "min";
  runCount: number;
  averageBaseline: number;
  averageEvolved: number;
  averageImprovement: number;
  bestScore: number;
};

function pointFrom(run: RunSummary): Point | null {
  const baseline = run.baseline_test_score;
  const evolved = run.champion_test_score;
  if (baseline == null || evolved == null) return null;

  return {
    taskName: run.task_name,
    baseline,
    evolved,
    improvement: run.metric_direction === "min" ? baseline - evolved : evolved - baseline,
    metric: run.metric,
    metricDirection: run.metric_direction,
  };
}

function summarizeByTask(points: Point[]): TaskSummary[] {
  const groups = new Map<string, Point[]>();
  for (const point of points) {
    // Metric and target are part of the group boundary. If a task definition
    // changes later, unlike outcomes are not silently averaged together.
    const key = `${point.taskName}\u0000${point.metric}\u0000${point.metricDirection}`;
    const group = groups.get(key) ?? [];
    group.push(point);
    groups.set(key, group);
  }

  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  return [...groups.entries()]
    .map(([key, outcomes]) => {
      const first = outcomes[0];
      const evolved = outcomes.map((outcome) => outcome.evolved);
      return {
        key,
        taskName: first.taskName,
        metric: first.metric,
        metricDirection: first.metricDirection,
        runCount: outcomes.length,
        averageBaseline: mean(outcomes.map((outcome) => outcome.baseline)),
        averageEvolved: mean(evolved),
        averageImprovement: mean(outcomes.map((outcome) => outcome.improvement)),
        bestScore: first.metricDirection === "min" ? Math.min(...evolved) : Math.max(...evolved),
      };
    })
    .sort((a, b) => a.taskName.localeCompare(b.taskName) || a.metric.localeCompare(b.metric));
}

function scoreNumber(value: number, metric: string): string {
  return isPercentageMetric(metric)
    ? (value * 100).toFixed(1)
    : value.toFixed(3);
}

function delta(value: number, metric: string): string {
  return `${value >= 0 ? "+" : ""}${scoreNumber(value, metric)}`;
}

export function ConsoleOverview() {
  const { data: runs = [], isLoading } = useSWR<RunSummary[]>("/api/runs?limit=500", {
    refreshInterval: 10000,
  });
  const [selectedKey, setSelectedKey] = useState("");
  const [tasksOpen, setTasksOpen] = useState(true);

  const summaries = useMemo(
    () => summarizeByTask(
      runs
        .filter((run) => (
          (run.status === "success" || run.status === "degraded")
          && Boolean(run.registry_version_tag)
        ))
        .map(pointFrom)
        .filter((point): point is Point => point !== null),
    ),
    [runs],
  );
  const selected = summaries.find((summary) => summary.key === selectedKey) ?? summaries[0] ?? null;
  const improved = summaries.filter((summary) => summary.averageImprovement > 0).length;

  const title = (
    <div className="flex items-center gap-2">
      <Kicker strong className="!text-sm">Model improvement by Zevo</Kicker>
      <Note size={14}>
        Improvement is the test-score change from Zero, the baseline model, to models evolved by Zevo. Tasks stay separate because their metrics and targets may differ.
      </Note>
    </div>
  );

  if (isLoading) {
    return (
      <Bezel className="p-6">
        {title}
        <div className="py-20 text-center text-sm text-slate-500">Reconstructing outcomes…</div>
      </Bezel>
    );
  }

  if (!selected) {
    return (
      <Bezel className="p-6">
        {title}
        <div className="py-20 text-center text-sm text-slate-500">
          <p>No completed task has a baseline and evolved test score yet.</p>
          <p className="mt-1.5 text-2xs text-slate-600">
            This summary appears after Zevo saves an evolved model with its test result.
          </p>
        </div>
      </Bezel>
    );
  }

  const percentageMetric = isPercentageMetric(selected.metric);
  const scoreUnit = percentageMetric ? "%" : undefined;
  const improvementUnit = percentageMetric ? "%" : undefined;
  const outcomeTextTone = selected.averageImprovement > 0
    ? "text-phosphor-300"
    : selected.averageImprovement < 0 ? "text-coral-300" : "text-slate-200";

  return (
    <Bezel className="p-6">
      {title}

      <div className="relative mt-5 grid grid-cols-1 gap-x-5 gap-y-5 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <span className="pointer-events-none absolute bottom-0 left-64 top-0 hidden w-px bg-hair lg:block" />

        <div className="lg:pr-5">
          <div className="[&_.kicker]:!text-[0.8rem]">
            <Readout
              label="# Tasks Improved"
              value={`${improved} / ${summaries.length}`}
              tone={improved === summaries.length ? "phosphor" : improved > 0 ? "brass" : "coral"}
              size="statLg"
              strongLabel
            />
          </div>
        </div>

        <div className="flex min-w-0 flex-col justify-between px-1 lg:pl-1">
          <div className="truncate font-mono text-lg text-ink" title={selected.taskName}>
            {selected.taskName}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3 pb-1 font-mono text-xs text-ink">
            <span>
              <span className="mr-2 text-slate-400">Metric</span>{fmtMetric(selected.metric)}
            </span>
            <span>·</span>
            <span>
              <span className="mr-2 text-slate-400">Target</span>{selected.metricDirection === "min" ? "Min" : "Max"}
            </span>
            <span>·</span>
            <span>
              {selected.runCount} run{selected.runCount === 1 ? "" : "s"}
            </span>
          </div>
        </div>

        <div className="lg:pr-5">
          <button
            onClick={() => setTasksOpen((open) => !open)}
            className="flex w-full items-center justify-between gap-3 text-left"
          >
            <Kicker strong className="!text-[0.8rem]">All Evaluated Tasks</Kicker>
            {tasksOpen
              ? <ChevronDown size={14} className="text-slate-500" />
              : <ChevronRight size={14} className="text-slate-500" />}
          </button>

          {tasksOpen && (
            <div className="mt-3 max-h-[18rem] overflow-y-auto pr-1">
              {summaries.map((summary, index) => {
                const active = summary.key === selected.key;
                return (
                  <button
                    key={summary.key}
                    onClick={() => setSelectedKey(summary.key)}
                    className={`flex w-full items-center gap-3 border-l py-2 pl-3 pr-2 text-left transition ${
                      active
                        ? "border-brass-400 text-ink"
                        : "border-hair text-slate-300 hover:border-slate-500 hover:text-ink"
                    }`}
                  >
                    <span className="w-5 shrink-0 font-mono text-2xs text-slate-500">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="block min-w-0 flex-1 truncate font-mono text-xs" title={summary.taskName}>
                      {summary.taskName}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="min-w-0 px-1 lg:pl-1">
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
            <div className="flex min-h-[12rem] flex-col rounded-lg bg-white/[0.035] px-5 py-5">
              <Kicker strong className="!text-sm">Average Improvement</Kicker>
              <div className="flex flex-1 items-center justify-center">
                <div className="flex items-baseline gap-1.5">
                  <span className={`readout text-5xl font-semibold ${outcomeTextTone}`}>
                    {delta(selected.averageImprovement, selected.metric)}
                  </span>
                  {improvementUnit && <span className="readout text-sm text-slate-500">{improvementUnit}</span>}
                </div>
              </div>
            </div>

            <div className="flex min-h-[12rem] flex-col rounded-lg bg-white/[0.035] px-5 py-5">
              <Kicker strong className="!text-sm">Average Test Score</Kicker>
              <div className="flex flex-1 items-center justify-center">
                <div className="flex items-baseline gap-1.5">
                  <span className="readout text-5xl font-semibold text-phosphor-300">
                    {scoreNumber(selected.averageEvolved, selected.metric)}
                  </span>
                  {scoreUnit && <span className="readout text-sm text-slate-500">{scoreUnit}</span>}
                </div>
              </div>
            </div>
            <div className="flex min-h-[12rem] flex-col rounded-lg bg-white/[0.035] px-5 py-5">
              <Kicker strong className="!text-sm">
                {selected.metricDirection === "min" ? "Lowest Test Score" : "Highest Test Score"}
              </Kicker>
              <div className="flex flex-1 items-center justify-center">
                <div className="flex items-baseline gap-1.5">
                  <span className="readout text-5xl font-semibold text-phosphor-300">
                    {scoreNumber(selected.bestScore, selected.metric)}
                  </span>
                  {scoreUnit && <span className="readout text-sm text-slate-500">{scoreUnit}</span>}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Bezel>
  );
}
