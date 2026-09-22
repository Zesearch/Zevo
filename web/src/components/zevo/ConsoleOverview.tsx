// ZEVO's system-level outcome summary. Runs stay on the Runs page; this panel
// groups completed test outcomes by task because two tasks may use different
// metrics or opposite target directions.
import { useState } from "react";
import useSWR from "swr";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { RunStatistics } from "../../lib/api";
import { fmtMetric, isPercentageMetric } from "../../lib/format";
import { Bezel, Kicker, Note, Readout } from "./primitives";

function scoreNumber(value: number, metric: string): string {
  return isPercentageMetric(metric)
    ? (value * 100).toFixed(1)
    : value.toFixed(3);
}

function delta(value: number, metric: string): string {
  return `${value >= 0 ? "+" : ""}${scoreNumber(value, metric)}`;
}

export function ConsoleOverview() {
  const { data: stats, isLoading, error, mutate } = useSWR<RunStatistics>("/api/runs/statistics", {
    refreshInterval: 10000,
  });
  const [selectedKey, setSelectedKey] = useState("");
  const [tasksOpen, setTasksOpen] = useState(true);

  const summaries = stats?.improvements ?? [];
  const selected = summaries.find((summary) => summary.key === selectedKey) ?? summaries[0] ?? null;
  const improved = summaries.filter((summary) => summary.averageImprovement > 0).length;

  const title = (
    <div className="flex items-center gap-2">
      <Kicker strong className="!text-sm">Model improvement</Kicker>
      <Note size={14}>
        Improvement is the test-score change from Zero, the baseline model, to models evolved by Zevo. Tasks stay separate because their metrics and targets may differ.
      </Note>
    </div>
  );

  if (error) return <Bezel className="p-6"><p role="alert">Could not load model outcomes.</p><button className="btn mt-3" onClick={() => void mutate()}>Retry</button></Bezel>;

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
