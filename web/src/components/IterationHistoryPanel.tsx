import { Fragment } from "react";
import { InlineMarkdown } from "./Markdown";
import {
  Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { IterationHistoryEntry } from "../lib/api";
import { Kicker } from "./zevo/primitives";
import { isPercentageMetric, shortModel } from "../lib/format";

/**
 * Zevo model-improvement iteration views, driven by run.history (which Evaluation
 * creates from measured scores and the orchestrator enriches). Split into two pieces because the run
 * detail page places them in different columns:
 *   - IterationChart: score per iteration, baseline as the leftmost point,
 *     the champion iteration marked in brass;
 *   - ExplorationLog: what the system actually tried each round.
 */

// Horizontal room each iteration gets in the score chart before it starts
// scrolling instead of squeezing.
const PX_PER_POINT = 90;

/** Oldest → newest; a new model lineage's baseline precedes its candidate. */
function ordered(history: IterationHistoryEntry[]): IterationHistoryEntry[] {
  return [...history].sort((a, b) => {
    if (a.iteration !== b.iteration) return a.iteration - b.iteration;
    if (a.source === b.source) return 0;
    return a.source === "baseline" ? -1 : 1;
  });
}

/** One line of the chart's key: the stroke as it is actually drawn, then the
 *  name. Hand-rolled rather than recharts' own `<Legend>`, which collects its
 *  entries from the chart's children and so listed the gradient Area as a third
 *  series called "test" (it shares the test line's dataKey, and `legendType`
 *  "none" is ignored in this version). */
function LegendKey({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-2 text-sm text-slate-400">
      <svg width="20" height="6" aria-hidden="true" className="shrink-0">
        <line
          x1="0" y1="3" x2="20" y2="3" stroke={color} strokeWidth="2"
          strokeDasharray={dashed ? "5 3" : undefined}
        />
      </svg>
      {label}
    </span>
  );
}

export function IterationChart({
  history, metricDirection = "max", metric = "",
}: {
  history: IterationHistoryEntry[];
  metricDirection?: "max" | "min";
  metric?: string;
}) {
  if (!history || history.length === 0) return null;
  const sorted = ordered(history);

  // `score` is the validation score the loop steered by; `test_score` is
  // the held-out measurement the harness took of the same iteration. Two series,
  // each meaning exactly one thing: the distance between them IS the story — how
  // much of the climb was the loop fitting its own set.
  //
  // The bold line used to fall back to validation wherever the held-out number
  // had not landed yet, which drew one continuous curve out of two different
  // measurements. With a legend naming them that is no longer a small
  // imprecision, it is a mislabel: it would read "test" over a point measured on
  // validation. A gap in the test line says the honest thing instead.
  // Plot the presentation unit directly. Ratio metrics use percentage points;
  // loss/perplexity/error magnitudes retain their own numeric scale.
  const percentage = isPercentageMetric(metric);
  const displayScale = percentage ? 100 : 1;
  const scoreOf = (n: unknown) =>
    typeof n === "number" && Number.isFinite(n) ? n * displayScale : undefined;
  const chart = sorted
    .filter((e) => scoreOf(e.score) !== undefined || scoreOf(e.test_score) !== undefined)
    .map((e) => ({
      label: e.source === "baseline" ? `base ${e.iteration}` : `it ${e.iteration}`,
      test: scoreOf(e.test_score),
      validation: scoreOf(e.score),
    }));
  if (chart.length === 0) return null;

  const nums = chart.flatMap((c) =>
    [c.test, c.validation].filter((n): n is number => typeof n === "number"));
  // The champion is the iteration that won on VALIDATION — the same definition
  // the backend uses for `champion_test_score` (routers/runs.py) and the same
  // one `zevo.engine.run.runner._record_holdout_score` is careful to keep.
  //
  // It used to be `Math.max(...tests)`, which crowns whichever round scored
  // highest on the HELD-OUT set. That is selecting on the test set one level
  // up: the run is allowed exactly one look at that number per iteration, and
  // picking the maximum over those looks turns it back into a set that was
  // optimised against. It also disagreed with the headline figure on the same
  // page, so a run whose validation winner was not its test maximum showed the
  // big number against one iteration and the word "champion" against another.
  const bestIdx = chart.reduce((bi, c, i) => {
    if (c.label === "base" || c.validation === undefined) return bi;
    if (bi < 0 || chart[bi]?.validation === undefined) return i;
    return metricDirection === "min"
      ? c.validation < (chart[bi].validation as number) ? i : bi
      : c.validation > (chart[bi].validation as number) ? i : bi;
  }, -1);
  // Use a clean percentage-point scale. For non-negative metrics such as
  // accuracy, start at zero and end on the next 1/2/5 × 10^n tick: a maximum
  // of 46 therefore ends at 50, not at an arbitrary padded 58. Metrics that
  // genuinely contain negative values still receive a symmetric-quality
  // lower bound instead of being clipped at zero.
  const lo = Math.min(...nums);
  const hi = Math.max(...nums);
  const scaleFloor = lo >= 0 ? 0 : lo;
  const span = hi - scaleFloor;
  const roughStep = span > 0
    ? span / 5
    : percentage ? 1 : Math.max(Math.abs(hi) / 5, 0.001);
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const fraction = roughStep / magnitude;
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  const tickStep = niceFraction * magnitude;
  const yMin = lo >= 0 ? 0 : Math.floor(lo / tickStep) * tickStep;
  const roundedMax = Math.ceil(hi / tickStep) * tickStep;
  const yMax = roundedMax > yMin ? roundedMax : yMin + tickStep;
  const yTicks = Array.from(
    { length: Math.round((yMax - yMin) / tickStep) + 1 },
    (_, index) => yMin + index * tickStep,
  );
  const chartValue = (value: number) => percentage
    ? `${value.toFixed(1)}%`
    : Number(value.toFixed(3)).toString();

  // Both series carry their number. Which SIDE of the point it goes on is
  // decided per point: away from the other series. A fixed "above" put the two
  // labels between the two points wherever the lines were close, and on a
  // baseline where validation sits above test they landed on top of each other.
  const valueLabel = (key: "test" | "validation", fill: string) => (p: any) => {
    const row = chart[p.index];
    const v = row?.[key];
    // An empty <g/>, not null: recharts types a label renderer as returning an
    // element, and an iteration missing this series still gets one call.
    if (typeof v !== "number") return <g key={p.index} />;
    const other = key === "test" ? row.validation : row.test;
    const tied = typeof other === "number" && Math.abs(v - other) < 1e-9;
    const above = tied
      ? key === "validation"
      : typeof other !== "number" || v > other;
    // Higher score above, lower score below. Equal dots split in opposite
    // directions. The x-axis tick is pushed farther down below the plot, so a
    // low baseline label remains readable without reversing this ordering.
    const dy = above ? -16 : 20;
    return (
      <text
        key={p.index}
        x={p.x}
        y={p.y + dy}
        fill={fill}
        fontSize={14}
        fontFamily="Spline Sans Mono"
        textAnchor="middle"
      >
        {chartValue(v)}
      </text>
    );
  };

  return (
    <div className="flex h-full flex-col">
      {/* Same kicker scale as the readout cards beside it. */}
      {/* The key sits in the header rather than inside the plot: the chart pans
          sideways once a run has more iterations than fit, and a legend drawn on
          the canvas scrolls away with it.

          One centred row under the title. Stacked it cost two lines of height
          between the title and the plot, which is the space the chart wants. */}
      <Kicker strong className="!text-sm">Score per iteration</Kicker>
      <div className="mt-1.5 flex items-center justify-center gap-5">
        <LegendKey color="#66788A" label="validation" dashed />
        <LegendKey color="#4FD1B5" label="test" />
      </div>
      {/* Scrolls sideways once the run has more iterations than fit: the inner
          track claims PX_PER_POINT per point, so a short run still fills the
          card (min-width under 100%) and a long one pans instead of cramming
          every label into the same width. */}
      <div className="mt-2 min-h-[200px] flex-1 overflow-x-auto">
      <div className="h-full" style={{ minWidth: chart.length * PX_PER_POINT }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chart} margin={{ top: 38, right: 12, left: -8, bottom: 4 }}>
          <defs>
            <linearGradient id="iter-acc" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4FD1B5" stopOpacity={0.32} />
              <stop offset="100%" stopColor="#4FD1B5" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1C2731" />
          <XAxis
            dataKey="label" stroke="#66788A" fontSize={15} tick={{ fill: "#B2C0CC" }}
            padding={{ left: 42, right: 22 }} tickMargin={22} height={50}
          />
          <YAxis
            stroke="#66788A" fontSize={15} tick={{ fill: "#B2C0CC" }}
            domain={[yMin, yMax]} allowDecimals={!percentage}
            ticks={yTicks}
            tickFormatter={(v) => percentage
              ? String(Math.round(Number(v)))
              : Number(Number(v).toFixed(3)).toString()}
            unit={percentage ? "%" : ""} width={62}
          />
          {/* The hover readout is the only place both numbers appear side by
              side with their names, so it is worth reading at a glance rather
              than squinting at 12px. */}
          <Tooltip
            contentStyle={{
              background: "#10161E", border: "1px solid #1C2731",
              borderRadius: 10, fontSize: 15, padding: "8px 12px",
            }}
            labelStyle={{ color: "#8B9CAB", fontSize: 14, marginBottom: 4 }}
            itemStyle={{ fontSize: 15, padding: "1px 0" }}
            formatter={(v: number, n: string) => [chartValue(v), n]}
          />
          {/* The shading under the test line, not a series of its own — it
              shares `test`'s dataKey, so left alone the legend lists "test"
              twice and the tooltip reports the same number twice. */}
          <Area
            type="monotone" dataKey="test" stroke="none" fill="url(#iter-acc)"
            isAnimationActive={false} connectNulls tooltipType="none" legendType="none"
            // Shading has no data points of its own. `dot={false}` is not
            // enough: recharts forces a dot on any series holding exactly ONE
            // value so a single point stays visible, which on a baseline-only
            // run put a third circle under the test dot. A renderer that draws
            // nothing is the only way to say no.
            dot={(p: any) => <g key={p.index} />} activeDot={false}
          />
          <Line
            type="monotone" dataKey="validation" name="validation" stroke="#66788A" strokeWidth={1.25}
            strokeDasharray="5 3" isAnimationActive={false} connectNulls
            // Same size and weight as an ordinary test dot, in grey. They are
            // two readings of equal standing, so they get the same mark; what
            // separates them is hue and the dashed stroke, not importance. At
            // r=2 it read as an afterthought next to the r=4 test dot.
            dot={{ r: 7, fill: "#66788A", stroke: "none" }}
            // Hovering must not SHRINK the point. recharts' default activeDot
            // is r=4, which was fine when the resting dot was smaller than that
            // and reads as a glitch now that it is 7.
            activeDot={{ r: 9, fill: "#66788A", stroke: "#0A0E13", strokeWidth: 2 }}
            label={valueLabel("validation", "#8B9CAB")}
          />
          <Line
            type="monotone" dataKey="test" name="test" stroke="#4FD1B5" strokeWidth={2}
            isAnimationActive={false} connectNulls
            activeDot={{ r: 9, fill: "#4FD1B5", stroke: "#0A0E13", strokeWidth: 2 }}
            label={valueLabel("test", "#A9E9DA")}
            dot={(p: any) => {
              // An iteration whose held-out measurement has not landed yet has
              // no point on this line. `connectNulls` bridges the line across
              // it; without this it would still draw a dot, at whatever
              // coordinate recharts made up for a missing value.
              if (typeof p.payload?.test !== "number") return <g key={p.index} />;
              const isBest = p.index === bestIdx;
              return (
                <circle
                  key={p.index}
                  cx={p.cx}
                  cy={p.cy}
                  // Green, and the same size as the validation dot beside it.
                  // The champion used to be a bigger BRASS circle, which read
                  // as a third thing on a chart whose legend says test is
                  // green. It keeps a halo instead: same mark, same hue, still
                  // findable.
                  r={7}
                  fill="#4FD1B5"
                  stroke="none"
                  style={isBest ? { filter: "drop-shadow(0 0 7px rgba(79,209,181,0.95))" } : undefined}
                />
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
      </div>
      </div>
    </div>
  );
}

/** A labelled chip. Model and method get their own hues so you can scan a long
 *  timeline for "which model" and "which method" without reading every word. */
function Chip({ label, value, tone }: { label: string; value: string; tone: "model" | "method" }) {
  const cls =
    tone === "model"
      ? "border-lilac-500/30 bg-lilac-500/10 text-lilac-300"
      : "border-skyx-500/30 bg-skyx-500/10 text-skyx-300";
  return (
    <span className={`rounded border px-1.5 py-0.5 font-mono text-[13px] ${cls}`}>
      <span className="opacity-60">{label}</span> {value}
    </span>
  );
}

/** Run Journal is a compact experiment summary, not the Data artifact audit log.
 *  Surface the first canonical method id; free-text audit steps stay solely on
 *  the Data recipe artifact. */
function journalDataMethod(methods: string[] | undefined): string {
  const primary = methods?.[0]?.trim();
  if (!primary) return "";
  return primary.replaceAll("_", " ");
}

/** The run's decisions as a vertical timeline: a rail down the left, one node
 *  per iteration. Always open, capped height, scrolls — a 20-iteration run
 *  would otherwise push everything below it off the page. */
/** One round's record: four labelled rows, one fact each. */
const JOURNAL_ROWS = [
  { key: "action", label: "ACTION" },
  { key: "result", label: "RESULT" },
  { key: "analysis", label: "ANALYSIS" },
  { key: "next", label: "NEXT" },
] as const;

function JournalEntry({ entry }: { entry: IterationHistoryEntry }) {
  const rows = JOURNAL_ROWS
    .map((r) => ({ ...r, text: String(entry[r.key] ?? "").trim() }))
    .filter((r) => r.text);
  if (rows.length === 0) {
    return <div className="mt-2 text-xs text-dim">Awaiting four-part Journal entry…</div>;
  }
  return (
    <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[15px] leading-relaxed">
      {rows.map((r) => (
        <Fragment key={r.key}>
          <dt className="select-none pt-px font-mono text-[12px] uppercase tracking-[0.14em] text-dim">
            {r.label}
          </dt>
          {/* ANALYSIS is the row a reader argues with, so it carries the
              brighter ink; the other three are what happened. */}
          <dd className={r.key === "analysis" ? "text-ink" : "text-slate-200"}>
            <InlineMarkdown>{r.text}</InlineMarkdown>
          </dd>
        </Fragment>
      ))}
    </dl>
  );
}

export function RunJournal({
  history, metricDirection = "max",
}: {
  history: IterationHistoryEntry[];
  metricDirection?: "max" | "min";
}) {
  if (!history || history.length === 0) {
    return <div className="py-8 text-center text-sm text-slate-500">No iterations recorded yet.</div>;
  }
  const rows = ordered(history);
  // Champion on the VALIDATION score — the same question the chart asks, and
  // the same definition the backend's `champion_test_score` uses. The loop is
  // allowed to choose on validation and is measured on test; choosing on test
  // instead (which this did) makes the held-out number a fitted one too.
  const num = (n: unknown) => (typeof n === "number" && Number.isFinite(n) ? n : undefined);
  const bestKey = (e: IterationHistoryEntry) => num(e.score);
  const candidates = rows.filter((e) => e.source !== "baseline").map(bestKey).filter((v): v is number => v !== undefined);
  const best: number | undefined = candidates.length
    ? (metricDirection === "min" ? Math.min(...candidates) : Math.max(...candidates))
    : undefined;

  return (
    // `pl-2` is not decoration: the rail markers sit a few pixels OUTSIDE the
    // list's own left border, and a scroll box clips both axes — without room
    // on the left every dot on the timeline gets sliced in half.
    <div className="max-h-[26rem] overflow-y-auto pl-2 pr-2">
      <ol className="relative border-l border-hair pl-6">
        {rows.map((e, i) => {
          const isBaseline = e.source === "baseline";
          const isBest = bestKey(e) !== undefined && bestKey(e) === best && !isBaseline;
          return (
            <li key={i} className={i === rows.length - 1 ? "relative" : "relative pb-5"}>
              {/* Node on the rail. Brass marks the champion iteration. */}
              <span
                className={`absolute -left-[1.9rem] top-1 h-2.5 w-2.5 rounded-full border-2 border-panel ${
                  isBest ? "bg-brass-400 shadow-glow-brass" : isBaseline ? "bg-slate-500" : "bg-phosphor-400"
                }`}
              />
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-xs text-slate-300">
                  {isBaseline ? `iteration ${e.iteration} · baseline` : `iteration ${e.iteration}`}
                  {isBest && <span className="ml-2 text-2xs text-brass-300">champion</span>}
                </span>
              </div>

              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {e.base_model && <Chip label="model" value={shortModel(e.base_model)} tone="model" />}
                {!isBaseline && journalDataMethod(e.method_ids) ? (
                  <Chip label="data" value={journalDataMethod(e.method_ids)} tone="method" />
                ) : null}
                {!isBaseline && e.training_method && <Chip label="train" value={e.training_method} tone="method" />}
                {isBaseline && (
                  <span className="rounded border border-hair px-1.5 py-0.5 font-mono text-[13px] text-slate-500">
                    no training, just the base model
                  </span>
                )}
              </div>

              {/* Four labelled rows, one fact each, aligned down the column so a
                  reader can scan ONE label across every round — every ACTION, or
                  every ANALYSIS — instead of re-reading four paragraphs to find
                  where the run turned. */}
              <JournalEntry entry={e} />
            </li>
          );
        })}
      </ol>
    </div>
  );
}
