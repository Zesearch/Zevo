import { useMemo } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { ExecutionEventDTO } from "../lib/api";

// Observatory palette: loss = brass, alert/val_loss = coral, live/healthy =
// phosphor, info = sky, extra = gold/lilac.
const METRIC_COLORS: Record<string, string> = {
  loss: "#E08D38",
  val_loss: "#F26F58",
  learning_rate: "#5FB0DD",
  grad_norm: "#4FD1B5",
  entropy: "#A98FD6",
  mean_token_accuracy: "#8FCDEB",
  num_tokens: "#E0B23C",
};

// Canonical metric names — different models/trainers emit the same thing under
// different keys (lr vs learning_rate, eval_loss vs val_loss). Normalise so the
// charts are consistent across drivers/models.
const METRIC_ALIAS: Record<string, string> = {
  lr: "learning_rate",
  // `val_loss` says what it is again: the trainer's `eval_dataset` is now the
  // run's VALIDATION set, reshaped for training by the data agent.
  eval_loss: "val_loss",
  validation_loss: "val_loss",
};
// What is NOT a metric. Everything else the trainer logged gets a chart —
// which is what the training agent is told will happen ("the UI draws one chart
// per key it sees, so just forward them all", playbook/agents/train/platform.md).
//
// It used to be a whitelist of seven SFT-family keys, so the promise held only
// for SFT: a `dpo` round forwards `rewards/chosen`, `rewards/margins`,
// `logps/*`, `logits/*` and a `grpo` round forwards `reward`, `kl`,
// `clip_ratio`, `completion_length` — every one of them dropped on the floor,
// leaving those methods with a single loss curve and no way to see whether the
// thing the method actually optimises was moving.
//
// A denylist inverts the failure: a new metric shows up unbidden (fine, that is
// the point) instead of vanishing unbidden (not fine, and invisible).
const NON_METRIC_KEYS = new Set([
  "step", "total", "epoch", "t",
  "train_runtime", "train_samples_per_second", "train_steps_per_second",
  "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
  "total_flos", "num_input_tokens_seen",
]);
// Keys that mark the END-OF-RUN summary __PROGRESS__ (HF trainer.train()'s final
// metrics). That row's aggregate num_tokens / loss don't belong on the per-step
// chart, so we skip it entirely.
// Metrics that cannot legitimately be negative, so a negative reading is the
// -1 sentinel rather than a measurement. `grad_norm` and the losses belong here
// too; entropy and mean_token_accuracy are also bounded below by zero.
const NON_NEGATIVE_METRICS = new Set([
  "learning_rate", "loss", "val_loss", "grad_norm", "num_tokens",
  "entropy", "mean_token_accuracy",
]);
const SUMMARY_KEYS = ["train_runtime", "total_flos", "train_samples_per_second", "train_steps_per_second"];
const PALETTE = ["#E08D38", "#5FB0DD", "#A98FD6", "#4FD1B5", "#F26F58", "#8FCDEB", "#E0B23C", "#F2C185"];

/** Compact scientific: drop the ".00" trailing zeros and the leading "+" on the
 *  exponent, so tiny lr / huge num_tokens labels stay narrow. */
function expo(v: number): string {
  const [m, e] = v.toExponential(2).split("e");
  return `${Number(m)}e${Number(e)}`;
}

function fmtMetric(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 0.001 || a >= 1e5) return expo(v);
  return Number(v.toFixed(4)).toString();
}

/** The same number, sized for an axis tick: FOUR DIGITS, never more.
 *
 *  A tick lives in a fixed-width gutter, so what has to be bounded is how many
 *  digits get printed — and bounding DECIMAL PLACES does not do that. Four
 *  decimals is four digits on `0.1234` and nine on `12345.6789`: one rule, one
 *  label that fits and one that is clipped and then reads as a different number.
 *  Significant digits bound it directly, and four of them is what an axis is
 *  read for — the shape of the curve, not the value, which the readout beside
 *  the title and the tooltip both give in full.
 *
 *  Counted as PRINTED digits, not significant ones: `0.9998` carries four
 *  significant digits and prints five, and it is the printed ones that take up
 *  the gutter. So drop precision a digit at a time until the label fits, and
 *  when no plain decimal does — the width is going to leading zeros or to
 *  trailing ones — name the magnitude with an exponent instead, which is both
 *  shorter and more honest about what was dropped.
 */
const digitCount = (s: string) => (s.match(/[0-9]/g) || []).length;

function fmtTick(v: number): string {
  if (v === 0) return "0";
  for (let p = 4; p >= 1; p--) {
    const s = Number(v.toPrecision(p)).toString();
    if (!s.includes("e") && digitCount(s) <= 4) return s;
  }
  for (let m = 2; m >= 0; m--) {
    const [mant, ex] = v.toExponential(m).split("e");
    const s = `${Number(mant)}e${Number(ex)}`;
    if (digitCount(s) <= 4) return s;
  }
  return expo(v);
}

/** Evenly spaced ticks across `0..total`, at multiples of one round interval.
 *
 *  The x-axis is fixed at the run's full length from the moment the total is
 *  known, so the chart does not redraw itself as points arrive: an auto-fitted
 *  axis reads 0..5 at step 5 and 0..400 at step 400, which changes the shape of
 *  the curve on every poll and makes one iteration impossible to compare against
 *  another. What adapts instead is the GAP — more steps, wider interval, same
 *  number of labels.
 *
 *  The gaps are equal by construction: one interval, repeated. No tick is added
 *  at the final step, because a 500-step run ticking 0/100/…/500 is regular
 *  while 0/100/…/492 is not, and the eye reads the short last gap as data.
 */
const TICK_TARGET = 6;

function axisTicks(total: number): number[] {
  if (!isFinite(total) || total <= 0) return [0];
  const mag = Math.pow(10, Math.floor(Math.log10(total / TICK_TARGET)));
  // Round intervals, and never below 1: a step is a whole number, so a fractional
  // interval labels positions that cannot occur — and accumulating 0.2 six times
  // lands on 0.6000000000000001, which is not even a uniform gap any more.
  const candidates = [...new Set(
    [1, 2, 5, 10, 20].map((m) => Math.max(1, Math.round(m * mag)))
  )];
  // Chosen by resulting COUNT, not by interval width. Taking the first interval
  // wide enough leaves a 1234-step run with three labels, where the next size
  // down gives seven — the target is a number of labels, so aim at it directly.
  const count = (s: number) => Math.floor(total / s) + 1;
  const step = candidates.reduce((best, c) =>
    Math.abs(count(c) - TICK_TARGET) < Math.abs(count(best) - TICK_TARGET) ? c : best
  );
  const out: number[] = [];
  for (let v = 0; v <= total; v += step) out.push(v);
  return out;
}

/** One small wandb-style panel: a single metric vs. step. */
function MetricChart({ metric, color, data, total }: {
  metric: string;
  color: string;
  data: Array<Record<string, number | undefined>>;
  /** The run's final step. The axis spans 0..total whatever has arrived. */
  total: number;
}) {
  const pts = data.filter((d) => typeof d[metric] === "number");
  if (pts.length === 0) return null;
  const last = pts[pts.length - 1][metric] as number;
  // A learning-rate schedule spans decades: a cosine run peaks at 8e-5 and ends
  // near 5e-9, and on a linear axis everything after the first few steps is
  // pinned to the bottom — the decay that IS the schedule reads as a flat line
  // at zero. Log makes it the shape it actually is. Only when every reading is
  // positive, which a warmup-from-zero run does not guarantee.
  const values = pts.map((d) => d[metric] as number);
  const logScale = metric === "learning_rate" && values.every((v) => v > 0)
    && Math.max(...values) / Math.min(...values) >= 100;
  return (
    <div className="rounded-lg border border-hair bg-canvas p-2">
      <div className="mb-1 flex items-center justify-between px-1">
        <span className="font-mono text-[15px] uppercase tracking-wider text-dim">{metric}</span>
        <span className="readout text-[15px]" style={{ color }}>{fmtMetric(last)}</span>
      </div>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data} margin={{ top: 4, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1C2731" />
          <XAxis dataKey="step" type="number" stroke="#66788A" fontSize={13}
            tick={{ fill: "#8B9CAB" }}
            domain={[0, total]} ticks={axisTicks(total)} allowDataOverflow />
          <YAxis stroke="#66788A" fontSize={13} tick={{ fill: "#8B9CAB" }} width={56}
            scale={logScale ? "log" : "auto"}
            domain={logScale ? ["auto", "auto"] : ["auto", "auto"]}
            allowDataOverflow={false} tickFormatter={fmtTick} />
          <Tooltip
            contentStyle={{ background: "#10161E", border: "1px solid #1C2731", borderRadius: 10, fontSize: 12 }}
            labelStyle={{ color: "#8B9CAB" }}
            labelFormatter={(s) => `step ${s}`}
            formatter={(v: number) => [fmtMetric(v), metric]}
          />
          <Line type="monotone" dataKey={metric} stroke={color} strokeWidth={2}
            dot={false} connectNulls isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Shared training/inference monitor: every reported numeric metric vs step,
 *  de-duped per step. Rendered identically in the run Timeline and the per-ticket
 *  view so the two never diverge. */
export function TrainingMonitor({ executionEvents }: { executionEvents: ExecutionEventDTO[] }) {
  const num = (v: unknown) => (typeof v === "number" && isFinite(v) ? v : undefined);

  // Runtime repair may launch Trainer again for the same Ticket. `attempt_id`
  // remains an internal curve identity so equal step numbers from two processes
  // are never spliced together, but it is not a user-facing concept: as soon as
  // a newer attempt is announced, its figure replaces the previous one.
  const latestAttemptId = useMemo(() => {
    let latest = "";
    for (const event of executionEvents) {
      if (
        (event.event_type === "attempt" || event.event_type === "progress")
        && event.attempt_id
      ) {
        latest = event.attempt_id;
      }
    }
    return latest;
  }, [executionEvents]);
  const scopedEvents = latestAttemptId
    ? executionEvents.filter(
        (event) => event.attempt_id === latestAttemptId,
      )
    : executionEvents;

  // One row per progress reading: {step, loss, ...every numeric extra}.
  // The same step can arrive twice — once live (POSTed mid-run) and once in the
  // end-of-run batch capture — so de-dup by step (last write wins).
  const byStep = new Map<number, Record<string, number | undefined>>();
  for (const p of scopedEvents) {
    if (p.event_type !== "progress") continue;
    const ex = (p.extras || {}) as Record<string, unknown>;
    // Skip the end-of-run summary row (its aggregate metrics skew the chart).
    if (SUMMARY_KEYS.some((k) => k in ex)) continue;
    // The row's OWN training loss, from the extras — not from `p.loss`.
    //
    // `p.loss` is the runner's convenience column, and it is not always a
    // training loss: a trainer's evaluation log line carries `eval_loss` and no
    // `loss` at all, so the runner fills the column from `eval_loss` to keep
    // the point chartable. Reading it back as `loss` plotted the VALIDATION
    // loss on the TRAINING curve — about fifteen times a run, once per eval —
    // and since the eval line lands on the same step as the training line it
    // had just followed, the real point was overwritten rather than joined.
    const exLoss = num(ex.loss);
    const exEval = num(ex.eval_loss) ?? num(ex.validation_loss);
    // Fall back to the column only for rows that carry no explicit loss of
    // either kind — an older marker, or a script that only set the column.
    const lossVal = exLoss ?? (exEval === undefined ? num(p.loss) : undefined);
    // Drop the loss=-1 sentinel rows the non-training phases emit (connecting,
    // loading_model, saving_model, …). They share current_step=0, so without
    // this guard they'd plot a phantom empty step-0 before the real step 1.
    const haveLoss = lossVal !== undefined && lossVal >= 0;
    const row: Record<string, number | undefined> = { step: p.current_step };
    if (haveLoss) row.loss = lossVal;
    for (const [rawK, v] of Object.entries(ex)) {
      const k = METRIC_ALIAS[rawK] ?? rawK;
      if (NON_METRIC_KEYS.has(k) || NON_METRIC_KEYS.has(rawK)) continue;
      const n = num(v);
      if (n === undefined) continue;
      // -1 is this system's "no reading", and some trainers report it as a
      // VALUE. A learning rate, a loss or a token count is never negative, so
      // a negative one is the sentinel wearing a number's clothes: an eval log
      // line carries no learning rate, and `lr: -1` on it drew the curve
      // diving off the bottom once per epoch. Drop the key, keep the row —
      // its other metrics are real.
      if (n < 0 && NON_NEGATIVE_METRICS.has(k)) continue;
      row[k] = n;
    }
    // Method-specific callbacks can emit a useful numeric row without either
    // loss key (reward, KL, entropy, learning rate, token accuracy, ...).
    // Keep it whenever at least one real metric survived the denylist.
    if (Object.keys(row).length === 1) continue;
    // MERGE, don't replace. Two readings can share a step — the training log
    // line and the evaluation that follows it — and they carry different
    // metrics. Overwriting kept whichever arrived last and threw the other
    // away, which is how the training loss went missing at every eval boundary.
    const prior = byStep.get(p.current_step);
    byStep.set(p.current_step, prior ? { ...prior, ...row } : row);
  }
  const data = [...byStep.values()].sort((a, b) => (a.step ?? 0) - (b.step ?? 0));

  if (data.length === 0) {
    const phase = scopedEvents[scopedEvents.length - 1]?.phase;
    return (
      <div className="text-xs text-dim">
        No loss readings yet{phase ? `, current phase: ` : "."}
        {phase && <span className="font-mono text-brass-300">{phase}</span>}
      </div>
    );
  }

  // The run's length, taken from the progress rows themselves — the trainer
  // reports `total_steps` alongside every reading, so the axis knows where the
  // run ENDS from the first point onward and never has to grow into it. Falling
  // back to the furthest step seen keeps a chart for a stage that reported no
  // total; there the axis does follow the data, because there is nothing else.
  const total = Math.max(
    ...scopedEvents.map((p) => p.total_steps || 0),
    ...data.map((d) => d.step ?? 0),
  );

  // Metric keys = loss first, then every other numeric field except the x-axis.
  const keys = new Set<string>();
  for (const row of data) for (const k of Object.keys(row)) if (k !== "step") keys.add(k);
  const metrics = ["loss", ...[...keys].filter((k) => k !== "loss").sort()];
  const colorFor = (m: string, i: number) => METRIC_COLORS[m] ?? PALETTE[i % PALETTE.length];

  // The last value each metric actually REPORTED, not the values on the last
  // row. The final reading is usually an evaluation line, which carries a loss
  // and nothing else — so `lr`, `grad_norm` and `entropy` dropped out of this
  // strip while their charts sat right below it, still full of data.
  const last: Record<string, number | undefined> = {};
  for (const row of data) {
    for (const [k, v] of Object.entries(row)) {
      if (typeof v === "number") last[k] = v;
    }
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2 font-mono text-[15px]">
        <span className="rounded bg-raised px-2 py-0.5 text-dim">step <span className="text-slate-100">{last.step}</span></span>
        {metrics.map((m, i) =>
          last[m] !== undefined ? (
            <span key={m} className="rounded px-2 py-0.5"
              style={{ background: "rgba(102,120,138,0.1)", color: colorFor(m, i) }}>
              {m} <span className="text-slate-100">{fmtMetric(last[m] as number)}</span>
            </span>
          ) : null
        )}
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {metrics.map((m, i) => (
          <MetricChart total={total} key={m} metric={m} color={colorFor(m, i)} data={data} />
        ))}
      </div>
    </div>
  );
}
