---
id: evaluation
name: Evaluation Runner
title: Quality Evaluator
reports_to: system
driver: evaluation_runner
model: no-llm
output_schema: zevo.contracts.evaluation:EvaluationResult
---

# Evaluation Runner

Evaluation is a system stage, not an LLM Agent. It receives a
typed `EvaluationTaskInput`; natural-language requests and Agent customization
are invalid.

## Inputs

- `predictions_path`: exact Inference predictions.
- `scoring_set`: matching full Validation or held-out Test records.
- `sample_submission`: authoritative prediction columns and order for this
  Validation or Test lane.
- `evaluation_script`: immutable scorer owned by this lane; empty selects its
  named built-in metric.
- `evaluator_sha256`: digest of the frozen custom evaluator; empty for built-ins.
- `answer_fields`, `metric`, and `evaluation_config`: immutable semantics of
  the owning Validation or Test contract.

Before any scorer runs, require predictions to match `sample_submission`
exactly in columns and order, match the scoring population's row count, and
preserve every shared identity column and row order. A custom script cannot
bypass this boundary. A built-in metric derives its single prediction column
from the sample rather than guessing from the prediction file. Multi-output
samples require a custom evaluator.

The runner must not modify predictions, answers, row order, evaluator logic,
metric scale, or column mappings. Validation and held-out Test use the same
execution protocol but may use different metrics, directions, samples, and
custom evaluator scripts.

## Custom scorer

When `evaluation_script` is non-empty, the runner writes an auditable
`evaluate.sh` and executes exactly:

```text
bash evaluate.sh
  -> python3 <evaluation_script> <predictions_path> <scoring_set> <metrics.json>
```

The command runs once in the foreground with a finite timeout. Stdout and
stderr go to `evaluate.log`. A non-zero exit, timeout, missing output, invalid
JSON, or invalid score is a failed Ticket. Never fall back after a custom
scorer fails, because doing so would change the measurement.
Task-owned scripts may call a versioned external model judge if their metric
explicitly requires it. In that case the scorer, not this runner, controls its
rubric, model, retry behavior, and API credential. A missing credential must
fail the Ticket; it must never cause a fallback to a lexical proxy. Such API
usage is billed by the provider and is not included in Zevo's Agent LLM cost
ledger.

## Built-in scorer

When `evaluation_script` is empty, call
`zevo.engine.method.eval_metrics.score_default` directly. Supported metrics are
`accuracy`, `exact_match`, `f1`, `token_f1`, `bleu`, and `rouge_l`. The only supported
fallback controls are `prediction_column`, `answer_column`, `strict`, and
`f1_average`; reject unknown or incorrectly typed fields and ambiguous answer
mappings.

## Output

Success requires a newly written UTF-8 JSON object at
`<work_dir>/metrics.json` with a finite numeric, non-boolean top-level `score`.
Do not normalize, clamp, round, or substitute another metric key. Return an
`EvaluationResult` containing the exact `metrics_path`; the reopened file is the
only score authority and the Result does not repeat `score`.

The normal Ticket/Heartbeat surface records phases, Bash command, output,
status, metrics, wrapper, and log. The runner itself makes no provider call;
custom scorers may do so as described above.
For a combined benchmark suite, emit one completion progress marker after each
member's metrics are validated. The marker carries its exact benchmark name
and one-based suite position; the dashboard counts these independently of the
single Evaluation ticket.

## Compute execution

Production Evaluation stays a deterministic runner. The engine binds the exact
Infrastructure route used by the source Inference, including checkpoint and
held-out evaluations. Missing or released routes fail closed; they never cause
scoring to fall back to the Zevo service or a cluster login node.

- Cluster: the runner stages `evaluate.sbatch`, the immutable evaluator and its
  declared inputs. The engine validates and submits through the existing Slurm
  controller. A retained per-run allocation is reused; evaluation queues behind
  its current workload. Otherwise the controller obtains a finite allocation.
  Login handles only staging/submission/status/transfer. Dependency installation,
  scoring and API judge calls run inside the allocation.
- Instance/cloud: use the bound SSH compute host. A ticket-marked, locked worker
  survives a disconnected SSH client; a new activation reconnects to that work.
  Keep the instance through inference and scoring, releasing at the end of the
  held-out evaluation chain according to the Run allocation policy.
- CPU scoring does not launch a GPU runtime. Cluster GPU reservations follow the
  existing allocation/site minimum (some GPU partitions require a GPU request).
  This change does not create a separate CPU pool or change score semantics.

The compute route must supply Python, tar and flock, plus the packages required
by the Task scorer. Zevo stages its own source and adds it to PYTHONPATH; it
does not install a package or modify the configured environment. Judge credentials are restricted
to the Run's resolved model providers, sent over SSH stdin to a private file,
loaded into the worker scope and removed. Cloud provisioning keys, database
credentials and service tokens are not forwarded.

Scoring uses the same configured environment as Data and Train. The runner does
not create virtual environments, install packages, or download Python. Custom
scorers must use the active interpreter instead of hard-coded backend runtime
paths. Missing dependencies fail explicitly and must be resolved in the shared
compute environment; there is no backend fallback.

Progress uses the ordinary execution markers, with a durable phase report.
Results return through the same schema and finite-score checks as local scoring.
A suite checkpoints completed **benchmarks**, binding each result to its inputs,
scorer, system runtime and metrics checksum. Reconnects/retries reuse only matching
receipts. A custom scorer's individual rows are not silently partitioned: its
aggregation may not be separable. API judge scripts retain their existing
per-request cache. Receipts require the remote workspace to survive; they cannot
recover data from a destroyed instance disk.

## Execution duration

Evaluation has no whole-stage execution timeout, for both custom and built-in
scorers on local, Slurm, and instance/cloud compute. The former
`ZEVO_EVALUATION_TIMEOUT_SECONDS` and `ZEVO_CODE_EVALUATION_TIMEOUT_SECONDS`
settings are no longer used. Explicit cancellation and resource-allocation
wall time still apply. Keep per-problem code execution limits and individual
API/transport request deadlines; these do not impose a deadline on the suite.
