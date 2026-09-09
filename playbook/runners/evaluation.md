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

Evaluation is a deterministic system stage, not an LLM Agent. It receives a
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
status, metrics, wrapper, and log. Model/token usage and LLM cost are always
zero because no prompt or provider call exists in this stage.
