"""
Input/output contracts for the deterministic Evaluation system runner.

The Evaluation runner's job per ticket, without an LLM call:
  1. Take an EvaluationTaskInput (mirrors orchestrator.EvaluatePayload).
  2. For a custom lane metric, run that lane's frozen evaluator via the
     output-file protocol:
       python3 <evaluation_script> <predictions_path> <scoring_set> <metrics_out>
  3. For a built-in lane metric, run the named deterministic implementation.
  4. Validate the freshly written metrics JSON and return its path. The file's
     top-level `score` remains the sole score authority.

This runner does NOT modify or interpret the custom evaluator.
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field

from zevo.contracts._base import AgentResult, AgentTaskInput


# ---------- Input ----------

class EvaluationTaskInput(AgentTaskInput):
    """Mirror of orchestrator.EvaluatePayload."""

    predictions_path: str = Field(..., description="Path to predictions CSV produced by inference.")
    scoring_set: str = Field(
        ...,
        description=(
            "Path to the FULL set this ticket scores — the half WITH the "
            "answers. Inference only ever saw the questions-only copy, which "
            "is why grading happens here and not there.\n\n"
            "Named for the JOB, not for one of the run's sets: a run has a "
            "validation set the loop tunes on and a held-out test set the "
            "engine judges it by, and this field carries whichever one this "
            "ticket is scoring. You are not told which, and it must not change "
            "how you score — the two numbers are only comparable because the "
            "same procedure produced them. It was called `test_set`, which "
            "read as a promise it could not keep on the majority of tickets, "
            "where it holds the validation set."
        ),
    )
    evaluation_script: str = Field(
        ..., description="Path to this scoring lane's frozen evaluator; empty for built-ins."
    )
    evaluator_sha256: str = Field(
        "", pattern=r"^(?:|[0-9a-f]{64})$",
        description="Frozen evaluator digest; empty only for built-in metrics.",
    )
    sample_submission: str = Field(
        ...,
        min_length=1,
        description=(
            "Lane-specific CSV schema contract. Predictions must match its "
            "columns and order before either built-in or custom scoring runs."
        ),
    )
    answer_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Ground-truth fields declared for this scoring lane and stamped "
            "from the Run. A custom scorer may use several. The built-in "
            "engine requires one selected answer field, either the sole "
            "entry here or `evaluation_config.answer_column`."
        ),
    )
    metric: str = Field(
        ...,
        min_length=1,
        description=(
            "Name of the Task's authoritative metric. With the built-in scorer "
            "it must be one of accuracy, exact_match, f1, token_f1, bleu, "
            "rouge_l, or mc_loglikelihood (alias accuracy_norm — length-"
            "normalized log-likelihood option scoring for multiple-choice QA, "
            "which reads per-option log-likelihoods Inference emits into the "
            "prediction column and argmaxes them). With a custom scorer it names "
            "the finite top-level `score` that the script emits; it is never "
            "silently replaced."
        ),
    )
    evaluation_config: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Built-in-only controls: `prediction_column` (str), "
            "`answer_column` (str), `strict` (bool), and `f1_average` "
            "('micro' or 'macro'). Reject unknown keys or mappings that "
            "conflict with declared answer_fields. Ignored when evaluation_script "
            "is non-empty because the task scorer owns metric semantics."
        ),
    )


# ---------- Output ----------

class EvaluationResult(AgentResult):
    """What the deterministic Evaluation runner reports back.

    The metrics file may preserve any task-specific fields, but success always
    requires a finite numeric top-level `score`. Its scale is task-defined.
    """

    status: Literal["succeeded", "failed"]

    metrics_path: str = Field(
        ...,
        description=(
            "Absolute local path to the freshly written metrics.json. Empty on "
            "failure. Evaluation tickets must return this durable file path."
        ),
    )
