"""The scoring contract an Auto-mode Run derives for itself.

A ``full_pipeline`` Run is created WITH its scoring contract: the user names the
metric, uploads the held-out Test set, and Run creation settles the
Validation/Test split before any Agent runs.  An ``auto`` Run is created with
only an objective.  Its first Ticket is a Data-agent ``scope_problem`` work
order whose output is one :class:`ScopingResult` -- the complete contract the
engine then settles onto the Run (see ``zevo.engine.run.scoping``).

The contract is deliberately closed and evidence-bearing:

* the held-out is either a REAL PUBLIC BENCHMARK materialized from the Hub
  (``eval_source="public_benchmark"``, with hub provenance) or, only when no
  suitable benchmark exists, a PRIVATE SYNTHESIZED set distilled from a named
  teacher (``eval_source="synthesized"``) -- and synthesis carries mandatory
  decontamination evidence plus teacher provenance, reusing the same guardrail
  vocabulary as :class:`zevo.contracts.data.TeacherDistillationIntent`;
* answers of a real benchmark are never fabricated: a benchmark result names
  the exact repo/config/split the rows came from;
* the Validation contract mirrors Test unless the agent states otherwise.

``python -m zevo.contracts.scoping validate <scoping_result.json>`` is the
side-effect-free validator the Data agent runs before reporting success, the
same way ``zevo.contracts.data validate-recipe`` guards a Data recipe.
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EvalSource = Literal["public_benchmark", "synthesized"]


def _clean_fields(values: list[str], *, name: str) -> list[str]:
    cleaned = [str(v).strip() for v in values]
    if not cleaned or any(not v for v in cleaned):
        raise ValueError(f"{name} must be a non-empty list of non-empty field names")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{name} must not contain duplicates")
    return cleaned


class BenchmarkProvenance(BaseModel):
    """Exactly which public rows became the held-out Test population."""

    model_config = ConfigDict(extra="forbid")
    hub_id: str = Field(min_length=1, description="Hugging Face dataset id, owner/name.")
    config: str = Field("", description="Named subset, when the repo ships several.")
    split: str = Field(min_length=1, description="Exact split the rows were read from.")
    revision: str = Field("", description="Repo revision/commit when known.")
    rows: int = Field(ge=1, description="Rows materialized into the held-out file.")
    license: str = ""
    url: str = ""

    @model_validator(mode="after")
    def normalize(self) -> "BenchmarkProvenance":
        self.hub_id = self.hub_id.strip()
        self.split = self.split.strip()
        if self.hub_id.count("/") != 1:
            raise ValueError("benchmark hub_id must be an owner/name Hugging Face id")
        return self


class SynthesisProvenance(BaseModel):
    """How a private held-out was distilled when no public benchmark fit."""

    model_config = ConfigDict(extra="forbid")
    teacher_model: str = Field(
        min_length=1,
        description="Stronger teacher whose verified generations became held-out items.",
    )
    output_format: str = Field(
        "instruction_distillation",
        description="Distillation shape, e.g. instruction_distillation or irac_rationale.",
    )
    generation_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Teacher decoding/sampling provenance; keys must be non-empty.",
    )
    seed_sources: list[str] = Field(
        default_factory=list,
        description=(
            "What generation was seeded from (topic lists, public corpora, hub ids). "
            "Never a Validation/Test population of another Run."
        ),
    )
    rows_generated: int = Field(ge=1)
    rows_verified: int = Field(
        ge=0,
        description="Rows whose answer passed the correctness/self-consistency check.",
    )
    rows_kept: int = Field(
        ge=1, description="Rows written to the held-out file after verification + decontamination.",
    )

    @model_validator(mode="after")
    def check_counts(self) -> "SynthesisProvenance":
        self.teacher_model = self.teacher_model.strip()
        if any(not str(key).strip() for key in self.generation_params):
            raise ValueError("synthesis generation_params keys must be non-empty")
        if self.rows_verified > self.rows_generated:
            raise ValueError("rows_verified cannot exceed rows_generated")
        if self.rows_kept > self.rows_generated:
            raise ValueError("rows_kept cannot exceed rows_generated")
        if any(not str(s).strip() for s in self.seed_sources):
            raise ValueError("seed_sources must contain non-empty strings")
        return self


class DecontaminationEvidence(BaseModel):
    """Proof that a synthesized held-out was checked against training sources."""

    model_config = ConfigDict(extra="forbid")
    checked: Literal[True] = Field(
        True,
        description="Always true: a synthesized held-out without decontamination is rejected.",
    )
    method: str = Field(
        min_length=1,
        description="How overlap was detected, e.g. exact + 13-gram near-duplicate.",
    )
    compared_against: list[str] = Field(
        min_length=1,
        description=(
            "Every public corpus or seed source used while constructing the "
            "held-out items. Eventual training overlap is checked separately "
            "by the engine after optimization Data is prepared."
        ),
    )
    overlap_removed: int = Field(ge=0)
    report_path: str = Field("", description="Optional local report of the check.")

    @model_validator(mode="after")
    def check_sources(self) -> "DecontaminationEvidence":
        self.method = self.method.strip()
        if any(not str(s).strip() for s in self.compared_against):
            raise ValueError("compared_against must contain non-empty strings")
        return self


class ScopingResult(BaseModel):
    """The complete scoring contract derived by ``scope_problem``.

    Everything Run creation would otherwise have demanded from the user is
    here, plus the provenance that makes an agent-defined held-out auditable.
    Paths are absolute local files inside the scoping Ticket's ``work_dir``;
    settlement moves the Test assets behind the private held-out boundary.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1

    # ── the Test (held-out) metric contract ──────────────────────────────────
    metric: str = Field(min_length=1, description="Held-out Test metric name.")
    metric_direction: Literal["max", "min"]
    metric_type: Literal["builtin", "custom"] = "builtin"
    evaluation_script: str = Field(
        "",
        description="Absolute path of a custom Test evaluator; empty for a built-in.",
    )

    # ── the Validation metric contract (typically mirrors Test) ─────────────
    validation_metric: str = Field(
        "", description="Leave blank to mirror the Test metric exactly.",
    )
    validation_metric_direction: Literal["", "max", "min"] = ""
    validation_metric_type: Literal["", "builtin", "custom"] = ""
    validation_evaluation_script: str = ""

    # ── the held-out population ─────────────────────────────────────────────
    eval_source: EvalSource
    test_set_path: str = Field(
        min_length=1,
        description=(
            "Absolute path of the FULL held-out set WITH ground truth. At least "
            "1,000 rows: settlement carves 20% (>= 200 rows) into Validation."
        ),
    )
    test_answer_fields: list[str] = Field(
        min_length=1,
        description="Ground-truth field(s): a CSV column or a JSON key.",
    )
    test_sample_submission_path: str = Field(
        min_length=1,
        description="Absolute path of the submission schema/example CSV built for this set.",
    )
    test_rows: int = Field(ge=1, description="Measured row count of test_set_path.")

    # ── evidence ────────────────────────────────────────────────────────────
    rationale: str = Field(
        min_length=1,
        description="Why this metric and this held-out population measure the objective.",
    )
    candidates_considered: list[str] = Field(
        default_factory=list,
        description="Public benchmarks searched/rejected before this choice, with a reason each.",
    )
    benchmark: BenchmarkProvenance | None = None
    synthesis: SynthesisProvenance | None = None
    decontamination: DecontaminationEvidence | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "ScopingResult":
        from zevo.contracts.orchestrator import BUILTIN_METRICS

        self.metric = self.metric.strip()
        self.validation_metric = self.validation_metric.strip()
        if not self.metric:
            raise ValueError("metric must not be blank")
        if self.metric_type == "builtin":
            if self.metric.lower() not in BUILTIN_METRICS:
                raise ValueError(
                    f"unknown built-in Test metric {self.metric!r}; built-ins are "
                    + ", ".join(sorted(BUILTIN_METRICS))
                )
            if self.evaluation_script.strip():
                raise ValueError("built-in metrics must not carry a custom evaluator")
        elif not self.evaluation_script.strip():
            raise ValueError("custom Test metrics require evaluation_script")

        # Validation mirrors Test unless every Validation field is stated.
        stated = (
            self.validation_metric, self.validation_metric_direction,
            self.validation_metric_type,
        )
        if any(stated) and not all(stated):
            raise ValueError(
                "validation_metric, validation_metric_direction and "
                "validation_metric_type must be given together, or all left blank "
                "to mirror the Test contract"
            )
        if self.validation_metric_type == "builtin":
            if self.validation_metric.lower() not in BUILTIN_METRICS:
                raise ValueError(
                    f"unknown built-in Validation metric {self.validation_metric!r}"
                )
            if self.validation_evaluation_script.strip():
                raise ValueError(
                    "built-in Validation metrics must not carry a custom evaluator"
                )
        elif self.validation_metric_type == "custom" and not (
            self.validation_evaluation_script.strip()
        ):
            raise ValueError("custom Validation metrics require validation_evaluation_script")
        elif not self.validation_metric_type and self.validation_evaluation_script.strip():
            raise ValueError(
                "validation_evaluation_script requires an explicit Validation contract"
            )
        if self.validation_metric_type and (
            self.validation_metric_type != self.metric_type
            or self.validation_metric.lower() != self.metric.lower()
            or self.validation_metric_direction != self.metric_direction
            or self.validation_evaluation_script.strip() != self.evaluation_script.strip()
        ):
            # Auto mode has no separately supplied Validation set: settlement
            # carves Validation out of the held-out population, and a carved
            # set is another sample of the same scoring problem.
            raise ValueError(
                "Validation is carved from the held-out Test population in auto "
                "mode, so its contract must mirror Test; leave the validation_* "
                "fields blank or equal to the Test contract"
            )

        self.test_answer_fields = _clean_fields(
            self.test_answer_fields, name="test_answer_fields",
        )
        for name in ("test_set_path", "test_sample_submission_path"):
            value = str(getattr(self, name)).strip()
            if not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute local path")
            setattr(self, name, value)
        if self.test_set_path == self.test_sample_submission_path:
            raise ValueError("test_set_path and test_sample_submission_path must differ")
        if any(not str(c).strip() for c in self.candidates_considered):
            raise ValueError("candidates_considered must contain non-empty strings")

        if self.eval_source == "public_benchmark":
            if self.benchmark is None:
                raise ValueError(
                    "public_benchmark requires benchmark provenance (hub id, split, rows)"
                )
            if self.synthesis is not None:
                raise ValueError("a public benchmark carries no synthesis provenance")
            if self.benchmark.rows != self.test_rows:
                raise ValueError(
                    "benchmark.rows must equal test_rows: the held-out file holds "
                    "exactly the materialized public rows, never fabricated ones"
                )
        else:
            if self.synthesis is None:
                raise ValueError(
                    "synthesized held-out requires synthesis provenance "
                    "(teacher model, generation params, counts)"
                )
            if self.decontamination is None:
                raise ValueError(
                    "synthesized held-out requires decontamination evidence against "
                    "every training source (held-out integrity is inviolate)"
                )
            if self.benchmark is not None:
                raise ValueError("a synthesized held-out carries no benchmark provenance")
            if self.synthesis.rows_kept != self.test_rows:
                raise ValueError("synthesis.rows_kept must equal test_rows")
        return self

    # ── convenience for the engine ─────────────────────────────────────────
    def effective_validation_contract(self) -> tuple[str, str, str, str]:
        """(metric_type, metric, direction, evaluation_script) for Validation."""
        if self.validation_metric_type:
            return (
                self.validation_metric_type, self.validation_metric,
                self.validation_metric_direction, self.validation_evaluation_script,
            )
        return (
            self.metric_type, self.metric, self.metric_direction, self.evaluation_script,
        )

    def provenance_summary(self) -> dict[str, Any]:
        """The audit record stored on the Run's holdout snapshot."""
        out: dict[str, Any] = {
            "eval_source": self.eval_source,
            "rationale": self.rationale,
            "candidates_considered": list(self.candidates_considered),
            "test_rows": self.test_rows,
        }
        if self.benchmark is not None:
            out["benchmark"] = self.benchmark.model_dump()
        if self.synthesis is not None:
            out["synthesis"] = self.synthesis.model_dump()
        if self.decontamination is not None:
            out["decontamination"] = self.decontamination.model_dump()
        return out


def load_scoping_result(path: str | Path) -> ScopingResult:
    source = Path(path)
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read scoping result {source}: {exc}") from exc
    return ScopingResult.model_validate(body)


def _main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate an Auto-mode ScopingResult")
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("scoping_result_path")
    args = parser.parse_args(argv)
    try:
        result = load_scoping_result(args.scoping_result_path)
        for name in ("test_set_path", "test_sample_submission_path"):
            if not Path(getattr(result, name)).is_file():
                raise ValueError(f"{name} does not exist: {getattr(result, name)}")
        if result.metric_type == "custom" and not Path(result.evaluation_script).is_file():
            raise ValueError(f"evaluation_script does not exist: {result.evaluation_script}")
    except (OSError, ValueError) as exc:
        print(f"INVALID ScopingResult: {exc}", file=sys.stderr)
        return 1
    print(
        f"VALID ScopingResult eval_source={result.eval_source} metric={result.metric} "
        f"rows={result.test_rows}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(_main())
