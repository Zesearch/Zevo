"""Typed contracts for versioned, Run-local Data preparation.

Optimization Data Tickets can see only the allowed training source/query and
training feedback.  Validation is prepared by the engine after Data returns,
so its rows, answers, schema, and statistics cannot influence source selection,
sampling, filtering, or transformation.  A private ``prepare_holdout_data``
invocation may strip Test answers for the held-out harness, but its evidence is
never exposed to the optimization loop.

An Auto-mode Run adds one engine-created ``scope_problem`` work order before
optimization. From the objective, that isolated scoping operation derives
the scoring contract as a :class:`zevo.contracts.scoping.ScopingResult`. The
engine moves its assets behind the held-out boundary before creating the
Orchestrator or any later optimization Data Ticket.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zevo.contracts._base import AgentResult, AgentTaskInput
from zevo.contracts.training_methods import METHOD_CONFIG_KEYS


DATA_METHOD_IDS = frozenset({
    "acquire_hf",
    "distill_augment",
    "inline_transform",
    "passthrough_jsonl",
    "pdf_to_qa",
    "reformat_csv",
    "reformat_jsonl",
    "synthesize_llm",
})
DataMethodId = Literal[
    "acquire_hf",
    "distill_augment",
    "inline_transform",
    "passthrough_jsonl",
    "pdf_to_qa",
    "reformat_csv",
    "reformat_jsonl",
    "synthesize_llm",
]
_DATA_CONFIGURATION_KEYS = frozenset({"method_ids", "target_size"})
DISTILLATION_OUTPUT_FORMATS = frozenset(
    {"irac_rationale", "instruction_distillation"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_sha256(value: str) -> bool:
    """True only for the canonical lowercase hexadecimal SHA-256 form."""
    return bool(_SHA256_RE.fullmatch(value or ""))


class TeacherDistillationIntent(BaseModel):
    """Opt-in teacher-distilled synthetic augmentation of TRAINING data only.

    Default OFF: an all-default instance reproduces the historical behavior in
    which the Data Agent never calls a stronger teacher model, so the baseline
    recipe is byte-identical to before this capability existed.  When an
    orchestrator/data-branch decision to scale data sets ``enabled=True``, this
    authorizes distilling supervision from a named stronger teacher into
    additional TRAINING rows — the single biggest documented accuracy lever for
    reasoning tasks (teacher-distilled IRAC chain-of-thought SFT; arXiv
    2504.04945).  The controlled reversal of the blanket "no stronger teacher"
    prohibition carries inviolable guardrails, enforced here and in the Data
    playbook:

    - held-out Validation/Test data and its answers are NEVER synthesized or
      altered (this object still contains no Validation/Test mutation controls);
    - every generated row is decontaminated against the frozen eval populations
      (``decontaminate_against_eval`` cannot be disabled while enabled);
    - the teacher model and generation params are recorded as provenance in the
      realized recipe (and echoed in ``DataResult``);
    - synthesis augments TRAINING data only.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    teacher_model: str = Field(
        "",
        description=(
            "Stronger teacher model id whose distilled outputs become new "
            "TRAINING rows. Required when enabled; recorded as provenance."
        ),
    )
    output_format: Literal["irac_rationale", "instruction_distillation"] = (
        "instruction_distillation"
    )
    target_augmentation_count: int = Field(
        0,
        ge=0,
        description=(
            "Absolute number of verified synthetic TRAINING rows to add. Must "
            "be >= 1 when enabled; 0 when disabled."
        ),
    )
    verify_answers: bool = Field(
        True,
        description=(
            "Keep only teacher generations whose final answer passes the task's "
            "correctness / self-consistency check (rejection filtering). The "
            "verifier/filter is the single most important synthesis step."
        ),
    )
    decontaminate_against_eval: bool = Field(
        True,
        description=(
            "REQUIRED held-out guardrail: drop any generated TRAINING row that "
            "overlaps the frozen Validation/Test scoring populations. May never "
            "be disabled while distillation is enabled — held-out integrity is "
            "inviolate."
        ),
    )
    generation_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Teacher decoding/sampling provenance (e.g. temperature, top_p, "
            "num_samples, self_consistency_k). Recorded in the recipe for "
            "reproducibility; keys must be non-empty."
        ),
    )

    @model_validator(mode="after")
    def validate_distillation(self) -> "TeacherDistillationIntent":
        if any(not str(key).strip() for key in self.generation_params):
            raise ValueError(
                "teacher_distillation generation_params keys must be non-empty"
            )
        if not self.enabled:
            # Disabled must stay inert so the default recipe is unchanged and
            # the intent signature of a baseline recipe is unaffected.
            if (
                self.teacher_model.strip()
                or self.target_augmentation_count
                or self.generation_params
            ):
                raise ValueError(
                    "teacher_distillation fields require enabled=true; leave them "
                    "empty to preserve the default no-synthesis behavior"
                )
            return self
        if not self.teacher_model.strip():
            raise ValueError(
                "teacher_distillation requires a teacher_model when enabled"
            )
        if self.target_augmentation_count < 1:
            raise ValueError(
                "teacher_distillation requires target_augmentation_count >= 1 "
                "when enabled"
            )
        if not self.decontaminate_against_eval:
            raise ValueError(
                "teacher_distillation must decontaminate generated rows against "
                "the frozen Validation/Test populations; "
                "decontaminate_against_eval cannot be disabled "
                "(held-out integrity is inviolate)"
            )
        return self


class DataRecipeIntent(BaseModel):
    """The complete requested training-data change for one iteration.

    Empty collections mean preserve the corresponding source behavior.  The
    shape is deliberately explicit while the nested ``sampling`` mapping stays
    open enough for method-appropriate strategies.  This object contains no
    Validation/Test mutation controls.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    direction: str = Field(
        "",
        description=(
            "One coherent data hypothesis. Empty means the baseline/all-data "
            "recipe or exact reuse, not permission to invent a change."
        ),
    )
    subset: str = Field(
        "",
        description="Deterministic subset expression or named slice; empty keeps all eligible rows.",
    )
    filters: list[str] = Field(default_factory=list)
    sampling: dict[str, Any] = Field(default_factory=dict)
    weighting: dict[str, float] = Field(default_factory=dict)
    transformations: list[str] = Field(default_factory=list)
    field_mapping: dict[str, str] = Field(default_factory=dict)
    seed: int = Field(0, ge=0)
    teacher_distillation: TeacherDistillationIntent = Field(
        default_factory=TeacherDistillationIntent,
        description=(
            "Opt-in teacher-distilled synthetic augmentation of TRAINING data. "
            "Default disabled (existing behavior, byte-identical baseline "
            "recipe). When enabled by explicit intent it authorizes distilling "
            "supervision from a stronger teacher into new training rows, never "
            "touching the held-out Validation/Test populations or their answers."
        ),
    )

    @model_validator(mode="after")
    def require_clean_recipe(self) -> "DataRecipeIntent":
        string_lists = {"filters": self.filters, "transformations": self.transformations}
        for name, values in string_lists.items():
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"data recipe {name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"data recipe {name} must not contain duplicates")
        if any(not str(key).strip() for key in self.sampling):
            raise ValueError("data recipe sampling keys must be non-empty")
        if any(
            not str(key).strip() or isinstance(value, bool) or float(value) < 0
            for key, value in self.weighting.items()
        ):
            raise ValueError("data recipe weighting requires non-negative numeric weights")
        if any(not key.strip() or not value.strip() for key, value in self.field_mapping.items()):
            raise ValueError("data recipe field_mapping keys and values must be non-empty")
        return self


class DataRecipe(DataRecipeIntent):
    """Exact recipe realized by Data and bound to one training artifact."""

    schema_version: Literal[2] = 2
    dataset_name: str = Field(
        min_length=1,
        description=(
            "Stable human-readable name of the realized source dataset, such "
            "as a Hub repository name or uploaded filename. Selection and "
            "transformation details remain in subset/filters and do not replace "
            "this name."
        ),
    )
    source_identity: str = Field(
        min_length=1,
        description=(
            "Engine-canonical lineage identity from the work order, copied "
            "from expected_source_identity without path conversion. The "
            "fingerprint binds the realized source bytes/revision; DataResult "
            "does not repeat either provenance field."
        ),
    )
    source_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Canonical SHA-256: exactly 64 lowercase hexadecimal characters "
            "with no 'sha256:' prefix. For a local source this is the digest "
            "of the exact source-file bytes."
        ),
    )
    training_method: str = Field(min_length=1)
    method_format: str = Field(min_length=1)
    method_ids: list[DataMethodId] = Field(
        min_length=1,
        description=(
            "Ordered canonical identifiers for the Data Skills or the generic "
            "inline_transform operation actually applied."
        ),
    )
    audit_steps: list[str] = Field(
        default_factory=list,
        description=(
            "Concise free-text execution details for human audit only. These "
            "steps never control reuse, identity, or hard validation."
        ),
    )

    @model_validator(mode="after")
    def require_unique_method_ids_and_clean_audit(self) -> "DataRecipe":
        if len(self.method_ids) != len(set(self.method_ids)):
            raise ValueError("method_ids must not contain duplicates")
        if any(not step.strip() for step in self.audit_steps):
            raise ValueError("audit_steps must contain only non-empty strings")
        return self


def data_intent_signature(value: dict[str, Any]) -> str:
    """Content identity used to decide reuse before executing Data."""
    canonical = {
        key: value.get(key)
        for key in (
            "dataset_source", "dataset", "dataset_split", "dataset_config",
            "data_query", "training_method", "recipe_intent",
            "configuration_suggestions", "configuration_pins",
        )
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def data_recipe_signature(recipe: DataRecipe | dict[str, Any], dataset_path: str) -> str:
    """Verified identity of the realized recipe and exact training bytes."""
    parsed = recipe if isinstance(recipe, DataRecipe) else DataRecipe.model_validate(recipe)
    digest = hashlib.sha256()
    realized = parsed.model_dump(mode="json")
    # `direction` and `audit_steps` explain the work but do not change it. Two
    # descriptions that realize byte-identical preparation choices are one data
    # identity, while the intent signature still records why Data was invoked.
    realized.pop("direction", None)
    realized.pop("audit_steps", None)
    # The name is display/provenance metadata. Renaming an otherwise identical
    # recipe must not create a different executable data identity.
    realized.pop("dataset_name", None)
    digest.update(json.dumps(
        realized, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8"))
    with open(dataset_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data_recipe(path: str | Path) -> DataRecipe:
    source = Path(path)
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read data recipe {source}: {exc}") from exc
    return DataRecipe.model_validate(body)


def _main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate a Data recipe and artifact identity")
    parser.add_argument("command", choices=["validate-recipe"])
    parser.add_argument("recipe_path")
    parser.add_argument("dataset_path")
    parser.add_argument(
        "--source-identity",
        required=True,
        help=(
            "Engine-canonical source lineage that DataRecipe.source_identity "
            "must copy byte-for-byte."
        ),
    )
    parser.add_argument(
        "--source-file",
        default="",
        help=(
            "Optional local source file whose exact bytes must match "
            "data_recipe.source_fingerprint."
        ),
    )
    args = parser.parse_args(argv)
    try:
        recipe = load_data_recipe(args.recipe_path)
        if recipe.source_identity != args.source_identity:
            raise ValueError(
                "source_identity must exactly copy the engine-supplied value: "
                f"expected {args.source_identity!r}, got {recipe.source_identity!r}"
            )
        if args.source_file:
            source = Path(args.source_file)
            if not source.is_file():
                raise ValueError(f"source file does not exist: {source}")
            expected_source_fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
            if recipe.source_fingerprint != expected_source_fingerprint:
                raise ValueError(
                    "source_fingerprint must equal the bare lowercase SHA-256 "
                    "of the exact source-file bytes: "
                    f"expected {expected_source_fingerprint}, got "
                    f"{recipe.source_fingerprint}"
                )
        signature = data_recipe_signature(recipe, args.dataset_path)
    except (OSError, ValueError) as exc:
        print(f"INVALID DataRecipe: {exc}", file=sys.stderr)
        return 1
    print(f"VALID DataRecipe data_signature={signature}")
    return 0


class NumericProfileSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit: Literal["characters", "tokens", "turns"]
    count: int = Field(ge=0)
    minimum: float = Field(ge=0)
    median: float = Field(ge=0)
    p90: float = Field(ge=0)
    p99: float = Field(ge=0)
    maximum: float = Field(ge=0)


class TrainingResponseLengthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["training_data_only"] = "training_data_only"
    summary: NumericProfileSummary


class InferenceDataProfile(BaseModel):
    """Closed answer-free evidence used by baseline Inference."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    source: Literal["validation_questions_only"] = "validation_questions_only"
    n_rows: int = Field(ge=1)
    task_shape: str = Field(min_length=1)
    record_fields: dict[str, str]
    input_fields: list[str] = Field(min_length=1)
    answer_fields_removed: list[str] = Field(min_length=1)
    turn_count: NumericProfileSummary | None = None
    input_length: NumericProfileSummary | None = None
    training_response_length: TrainingResponseLengthSummary | None = None
    submission_format: Literal["csv"] = "csv"
    submission_columns: list[str] = Field(min_length=1)
    prediction_encoding: str = Field(min_length=1)
    row_order_preserved: Literal[True] = True
    stable_ids_present: bool
    contains_answer_values: Literal[False] = False
    contains_evaluation_logic: Literal[False] = False

    @model_validator(mode="after")
    def require_answer_free_profile(self) -> "InferenceDataProfile":
        if set(self.answer_fields_removed) & set(self.record_fields):
            raise ValueError("answer fields remain in the inference profile")
        if len(self.submission_columns) != len(set(self.submission_columns)):
            raise ValueError("submission_columns must not contain duplicates")
        return self


DATA_OPERATIONS = ("prepare_run_data", "prepare_holdout_data", "scope_problem")
DataOperation = Literal["prepare_run_data", "prepare_holdout_data", "scope_problem"]
# Fields that exist only for the Auto-mode scoping work order. Every other
# operation must leave them empty so the historical payload shapes are unchanged.
SCOPING_ONLY_FIELDS = ("task_objective", "test_query", "constraints")


class DataTaskInput(AgentTaskInput):
    operation: DataOperation
    run_id: str = Field(min_length=1)
    test_set_name: str = ""

    # ── scope_problem (Auto mode) only ───────────────────────────────────────
    task_objective: str = Field(
        "",
        description=(
            "Auto mode: the user's objective, the ONLY required input to "
            "scope_problem. Empty for every other operation."
        ),
    )
    test_query: str = Field(
        "",
        description=(
            "Auto mode: optional user guidance for the held-out Test "
            "population, format, provenance, or construction."
        ),
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Auto mode: optional user constraints, e.g. 'no external APIs'.",
    )
    scoping_result_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Auto mode: exact JSON Schema of the scoping_result.json artifact.",
    )
    scoping_result_validation_command: str = Field(
        "",
        description=(
            "Auto mode: exact side-effect-free validator to run on the written "
            "scoping_result.json before reporting success."
        ),
    )

    dataset_source: str = ""
    dataset: str = ""
    dataset_split: str = ""
    dataset_config: str = ""
    data_query: str = ""
    expected_source_identity: str = Field(
        "",
        description=(
            "Engine-canonical source lineage for this work order. For "
            "prepare_run_data, copy it byte-for-byte into "
            "data_recipe.source_identity; do not resolve or rewrite it."
        ),
    )
    training_method: str = ""
    recipe_intent: DataRecipeIntent = Field(default_factory=DataRecipeIntent)
    branch_transition: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Engine-validated exhausted-branch audit record. level=none for an "
            "initial or continuing branch."
        ),
    )
    data_intent_signature: str = Field(
        "",
        max_length=64,
        pattern=r"^(?:|[0-9a-f]{64})$",
        description=(
            "Engine-computed recipe-intent identity. Copy the supplied bare "
            "64-character lowercase SHA-256 exactly; empty only for held-out Data."
        ),
    )
    expected_source_fingerprint: str = Field(
        "",
        max_length=64,
        pattern=r"^(?:|[0-9a-f]{64})$",
        description=(
            "Engine-computed canonical fingerprint for a local source: exactly "
            "64 lowercase hexadecimal SHA-256 characters with no prefix. Copy "
            "this value exactly into data_recipe.source_fingerprint. Empty only "
            "when the engine cannot precompute a remote/acquired source."
        ),
    )
    data_recipe_schema: dict[str, Any]
    data_recipe_validation_command: str = Field(min_length=1)
    artifacts_validation_command: str = Field(
        min_length=1,
        description=(
            "Exact side-effect-free validator. For prepare_run_data it can see "
            "only the training artifact; for private held-out preparation it "
            "validates exact answer removal. Replace only named output paths."
        ),
    )
    validation_policy: Literal["supplied"] = "supplied"
    validation_fraction: Literal[0.0] = 0.0
    scoring_set: str = ""
    answer_fields: list[str] = Field(default_factory=list)
    metric_type: Literal["builtin", "custom"] = "builtin"
    # Private held-out preparation requires this value. Optimization Data sees
    # no scoring metric; scope_problem derives one rather than receiving it.
    metric: str = ""
    evaluation_script: str = ""
    evaluator_sha256: str = Field("", pattern=r"^(?:|[0-9a-f]{64})$")
    sample_submission: str = ""

    configuration_suggestions: dict[str, Any] = Field(default_factory=dict)
    configuration_pins: dict[str, Any] = Field(default_factory=dict)
    work_dir: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lane_scope(self) -> "DataTaskInput":
        if self.operation == "scope_problem":
            if not self.task_objective.strip():
                raise ValueError("scope_problem requires task_objective")
            if any(not str(c).strip() for c in self.constraints):
                raise ValueError("constraints must contain non-empty strings")
            if not self.scoping_result_schema:
                raise ValueError("scope_problem requires scoping_result_schema")
            if not self.scoping_result_validation_command.strip():
                raise ValueError("scope_problem requires scoping_result_validation_command")
            if (
                self.dataset or self.dataset_source or self.dataset_split
                or self.dataset_config or self.data_query or self.training_method
                or self.test_set_name
                or self.scoring_set or self.answer_fields or self.metric
                or self.evaluation_script or self.evaluator_sha256
                or self.sample_submission or self.configuration_suggestions
                or self.configuration_pins or self.expected_source_identity
                or self.data_intent_signature or self.expected_source_fingerprint
                or self.recipe_intent != DataRecipeIntent()
            ):
                raise ValueError(
                    "scope_problem derives the scoring contract itself and must "
                    "not receive training-source, method, metric, or scoring fields"
                )
            return self
        if any(
            getattr(self, name) for name in SCOPING_ONLY_FIELDS
        ) or self.scoping_result_schema or self.scoping_result_validation_command:
            raise ValueError(
                f"{self.operation} must not carry Auto-mode scoping fields"
            )
        if self.operation == "prepare_holdout_data" and not self.metric.strip():
            raise ValueError("prepare_holdout_data requires metric")
        if self.operation == "prepare_run_data" and not (
            self.dataset or self.data_query
        ):
            raise ValueError("prepare_run_data requires dataset or data_query")
        if self.operation == "prepare_run_data":
            method = self.training_method.strip().lower()
            if not method:
                raise ValueError("prepare_run_data requires training_method")
            if method not in METHOD_CONFIG_KEYS:
                raise ValueError(
                    f"unsupported training_method={method!r}; installed methods: "
                    + ", ".join(sorted(METHOD_CONFIG_KEYS))
                )
            if not is_sha256(self.data_intent_signature):
                raise ValueError(
                    "prepare_run_data requires a lowercase hexadecimal SHA-256 "
                    "data_intent_signature"
                )
            if self.expected_source_fingerprint and not is_sha256(
                self.expected_source_fingerprint
            ):
                raise ValueError(
                    "expected_source_fingerprint must be exactly 64 lowercase "
                    "hexadecimal SHA-256 characters with no prefix"
                )
            if not self.expected_source_identity.strip():
                raise ValueError("prepare_run_data requires expected_source_identity")
            leaked = {
                "test_set_name": self.test_set_name,
                "scoring_set": self.scoring_set,
                "answer_fields": self.answer_fields,
                "metric": self.metric,
                "evaluation_script": self.evaluation_script,
                "evaluator_sha256": self.evaluator_sha256,
                "sample_submission": self.sample_submission,
            }
            exposed = sorted(name for name, value in leaked.items() if value)
            if exposed:
                raise ValueError(
                    "prepare_run_data is Validation-blind; forbidden fields: "
                    + ", ".join(exposed)
                )
        unknown_config = sorted(
            (set(self.configuration_suggestions) | set(self.configuration_pins))
            - _DATA_CONFIGURATION_KEYS
        )
        if unknown_config:
            raise ValueError(
                "Data configuration may contain only method_ids/target_size; "
                f"training or inference hyperparameters are forbidden: {unknown_config}"
            )
        for owner, values in (
            ("configuration_suggestions", self.configuration_suggestions),
            ("configuration_pins", self.configuration_pins),
        ):
            if "method_ids" not in values:
                continue
            method_ids = values["method_ids"]
            if (
                not isinstance(method_ids, list)
                or any(
                    not isinstance(method_id, str)
                    or method_id not in DATA_METHOD_IDS
                    for method_id in method_ids
                )
                or len(method_ids) != len(set(method_ids))
            ):
                raise ValueError(
                    f"{owner}.method_ids must be a unique list of installed "
                    "canonical Data method ids"
                )
        if self.operation == "prepare_holdout_data" and (
            self.dataset or self.data_query or self.dataset_source
            or self.expected_source_identity or self.training_method
            or self.configuration_suggestions or self.configuration_pins
        ):
            raise ValueError(
                "held-out Data must not receive training-source, method, or configuration fields"
            )
        if self.operation == "prepare_holdout_data" and (
            self.validation_policy != "supplied"
            or not self.scoring_set
            or not self.answer_fields
            or not self.test_set_name.strip()
        ):
            raise ValueError(
                "held-out Data requires test_set_name, a supplied scoring set, "
                "and answer fields"
            )
        self.training_method = self.training_method.strip().lower()
        return self


class DataResult(AgentResult):
    status: Literal["succeeded", "failed"]
    operation: DataOperation

    # Auto mode: the one artifact a successful scope_problem must produce. The
    # engine validates it as a ScopingResult and settles the Run's scoring
    # contract from it; every other operation leaves it empty.
    scoping_result_path: str = Field(
        "",
        description=(
            "Absolute path of the written scoping_result.json (a ScopingResult). "
            "Required for a successful scope_problem; empty otherwise."
        ),
    )

    training_dataset_path: str = ""
    validation_source_path: str = ""
    validation_answer_fields: list[str] = Field(default_factory=list)
    validation_dataset_path: str = ""
    scoring_public_path: str = ""
    inference_data_profile_path: str = ""
    sample_submission_path: str = ""
    prepare_script_path: str = ""
    data_recipe_path: str = ""

    n_rows_in: int = Field(0, ge=0)
    n_rows_out: int = Field(0, ge=0)
    system_scoring_duplicates_removed: int = Field(
        0,
        ge=0,
        description=(
            "Engine-owned count attached after Data returns; Data must report zero."
        ),
    )

    # Teacher-distillation provenance and held-out decontamination outcome.
    # Defaults describe an ordinary (non-synthesized) run; populated only when a
    # teacher-distilled augmentation actually generated rows.
    synthesis_teacher_model: str = Field(
        "",
        description=(
            "Teacher model id that produced distilled TRAINING rows this run; "
            "empty when no teacher distillation ran."
        ),
    )
    synthesis_output_format: str = ""
    synthesis_generated_rows: int = Field(0, ge=0)
    decontamination_checked: bool = Field(
        False,
        description=(
            "Engine-owned flag attached after Data returns and the finished "
            "Training artifact is checked against private Validation/Test. "
            "Data must report false."
        ),
    )
    decontamination_removed_rows: int = Field(
        0,
        ge=0,
        description=(
            "Engine-owned held-out overlap count attached after Data returns; "
            "Data must report zero."
        ),
    )

    @model_validator(mode="after")
    def require_operation_artifacts(self) -> "DataResult":
        if self.status != "succeeded":
            return self
        if self.operation == "scope_problem":
            if not self.scoping_result_path.strip():
                raise ValueError("successful scope_problem requires scoping_result_path")
            if (
                self.training_dataset_path or self.validation_source_path
                or self.validation_answer_fields or self.validation_dataset_path
                or self.scoring_public_path or self.inference_data_profile_path
                or self.data_recipe_path or self.synthesis_generated_rows
                or self.synthesis_teacher_model or self.decontamination_checked
                or self.decontamination_removed_rows
                or self.system_scoring_duplicates_removed
            ):
                raise ValueError(
                    "scope_problem produces only scoping_result_path; training, "
                    "Validation, questions-only, recipe, and TRAINING-synthesis "
                    "provenance fields belong to prepare_run_data (held-out "
                    "synthesis provenance lives inside the ScopingResult)"
                )
            return self
        if self.scoping_result_path:
            raise ValueError("scoping_result_path is valid only for scope_problem")
        if self.operation == "prepare_run_data":
            if (
                self.system_scoring_duplicates_removed
                or self.decontamination_checked
                or self.decontamination_removed_rows
            ):
                raise ValueError(
                    "Data cannot report engine-owned decontamination results"
                )
            missing = [
                name for name, value in (
                    ("training_dataset_path", self.training_dataset_path),
                    ("data_recipe_path", self.data_recipe_path),
                ) if not value
            ]
            if missing:
                raise ValueError(
                    "successful prepare_run_data requires " + ", ".join(missing)
                )
            system_owned = [
                name for name, value in (
                    ("validation_source_path", self.validation_source_path),
                    ("validation_answer_fields", self.validation_answer_fields),
                    ("validation_dataset_path", self.validation_dataset_path),
                    ("scoring_public_path", self.scoring_public_path),
                    ("inference_data_profile_path", self.inference_data_profile_path),
                    ("sample_submission_path", self.sample_submission_path),
                ) if value
            ]
            if system_owned:
                raise ValueError(
                    "prepare_run_data must not author engine-owned Validation "
                    "artifacts: " + ", ".join(system_owned)
                )
        elif not self.scoring_public_path:
            raise ValueError("successful held-out Data requires questions-only scoring data")
        elif (
            self.training_dataset_path
            or self.validation_source_path
            or self.validation_answer_fields
            or self.validation_dataset_path
            or self.synthesis_generated_rows
            or self.synthesis_teacher_model
            or self.decontamination_removed_rows
        ):
            raise ValueError("held-out Data must not produce trainable datasets")
        if bool(self.validation_source_path) != bool(self.validation_answer_fields):
            raise ValueError(
                "validation_source_path and validation_answer_fields must be "
                "reported together"
            )
        # Data owns teacher provenance. The engine attaches held-out
        # decontamination evidence only after this Agent result is accepted.
        if self.synthesis_generated_rows:
            if not self.synthesis_teacher_model.strip():
                raise ValueError(
                    "teacher-distilled rows require synthesis_teacher_model "
                    "provenance"
                )
        return self


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(_main())
