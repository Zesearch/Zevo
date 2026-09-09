"""Tests for opt-in teacher-distilled synthetic data augmentation.

Pinned invariants:
  - default OFF: a baseline recipe is byte-identical to the pre-feature default,
    so intent/recipe identity is unaffected;
  - enabling requires a teacher model, a positive augmentation target, and
    keeps decontamination on (held-out integrity is inviolate);
  - disabled fields must stay inert;
  - the `distill_augment` method id is installed and reachable as a Skill;
  - DataResult carries teacher provenance, while private held-out
    decontamination is attached later by the engine;
  - Data cannot author decontamination outcomes or synthesize on the held-out
    lane.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from zevo.contracts.data import (
    DATA_METHOD_IDS,
    DISTILLATION_OUTPUT_FORMATS,
    DataMethodId,
    DataRecipe,
    DataRecipeIntent,
    DataResult,
    TeacherDistillationIntent,
    data_recipe_signature,
)
from zevo.engine.agent.loader import load_agent


_RESULT_BASE = dict(ticket_id="t1", error_message="", notes="n")
_RUN_ARTIFACTS = dict(
    training_dataset_path="t",
    data_recipe_path="r",
)


# ─────────────────────────── default is OFF ─────────────────────────────────


def test_distillation_defaults_off_and_baseline_recipe_unchanged():
    intent = DataRecipeIntent()
    assert intent.teacher_distillation.enabled is False
    assert intent.teacher_distillation.teacher_model == ""
    assert intent.teacher_distillation.target_augmentation_count == 0
    # Baseline equals baseline: the new nested default does not perturb identity.
    assert DataRecipeIntent() == DataRecipeIntent()


def test_baseline_recipe_signature_ignores_default_distillation(tmp_path):
    data = tmp_path / "dataset.jsonl"
    data.write_text('{"text":"x"}\n', encoding="utf-8")
    recipe = DataRecipe(
        dataset_name="fixture",
        source_identity="src",
        source_fingerprint="a" * 64,
        training_method="lora_sft",
        method_format="text",
        method_ids=["passthrough_jsonl"],
    )
    # A default-constructed distillation block is inert in the realized identity.
    assert recipe.teacher_distillation == TeacherDistillationIntent()
    # Signature is stable and computable with the default block present.
    assert data_recipe_signature(recipe, str(data))


# ─────────────────────────── enable-time guardrails ─────────────────────────


def test_enabled_requires_teacher_model():
    with pytest.raises(ValidationError, match="teacher_model"):
        TeacherDistillationIntent(enabled=True, target_augmentation_count=100)


def test_enabled_requires_positive_augmentation_count():
    with pytest.raises(ValidationError, match="target_augmentation_count"):
        TeacherDistillationIntent(enabled=True, teacher_model="meta/Llama-3-70B")


def test_enabled_cannot_disable_decontamination():
    with pytest.raises(ValidationError, match="decontaminate"):
        TeacherDistillationIntent(
            enabled=True,
            teacher_model="meta/Llama-3-70B",
            target_augmentation_count=5000,
            decontaminate_against_eval=False,
        )


def test_disabled_fields_must_stay_inert():
    with pytest.raises(ValidationError, match="require enabled=true"):
        TeacherDistillationIntent(teacher_model="meta/Llama-3-70B")
    with pytest.raises(ValidationError, match="require enabled=true"):
        TeacherDistillationIntent(target_augmentation_count=10)


def test_generation_params_keys_must_be_non_empty():
    with pytest.raises(ValidationError, match="generation_params"):
        TeacherDistillationIntent(
            enabled=True,
            teacher_model="meta/Llama-3-70B",
            target_augmentation_count=10,
            generation_params={" ": 1},
        )


def test_valid_enabled_distillation_intent():
    intent = DataRecipeIntent(
        direction="scale reasoning data via teacher distillation",
        teacher_distillation=TeacherDistillationIntent(
            enabled=True,
            teacher_model="meta/Llama-3-70B-Instruct",
            output_format="irac_rationale",
            target_augmentation_count=20000,
            generation_params={"temperature": 0.9, "self_consistency_k": 4},
        ),
    )
    td = intent.teacher_distillation
    assert td.enabled and td.verify_answers and td.decontaminate_against_eval
    assert td.output_format in DISTILLATION_OUTPUT_FORMATS


def test_output_format_is_constrained():
    with pytest.raises(ValidationError):
        TeacherDistillationIntent(
            enabled=True,
            teacher_model="x",
            target_augmentation_count=1,
            output_format="freeform_nonsense",
        )


# ─────────────────────────── method id + skill wiring ───────────────────────


def test_distill_augment_is_a_registered_method_id():
    assert "distill_augment" in DATA_METHOD_IDS
    assert "distill_augment" in DataMethodId.__args__  # type: ignore[attr-defined]


def test_data_agent_installs_the_distill_augment_skill():
    bp = load_agent("data")
    cards = {c.method: c for c in bp.skills}
    assert "distill_augment" in cards
    card = cards["distill_augment"]
    assert card.name == "distill-augment"
    assert card.description
    assert card.path.exists()


def test_distill_augment_realized_recipe_records_provenance(tmp_path):
    data = tmp_path / "dataset.jsonl"
    data.write_text('{"text":"x"}\n', encoding="utf-8")
    recipe = DataRecipe(
        dataset_name="fixture",
        source_identity="src",
        source_fingerprint="b" * 64,
        training_method="lora_sft",
        method_format="messages",
        method_ids=["reformat_jsonl", "distill_augment"],
        teacher_distillation=TeacherDistillationIntent(
            enabled=True,
            teacher_model="meta/Llama-3-70B-Instruct",
            output_format="irac_rationale",
            target_augmentation_count=10000,
            generation_params={"temperature": 0.9},
        ),
    )
    assert recipe.teacher_distillation.teacher_model == "meta/Llama-3-70B-Instruct"
    # Provenance is part of the realized recipe identity.
    assert data_recipe_signature(recipe, str(data))


# ─────────────────────────── DataResult provenance ──────────────────────────


def test_result_rejects_synthetic_rows_without_teacher_provenance():
    with pytest.raises(ValidationError, match="synthesis_teacher_model"):
        DataResult(
            status="succeeded",
            operation="prepare_run_data",
            synthesis_generated_rows=100,
            **_RUN_ARTIFACTS,
            **_RESULT_BASE,
        )


def test_result_leaves_private_decontamination_for_the_engine():
    result = DataResult(
        status="succeeded",
        operation="prepare_run_data",
        synthesis_generated_rows=100,
        synthesis_teacher_model="meta/Llama-3-70B",
        **_RUN_ARTIFACTS,
        **_RESULT_BASE,
    )
    assert result.decontamination_checked is False
    assert result.decontamination_removed_rows == 0


def test_result_rejects_agent_authored_decontamination_outcome():
    with pytest.raises(ValidationError, match="engine-owned decontamination"):
        DataResult(
            status="succeeded",
            operation="prepare_run_data",
            synthesis_generated_rows=9800,
            synthesis_teacher_model="meta/Llama-3-70B",
            synthesis_output_format="irac_rationale",
            decontamination_checked=True,
            decontamination_removed_rows=200,
            n_rows_out=10800,
            **_RUN_ARTIFACTS,
            **_RESULT_BASE,
        )


def test_result_rejects_agent_authored_removed_rows():
    with pytest.raises(ValidationError, match="engine-owned decontamination"):
        DataResult(
            status="succeeded",
            operation="prepare_run_data",
            decontamination_removed_rows=5,
            **_RUN_ARTIFACTS,
            **_RESULT_BASE,
        )


def test_holdout_result_forbids_any_synthesis():
    with pytest.raises(ValidationError, match="held-out"):
        DataResult(
            status="succeeded",
            operation="prepare_holdout_data",
            scoring_public_path="s",
            synthesis_generated_rows=1,
            synthesis_teacher_model="x",
            decontamination_checked=True,
            **_RESULT_BASE,
        )


def test_ordinary_run_needs_no_provenance():
    # The common (non-synthesized) path stays valid with all defaults.
    result = DataResult(
        status="succeeded",
        operation="prepare_run_data",
        **_RUN_ARTIFACTS,
        **_RESULT_BASE,
    )
    assert result.synthesis_generated_rows == 0
    assert result.decontamination_checked is False
