"""Contract tests for InferenceTaskInput.configuration_suggestions handling.

Reused (checkpoint) inference must execute the existing baseline YAML exactly.
A stray non-empty ``configuration_suggestions`` from the Orchestrator is
baseline-only advice with no effect in reuse mode, so the validator drops it and
proceeds rather than hard-failing and wedging the ticket on every wakeup.
Baseline (select) inference still carries suggestions unchanged, and every other
reuse guarantee still raises exactly as before.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from zevo.contracts.configuration import (
    InferenceRunConfig,
    inference_mapping_contract,
)
from zevo.contracts.inference import InferenceTaskInput


_CONFIG_VALIDATION_COMMAND = (
    "python -m zevo.contracts.configuration validate inference "
    "<absolute-yaml-path>"
)
_PREDICTIONS_VALIDATION_COMMAND = (
    "python -m zevo.engine.artifact_validation validate-predictions "
    "--predictions <absolute-predictions-csv-path>"
)


def _write_inference_config(tmp_path: Path) -> Path:
    path = tmp_path / "inference_config.yaml"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    return path


def _reuse_kwargs(tmp_path: Path, **overrides: object) -> dict[str, object]:
    """Baseline-valid kwargs for a checkpoint/reuse InferenceTaskInput."""
    config_path = _write_inference_config(tmp_path)
    kwargs: dict[str, object] = dict(
        ticket_id="infer-reuse-001",
        run_id="r",
        iteration=1,
        model_source="checkpoint",
        configuration_mode="reuse",
        base_model="Qwen/Qwen3-0.6B-Base",
        checkpoint_path="/tmp/model",
        scoring_set="/tmp/questions.jsonl",
        sample_submission="/tmp/submission.csv",
        inference_config_path=str(config_path),
        inference_config_schema=InferenceRunConfig.model_json_schema(),
        inference_mapping_contract=inference_mapping_contract(),
        config_validation_command=_CONFIG_VALIDATION_COMMAND,
        memory_helper_path=str(tmp_path / "zevo_inference_memory.py"),
        predictions_validation_command=_PREDICTIONS_VALIDATION_COMMAND,
        device_info_path="/tmp/device.json",
        work_dir=str(tmp_path),
    )
    kwargs.update(overrides)
    return kwargs


def _baseline_kwargs(tmp_path: Path, **overrides: object) -> dict[str, object]:
    """Baseline-valid kwargs for a base_model/select InferenceTaskInput."""
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    kwargs: dict[str, object] = dict(
        ticket_id="infer-base-001",
        run_id="r",
        iteration=0,
        model_source="base_model",
        configuration_mode="select",
        base_model="Qwen/Qwen3-0.6B-Base",
        scoring_set="/tmp/questions.jsonl",
        sample_submission="/tmp/submission.csv",
        inference_data_profile_path=str(profile),
        inference_config_schema=InferenceRunConfig.model_json_schema(),
        inference_mapping_contract=inference_mapping_contract(),
        config_validation_command=_CONFIG_VALIDATION_COMMAND,
        memory_helper_path=str(tmp_path / "zevo_inference_memory.py"),
        predictions_validation_command=_PREDICTIONS_VALIDATION_COMMAND,
        device_info_path="/tmp/device.json",
        work_dir=str(tmp_path),
    )
    kwargs.update(overrides)
    return kwargs


def test_reuse_drops_nonempty_suggestions_and_validates(tmp_path: Path) -> None:
    inp = InferenceTaskInput(
        **_reuse_kwargs(
            tmp_path,
            configuration_suggestions={"direction": "try a higher temperature"},
        )
    )
    assert inp.configuration_suggestions == {}


def test_reuse_clears_every_suggestion_key(tmp_path: Path) -> None:
    inp = InferenceTaskInput(
        **_reuse_kwargs(
            tmp_path,
            configuration_suggestions={"direction": "x", "temperature": 0.8},
        )
    )
    assert inp.configuration_suggestions == {}


def test_reuse_empty_suggestions_still_valid(tmp_path: Path) -> None:
    inp = InferenceTaskInput(**_reuse_kwargs(tmp_path))
    assert inp.configuration_suggestions == {}


def test_reuse_effectively_empty_suggestions_untouched(tmp_path: Path) -> None:
    # Zero/empty/null values already mean self-select; they are not "advice",
    # so validation passes and the mapping is preserved as given.
    inp = InferenceTaskInput(
        **_reuse_kwargs(
            tmp_path,
            configuration_suggestions={"direction": "", "temperature": 0},
        )
    )
    assert inp.configuration_suggestions == {"direction": "", "temperature": 0}


def test_baseline_suggestions_still_allowed(tmp_path: Path) -> None:
    suggestions = {"direction": "prefer greedy decoding"}
    inp = InferenceTaskInput(
        **_baseline_kwargs(tmp_path, configuration_suggestions=suggestions)
    )
    assert inp.configuration_suggestions == suggestions


def test_reuse_requires_iteration_ge_1(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="iteration >= 1"):
        InferenceTaskInput(
            **_reuse_kwargs(
                tmp_path,
                iteration=0,
                configuration_suggestions={"direction": "x"},
            )
        )


def test_reuse_requires_checkpoint_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="checkpoint_path"):
        InferenceTaskInput(
            **_reuse_kwargs(
                tmp_path,
                checkpoint_path="",
                configuration_suggestions={"direction": "x"},
            )
        )


def test_reuse_requires_inference_config_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="inference_config"):
        InferenceTaskInput(
            **_reuse_kwargs(
                tmp_path,
                inference_config_path="",
                configuration_suggestions={"direction": "x"},
            )
        )
