from __future__ import annotations

import json

import pytest
import yaml

from zevo.contracts.configuration import load_inference_config, _main
from zevo.contracts.inference_core import inference_core, validate_inference_core
from tests.test_sequential_yaml_workflow import _write_inference_config


@pytest.fixture
def config(tmp_path):
    return load_inference_config(str(_write_inference_config(tmp_path)))


def test_core_excludes_benchmark_data_and_budget(config):
    core = inference_core(config)
    changed = config.model_copy(deep=True)
    changed.measurement.max_new_tokens *= 4
    changed.measurement.inference_config.update(
        input_fields=["other"], inference_query="Other task: {other}", answer_regex="[ABC]")
    changed.implementation_config["llm_kwargs"] = {"max_model_len": 4096}
    changed.prompt_example.input_values = {"private": "NEVER_SHARE"}
    validate_inference_core(changed, core)
    assert "NEVER_SHARE" not in json.dumps(inference_core(changed))
    assert "prompt_example" not in core
    assert "inference_config" not in core


@pytest.mark.parametrize("field,value", [
    ("system_prompt", "Reason briefly and finish with Answer: <letter>."),
    ("stop_token_ids", [2, 3]),
    ("temperature", 0.7),
    ("template_kwargs", {"enable_thinking": True}),
    ("chat_template_hash", "b" * 64),
    ("ignore_eos", True),
    ("add_generation_prompt", False),
])
def test_core_rejects_independent_heldout_choices(config, field, value):
    expected = inference_core(config)
    changed = config.model_copy(deep=True)
    if field == "system_prompt":
        changed.prompt.system_prompt = value
    elif field == "temperature":
        changed.measurement.temperature = value
    elif field == "ignore_eos":
        changed.implementation_config["sampling_params"] = {field: value}
    elif field == "add_generation_prompt":
        changed.implementation_config[field] = value
    else:
        setattr(changed, field, value)
    with pytest.raises(ValueError, match="shared Validation core"):
        validate_inference_core(changed, expected)


def test_sampling_namespaces_cannot_override_stop_behavior(config):
    config.implementation_config.update(sampling_extra={"ignore_eos": False},
                                        sampling_params={"ignore_eos": True})
    with pytest.raises(ValueError, match="conflicting sampling"):
        inference_core(config)


def test_agent_validator_enforces_shared_core(tmp_path, config):
    path = _write_inference_config(tmp_path)
    core_path = tmp_path / "core.json"
    core_path.write_text(json.dumps(inference_core(config)))
    args = ["validate", "inference", str(path), "--shared-core", str(core_path)]
    assert _main(args) == 0
    data = yaml.safe_load(path.read_text())
    data["measurement"]["seed"] = 17
    path.write_text(yaml.safe_dump(data))
    assert _main(args) == 1


@pytest.mark.parametrize("member_state", ["matching", "different", "missing"])
def test_submit_checks_every_member_before_gpu(tmp_path, config, member_state):
    from zevo.contracts.inference import InferenceTaskInput, InferenceResult
    from zevo.engine.run.runner import _extract_summary_artifact_meta
    from tests.test_inference_reuse_suggestions import _baseline_kwargs

    primary = _write_inference_config(tmp_path)
    member_dir = tmp_path / "member"
    member_dir.mkdir()
    if member_state != "missing":
        member_path = _write_inference_config(member_dir)
        data = yaml.safe_load(member_path.read_text())
        data["measurement"]["max_new_tokens"] = 2048
        if member_state == "different":
            data["measurement"]["seed"] = 17
        member_path.write_text(yaml.safe_dump(data))
    inp = InferenceTaskInput(**_baseline_kwargs(
        tmp_path, shared_inference_core=inference_core(config),
        suite_members=[dict(name="other", test_set_name="other",
                            scoring_set="/other.csv", sample_submission="/sample.csv",
                            inference_data_profile_path="/profile.json",
                            predictions_validation_command="validate-predictions",
                            configuration_mode="select", work_dir=str(member_dir))],
    ))
    result = InferenceResult(ticket_id=inp.ticket_id, status="deferred", notes="", error_message="",
                             inference_config_path=str(primary))
    status, summary, _, _ = _extract_summary_artifact_meta(
        result, inp=inp, agent_id="inference", work_dir=str(tmp_path))
    if member_state == "matching":
        assert status == "deferred", summary
    else:
        assert status == "failed"
        assert "shared inference core validation failed" in summary
