from pathlib import Path

import pytest

from tests.test_sequential_yaml_workflow import _write_inference_config
from zevo.contracts.configuration import TrainRunConfig, file_sha256, train_method_contracts
from zevo.contracts.train import TrainExecutionContract, TrainTaskInput
from zevo.engine.agent.drivers.stub import _stub_train
from zevo.engine.method.branch import active_training_method, method_branch_problem
from zevo.engine.run.failure_policy import classify_failure
from zevo.engine.run.runner import _validate_specialist_yaml


TRANSITION = {
    "level": "method", "next_branch": "rft", "exhausted_branch": "sft",
    "validation_evidence": "SFT data and optimization directions exhausted after plateau.",
}


def case(tmp_path: Path, active="rft", transition=None):
    inference = _write_inference_config(tmp_path)
    common = dict(operation="train", base_model="Qwen/Qwen3-0.6B-Base",
                  inference_config_path=str(inference), data_signature="d" * 64)
    parent = _stub_train({**common, "ticket_id": "train-r-009", "iteration": 9,
                          "model_source": "base_model", "training_method_pin": "sft",
                          "parent_selection_rationale": "Baseline weights"}, tmp_path / "parent")
    payload = dict(**common, ticket_id="train-r-013", iteration=13,
                   model_source="checkpoint", parent_checkpoint_path=parent.checkpoint_path,
                   parent_train_config_path=parent.train_config_path,
                   parent_selection_rationale="Reuse the SFT champion while exploring RFT.",
                   active_training_method=active, configuration_suggestions={"training_method": "rft"},
                   branch_transition=transition or {"level": "none"})
    result = _stub_train(payload, tmp_path / "current")
    inp = TrainTaskInput(
        **payload, run_id="r", dataset_path="/tmp/data.jsonl", dataset_rows=1,
        validation_dataset_path="/tmp/validation.jsonl", validation_answer_fields=["answer"],
        expected_inference_config_sha256=file_sha256(inference),
        train_config_schema=TrainRunConfig.model_json_schema(),
        training_method_contracts=train_method_contracts(), config_validation_command="validate",
        telemetry_helper_path="/tmp/telemetry.py", checkpoint_helper_path="/tmp/checkpoint.py",
        execution_contract=TrainExecutionContract(required_environment={"RUN_ID": "r"}, slurm_step_name="zevo-train"),
        device_info_path="/tmp/device.json", work_dir=str(tmp_path / "current"), generation_backend="vllm",
    )
    return inp, result


@pytest.mark.parametrize("status", ["succeeded", "deferred"])
def test_rft_can_reuse_sft_champion_at_submit_and_collect(tmp_path, status):
    inp, result = case(tmp_path)
    if status == "deferred":
        result = result.model_copy(update={"status": status, "checkpoint_path": ""})
    warnings = []
    config = _validate_specialist_yaml(result, inp, advisory_warnings=warnings)
    assert config["training_method"] == "rft"
    assert config["method_diversity_status"] == "retained_in_branch"
    assert not warnings


@pytest.mark.parametrize("status", ["succeeded", "deferred"])
def test_actual_method_change_requires_evidence_before_compute(tmp_path, status):
    inp, result = case(tmp_path, active="sft")
    result = result.model_copy(update={"status": status})
    with pytest.raises(ValueError, match="upstream method branch conflict") as exc:
        _validate_specialist_yaml(result, inp)
    assert classify_failure(str(exc.value), agent_id="train").route == "orchestrator"
    approved = inp.model_copy(update={"branch_transition": TRANSITION})
    assert _validate_specialist_yaml(result, approved)["training_method"] == "rft"


def test_parent_identity_is_still_enforced(tmp_path):
    inp, result = case(tmp_path)
    with pytest.raises(ValueError, match="parent_model breaks the model chain"):
        _validate_specialist_yaml(result, inp.model_copy(update={"parent_checkpoint_path": "/other/model"}))


def test_wrong_specialist_method_is_self_repairable(tmp_path):
    inp, result = case(tmp_path, active="sft")
    inp = inp.model_copy(update={"configuration_suggestions": {"training_method": "sft"}})
    with pytest.raises(ValueError, match="explicitly selects") as exc:
        _validate_specialist_yaml(result, inp)
    assert classify_failure(str(exc.value), agent_id="train").route == "self"


def test_history_uses_latest_round_not_champion_or_future_round():
    history = [
        {"source": "trained", "iteration": 9, "training_method": "sft", "score": .9},
        {"source": "trained", "iteration": 12, "training_method": "rft", "score": .8},
        {"source": "trained", "iteration": 13, "training_method": "dpo", "score": .7},
    ]
    assert active_training_method(history, before_iteration=13) == "rft"
    assert active_training_method(history[:2]) == "rft"
    assert not method_branch_problem("rft", "rft")
    assert method_branch_problem("sft", "rft", transition={"level": "method"})
    assert not method_branch_problem("sft", "rft", transition=TRANSITION)
    assert not method_branch_problem("sft", "rft", pin="rft")


@pytest.mark.asyncio
async def test_invalid_stored_direction_is_rejected_before_resource_planning(monkeypatch, tmp_path):
    from zevo.db import Run, Ticket
    from zevo.engine.run import adaptive_training, runner

    async def no_adaptive_contract(*args):
        return {}

    monkeypatch.setattr(adaptive_training, "build_validation_contract", no_adaptive_contract)
    run = Run(history=[{"source": "trained", "iteration": 12, "training_method": "sft"}])
    ticket = Ticket(id="train-r-013", iteration=13)
    # No DB/device route is supplied: the conflict must be caught before either
    # can be used, including when a stored work order bypasses the creation API.
    with pytest.raises(ValueError, match="upstream method branch conflict"):
        await runner._build_train_input(
            ticket, {"configuration_suggestions": {"training_method": "rft"}},
            {}, str(tmp_path), "vllm", run, None,
        )
