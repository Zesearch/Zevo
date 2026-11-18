from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from zevo.contracts.agent_submission import DATA_ENGINE_FIELDS, submission_schema
from zevo.contracts.data import DataResult
from zevo.contracts.infrastructure import InfraResult
from zevo.contracts.train import TrainResult
from zevo.contracts.inference import InferenceResult
from zevo.contracts.model_registry import RegisterResult
from zevo.engine.agent.drivers.base import DriverRunResult, submission_blueprint
from zevo.engine.agent.drivers.codex_cli import _codex_output_schema
from zevo.engine.agent.loader import load_agent


def common(status="succeeded", **fields):
    return dict(status=status, ticket_id="data-r-001", notes="",
                error_message="failure" if status == "failed" else "", **fields)


@pytest.mark.parametrize("operation,artifacts", [
    ("prepare_run_data", dict(training_dataset_path="/data", data_recipe_path="/recipe")),
    ("scope_problem", dict(scoping_result_path="/scope")),
    ("prepare_holdout_data", dict(scoring_public_path="/questions")),
])
def test_data_submission_excludes_engine_fields_and_retains_internal_shape(operation, artifacts):
    schema = submission_schema(DataResult, operation)
    strict = _codex_output_schema(schema)
    assert not DATA_ENGINE_FIELDS.intersection(strict["properties"])
    output = schema(**common(operation=operation, **artifacts))
    canonical = DriverRunResult(output=output).output
    assert isinstance(canonical, DataResult)
    assert canonical.operation == operation
    assert canonical.decontamination_checked is False
    # Engine enrichment and serialized records retain the old shape.
    enriched = canonical.model_copy(update={"decontamination_checked": True,
                                            "decontamination_removed_rows": 2})
    assert enriched.model_dump()["decontamination_removed_rows"] == 2


@pytest.mark.parametrize("field", sorted(DATA_ENGINE_FIELDS))
@pytest.mark.parametrize("status", ["succeeded", "failed", "deferred"])
def test_agent_cannot_submit_engine_fields_even_on_failure_or_defer(field, status):
    schema = submission_schema(DataResult, "prepare_run_data")
    with pytest.raises(ValidationError) as error:
        schema(**common(status, operation="prepare_run_data", **{field: 0}))
    assert any(e["loc"] == (field,) and e["type"] == "extra_forbidden"
               for e in error.value.errors())


def test_data_wrong_operation_and_artifacts_are_rejected():
    schema = submission_schema(DataResult, "scope_problem")
    with pytest.raises(ValidationError, match="training_dataset_path"):
        schema(**common(scoping_result_path="/scope", training_dataset_path="/train"))
    with pytest.raises(ValidationError, match="scope_problem"):
        schema(**common(operation="prepare_run_data", scoping_result_path="/scope"))
    with pytest.raises(ValidationError, match="scoping_result_path"):
        schema(**common())


def test_remote_data_artifact_validation_is_preserved():
    schema = submission_schema(DataResult, "prepare_run_data")
    with pytest.raises(ValidationError, match="remote_dataset_spec_path"):
        schema(**common(remote_dataset_path="/remote/data", data_recipe_path="/recipe"))
    deferred = schema(**common("deferred", slurm_script_path="/data.sbatch"))
    assert deferred.to_result().status == "deferred"


def test_infra_success_failure_and_release_have_distinct_resource_reporting():
    provision = submission_schema(InfraResult, "provision")
    release = submission_schema(InfraResult, "release")
    assert "provider" not in provision.model_fields
    assert "device_info_path" not in release.model_fields
    assert provision(**common(device_info_path="/device")).to_result().provider == ""
    resource = dict(provider="cloud", instance_id="gpu-1", cloud_backend="vastai")
    failed = provision(**common("failed", failure_resource={**resource, "auto_release": True}))
    assert failed.to_result().instance_id == "gpu-1"
    assert failed.to_result().auto_release
    with pytest.raises(ValidationError, match="failure_resource"):
        provision(**common(device_info_path="/device", failure_resource=resource))
    assert release(**common(released_resource=resource)).to_result().provider == "cloud"
    with pytest.raises(ValidationError, match="provider and instance_id"):
        release(**common())


def evidence(scope, **fields):
    return dict(scope=scope, reason="diagnosed", evidence=["/log"],
                next_action="repair", **fields)


@pytest.mark.parametrize("result,operation,extra", [
    (DataResult, "prepare_run_data", {}), (InfraResult, "provision", {}),
    (TrainResult, "train", {}), (InferenceResult, "run_inference", {}),
    (RegisterResult, "", {"registry_path": "", "register_script_path": ""}),
])
def test_recovery_retry_and_review_cannot_be_combined(result, operation, extra):
    schema = submission_schema(result, operation)
    assert "execution_retry_reason" not in schema.model_fields
    current = schema(**common("failed", **extra, failure_recovery=evidence(
        "current_task", execution_retry_reason="retry corrected command")))
    assert current.to_result().execution_retry_reason == "retry corrected command"
    review = schema(**common("failed", **extra, failure_recovery=evidence("upstream_input")))
    assert review.to_result().execution_retry_reason == ""
    with pytest.raises(ValidationError):
        schema(**common("failed", **extra, failure_recovery=evidence(
            "upstream_input", execution_retry_reason="invalid retry")))
    with pytest.raises(ValidationError, match="only valid for failed"):
        schema(**common("succeeded", **extra, failure_recovery=evidence("upstream_input")))
    _codex_output_schema(schema)  # All branches remain compatible with strict output.


def test_binding_does_not_mutate_blueprint_or_affect_supervisor():
    blueprint = load_agent("data")
    bound = submission_blueprint(blueprint, SimpleNamespace(operation="scope_problem"))
    assert bound is not blueprint
    assert blueprint.output_schema is DataResult
    assert "remote_dataset_path" not in bound.output_schema.model_fields
    supervisor = load_agent("orchestrator")
    assert submission_blueprint(supervisor, SimpleNamespace()) is supervisor


def test_train_configuration_review_roundtrips_without_retry_permission():
    proposal = dict(original_config_path="/original.yaml", proposed_config_path="/candidate.yaml",
                    original_sha256="a" * 64, proposed_sha256="b" * 64,
                    changes=["learning rate"], restart_mode="restart_parent")
    recovery = evidence("fixed_configuration", training_configuration_change=proposal)
    schema = submission_schema(TrainResult, "train")
    result = schema(**common("failed", failure_recovery=recovery)).to_result()
    assert result.failure_recovery.training_configuration_change.proposed_sha256 == "b" * 64
    assert result.execution_retry_reason == ""
    with pytest.raises(ValidationError):
        submission_schema(DataResult, "prepare_run_data")(
            **common("failed", failure_recovery=recovery))


def test_resource_replacement_still_requires_reason():
    schema = submission_schema(TrainResult, "train")
    with pytest.raises(ValidationError, match="replace_resource requires execution_retry_reason"):
        schema(**common("failed", failure_recovery=evidence(
            "current_task", execution_retry_action="replace_resource")))
    result = schema(**common("failed", failure_recovery=evidence(
        "current_task", execution_retry_action="replace_resource",
        execution_retry_reason="assigned device is broken"))).to_result()
    assert result.execution_retry_action == "replace_resource"


@pytest.mark.parametrize("release", [False, True])
def test_infra_binding_uses_real_runtime_input(release, tmp_path):
    from zevo.contracts.infrastructure import InfraTaskInput
    inp = InfraTaskInput(
        ticket_id="infra-r-001", run_id="r", release=release,
        purpose="" if release else "train", provider="cluster", num_gpus=1,
        auto_release=False, instance_id="existing" if release else "",
        work_dir=str(tmp_path), remote_dir="/scratch/zevo",
        ssh_host="login.example.com", ssh_port=22, ssh_user="user", ssh_key_path="/k",
        gpu_lease_request_schema={}, gpu_lease_grant_schema={},
        infra_instance_create_schema={}, infra_instance_patch_schema={},
        infra_instance_response_schema={}, device_info_schema={},
        device_info_validation_command="true",
    )
    assert not hasattr(inp, "operation")
    bound = submission_blueprint(load_agent("infrastructure"), inp)
    schema = bound.output_schema
    operation = "release" if release else "provision"
    assert schema.model_fields["operation"].default == operation
    assert ("device_info_path" in schema.model_fields) is not release
    output = schema(**common(**(
        {"released_resource": {"provider": "cluster", "instance_id": "existing"}}
        if release else {"device_info_path": "/device"})))
    assert DriverRunResult(output=output).output.operation == operation


def test_invalid_binding_is_explicitly_pre_dispatch():
    from zevo.engine.agent.drivers.base import SubmissionContractError
    with pytest.raises(SubmissionContractError, match="unsupported Infra"):
        submission_blueprint(load_agent("infrastructure"), SimpleNamespace())
