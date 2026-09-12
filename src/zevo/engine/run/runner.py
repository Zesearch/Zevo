"""Ticket heartbeat runner.

Responsibility: take one immutable execution Ticket from the DB and run it
through one driver heartbeat. Configuration choices are persisted as YAML
artifacts, not as hidden conversation state or a second planning activation.

Execution goes through the identity.md + Driver
abstraction (so drivers — claude_cli / codex_cli / bedrock / openrouter /
stub — are swappable).

API:
    await run_ticket(session, ticket_id) -> Ticket (with status updated)

Side effects (all in Postgres):
    - heartbeat_runs row (start/finish/exit_code/stdout_path)
    - execution_events rows (one logical phase/progress reading; replays merge)
    - heartbeat_results row (the validated structured output)
    - WorkProduct rows for verified result artifact paths
    - ticket.status updated to 'succeeded' or 'failed', summary set

The runner is intentionally agnostic of the scheduler. The daemon may process
ready tickets from different Runs concurrently; the workflow inside one Run
remains serial. `agent run` calls it for one ticket on demand.
"""
from __future__ import annotations

import ast
import asyncio
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any




def _warn(where: str, exc: BaseException) -> None:
    """Log a swallowed exception so we never silently drop errors.

    Background tasks (event flusher, cancel poller, queue puts) all
    use this — they can't raise without taking the whole heartbeat
    down, but they MUST leave a stderr trace so operators can diagnose
    missing transcript events or stuck cancels.
    """
    print(f"[runner.{where}] WARN: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from zevo.paths import work_dir_root as paths_work_dir_root
from zevo.api.artifacts import host_path
from zevo.providers import resolve_ssh_key
from zevo.engine.agent.loader import AgentBlueprint, REPO_ROOT, load_agent
from zevo.engine.agent.memory import (
    applicability_for,
    load_memory_context,
    persist_memory_updates,
)
from zevo.db import (
    AgentWakeupRequest,
    ExecutionEvent,
    HeartbeatResult,
    HeartbeatRun,
    InfraInstance,
    Run,
    SshHost,
    Ticket,
    TicketMessage,
    TicketNotice,
    TranscriptEvent,
    WorkProduct,
)
from zevo.engine.agent.drivers import Driver, get_driver
from zevo.engine.observe.live_markers import LiveMarkerReader
from zevo.engine.observe.markers import scan_text
from zevo.engine.cost.pricing import accumulate_usage, estimate_cost
from zevo.engine.run.scheduler.bindings import input_path, resolve_input_bindings
from zevo.engine.observe.score_monitor import extract_score, materialize_metrics_artifact
from zevo.engine.observe.run_metrics import (
    baseline_and_best_test,
    upsert_history_fact,
    validation_best_score,
)
from zevo.engine.method.score_direction import is_better
from zevo.engine.run.failure_policy import (
    MAX_PAYLOAD_VALIDATION_ATTEMPTS,
    MAX_REPAIR_ATTEMPTS,
    PAYLOAD_VALIDATION_FAILURE_SIGNATURE,
    classify_failure,
    repair_instruction,
)
from zevo.engine.observe import transcript_bus
from zevo.engine.artifact_validation import (
    materialize_system_scoring_artifacts,
    sanitize_training_against_scoring,
    validate_data_artifacts,
    validate_prediction_artifacts,
)

# Schema imports + per-task-type adapters live here so the runner can
# build the right TypedInput from the ticket.payload dict.
from zevo.contracts.data import (
    DataRecipe,
    DataTaskInput,
    DataResult,
    InferenceDataProfile,
    data_recipe_signature,
    load_data_recipe,
)
from zevo.contracts.evaluation import EvaluationTaskInput, EvaluationResult
from zevo.contracts.train import (
    LossSeriesSummary,
    TrainingDiagnostics,
    TrainExecutionContract,
    TrainTaskInput,
    TrainResult,
)
from zevo.contracts.freeform import FreeformInput
from zevo.contracts.inference import (
    InferenceTaskInput,
    InferenceResult,
    load_generation_diagnostics,
    summarize_generation_diagnostics,
)
from zevo.contracts.infrastructure import (
    CreateInfraInstanceBody,
    GpuLeaseGrant,
    GpuLeaseRequest,
    InfraInstanceDTO,
    InfraTaskInput,
    InfraResult,
    InfrastructureDeviceInfo,
    PatchInfraInstanceBody,
    SlurmStageJobContract,
    SLURM_STATUS_FILENAME,
    slurm_lifecycle_prologue,
    validate_device_info,
)
from zevo.contracts.configuration import (
    InferenceRunConfig,
    TrainRunConfig,
    file_sha256,
    load_inference_config,
    load_train_config,
    inference_mapping_contract,
    train_method_contracts,
    validate_adaptive_vllm_memory_config,
    validate_cluster_train_config,
)
from zevo.contracts.model_registry import (
    RegisterTaskInput,
    RegisterResult,
    RegistryEntry,
    model_tag_for_run,
    validate_registry_entry,
)
from zevo.contracts.orchestrator import (
    OrchestratorTaskInput,
    PatchRunRequest,
    SupervisorAction,
)
from zevo.contracts.tickets import (
    ArtifactBinding,
    CreateTicketBody,
    CreateTicketResponse,
    RunMode,
    TERMINAL_RUN_STATUSES,
    TERMINAL_TICKET_STATUSES,
    validate_stored_payload,
    specialist_request_payload_schemas,
    specialist_input_binding_contracts,
    specialist_stored_payload_schemas,
)


# --------------------- payload adapters ----------------------------------
# Each adapter converts the orchestrator's payload dict (stored as JSONB
# in tickets.payload) into the typed input the agent expects. The shape
# is dictated by the orchestrator's platform.md cheat-sheet; mismatches
# between cheat-sheet and Pydantic schemas are an ongoing risk. The `_require`
# helper turns any future drift into a readable
# error instead of a bare KeyError deep in stack.

def _require(payload: dict, key: str, ticket: Ticket):
    """Return payload[key] or raise with a message naming the agent, the
    ticket, the missing key, and the actual payload keys."""
    if key not in payload:
        raise KeyError(
            f"{ticket.agent_id} ticket {ticket.id!r} payload missing "
            f"required field {key!r}. Got keys: {sorted(payload.keys())}"
        )
    return payload[key]


def _customization_kwargs(customization: dict) -> dict:
    """Pass the stored per-Agent customization through without renaming it."""
    return {"customization": dict(customization or {})}


def _canonical_data_source_identity(payload: dict) -> str:
    """Return the engine-owned display identity for one Data source.

    Explicit local/Hugging Face sources retain their submitted identity, with
    only the repository's single container-to-host path conversion applied.
    A free-form acquisition query is represented by a stable digest so prose,
    whitespace, or secrets do not leak into model lineage.
    """
    source = str(
        payload.get("dataset_source") or payload.get("dataset") or ""
    ).strip()
    if source:
        return host_path(source)
    query = str(payload.get("data_query") or "").strip()
    if query:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"query:sha256:{digest}"
    return ""


def _same_local_file(left: str, right: str) -> bool:
    """Whether two reported local paths identify byte-identical files."""
    if not left or not right:
        return False
    try:
        left_path = Path(left)
        right_path = Path(right)
        if left_path.resolve() == right_path.resolve():
            return True
        return (
            left_path.is_file()
            and right_path.is_file()
            and file_sha256(left_path) == file_sha256(right_path)
        )
    except OSError:
        return False


def _build_data_input(
    ticket: Ticket, payload: dict, inputs: dict, work_dir: str, run: Run,
    specialist_context: dict | None = None,
) -> BaseModel:
    held_out = ticket.lane == "held_out_test"
    operation = str(_require(payload, "operation", ticket))
    if operation == "scope_problem":
        if held_out:
            raise ValueError("Data scope_problem runs on the optimization lane only")
        return _build_scoping_input(ticket, payload, work_dir, specialist_context)
    expected_operation = "prepare_holdout_data" if held_out else "prepare_run_data"
    if operation != expected_operation:
        raise ValueError(
            f"Data payload operation={operation!r} conflicts with lane={ticket.lane!r}; "
            f"expected {expected_operation!r}"
        )
    dataset = "" if held_out else str(payload.get("dataset") or "")
    local_dataset = Path(dataset) if dataset else None
    expected_source_fingerprint = (
        file_sha256(local_dataset)
        if local_dataset is not None and local_dataset.is_file()
        else ""
    )
    recipe_validation_command = (
        "python -m zevo.contracts.data validate-recipe "
        "<absolute-recipe-json-path> <absolute-training-dataset-path>"
    )
    expected_source_identity = (
        "" if held_out else _canonical_data_source_identity(payload)
    )
    if expected_source_identity:
        recipe_validation_command += (
            " --source-identity " + shlex.quote(expected_source_identity)
        )
    if expected_source_fingerprint:
        recipe_validation_command += (
            " --source-file " + shlex.quote(str(local_dataset.resolve()))
        )
    validation_policy = str(payload.get("validation_policy") or "supplied")
    validation_fraction = float(payload.get("validation_fraction", 0.0))
    scoring_set = ""
    answer_fields: list[str] = []
    if held_out:
        scoring_set = str(_require(payload, "scoring_set", ticket))
        answer_fields = list(_require(payload, "answer_fields", ticket))
        artifact_validation_command = (
            "python -m zevo.engine.artifact_validation validate-data"
            f" --operation {shlex.quote(operation)}"
            f" --scoring-source {shlex.quote(scoring_set)}"
            " --questions <absolute-questions-only-path>"
            " --sample-submission <absolute-sample-submission-path>"
            " --profile <absolute-inference-data-profile-path>"
        )
        for answer_field in answer_fields:
            artifact_validation_command += (
                " --answer-field " + shlex.quote(str(answer_field))
            )
    else:
        artifact_validation_command = (
            "python -m zevo.engine.artifact_validation validate-training-data"
            " --training-dataset <absolute-training-dataset-path>"
        )
    return DataTaskInput(
        ticket_id=ticket.id,
        **_customization_kwargs(ticket.customization or {}),
        run_context=dict(specialist_context or {}),
        operation=operation,
        run_id=str(ticket.run_id or ""),
        test_set_name=(
            str(payload.get("test_set_name") or "test") if held_out else ""
        ),
        dataset_source="" if held_out else str(payload.get("dataset_source") or ""),
        dataset=dataset,
        dataset_split="" if held_out else str(payload.get("dataset_split") or ""),
        dataset_config="" if held_out else str(payload.get("dataset_config") or ""),
        data_query="" if held_out else str(payload.get("data_query") or ""),
        expected_source_identity=expected_source_identity,
        training_method="" if held_out else str(payload.get("training_method") or ""),
        branch_transition=(
            {} if held_out else dict(payload.get("branch_transition") or {})
        ),
        recipe_intent=(
            {} if held_out else dict(payload.get("recipe_intent") or {})
        ),
        data_intent_signature=(
            "" if held_out else str(payload.get("data_intent_signature") or "")
        ),
        expected_source_fingerprint=(
            "" if held_out else expected_source_fingerprint
        ),
        data_recipe_schema=DataRecipe.model_json_schema(),
        data_recipe_validation_command=recipe_validation_command,
        artifacts_validation_command=artifact_validation_command,
        validation_policy=validation_policy,
        validation_fraction=validation_fraction,
        scoring_set=scoring_set,
        answer_fields=answer_fields,
        metric_type=(
            str(payload.get("metric_type") or "builtin") if held_out else "builtin"
        ),
        metric=(str(payload.get("metric") or "") if held_out else ""),
        evaluation_script=(
            str(payload.get("evaluation_script") or "") if held_out else ""
        ),
        evaluator_sha256=(
            str(payload.get("evaluator_sha256") or "") if held_out else ""
        ),
        sample_submission=(
            str(payload.get("sample_submission") or "") if held_out else ""
        ),
        configuration_suggestions=(
            {} if held_out else dict(payload.get("configuration_suggestions") or {})
        ),
        configuration_pins=(
            {} if held_out else dict(payload.get("configuration_pins") or {})
        ),
        work_dir=work_dir,
    )


def _build_scoping_input(
    ticket: Ticket, payload: dict, work_dir: str,
    specialist_context: dict | None = None,
) -> BaseModel:
    """The Auto-mode work order: derive the scoring contract from the objective.

    Nothing about metric or scoring set is stamped here — deriving them IS the
    task. Training pins and hints are intentionally withheld as well: Auto
    replaces Test setup, and evaluation must not adapt itself to a proposed
    optimization configuration. The agent receives only the objective,
    optional Test query, constraints, exact ScopingResult schema, and its
    output validator.
    """
    from zevo.contracts.scoping import ScopingResult

    return DataTaskInput(
        ticket_id=ticket.id,
        **_customization_kwargs(ticket.customization or {}),
        # Generic specialist context contains agent_objective and other
        # optimization evidence. Do not let that become an indirect channel
        # around the evaluation-only scope above.
        run_context={},
        operation="scope_problem",
        run_id=str(ticket.run_id or ""),
        task_objective=str(payload.get("task_objective") or ""),
        test_query=str(payload.get("test_query") or ""),
        constraints=[str(c) for c in (payload.get("constraints") or [])],
        scoping_result_schema=ScopingResult.model_json_schema(),
        scoping_result_validation_command=(
            "python -m zevo.contracts.scoping validate <absolute-scoping-result-json-path>"
        ),
        data_recipe_schema=DataRecipe.model_json_schema(),
        data_recipe_validation_command="(not used by scope_problem)",
        artifacts_validation_command="(not used by scope_problem)",
        work_dir=work_dir,
    )


def _answer_fields(holdout: dict, lane: str) -> list[str]:
    """Return the canonical ground-truth fields for one scoring lane."""
    return list(holdout.get(f"{lane}_answer_fields") or [])


def _validation_files(holdout: dict) -> tuple[str, str]:
    """Return Validation's settled evaluator and submission shape."""
    script = str(holdout.get("validation_evaluation_script") or "")
    submission = str(holdout.get("validation_sample_submission") or "")
    return script, submission


def _num_gpus(run: Run) -> int:
    """Return the Run-level GPU maximum; zero means no upper bound."""
    return max(0, int(run.num_gpus or 0))


# The payload keys a stage may fill in from what it actually ran. Restricted on
# purpose: `__CONFIG__` also carries things that are not ticket inputs (a sample
# prompt, row counts, a timestamp), and copying those in would turn the payload
# from "the ticket" into "a second transcript".
async def _build_infra_input(
    ticket: Ticket, payload: dict, inputs: dict, work_dir: str,
    run: Run, session: AsyncSession, specialist_context: dict | None = None,
) -> BaseModel:
    # Runtime ownership lives on Run exactly once and is stamped here.
    provider = str(run.gpu_provider or "instance")
    if provider not in ("cluster", "cloud", "instance"):
        raise ValueError(f"run {ticket.run_id}: unsupported gpu_provider={provider!r}")

    release = payload.get("operation") == "release"
    cloud_backend = str(payload.get("cloud_backend") or "").strip().lower()
    if cloud_backend not in ("", "vastai", "lambda"):
        raise ValueError(
            f"ticket {ticket.id}: unsupported cloud_backend={cloud_backend!r}"
        )
    available_cloud_backends: list[str] = []
    preferred_cloud_backend = ""
    if provider == "cloud" and not release:
        if os.environ.get("VASTAI_API_KEY", "").strip():
            available_cloud_backends.append("vastai")
        if (
            os.environ.get("LAMBDA_API_KEY", "").strip()
            or os.environ.get("LAMBDA_CLOUD_API_KEY", "").strip()
        ):
            available_cloud_backends.append("lambda")
        # A per-run pin (from the launch picker, stored on the run) chooses
        # Vast.ai vs Lambda for this run; otherwise the deployment default.
        preferred_cloud_backend = (
            str(run.cloud_backend or "").strip().lower()
            or os.environ.get("ZEVO_CLOUD_BACKEND", "").strip().lower()
        )
        if preferred_cloud_backend not in ("", "vastai", "lambda"):
            raise ValueError(
                "ZEVO_CLOUD_BACKEND must be empty, 'vastai', or 'lambda'"
            )

    # Cluster/Instance SSH data comes from the verified profile selected on the
    # Run. Empty ssh_host_id retains the deployment-level .env fallback used by
    # the CLI and existing installations.
    ssh_host = ""
    ssh_port = 0
    ssh_user = ""
    ssh_key_path = ""
    ssh_password_path = ""
    remote_dir = ""
    env_setup = ""
    container_image = ""
    slurm_partition = ""
    slurm_account = ""
    slurm_qos = ""
    if provider in ("cluster", "instance"):
        ssh_host_id = str(run.ssh_host_id or "").strip()
        box = None
        if ssh_host_id:
            box = (await session.execute(
                select(SshHost).where(SshHost.id == ssh_host_id)
            )).scalar_one_or_none()
            if box is None or box.status != "verified":
                raise ValueError(
                    f"infra ticket {ticket.id!r}: ssh_host_id {ssh_host_id!r} "
                    "does not resolve to a verified SSH connection"
                )
            if box.category != provider:
                raise ValueError(
                    f"infra ticket {ticket.id!r}: SSH connection {box.label!r} "
                    f"is {box.category}, not {provider}"
                )

        if box is not None:
            ssh_host = box.host
            ssh_port = box.port
            ssh_user = box.username
            ssh_key_path = box.key_path
            ssh_password_path = box.password_path
            remote_dir = box.remote_dir
            env_setup = box.env_setup
            container_image = box.container_image
        else:
            prefix = provider.upper()
            ssh_host = os.environ.get(f"ZEVO_{prefix}_SSH_HOST", "")
            ssh_port = int(os.environ.get(f"ZEVO_{prefix}_SSH_PORT", "22") or "22")
            ssh_user = os.environ.get(f"ZEVO_{prefix}_SSH_USER", "")
            ssh_password_path = os.environ.get(
                f"ZEVO_{prefix}_SSH_PASSWORD_FILE", ""
            ).strip()
            ssh_key_path = (
                "" if ssh_password_path
                else resolve_ssh_key(f"ZEVO_{prefix}_SSH_KEY")
            )
            remote_dir = os.environ.get(f"ZEVO_{prefix}_REMOTE_DIR", "")
            env_setup = os.environ.get(f"ZEVO_{prefix}_ENV_SETUP", "")
            container_image = os.environ.get(
                f"ZEVO_{prefix}_CONTAINER_IMAGE", ""
            )

        if provider == "cluster":
            slurm_partition = slurm_partition or os.environ.get("ZEVO_CLUSTER_SLURM_PARTITION", "")
            slurm_account = slurm_account or os.environ.get("ZEVO_CLUSTER_SLURM_ACCOUNT", "")
            slurm_qos = slurm_qos or os.environ.get("ZEVO_CLUSTER_SLURM_QOS", "")

    if release or provider in ("cluster", "instance"):
        auto_release = False
    else:
        # System-owned cloud rentals are ephemeral by default. Requiring every
        # planner to remember this boolean turned an
        # omitted optional key into a leaked resource once reconciliation began
        # honoring explicit false.
        auto_release = True

    return InfraTaskInput(
        ticket_id=ticket.id,
        run_id=str(ticket.run_id or ""),
        purpose=str(payload.get("purpose") or ""),
        **_customization_kwargs(ticket.customization or {}),
        provider=provider,
        cloud_backend=cloud_backend if provider == "cloud" and release else "",
        available_cloud_backends=available_cloud_backends,
        preferred_cloud_backend=preferred_cloud_backend,
        release=release,
        num_gpus=_num_gpus(run),
        max_queue_wait_hours=float(run.max_queue_wait_hours or 0.0),
        resource_context=dict(specialist_context or {}),
        gpu_lease_request_schema=GpuLeaseRequest.model_json_schema(),
        gpu_lease_grant_schema=GpuLeaseGrant.model_json_schema(),
        infra_instance_create_schema=CreateInfraInstanceBody.model_json_schema(),
        infra_instance_patch_schema=PatchInfraInstanceBody.model_json_schema(),
        infra_instance_response_schema=InfraInstanceDTO.model_json_schema(),
        device_info_schema=InfrastructureDeviceInfo.model_json_schema(),
        device_info_validation_command=(
            "python -m zevo.contracts.infrastructure validate-device "
            "<absolute-device-info-json-path> --run-id "
            f"{shlex.quote(str(ticket.run_id or ''))} --ticket-id "
            f"{shlex.quote(ticket.id)} --provider {shlex.quote(provider)} "
            f"--num-gpus {_num_gpus(run)} --purpose "
            f"{shlex.quote(str(payload.get('purpose') or ''))}"
        ),
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        ssh_user=ssh_user,
        ssh_key_path=ssh_key_path,
        ssh_password_path=ssh_password_path,
        slurm_partition=slurm_partition if provider == "cluster" else "",
        slurm_account=slurm_account if provider == "cluster" else "",
        slurm_qos=slurm_qos if provider == "cluster" else "",
        instance_id=payload.get("instance_id", ""),
        auto_release=auto_release,
        remote_dir=remote_dir,
        env_setup=env_setup,
        container_image=container_image if provider == "cluster" else "",
        work_dir=work_dir,
    )


async def _slurm_stage_job_contract(
    *, run: Run, ticket: Ticket, work_dir: str, filename: str,
    device_info_path: str, session: AsyncSession,
) -> SlurmStageJobContract:
    """Stamp a new submission or the newest backend-observed Slurm job."""
    cluster = str(run.gpu_provider or "instance") == "cluster"
    try:
        info = InfrastructureDeviceInfo.model_validate_json(
            Path(device_info_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"cannot resolve GPU plan from {device_info_path!r}: {exc}"
        ) from exc
    expected_provider = str(run.gpu_provider or "instance")
    if info.provider != expected_provider:
        raise ValueError(
            f"stage provider {expected_provider!r} differs from Infrastructure "
            f"provider {info.provider!r}"
        )
    selected_gpus = int(info.resource_plan.num_gpus)
    maximum_gpus = _num_gpus(run)
    if selected_gpus < 1 or (maximum_gpus and selected_gpus > maximum_gpus):
        limit = str(maximum_gpus) if maximum_gpus else "unlimited"
        raise ValueError(
            f"Infrastructure selected {selected_gpus} GPUs outside Run maximum {limit}"
        )
    job_row = None
    if cluster:
        job_row = (await session.execute(
            select(InfraInstance)
            .where(
                InfraInstance.provider == "cluster",
                InfraInstance.run_id == run.id,
                InfraInstance.ticket_id == ticket.id,
                InfraInstance.instance_id != "",
            )
            .order_by(InfraInstance.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
    meta = dict(job_row.meta or {}) if job_row is not None else {}
    if cluster and info.cluster is None:
        raise ValueError("cluster stage has no cluster execution route")
    status_path = (
        str(PurePosixPath(info.cluster.workdir) / ticket.id / SLURM_STATUS_FILENAME)
        if cluster and info.cluster is not None else ""
    )
    remote_ticket_dir = str(PurePosixPath(status_path).parent) if status_path else ""
    return SlurmStageJobContract(
        enabled=cluster,
        phase="collect" if job_row is not None else "submit",
        script_path=str(Path(work_dir) / filename) if cluster else "",
        job_name=f"zevo-{ticket.id}" if cluster else "",
        status_path=status_path,
        stdout_path=(
            str(PurePosixPath(remote_ticket_dir) / "slurm-%j.out")
            if cluster else ""
        ),
        stderr_path=(
            str(PurePosixPath(remote_ticket_dir) / "slurm-%j.err")
            if cluster else ""
        ),
        lifecycle_prologue=slurm_lifecycle_prologue(status_path) if cluster else "",
        bookkeeping_row_id=job_row.id if job_row is not None else "",
        job_id=job_row.instance_id if job_row is not None else "",
        scheduler_state=(
            str(meta.get("scheduler_state") or job_row.status).upper()
            if job_row is not None else ""
        ),
        scheduler_exit_code=str(meta.get("scheduler_exit_code") or ""),
        scheduler_reason=str(meta.get("scheduler_reason") or ""),
        num_gpus=selected_gpus,
        nodes=int(info.resource_plan.nodes),
        max_queue_wait_hours=float(run.max_queue_wait_hours or 24.0),
        infra_instance_create_schema=(
            CreateInfraInstanceBody.model_json_schema() if cluster else {}
        ),
        infra_instance_patch_schema=(
            PatchInfraInstanceBody.model_json_schema() if cluster else {}
        ),
        infra_instance_response_schema=(
            InfraInstanceDTO.model_json_schema() if cluster else {}
        ),
    )


async def _build_train_input(
    ticket: Ticket, payload: dict, inputs: dict, work_dir: str, generation_backend: str,
    run: Run, session: AsyncSession, specialist_context: dict | None = None,
) -> BaseModel:
    iteration = int(ticket.iteration or 0)
    parent_model = payload.get("model_source") == "checkpoint"
    wandb_values = {
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "").strip(),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "").strip(),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "").strip(),
    }
    configured_wandb = {key for key, value in wandb_values.items() if value}
    if configured_wandb and len(configured_wandb) != len(wandb_values):
        missing = sorted(set(wandb_values) - configured_wandb)
        raise ValueError(
            "Weights & Biases configuration is incomplete; set or clear together: "
            + ", ".join(missing)
        )
    wandb_enabled = len(configured_wandb) == len(wandb_values)
    wandb_run_id = f"{str(ticket.run_id or '')[:8]}-{ticket.id}"[:128]
    wandb_url = (
        f"https://wandb.ai/{wandb_values['WANDB_ENTITY']}/"
        f"{wandb_values['WANDB_PROJECT']}/runs/{wandb_run_id}"
        if wandb_enabled else ""
    )
    required_environment = {
        "ZEVO_RUN_ID": str(ticket.run_id or ""),
        "ZEVO_TICKET_ID": ticket.id,
        "RUN_ID": str(ticket.run_id or ""),
        "TICKET_ID": ticket.id,
    }
    if wandb_enabled:
        required_environment.update({
            "WANDB_ENTITY": wandb_values["WANDB_ENTITY"],
            "WANDB_PROJECT": wandb_values["WANDB_PROJECT"],
            "WANDB_RUN_ID": wandb_run_id,
            "WANDB_NAME": f"Zevo {ticket.id}",
            "WANDB_MODE": "online",
        })
    device_info_path = input_path(inputs, "device_info")
    slurm_job = await _slurm_stage_job_contract(
        run=run, ticket=ticket, work_dir=work_dir, filename="train.sbatch",
        device_info_path=device_info_path, session=session,
    )
    return TrainTaskInput(
        ticket_id=ticket.id,
        **_customization_kwargs(ticket.customization or {}),
        run_context=dict(specialist_context or {}),
        operation="train",
        run_id=str(ticket.run_id or ""),
        iteration=iteration,
        dataset_path=input_path(inputs, "training_dataset"),
        validation_dataset_path=input_path(inputs, "validation_dataset"),
        validation_answer_fields=_answer_fields(
            dict(run.holdout or {}), "validation",
        ),
        data_signature=str(_require(payload, "data_signature", ticket)),
        inference_config_path=input_path(inputs, "inference_config"),
        expected_inference_config_sha256=file_sha256(
            input_path(inputs, "inference_config")
        ),
        train_config_schema=TrainRunConfig.model_json_schema(),
        training_method_contracts=train_method_contracts(),
        config_validation_command=(
            "python -m zevo.contracts.configuration validate train "
            "<absolute-yaml-path>"
            + (" --cluster" if slurm_job.enabled else "")
        ),
        telemetry_helper_path=str(Path(work_dir) / "zevo_train_telemetry.py"),
        execution_contract=TrainExecutionContract(
            required_environment=required_environment,
            secret_environment_names=["WANDB_API_KEY"] if wandb_enabled else [],
            tracking_provider="weights_and_biases" if wandb_enabled else "",
            tracking_url=wandb_url,
            slurm_step_name=f"zevo-{ticket.id}",
        ),
        base_model=_require(payload, "base_model", ticket),
        model_source=str(payload.get("model_source") or "base_model"),
        parent_selection_rationale=str(
            _require(payload, "parent_selection_rationale", ticket)
        ),
        parent_checkpoint_path=(
            input_path(inputs, "parent_checkpoint") if parent_model else ""
        ),
        parent_train_config_path=(
            input_path(inputs, "parent_train_config") if parent_model else ""
        ),
        branch_transition=dict(payload.get("branch_transition") or {}),
        training_method_pin=str(payload.get("training_method_pin") or ""),
        method_config_pins=dict(payload.get("method_config_pins") or {}),
        loss_objective_pins=dict(payload.get("loss_objective_pins") or {}),
        configuration_suggestions=dict(payload.get("configuration_suggestions") or {}),
        configuration_pins=dict(payload.get("configuration_pins") or {}),
        device_info_path=device_info_path,
        slurm_job=slurm_job,
        work_dir=work_dir,
        generation_backend=generation_backend,
    )


def _eval_set_of(payload: dict, ticket: Ticket) -> str:
    """Return the canonical stored ``scoring_set`` for this Ticket."""
    value = str(payload.get("scoring_set") or "").strip()
    if value:
        return value
    raise KeyError(f"ticket {ticket.id!r} has no scoring_set and no Run scoring specification")


async def _build_inference_input(
    ticket: Ticket, payload: dict, inputs: dict, work_dir: str, generation_backend: str,
    run: Run, session: AsyncSession, specialist_context: dict | None = None,
) -> BaseModel:
    # Ticket creation has already resolved the lane's immutable questions-only
    # set and output template. Read the stored work order verbatim; consulting
    # Run state again would make Ticket.payload disagree with what actually ran.
    scoring_set = _eval_set_of(payload, ticket)
    submission = str(payload.get("sample_submission") or "")
    device_info_path = input_path(inputs, "device_info")
    inference_config_path = input_path(
        inputs, "inference_config", required=False,
    )
    return InferenceTaskInput(
        ticket_id=ticket.id,
        **_customization_kwargs(ticket.customization or {}),
        run_context=dict(specialist_context or {}),
        operation="run_inference",
        run_id=str(ticket.run_id or ""),
        iteration=int(ticket.iteration or 0),
        test_set_name=str(payload.get("test_set_name") or ""),
        model_source=str(payload.get("model_source") or "base_model"),
        configuration_mode=(
            "reuse" if inference_config_path else "select"
        ),
        scoring_set=scoring_set,
        sample_submission=submission,
        checkpoint_path=(
            input_path(inputs, "checkpoint")
            if payload.get("model_source") == "checkpoint" else ""
        ),
        base_model=_require(payload, "base_model", ticket),
        branch_transition=dict(payload.get("branch_transition") or {}),
        inference_data_profile_path=input_path(
            inputs, "inference_data_profile", required=False,
        ),
        inference_config_path=inference_config_path,
        inference_config_schema=InferenceRunConfig.model_json_schema(),
        inference_mapping_contract=inference_mapping_contract(),
        config_validation_command=(
            "python -m zevo.contracts.configuration validate inference "
            "<absolute-yaml-path>"
            + (
                " --adaptive-vllm-memory"
                if generation_backend == "vllm" else ""
            )
        ),
        memory_helper_path=(
            str(Path(work_dir) / "zevo_inference_memory.py")
            if generation_backend == "vllm" else ""
        ),
        predictions_validation_command=(
            "python -m zevo.engine.artifact_validation validate-predictions "
            "--predictions <absolute-predictions-csv-path> "
            f"--questions {shlex.quote(scoring_set)} "
            f"--sample-submission {shlex.quote(submission)}"
        ),
        reusable_predict_script_path=input_path(
            inputs, "predict_script", required=False,
        ),
        option_scoring_helper_path=str(
            Path(work_dir) / "zevo_option_scoring.py"
        ),
        configuration_suggestions=dict(payload.get("configuration_suggestions") or {}),
        configuration_pins=dict(payload.get("configuration_pins") or {}),
        device_info_path=device_info_path,
        slurm_job=await _slurm_stage_job_contract(
            run=run, ticket=ticket, work_dir=work_dir, filename="predict.sbatch",
            device_info_path=device_info_path, session=session,
        ),
        work_dir=work_dir,
        generation_backend=generation_backend,
    )


def _build_evaluation_input(
    ticket: Ticket, payload: dict, inputs: dict, work_dir: str, run: Run,
) -> EvaluationTaskInput:
    # EvaluationTaskInput has no work_dir field. Evaluation gets the FULL set
    # — the half with the answers — whichever split this ticket is scoring.
    #
    # Ticket creation stamps the owning lane's full set, evaluator, sample, and answer fields.
    # The persisted payload is therefore sufficient to reproduce this stage.
    if ticket.customization:
        raise ValueError(
            "Evaluation is a deterministic system stage and cannot carry "
            "Agent customization"
        )
    scoring_set = _eval_set_of(payload, ticket)
    script = str(payload.get("evaluation_script") or "")
    answer_fields = list(payload.get("answer_fields") or [])
    return EvaluationTaskInput(
        ticket_id=ticket.id,
        **_customization_kwargs(ticket.customization or {}),
        predictions_path=input_path(inputs, "predictions"),
        test_set_name=str(payload.get("test_set_name") or ""),
        scoring_set=scoring_set,
        evaluation_script=script,
        evaluator_sha256=str(payload.get("evaluator_sha256") or ""),
        sample_submission=str(payload.get("sample_submission") or ""),
        answer_fields=answer_fields,
        # Task-owned metric, stamped on every evaluation invocation. The script
        # emits this metric's stable top-level `score`, or the built-in scorer
        # computes the named metric when no script is supplied.
        metric=_require(payload, "metric", ticket),
        evaluation_config=dict(payload.get("evaluation_config") or {}),
    )


def _build_registry_input(
    ticket: Ticket, payload: dict, inputs: dict, work_dir: str
) -> RegisterTaskInput:
    device_info_path = input_path(inputs, "device_info", required=False)
    expected_version_tag = model_tag_for_run(str(ticket.run_id or ""))
    return RegisterTaskInput(
        ticket_id=ticket.id,
        run_id=str(ticket.run_id or ""),
        iteration=int(ticket.iteration or 0),
        expected_version_tag=expected_version_tag,
        registry_entry_schema=RegistryEntry.model_json_schema(),
        registry_validation_command=(
            "python -m zevo.contracts.model_registry validate-entry "
            "<absolute-registry-yaml-path> --run-id "
            f"{shlex.quote(str(ticket.run_id or ''))} --version-tag "
            f"{shlex.quote(expected_version_tag)}"
        ),
        **_customization_kwargs(ticket.customization or {}),
        checkpoint_path=input_path(inputs, "checkpoint"),
        train_config_path=input_path(inputs, "train_config"),
        checkpoint_is_remote=bool(payload.get("checkpoint_is_remote", False)),
        metrics_path=input_path(inputs, "metrics"),
        device_info_path=device_info_path,
        base_model=_require(payload, "base_model", ticket),
        training_method=_require(payload, "training_method", ticket),
        # Registry writes a host-visible YAML snapshot. Resolve the display form
        # here so the specialist can copy this provenance verbatim instead of
        # having to guess which input paths need /app stripped.
        dataset_source=host_path(str(payload.get("dataset_source") or "")),
        task_objective=_require(payload, "task_objective", ticket),
        metric=_require(payload, "metric", ticket),
        metric_direction=_require(payload, "metric_direction", ticket),
        registry_path=str(Path(work_dir) / "registry.yaml"),
        work_dir=work_dir,
    )


def _build_orchestrate_input(ticket: Ticket, payload: dict) -> OrchestratorTaskInput:
    user_request = dict(payload.get("user_request") or {})
    # Enforce the boundary at invocation time too, so an already-created Run
    # whose supervisor payload predates the isolation change cannot keep seeing
    # Validation on later wakeups.
    for key, empty in (
        ("validation_set", ""),
        ("validation_answer_fields", []),
        ("validation_sample_submission", ""),
        ("validation_evaluation_script", ""),
        ("validation_evaluator_sha256", ""),
    ):
        user_request[key] = empty
    return OrchestratorTaskInput(
        ticket_id=ticket.id,
        iteration=int(ticket.iteration or 0),
        trigger=payload.get("trigger") or {},
        task_objective=str(payload.get("task_objective") or ""),
        agent_objective=str(payload.get("agent_objective") or ""),
        user_request=user_request,
        history=payload.get("history") or [],
        run_id=ticket.run_id,
        supervisor_ticket_id=ticket.id,
        ticket_creation_schema=CreateTicketBody.model_json_schema(),
        ticket_record_schema=CreateTicketResponse.model_json_schema(),
        ticket_creation_response_id_command=(
            "python -m zevo.contracts.tickets created-ticket-id"
        ),
        specialist_request_payload_schemas=specialist_request_payload_schemas(),
        specialist_stored_payload_schemas=specialist_stored_payload_schemas(),
        specialist_input_binding_contracts=specialist_input_binding_contracts(),
        artifact_binding_schema=ArtifactBinding.model_json_schema(),
        run_patch_schema=PatchRunRequest.model_json_schema(),
        mode=payload["mode"],
        customizations=payload.get("customizations") or {},
        validation_rows=int(payload.get("validation_rows", 0) or 0),
        budget=payload.get("budget") or {},
        dataset_profile=payload.get("dataset_profile") or {},
        runtime=payload["runtime"],
    )


def _build_freeform_input(
    *, agent_id: str, ticket: Ticket, payload: dict, work_dir: str
) -> FreeformInput:
    attachments = payload.get("attachments") or []
    if not isinstance(attachments, list):
        attachments = [str(attachments)]
    return FreeformInput(
        ticket_id=ticket.id,
        agent_id=agent_id,
        request=str(payload.get("request") or ""),
        attachments=[str(a) for a in attachments],
        work_dir=work_dir,
        run_id=ticket.run_id or "",
    )


async def _build_input(
    *, agent_id: str, ticket: Ticket, payload: dict, inputs: dict,
    work_dir: str,
    generation_backend: str, run: Run, session: AsyncSession,
    specialist_context: dict | None = None,
) -> BaseModel:
    # Freeform tickets (a direct user call) bypass the per-agent typed
    # builders -- a single FreeformInput is built and passed through. The agent's
    # ## FREEFORM MODE doc section tells it how to interpret the NL request.
    if ticket.input_format == "freeform":
        return _build_freeform_input(
            agent_id=agent_id, ticket=ticket, payload=payload, work_dir=work_dir
        )
    if agent_id == "orchestrator":
        return _build_orchestrate_input(ticket, payload)
    if agent_id == "data":
        return _build_data_input(
            ticket, payload, inputs, work_dir, run, specialist_context,
        )
    if agent_id == "infrastructure":
        return await _build_infra_input(
            ticket, payload, inputs, work_dir, run, session, specialist_context,
        )
    if agent_id == "train":
        return await _build_train_input(
            ticket, payload, inputs, work_dir, generation_backend, run,
            session, specialist_context,
        )
    if agent_id == "inference":
        return await _build_inference_input(
            ticket, payload, inputs, work_dir, generation_backend, run,
            session, specialist_context,
        )
    if agent_id == "evaluation":
        return _build_evaluation_input(ticket, payload, inputs, work_dir, run)
    if agent_id == "registry":
        return _build_registry_input(ticket, payload, inputs, work_dir)
    raise ValueError(f"no input builder for agent_id={agent_id!r}")


def _resolve_heldout_payload_assets(ticket: Ticket, payload: dict) -> dict:
    """Resolve logical Test paths only at held-out execution time.

    Ticket creation may run in the optimization scheduler, which deliberately
    has no private holdout mount. Resolving there stamps an unreadable logical
    path into the work order. The held-out scheduler owns the mount, so it
    resolves a transient payload copy immediately before building Agent input;
    the private physical path is never persisted back to the Ticket.
    """
    resolved = dict(payload or {})
    if ticket.lane != "held_out_test":
        return resolved

    from zevo.holdout_storage import resolve_asset

    for key in ("scoring_set", "sample_submission", "evaluation_script"):
        value = str(resolved.get(key) or "").strip()
        if value:
            resolved[key] = resolve_asset(value)
    return resolved


async def _build_specialist_context(
    session: AsyncSession, *, run: Run, ticket: Ticket, payload: dict,
) -> dict:
    """Build engine-owned evidence without turning suggestions into commands."""

    if ticket.lane == "held_out_test":
        return {}

    supervisor = (await session.execute(select(Ticket).where(
        Ticket.run_id == run.id,
        Ticket.agent_id == "orchestrator",
    ).limit(1))).scalar_one_or_none()
    supervisor_payload = dict(supervisor.payload or {}) if supervisor else {}
    user_request = dict(supervisor_payload.get("user_request") or {})
    for key in (
        "test_set", "test_answer_fields", "test_sample_submission",
        "evaluation_script", "evaluator_sha256",
    ):
        user_request.pop(key, None)
    if ticket.agent_id == "inference":
        for key in (
            "validation_set", "validation_answer_fields",
            "validation_sample_submission",
        ):
            user_request.pop(key, None)

    previous_iteration = max(0, int(ticket.iteration or 0) - 1)
    prior_outputs: dict[str, dict] = {}
    if int(ticket.iteration or 0) > 0:
        rows = (await session.execute(
            select(Ticket.agent_id, HeartbeatResult.output)
            .join(HeartbeatResult, HeartbeatResult.ticket_id == Ticket.id)
            .where(
                Ticket.run_id == run.id,
                Ticket.iteration == previous_iteration,
                Ticket.lane == "optimization",
                Ticket.agent_id.in_(("train", "evaluation")),
                HeartbeatResult.status.in_(("succeeded", "degraded")),
            )
            .order_by(HeartbeatResult.created_at.desc())
        )).all()
        for agent_id, output in rows:
            prior_outputs.setdefault(str(agent_id), dict(output or {}))

    from zevo.engine.cost.budget import snapshot_for_run
    budget = await snapshot_for_run(session, run.id)
    validation_history = list(run.history or [])
    # An auto Run has no metric until its scoping Ticket derives one; showing
    # the column default here would read as a decision already made.
    scoring_pending = not bool(getattr(run, "scoring_settled", True))
    context: dict = {
        "task_objective": run.task_objective,
        "agent_objective": run.agent_objective,
        "metric": "" if scoring_pending else run.validation_metric,
        "metric_direction": "" if scoring_pending else run.validation_metric_direction,
        "dataset_profile": dict(supervisor_payload.get("dataset_profile") or {}),
        "prior_train_result": prior_outputs.get("train", {}),
        "prior_validation_result": prior_outputs.get("evaluation", {}),
        "validation_history": validation_history,
        "used_training_methods": [
            str(row.get("training_method"))
            for row in validation_history
            if isinstance(row, dict) and str(row.get("training_method") or "")
        ],
        "runtime": {
            "gpu_provider": run.gpu_provider,
            "num_gpus": run.num_gpus,
            "generation_backend": run.generation_backend,
            "max_queue_wait_hours": float(run.max_queue_wait_hours or 0.0),
        },
        "budget": {
            "max_cost_usd": budget.max_cost_usd,
            "spent_usd": budget.spent_usd,
            "remaining_usd": budget.remaining_usd,
            "projected_next_iteration_usd": budget.projected_next_iteration_usd,
            "can_afford_next_iteration": budget.can_afford_next_iteration,
            "projection_basis": budget.projection_basis,
            "max_runtime_hours": budget.max_runtime_hours,
            "max_queue_wait_hours": budget.max_queue_wait_hours,
            "queue_wait_hours": budget.queue_wait_hours,
            "elapsed_runtime_hours": budget.elapsed_runtime_hours,
            "remaining_runtime_hours": budget.remaining_runtime_hours,
            "over_time_limit": budget.over_time_limit,
            "projected_next_iteration_hours": budget.projected_next_iteration_hours,
            "can_finish_next_iteration_in_time": budget.can_finish_next_iteration_in_time,
            "runtime_projection_basis": budget.runtime_projection_basis,
        },
        "iteration_horizon": {
            "budget": int(run.iteration_budget or 0),
            "completed": int(run.iterations_completed or 0),
            "remaining": max(
                0,
                int(run.iteration_budget or 0) - int(run.iterations_completed or 0),
            ) if int(run.iteration_budget or 0) > 0 else 0,
        },
    }
    if ticket.agent_id == "data":
        # The cached dataset profile can include scoring-set-derived task-shape
        # hints in older Runs. Data may use scalar evaluation outcomes to judge
        # whether a completed training experiment helped, but it must select
        # and transform training sources without Validation content/statistics.
        context.pop("dataset_profile", None)
        prior_validation = dict(context.get("prior_validation_result") or {})
        context["prior_validation_result"] = {
            key: value for key, value in prior_validation.items()
            if key in {"status", "metric", "score", "metrics", "summary"}
        }
    # Infrastructure needs model/data scale as advisory capacity evidence. For
    # every execution Specialist, the complete stored payload is the sole
    # authority; repeating user pins in run_context would create a hidden
    # second configuration source.
    if ticket.agent_id == "infrastructure":
        context["user_request"] = user_request
        context["decision_pins"] = dict(run.decision_pins or {})
    return context


def _artifact_binding(role: str, source: Ticket) -> dict[str, str]:
    return {"artifact_role": role, "source_ticket_id": source.id}


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _training_diagnostics_from_events(events: list[Any]) -> TrainingDiagnostics:
    """Summarize the newest concrete Trainer attempt without Agent inference."""

    progress: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, dict):
            row = event
        else:
            row = {
                "attempt_id": getattr(event, "attempt_id", ""),
                "event_type": getattr(event, "event_type", ""),
                "phase": getattr(event, "phase", ""),
                "current_step": getattr(event, "current_step", 0),
                "total_steps": getattr(event, "total_steps", 0),
                "loss": getattr(event, "loss", -1.0),
                "extras": getattr(event, "extras", {}) or {},
                "ts": getattr(event, "ts", None),
            }
        if str(row.get("event_type") or "") == "progress":
            progress.append(row)
    if not progress:
        return TrainingDiagnostics()

    # Marker ingestion preserves process order even when a remote clock is
    # unavailable and several swept rows receive nearly identical timestamps.
    # The last observed progress row therefore identifies the newest attempt
    # more reliably than comparing clocks across machines.
    newest = progress[-1]
    attempt_id = str(newest.get("attempt_id") or "")
    selected = sorted(
        [row for row in progress if str(row.get("attempt_id") or "") == attempt_id],
        key=lambda row: (int(row.get("current_step") or 0), str(row.get("ts") or "")),
    )
    # One remote marker may be observed live by the Slurm watcher and then
    # replayed when Collect reads the completed log. Collapse both copies by
    # Trainer step. Training and evaluation records at the same step enrich the
    # same row rather than counting as two measurements.
    by_step: dict[int, dict[str, float | None]] = {}
    for row in selected:
        step = max(0, int(row.get("current_step") or 0))
        extras = dict(row.get("extras") or {})
        validation = next((
            value for key in ("eval_loss", "validation_loss", "val_loss")
            if (value := _finite_number(extras.get(key))) is not None
        ), None)
        # Prefer the per-step loss. `train_loss` on Trainer's terminal summary
        # is a whole-run aggregate and must not replace the final curve point.
        training = _finite_number(extras.get("loss"))
        is_terminal_summary = any(
            key in extras for key in (
                "train_runtime", "total_flos",
                "train_samples_per_second", "train_steps_per_second",
            )
        )
        if training is None and not is_terminal_summary:
            training = _finite_number(extras.get("train_loss"))
        if training is None and validation is None and not is_terminal_summary:
            value = _finite_number(row.get("loss"))
            if value is not None and value >= 0:
                training = value
        point = by_step.setdefault(step, {"training": None, "validation": None})
        if training is not None and training >= 0:
            point["training"] = training
        if validation is not None and validation >= 0:
            point["validation"] = validation

    training_points = [
        (step, float(point["training"]))
        for step, point in sorted(by_step.items())
        if point["training"] is not None
    ]
    validation_points = [
        (step, float(point["validation"]))
        for step, point in sorted(by_step.items())
        if point["validation"] is not None
    ]

    def summarize(points: list[tuple[int, float]]) -> LossSeriesSummary:
        if not points:
            return LossSeriesSummary()
        minimum_step, minimum = min(points, key=lambda point: point[1])
        return LossSeriesSummary(
            points=len(points),
            first=points[0][1],
            final=points[-1][1],
            minimum=minimum,
            minimum_step=minimum_step,
        )

    train_summary = summarize(training_points)
    validation_summary = summarize(validation_points)
    observations: list[str] = []
    if train_summary.points >= 2 and train_summary.first is not None:
        if train_summary.final < train_summary.first:
            observations.append("training_loss_decreased")
        elif train_summary.final > train_summary.first:
            observations.append("training_loss_increased")
    if validation_summary.points >= 2 and validation_summary.first is not None:
        if validation_summary.final < validation_summary.first:
            observations.append("validation_loss_decreased")
        elif validation_summary.final > validation_summary.first:
            observations.append("validation_loss_increased")
        if (
            validation_summary.minimum is not None
            and validation_summary.final is not None
            and validation_summary.final > validation_summary.minimum
            and validation_summary.minimum_step != validation_points[-1][0]
        ):
            observations.append("validation_loss_rose_after_minimum")
    return TrainingDiagnostics(
        attempt_id=attempt_id,
        final_step=max((int(row.get("current_step") or 0) for row in selected), default=0),
        total_steps=max((int(row.get("total_steps") or 0) for row in selected), default=0),
        training_loss=train_summary,
        validation_loss=validation_summary,
        observations=observations,
    )


def _validate_specialist_yaml(
    result: BaseModel, inp: BaseModel, *, advisory_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Validate durable YAML, separating experiment invariants from metadata.

    Prompt/measurement reuse, lineage, user pins, and realized method semantics
    are hard constraints. Explanation and method-branch-label drift is advisory: it
    remains visible without invalidating an otherwise reproducible experiment.
    """
    warnings = advisory_warnings if advisory_warnings is not None else []
    def assigned(value: object) -> bool:
        return value not in (None, "", 0, {}, [])

    def require_suggestion_decisions(config: Any, suggestions: dict[str, Any]) -> None:
        expected = {key for key, value in suggestions.items() if assigned(value)}
        reported = {decision.key for decision in config.suggestion_decisions}
        missing = sorted(expected - reported)
        if missing:
            warnings.append(
                "configuration YAML does not record decisions for suggestions: "
                + ", ".join(missing)
            )

    if isinstance(result, InferenceResult) and result.status == "succeeded":
        if not Path(result.inference_config_path).is_absolute():
            raise ValueError("inference_config_path must be absolute")
        config = load_inference_config(result.inference_config_path)
        if not isinstance(inp, InferenceTaskInput):
            raise ValueError("InferenceResult received a non-Inference input")
        diagnostics = load_generation_diagnostics(
            result.generation_diagnostics_path
        )
        if len(diagnostics.records) != result.n_requests:
            raise ValueError(
                "generation diagnostics row count differs from "
                "InferenceResult.n_requests"
            )
        if "--adaptive-vllm-memory" in inp.config_validation_command:
            validate_adaptive_vllm_memory_config(config)
        if config.base_model != inp.base_model:
            raise ValueError("inference_config.yaml base_model differs from Ticket input")
        if config.generation_backend != inp.generation_backend:
            raise ValueError(
                "inference_config.yaml generation_backend differs from Run runtime"
            )
        pins = dict(inp.configuration_pins or {})
        for key in ("prompt_framing", "system_prompt"):
            if key in pins and getattr(config.prompt, key) != pins[key]:
                raise ValueError(f"inference_config.yaml violates pinned {key}")
        for key, value in dict(pins.get("inference_config") or {}).items():
            if config.measurement.inference_config.get(key) != value:
                raise ValueError(
                    f"inference_config.yaml violates pinned inference_config.{key}"
                )
        for key, value in dict(pins.get("decoding_config") or {}).items():
            if getattr(config.measurement, key) != value:
                raise ValueError(
                    f"inference_config.yaml violates pinned decoding_config.{key}"
                )
        require_suggestion_decisions(config, inp.configuration_suggestions)
        if inp.inference_config_path:
            supplied = load_inference_config(inp.inference_config_path)
            if config.model_dump(mode="json") != supplied.model_dump(mode="json"):
                raise ValueError(
                    "trained-model inference changed baseline inference_config.yaml"
                )
            if file_sha256(result.inference_config_path) != file_sha256(
                inp.inference_config_path
            ):
                raise ValueError(
                    "trained-model inference must report the exact supplied config file"
                )
        if (
            inp.reusable_predict_script_path
            and not _same_local_file(
                result.predict_script_path, inp.reusable_predict_script_path,
            )
            and not result.notes.strip()
        ):
            warnings.append(
                "Inference regenerated predict.py without documenting the concrete "
                "incompatibility that prevented reuse"
            )
        if "--adaptive-vllm-memory" in inp.config_validation_command:
            problem = _python_memory_helper_problem(result.predict_script_path)
            if problem:
                raise ValueError(problem)
        return config.model_dump(mode="json")

    if isinstance(result, TrainResult) and result.status == "succeeded":
        if not Path(result.train_config_path).is_absolute():
            raise ValueError("train_config_path must be absolute")
        config = load_train_config(result.train_config_path)
        if not isinstance(inp, TrainTaskInput):
            raise ValueError("TrainResult received a non-Train input")
        if result.tracking_url != inp.execution_contract.tracking_url:
            raise ValueError(
                "TrainResult.tracking_url must exactly match the engine-owned "
                "Weights & Biases URL (or remain empty when tracking is disabled)"
            )
        if inp.slurm_job.enabled:
            validate_cluster_train_config(config)
        inference = load_inference_config(inp.inference_config_path)
        if config.iteration != inp.iteration:
            raise ValueError("train_config.yaml iteration differs from Ticket iteration")
        if config.data_signature != inp.data_signature:
            raise ValueError("train_config.yaml data_signature differs from bound Data")
        if config.prompt != inference.prompt:
            raise ValueError(
                "train_config.yaml prompt must equal baseline inference_config.yaml prompt"
            )
        for key in (
            "tokenizer_source", "chat_template_source", "chat_template_hash",
            "template_kwargs", "special_token_ids",
        ):
            if getattr(config, key) != getattr(inference, key):
                raise ValueError(
                    f"train_config.yaml {key} must equal baseline "
                    "inference_config.yaml"
                )
        if (
            config.prompt_alignment.rendered_prompt
            != inference.prompt_example.rendered_prompt
        ):
            raise ValueError(
                "train_config.yaml prompt_alignment.rendered_prompt must equal "
                "baseline inference_config.yaml prompt_example.rendered_prompt"
            )
        if config.generation_backend != inp.generation_backend:
            raise ValueError(
                "train_config.yaml generation_backend differs from Run runtime"
            )
        current_inference_config_sha256 = file_sha256(inp.inference_config_path)
        if current_inference_config_sha256 != inp.expected_inference_config_sha256:
            raise ValueError(
                "baseline inference_config.yaml bytes changed after the Train "
                "work order was created"
            )
        if config.inference_config_sha256 != inp.expected_inference_config_sha256:
            raise ValueError(
                "train_config.yaml inference_config_sha256 must exactly copy "
                "expected_inference_config_sha256"
            )
        expected_parent = inp.parent_checkpoint_path or inp.base_model
        if config.parent_model != expected_parent:
            raise ValueError("train_config.yaml parent_model breaks the model chain")
        expected_parent_kind = (
            "run_checkpoint" if inp.model_source == "checkpoint" else "baseline"
        )
        if config.parent_kind != expected_parent_kind:
            raise ValueError(
                "train_config.yaml parent_kind differs from the selected model source"
            )
        if config.parent_selection_rationale != inp.parent_selection_rationale:
            warnings.append(
                "train_config.yaml rephrased the Orchestrator's parent selection "
                "rationale; parent_model and parent_kind still match"
            )
        if inp.training_method_pin:
            if config.training_method != inp.training_method_pin:
                raise ValueError("train_config.yaml violates the pinned training method")
            if config.method_diversity_status != "user_pinned":
                warnings.append(
                    "a user-pinned training method requires "
                    "method_diversity_status=user_pinned"
                )
        elif inp.iteration == 1:
            if config.method_diversity_status != "initial":
                warnings.append(
                    "the first unpinned Train iteration requires "
                    "method_diversity_status=initial"
                )
        elif inp.parent_train_config_path:
            previous = load_train_config(inp.parent_train_config_path)
            if config.training_method != previous.training_method:
                suggested_method = str(
                    inp.configuration_suggestions.get("training_method") or ""
                ).strip().lower()
                if suggested_method != config.training_method:
                    raise ValueError(
                        "an unpinned method may change only when the Orchestrator "
                        "explicitly selects the next exhausted-search branch"
                    )
                transition_level = str(
                    (inp.branch_transition or {}).get("level") or "none"
                )
                if transition_level not in {"method", "base_model"}:
                    raise ValueError(
                        "an unpinned method change requires an engine-validated "
                        "method/base_model branch_transition"
                    )
                if config.method_diversity_status != "varied":
                    warnings.append(
                        "an exhausted-branch method change requires "
                        "method_diversity_status=varied"
                    )
                if "exhaust" not in config.method_selection_rationale.lower():
                    raise ValueError(
                        "a method-branch transition must record the exhaustion "
                        "evidence in method_selection_rationale"
                    )
            elif config.method_diversity_status not in {
                "retained_in_branch", "retained_for_constraints",
            }:
                warnings.append(
                    "an active unpinned method branch requires "
                    "method_diversity_status=retained_in_branch and a rationale"
                )
        for key, value in inp.method_config_pins.items():
            if config.method_config.get(key) != value:
                raise ValueError(
                    f"train_config.yaml violates pinned method_config.{key}"
                )
        for key, value in inp.loss_objective_pins.items():
            if config.loss_contract.objective_config.get(key) != value:
                raise ValueError(
                    f"train_config.yaml violates pinned loss_objective_config.{key}"
                )
        for key, value in inp.configuration_pins.items():
            if key in type(config.training).model_fields and getattr(config.training, key) != value:
                raise ValueError(f"train_config.yaml violates pinned training.{key}")
        retention = config.training.checkpoint_retention
        intermediates = list(result.intermediate_checkpoints or [])
        if retention.strategy == "none" and intermediates:
            raise ValueError(
                "TrainResult declared intermediate checkpoints while "
                "training.checkpoint_retention.strategy=none"
            )
        if retention.strategy != "none":
            if not intermediates:
                raise ValueError(
                    "configured checkpoint retention produced no verified "
                    "intermediate checkpoints"
                )
            if len(intermediates) > retention.max_intermediate_checkpoints:
                raise ValueError(
                    "TrainResult exceeds checkpoint_retention.max_intermediate_checkpoints"
                )
            realized_save_only_model = config.training.implementation_config.get(
                "save_only_model"
            )
            if inp.slurm_job.enabled and realized_save_only_model is not False:
                raise ValueError(
                    "finite Slurm checkpoint retention requires realized "
                    "training.implementation_config.save_only_model=false so "
                    "walltime continuation preserves optimizer, scheduler, "
                    "scaler, RNG, and Trainer state"
                )
        seen_intermediate_paths: set[str] = set()
        final_checkpoint = Path(result.checkpoint_path)
        if not final_checkpoint.is_absolute():
            raise ValueError("checkpoint_path must be absolute")
        for checkpoint in intermediates:
            if checkpoint.path in seen_intermediate_paths:
                raise ValueError("intermediate checkpoint paths must be unique")
            seen_intermediate_paths.add(checkpoint.path)
            candidate = Path(checkpoint.path)
            if not candidate.is_absolute():
                raise ValueError("intermediate checkpoint paths must be absolute")
            if (
                candidate == final_checkpoint
                or final_checkpoint in candidate.parents
                or candidate in final_checkpoint.parents
                or final_checkpoint.parent not in candidate.parents
            ):
                raise ValueError(
                    "intermediate checkpoints must live in a separate subtree "
                    "beside the final model, never inside it or around it"
                )
            if retention.strategy == "steps" and (
                checkpoint.step < 1 or checkpoint.step % retention.save_steps != 0
            ):
                raise ValueError(
                    "a step-retained checkpoint must match checkpoint_retention.save_steps"
                )
            if retention.strategy == "epoch" and checkpoint.epoch <= 0:
                raise ValueError("an epoch-retained checkpoint requires epoch > 0")
            if result.checkpoint_is_remote:
                if inp.ticket_id not in checkpoint.path:
                    raise ValueError(
                        "a remote intermediate checkpoint path must contain ticket_id"
                    )
            else:
                problem = _artifact_problem(checkpoint.path)
                if problem:
                    raise ValueError(
                        f"intermediate checkpoint is {problem}: {checkpoint.path}"
                    )
        script_path = Path(result.train_script_path)
        if not script_path.is_absolute():
            raise ValueError("train_script_path must be absolute")
        try:
            script_body = script_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read train_script_path: {exc}") from exc
        if "ZevoTrainerTelemetryCallback" not in script_body:
            raise ValueError(
                "train.py must import and install the system-owned "
                "ZevoTrainerTelemetryCallback"
            )
        require_suggestion_decisions(config, inp.configuration_suggestions)
        return config.model_dump(mode="json")
    return {}


def _python_memory_helper_problem(path: str) -> str:
    """Require a semantic call to Zevo's vLLM memory calculator.

    A substring check was too weak: a comment or an unused import satisfied it
    while ``predict.py`` could still hard-code ``gpu_memory_utilization``.
    """
    try:
        body = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return f"cannot read predict_script_path: {exc}"
    try:
        tree = ast.parse(body, filename=path)
    except SyntaxError as exc:
        return f"cannot parse predict_script_path: {exc}"

    direct_names: set[str] = set()
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "zevo_inference_memory":
            for imported in node.names:
                if imported.name == "gpu_memory_utilization":
                    direct_names.add(imported.asname or imported.name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "zevo_inference_memory":
                    module_names.add(imported.asname or imported.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in direct_names:
            return ""
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "gpu_memory_utilization"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in module_names
        ):
            return ""
    return (
        "vLLM predict.py must import and call the system-owned "
        "zevo_inference_memory.gpu_memory_utilization helper"
    )


def _slurm_stage_result_problem(
    inp: BaseModel, reported_path: str, executable_name: str,
    *, configuration_path: str = "",
) -> str:
    """Validate the auditable, finite Slurm script required by cluster stages."""
    contract = getattr(inp, "slurm_job", None)
    if not isinstance(contract, SlurmStageJobContract):
        return "stage input has no Slurm execution contract"
    if not contract.enabled:
        return (
            "non-cluster stage unexpectedly reported a Slurm script"
            if reported_path else ""
        )
    if reported_path != contract.script_path:
        return "cluster stage must report the engine-assigned Slurm script path"
    path = Path(reported_path)
    try:
        body = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"cannot read cluster stage Slurm script: {exc}"
    required = (
        "#SBATCH", contract.job_name, executable_name,
        contract.status_path, contract.lifecycle_prologue,
        f"#SBATCH --output={contract.stdout_path}",
        f"#SBATCH --error={contract.stderr_path}",
    )
    missing = [token for token in required if token not in body]
    if missing:
        return f"cluster stage Slurm script is missing {missing}"
    lowered = body.lower()
    if "sleep infinity" in lowered or "--wrap" in lowered:
        return "cluster stage Slurm script must be a finite direct job, not a holder/wrap"
    if executable_name == "train.py":
        if not configuration_path:
            return "cluster Train must report train_config_path before submission handoff"
        try:
            validate_cluster_train_config(load_train_config(configuration_path))
        except (OSError, ValueError) as exc:
            return f"cluster Train configuration is invalid: {exc}"
    elif (
        executable_name == "predict.py"
        and "--adaptive-vllm-memory"
        in str(getattr(inp, "config_validation_command", ""))
    ):
        if not configuration_path:
            return (
                "cluster Inference must report inference_config_path before "
                "submission handoff"
            )
        try:
            validate_adaptive_vllm_memory_config(
                load_inference_config(configuration_path)
            )
        except (OSError, ValueError) as exc:
            return f"cluster Inference memory configuration is invalid: {exc}"
        predict_problem = _python_memory_helper_problem(
            str(path.with_name("predict.py"))
        )
        if predict_problem:
            return predict_problem
        if not (
            "zevo_inference_memory" in body
            and "required_free_memory_gib" in body
        ):
            return (
                "cluster Inference Slurm script must call the system-owned "
                "zevo_inference_memory.required_free_memory_gib helper for "
                "free-memory preflight"
            )
    return ""


def _infra_provision_summary(info: InfrastructureDeviceInfo) -> str:
    """Concise, engine-authored one-line summary of a successful provision.

    The full resolved resource plan is a structured object; its durable homes
    are ``device_info.json`` and the Ticket's ``resource_plan`` artifact meta.
    This human-facing Ticket summary deliberately does NOT dump the plan text,
    so the plan never leaks into the summary/error field on success.
    """
    plan = info.resource_plan
    provider = info.provider + (f"/{info.cloud_backend}" if info.cloud_backend else "")
    return (
        f"provision succeeded on {provider}: "
        f"num_gpus={plan.num_gpus}, min_vram_gb={plan.min_vram_gb}, "
        f"min_ram_gb={plan.min_ram_gb}, min_cpus={plan.min_cpus}, "
        f"time_limit_hours={plan.time_limit_hours}"
    )


def _extract_summary_artifact_meta(
    result: BaseModel, *, inp: BaseModel, agent_id: str, work_dir: str,
    run_id: str = "",
) -> tuple[str, str, str, dict]:
    """Pull (status, summary, artifact_path, artifact_meta).

    Evaluation is special: its output is a durable metrics.json path. Read the
    file so Registry can consume the `metrics` ArtifactBinding cleanly.
    """
    status = getattr(result, "status", "succeeded")

    if isinstance(result, EvaluationResult):
        artifact, metrics = materialize_metrics_artifact(result.metrics_path, work_dir)
        # The durable file is the only score authority.
        score = extract_score({"score": metrics.get("score")})
        meta = dict(metrics)
        if status == "succeeded" and score is None:
            status = "failed"
            summary = (
                "evaluation reported success but metrics.json has no finite "
                "numeric `score`"
            )
        else:
            summary = result.notes or (f"score={score:.4f}" if score is not None else "")
        return status, summary, artifact, meta

    meta: dict = {}
    if isinstance(result, DataResult):
        meta = {
            "operation": result.operation,
            "n_rows_in": result.n_rows_in,
            "n_rows_out": result.n_rows_out,
        }
    elif isinstance(result, InfraResult):
        meta = {
            "operation": result.operation,
            "provider": result.provider,
            "cloud_backend": getattr(result, "cloud_backend", "") or "",
            "instance_id": result.instance_id,
            "auto_release": result.auto_release,
        }
        if status in ("succeeded", "degraded"):
            problems: list[str] = []
            provision_summary = ""
            if result.operation == "provision":
                if not isinstance(inp, InfraTaskInput):
                    problems.append("provision received a non-Infrastructure input")
                if not result.device_info_path:
                    problems.append("device_info_path is empty")
                elif not Path(result.device_info_path).is_absolute():
                    problems.append("device_info_path is not absolute")
                if (
                    isinstance(inp, InfraTaskInput)
                    and result.device_info_path
                    and Path(result.device_info_path).is_absolute()
                ):
                    try:
                        info = validate_device_info(
                            result.device_info_path,
                            run_id=run_id,
                            ticket_id=result.ticket_id,
                            provider=inp.provider,
                            num_gpus=inp.num_gpus,
                            purpose=inp.purpose,
                        )
                        if info.auto_release != inp.auto_release:
                            problems.append(
                                "device_info.auto_release differs from the Ticket policy"
                            )
                        if info.provider == "cloud" and info.cloud_backend not in set(
                            inp.available_cloud_backends
                        ):
                            problems.append("device_info selected an unavailable cloud backend")
                        if (
                            info.provider == "cluster"
                            and info.resource_plan.time_limit_hours < 1
                        ):
                            problems.append("cluster resource plan requires positive wall-time")
                        if info.gpu is not None:
                            minimum_vram_mb = info.resource_plan.min_vram_gb * 1024
                            if any(
                                device.vram_mb < minimum_vram_mb
                                for device in info.gpu.devices
                            ):
                                problems.append(
                                    "assigned GPU does not meet resource_plan.min_vram_gb"
                                )
                        meta = {
                            "operation": result.operation,
                            "provider": info.provider,
                            "cloud_backend": info.cloud_backend,
                            "instance_id": info.instance_id,
                            "has_gpu": bool(info.gpu),
                            "gpu_count": info.gpu.gpu_count if info.gpu else 0,
                            "gpu_name": info.gpu.gpu_name if info.gpu else "",
                            "vram_gb": info.gpu.vram_gb if info.gpu else 0,
                            "cuda_version": info.cuda.cuda_version if info.cuda else "",
                            "host": info.host,
                            "dph_total": info.cost.dph_total,
                            "auto_release": info.auto_release,
                            "resource_plan": info.resource_plan.model_dump(mode="json"),
                        }
                        provision_summary = _infra_provision_summary(info)
                    except ValueError as exc:
                        problems.append(f"device_info.json is invalid: {exc}")
            else:
                if not isinstance(inp, InfraTaskInput):
                    problems.append("release received a non-Infrastructure input")
                elif (
                    result.provider != inp.provider
                    or result.instance_id != inp.instance_id
                    or result.cloud_backend != inp.cloud_backend
                ):
                    problems.append("release lifecycle identity differs from the Ticket")
            if problems:
                status = "failed"
                return (
                    status,
                    "infrastructure reported usable success but " + "; ".join(problems),
                    result.device_info_path,
                    meta,
                )
            if result.operation == "provision" and provision_summary:
                # The resolved plan's home is device_info.json / the
                # resource_plan artifact meta above. Never let the agent's
                # free-text notes carry the plan into the Ticket summary/error.
                return status, provision_summary, result.device_info_path, meta
    elif isinstance(result, TrainResult):
        slurm_problem = _slurm_stage_result_problem(
            inp, result.slurm_script_path, "train.py",
            configuration_path=result.train_config_path,
        )
        if status in {"succeeded", "deferred"} and slurm_problem:
            return "failed", slurm_problem, result.checkpoint_path, {}
        meta = {
            "n_examples": result.n_examples,
            "final_loss": result.final_loss,
            "training_seconds": result.training_seconds,
            "train_config_path": result.train_config_path,
            "tracking_url": result.tracking_url,
        }
    elif isinstance(result, InferenceResult):
        slurm_problem = _slurm_stage_result_problem(
            inp, result.slurm_script_path, "predict.py",
            configuration_path=result.inference_config_path,
        )
        if status in {"succeeded", "deferred"} and slurm_problem:
            return "failed", slurm_problem, result.predictions_path, {}
        meta = {
            "n_rows": result.n_rows,
            "n_requests": result.n_requests,
            "n_unparseable": result.n_unparseable,
            "inference_config_path": result.inference_config_path,
        }
        if isinstance(inp, InferenceTaskInput) and inp.reusable_predict_script_path:
            reused = _same_local_file(
                result.predict_script_path, inp.reusable_predict_script_path,
            )
            # Provenance comes from the resolved ArtifactBinding and verified
            # file bytes, never from Agent-authored echo fields.
            meta["predict_script_reuse"] = {
                "reused": reused,
                "source_path": inp.reusable_predict_script_path if reused else "",
            }
    elif isinstance(result, RegisterResult):
        meta = {}
        if status == "succeeded":
            problems: list[str] = []
            if not result.registry_path:
                problems.append("registry_path is empty")
            elif not Path(result.registry_path).is_absolute():
                problems.append("registry_path is not absolute")
            elif Path(result.registry_path).resolve() != (
                Path(work_dir) / "registry.yaml"
            ).resolve():
                problems.append(
                    "registry_path must be this Ticket's work_dir/registry.yaml"
                )
            if not isinstance(inp, RegisterTaskInput):
                problems.append("Registry result received a non-Registry input")
            if not run_id:
                problems.append("run_id is unavailable for version_tag validation")
            else:
                expected_tag = model_tag_for_run(run_id)
                if result.registry_path:
                    try:
                        selected = validate_registry_entry(
                            result.registry_path,
                            run_id=run_id,
                            version_tag=expected_tag,
                        )
                        if not isinstance(inp, RegisterTaskInput):
                            raise ValueError("missing Registry execution input")
                        metrics = json.loads(
                            Path(inp.metrics_path).read_text(encoding="utf-8")
                        )
                        candidate_score = extract_score(
                            {"score": metrics.get("score")}
                            if isinstance(metrics, dict) else {}
                        )
                        if candidate_score is None:
                            raise ValueError("champion metrics has no finite score")
                        expected_provenance = {
                            "base_model": inp.base_model,
                            "training_method": inp.training_method,
                            "dataset_source": inp.dataset_source,
                            "task_objective": inp.task_objective,
                            "metric": inp.metric,
                            "metric_direction": inp.metric_direction,
                        }
                        retained = selected.ticket_id == result.ticket_id
                        if not retained:
                            raise ValueError(
                                "final Registry manifest must retain this Ticket's "
                                "engine-selected Validation champion"
                            )
                        actual = {
                            key: getattr(selected, key)
                            for key in expected_provenance
                        }
                        if selected.iteration != inp.iteration:
                            raise ValueError("selected champion iteration differs")
                        if actual != expected_provenance:
                            differences = {
                                key: {
                                    "expected": expected_provenance[key],
                                    "actual": actual[key],
                                }
                                for key in expected_provenance
                                if actual[key] != expected_provenance[key]
                            }
                            raise ValueError(
                                "selected champion provenance differs: "
                                + json.dumps(differences, sort_keys=True)
                            )
                        if selected.eval.model_dump(mode="json") != metrics:
                            raise ValueError(
                                "selected champion eval differs from metrics.json"
                            )
                        stored_model = Path(selected.model_path)
                        local_model = (
                            stored_model
                            if stored_model.is_absolute()
                            else (REPO_ROOT / stored_model).resolve()
                        )
                        model_problem = _artifact_problem(str(local_model))
                        if model_problem:
                            raise ValueError(
                                f"selected champion model is {model_problem}: {local_model}"
                            )
                        meta = {
                            "version_tag": expected_tag,
                            "candidate_score": candidate_score,
                            "retained": retained,
                            "model_path": str(local_model) if retained else "",
                        }
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        problems.append(f"registry entry schema is invalid: {exc}")
            if problems:
                status = "failed"
                return (
                    status,
                    "registry reported success but " + "; ".join(problems),
                    result.registry_path,
                    meta,
                )

    # Try common artifact fields in priority order.
    artifact = ""
    for field in (
        "training_dataset_path",  # DataResult (prepared dataset.jsonl)
        "scoping_result_path",    # DataResult scope_problem (auto mode)
        # A private held-out Data ticket has no training dataset; its
        # questions-only copy is the primary artifact.
        "scoring_public_path",
        "inference_data_profile_path",  # answer-free Run Setup evidence
        "checkpoint_path",    # TrainResult (model dir)
        "predictions_path",   # InferenceResult
        "metrics_path",       # EvaluationResult (the JSON file)
        "device_info_path",   # InfraResult (device_info.json)
        "registry_path",      # RegisterResult (the YAML file, less interesting)
    ):
        v = getattr(result, field, "")
        if isinstance(v, str) and v:
            artifact = v
            break
    summary = getattr(result, "notes", "") or getattr(result, "summary", "") or ""
    return status, summary, artifact, meta


def _set_current_ticket_outcome_text(
    ticket: Ticket, *, status: str, summary: str,
) -> None:
    """Persist only the current activation's summary/error on a Ticket.

    Older activation failures remain in HeartbeatResult. A later successful
    rerun must not leave its Ticket looking failed through stale error text.
    """
    ticket.summary = (summary or "")[:1000]
    ticket.error_message = (summary or "")[:2000] if status == "failed" else ""


# Every LOCAL artifact each agent produces — script(s) AND output(s) — so a
# ticket's work_products are COMPLETE + CONSISTENT, not just one file (e.g.
# inference shows predict.py + predictions.csv; train shows train.py, and
# the model which is a remote pointer). The OUTPUT is listed FIRST — that's the
# "primary" the fabrication guard checks. Evaluation is handled separately (its
# metrics are materialized to a path by _extract_summary_artifact_meta).
_TICKET_ARTIFACTS: dict[str, list[tuple[str, str]]] = {
    "infrastructure": [("device_info_path", "device_info")],
    # Optimization Data authors only Training/recipe/script. The engine appends
    # the hidden Validation artifacts after the Data result is final so lineage
    # remains convenient without granting Data access to the scoring set.
    # Private held-out Data authors only its questions-only copy.
    "data":           [("training_dataset_path", "training_dataset"), ("prepare_script_path", "script"),
                       # Auto mode: the derived scoring contract. Its own role,
                       # so no binding contract can ever treat it as training data.
                       ("scoping_result_path", "scoping_result"),
                       ("data_recipe_path", "data_recipe"),
                       ("validation_source_path", "validation_source"),
                       ("scoring_public_path", "scoring_questions"),
                       ("inference_data_profile_path", "inference_data_profile"),
                       ("validation_dataset_path", "validation_dataset"),
                       ("sample_submission_path", "sample_submission")],
    "train":     [("checkpoint_path", "checkpoint"), ("train_config_path", "train_config"),
                   ("train_script_path", "script"),
                   ("slurm_script_path", "slurm_script"), ("log_path", "log")],
    "inference":      [("predictions_path", "predictions"),
                       ("generation_diagnostics_path", "generation_diagnostics"),
                       ("inference_config_path", "inference_config"),
                       ("predict_script_path", "script"),
                       ("slurm_script_path", "slurm_script"), ("log_path", "log")],
    "registry": [("registry_path", "registry_entry"),
                 ("register_script_path", "script")],
}


def _apply_run_score_monitor(
    run: Run, ticket: Ticket, result: BaseModel, meta: dict,
) -> None:
    """Update run-level score/tag fields from trusted typed outputs.

    These are the VALIDATION fields — the run's optimization signal. Held-out
    tickets skip this entirely; their numbers go through `_record_holdout_score`
    so that a test result can never be mistaken for a tuning result.
    """
    if ticket.lane == "held_out_test":
        return

    incumbent = None if run.best_validation_score is None else float(run.best_validation_score)
    if isinstance(result, EvaluationResult) and result.status == "succeeded":
        score = extract_score(meta)
        if score is not None:
            if is_better(score, incumbent, run.validation_metric_direction):
                run.best_validation_score = score
            # The Run headline is the validation-selected champion, not the
            # most recently measured candidate. The full latest series lives
            # in ScoreEvent; using it here made Runs disagree with Models after
            # Registry rejected a regression.
    elif isinstance(result, RegisterResult) and result.status == "succeeded":
        # Every candidate gets a registry row; only the retained candidate is
        # the run's winner. Pointing this field at the latest superseded row
        # made the Run page disagree with the model actually left on disk.
        if meta.get("retained") and meta.get("version_tag"):
            run.registry_version_tag = str(meta["version_tag"])
        score = extract_score(meta, meta.get("candidate_score"))
        if score is not None:
            if is_better(score, incumbent, run.validation_metric_direction):
                run.best_validation_score = score


def _sync_best_validation_score(run: Run) -> None:
    """Recompute the validation headline strictly from validation history."""
    run.best_validation_score = validation_best_score(
        run.history, run.validation_metric_direction,
    )


def _relocate_artifact(path: str, work_dir: str) -> str:
    """The claimed path, or the same artifact found under `work_dir`.

    Agents report where they think they wrote something, and that string is
    routinely a shade off the real path — an eval agent reported
    `/app/data/runs/eval-002/metrics.json` for a file that was really at
    `/app/data/runs/<run-id>/eval-002/metrics.json`. The work was done; only the
    delivery note was wrong, and failing the ticket for it fails the whole run.

    So before calling an artifact missing, look for it where the agent actually
    worked: try the claimed path's trailing segments against the ticket's work
    dir, longest first, so a nested output keeps its shape and a bare filename
    is the last resort. Only paths that stay INSIDE the work dir are accepted —
    this forgives a typo, it does not let an agent claim someone else's file.
    Returns the original path unchanged when nothing matches.
    """
    if not path:
        return path
    p = Path(path)
    if p.exists():
        return path
    wd = Path(work_dir)
    try:
        wd_real = wd.resolve()
    except OSError:
        return path
    parts = [x for x in p.parts if x not in ("/", "")]
    for i in range(len(parts)):
        cand = wd.joinpath(*parts[i:])
        if not cand.exists():
            continue
        try:
            if not str(cand.resolve()).startswith(str(wd_real)):
                continue
        except OSError:
            continue
        return str(cand)
    return path


def _referenced_paths(guarded: list[str], events: list[dict]) -> list[str]:
    """Which of `guarded` this activation's tool calls actually named.

    Matched on a PATH-ISH substring, not on the bare file name. The check used
    to compare `Path(p).name` against the whole tool input, which on the
    commonest names in this system — `eval.py`, `test.csv` — flags any command
    that merely contains the string: the agent writing its own `eval.py`,
    a comment body quoting one, a `find` whose pattern spells it. Those false
    positives are expensive, because the comment they raise tells the reader
    the run's headline number is unattributable.

    So require the enclosing directory too when the guarded path has one. Two
    different files called `test.csv` in two different directories stop
    colliding, while a stage that really did open the guarded path still names
    it in full — nobody reads a file without saying where it is.
    """
    if not guarded:
        return []
    inputs = [
        str((e.get("payload") or {}).get("input") or "")
        for e in events if e.get("type") == "tool_call"
    ]
    hits: set[str] = set()
    for path in guarded:
        if not path:
            continue
        p = Path(path)
        # `<parent>/<name>` when there is a parent, else the bare name. Both
        # forms are what a shell command would contain.
        needle = f"{p.parent.name}/{p.name}" if p.parent.name else p.name
        if any(needle in inp for inp in inputs):
            hits.add(p.name)
    return sorted(hits)


def _artifact_problem(path: str) -> str:
    """Return a short reason if a claimed artifact is missing/empty, else ''.

    GenerationBackend-level guard against agents that hallucinate success without
    actually producing their output file (e.g. a lazy data agent that
    reports success but never writes dataset.jsonl). Handles both file
    artifacts (must exist + be non-empty) and directory artifacts like a
    LoRA adapter dir (must exist + contain at least one entry).
    """
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.is_dir():
        try:
            if not any(p.iterdir()):
                return "an empty directory"
        except OSError:
            return "unreadable"
        return ""
    try:
        if p.stat().st_size == 0:
            return "empty"
    except OSError:
        return "unreadable"
    return ""


def _essential_artifact_fields(result: BaseModel) -> list[str]:
    """Artifacts whose absence changes execution, lineage, or measured output."""
    if isinstance(result, DataResult):
        if result.operation == "scope_problem":
            return ["scoping_result_path"]
        fields = ["scoring_public_path"]
        if result.operation == "prepare_run_data":
            fields.extend([
                "training_dataset_path", "validation_dataset_path",
                "inference_data_profile_path",
                "data_recipe_path",
            ])
            if result.validation_source_path:
                fields.append("validation_source_path")
        if result.sample_submission_path:
            fields.append("sample_submission_path")
        return fields
    if isinstance(result, TrainResult):
        fields = ["train_config_path", "train_script_path", "log_path"]
        if result.slurm_script_path:
            fields.append("slurm_script_path")
        if not result.checkpoint_is_remote:
            fields.append("checkpoint_path")
        return fields
    if isinstance(result, InferenceResult):
        fields = [
            "predictions_path", "inference_config_path",
            "generation_diagnostics_path",
        ]
        if result.slurm_script_path:
            fields.append("slurm_script_path")
        return fields
    if isinstance(result, InfraResult) and result.operation == "provision":
        return ["device_info_path"]
    if isinstance(result, RegisterResult):
        return ["registry_path"]
    return []


def _bind_system_validation_artifacts(
    result: DataResult, *, run: Run, work_dir: str,
) -> DataResult:
    """Attach frozen Validation artifacts only after Data has completed.

    The Data invocation is deliberately unable to see these paths. This system
    step strips answers, profiles the public question shape, and binds the raw
    full Validation view for trainer-side eval only after the selected training
    artifact and recipe already exist.
    """
    holdout = dict(run.holdout or {})
    validation_source = str(holdout.get("validation_set") or "")
    answer_fields = list(holdout.get("validation_answer_fields") or [])
    sample_submission = str(holdout.get("validation_sample_submission") or "")
    if not validation_source or not answer_fields or not sample_submission:
        raise ValueError(
            "engine Validation contract is incomplete after Data selection"
        )

    sanitized = sanitize_training_against_scoring(
        training_dataset=result.training_dataset_path,
        validation_source=validation_source,
        blocked_fingerprints=list(
            holdout.get("test_semantic_fingerprints") or []
        ),
        out_dir=work_dir,
    )

    public = str(holdout.get("validation_public") or "")
    profile = str(holdout.get("inference_data_profile") or "")
    if not (
        public and profile and Path(public).is_file() and Path(profile).is_file()
    ):
        prepared = materialize_system_scoring_artifacts(
            scoring_source=validation_source,
            answer_fields=answer_fields,
            sample_submission=sample_submission,
            out_dir=work_dir,
        )
        public = prepared.questions_path
        profile = prepared.profile_path
        validation_source = prepared.validation_dataset_path
        sample_submission = prepared.sample_submission_path

    return result.model_copy(update={
        "training_dataset_path": sanitized.path,
        "system_scoring_duplicates_removed": (
            sanitized.removed_scoring_duplicates
        ),
        "decontamination_checked": True,
        "decontamination_removed_rows": (
            sanitized.removed_scoring_duplicates
        ),
        "validation_source_path": validation_source,
        "validation_answer_fields": answer_fields,
        # This is the frozen raw scoring population. Train owns any temporary
        # conversion into its method-specific eval representation.
        "validation_dataset_path": validation_source,
        "scoring_public_path": public,
        "inference_data_profile_path": profile,
        "sample_submission_path": sample_submission,
    })


def _tabular_shape(path: str) -> tuple[int, set[str]] | None:
    """Read count/columns for formats whose semantics we can verify exactly."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            return len(rows), set(reader.fieldnames or [])
    if suffix == ".jsonl":
        # Iterate physical file records. str.splitlines() also treats U+2028
        # and U+2029 inside valid JSON strings as row separators and used to
        # reject otherwise-correct Data artifacts.
        with source.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif suffix == ".json":
        loaded = json.loads(source.read_text(encoding="utf-8"))
        rows = loaded if isinstance(loaded, list) else []
    else:
        return None
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain tabular object records")
    fields = set().union(*(set(row) for row in rows)) if rows else set()
    return len(rows), fields


# --------------------- the runner ---------------------------------------

_SLURM_TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}


async def _latest_slurm_stage_job(
    session: AsyncSession, ticket: Ticket,
) -> InfraInstance | None:
    """Return the newest finite Slurm job registered for this exact Ticket."""
    rows = (await session.execute(
        select(InfraInstance)
        .where(
            InfraInstance.provider == "cluster",
            InfraInstance.run_id == ticket.run_id,
            InfraInstance.ticket_id == ticket.id,
            InfraInstance.instance_id != "",
        )
        .order_by(InfraInstance.created_at.desc())
        .limit(5)
    )).scalars().all()
    return next(
        (row for row in rows if bool((row.meta or {}).get("stage_job"))),
        None,
    )


def _slurm_job_is_live(row: InfraInstance | None) -> bool:
    if row is None or row.released_at is not None:
        return False
    state = str((row.meta or {}).get("scheduler_state") or "").strip().upper()
    state = state.split()[0].rstrip("+") if state else ""
    return state not in _SLURM_TERMINAL_STATES


def _commit_slurm_submission(
    row: InfraInstance, *, heartbeat_id: str,
    stdout_path: str, stderr_path: str,
) -> str:
    """Hand one validated finite job from the submit activation to watcher.

    Registering a JOBID is not enough: it can happen while the Agent is still
    writing its typed Result and the runner is still validating the local
    config/script contract. The watcher must not change Ticket ownership in
    that interval. This marker is written only after those checks passed (or
    after the runner deliberately preserved an already-submitted live job).
    """
    meta = dict(row.meta or {})
    meta["submission_committed"] = True
    meta["submission_heartbeat_id"] = heartbeat_id
    meta["submission_committed_at"] = datetime.now(timezone.utc).isoformat()
    # The engine owns the log paths just as it owns the watcher handoff.  They
    # are resolved only after Slurm returns the concrete JOBID, so every retry
    # keeps separate stdout/stderr and the watcher follows the exact active job.
    meta["log_path"] = stdout_path.replace("%j", row.instance_id)
    meta["stderr_path"] = stderr_path.replace("%j", row.instance_id)
    row.meta = meta
    state = str(meta.get("scheduler_state") or "PENDING").strip().upper()
    return state.split()[0].rstrip("+") if state else "PENDING"


async def _add_slurm_stage_message_once(
    session: AsyncSession,
    *,
    ticket: Ticket,
    job: InfraInstance,
    prefix: str,
    body: str,
) -> None:
    """Publish one engine-owned lifecycle message per JOBID and state."""
    recent = (await session.execute(
        select(TicketMessage)
        .where(TicketMessage.ticket_id == ticket.id)
        .order_by(TicketMessage.created_at.desc())
        .limit(30)
    )).scalars().all()
    wanted = prefix.lower() + ":"
    if any(
        message.body.lstrip().lower().startswith(wanted)
        and job.instance_id in message.body
        for message in recent
    ):
        return
    session.add(TicketMessage(
        ticket_id=ticket.id,
        author="system",
        body=body,
    ))

async def _apply_ticket_failure_policy(
    session: AsyncSession,
    ticket: Ticket,
    *,
    error_message: str,
    cancelled: bool = False,
) -> bool:
    """Set the Ticket's failure/repair state; return True when reactivation is due."""
    disposition = classify_failure(
        error_message, agent_id=ticket.agent_id, cancelled=cancelled,
    )
    if cancelled:
        ticket.repair_attempts = int(ticket.repair_attempts or 0)
        ticket.repair_route = "terminal"
        return False
    if (
        disposition.route == "self"
        and int(ticket.repair_attempts or 0) < MAX_REPAIR_ATTEMPTS
    ):
        ticket.repair_attempts = int(ticket.repair_attempts or 0) + 1
        ticket.repair_route = "self"
        ticket.status = "repairing"
        session.add(TicketMessage(
            ticket_id=ticket.id,
            author="system",
            body=repair_instruction(
                agent_id=ticket.agent_id,
                attempt=int(ticket.repair_attempts),
                error_message=error_message,
            ),
        ))
        return True
    ticket.status = "failed"
    ticket.repair_route = (
        "orchestrator"
        if disposition.route == "self"
        and int(ticket.repair_attempts or 0) >= MAX_REPAIR_ATTEMPTS
        else disposition.route
    )
    return False


class StoredPayloadValidationError(ValueError):
    """A ticket's STORED typed payload failed schema validation before the
    agent ran.

    Deterministic, not transient: re-draining the identical payload fails
    identically, so it must be escalated + bounded rather than silently
    retried. Raised so the daemon records THIS wakeup failed with the
    payload-validation signature (feeding both the runner's consecutive-failure
    counter and the reconciler's safety net).
    """


async def _count_validation_failed_wakeups(
    session: AsyncSession, ticket_id: str | None
) -> int:
    """How many of this ticket's wakeups already failed typed-payload validation.

    Counted from the wakeup rows the daemon marks `failed` and stamps with
    `PAYLOAD_VALIDATION_FAILURE_SIGNATURE`. This survives a rerun (old failed
    rows stay), so a ticket that keeps being re-queued with a still-invalid
    payload is bounded across its whole life, and a corrected *new* ticket (a
    different id) starts its own count from zero.
    """
    if not ticket_id:
        return 0
    n = (await session.execute(
        select(func.count()).select_from(AgentWakeupRequest).where(
            AgentWakeupRequest.ticket_id == ticket_id,
            AgentWakeupRequest.status == "failed",
            AgentWakeupRequest.reason.contains(PAYLOAD_VALIDATION_FAILURE_SIGNATURE),
        )
    )).scalar_one()
    return int(n or 0)


async def _escalate_stored_payload_validation_failure(
    session: AsyncSession, tk: Ticket, run: Run, exc: Exception,
) -> None:
    """Escalate a ticket whose stored typed payload failed validation at wakeup.

    Typed-payload validation (validate_stored_payload, and the per-agent
    *TaskInput builders that enforce the same contract) runs BEFORE the agent
    executes. A `queued` ticket that fails it used to stay `queued`: the wakeup
    was marked failed but the ticket was untouched, and the alarm-clock cron
    (~5 min) re-enqueued the identical payload forever -- no escalation, no
    re-plan, and on a rented GPU an unbounded idle burn.

    Instead: record the validation error on the ticket, wake the run's
    orchestrator with the error as feedback so it can re-emit a corrected child
    (mirroring the repair/handoff wake pattern), and after
    MAX_PAYLOAD_VALIDATION_ATTEMPTS consecutive failures fail the ticket
    terminally rather than looping. Always raises StoredPayloadValidationError
    so the wakeup is recorded failed with the signature the counters read.
    """
    detail = " ".join(str(exc).split())[:1500]
    prior = await _count_validation_failed_wakeups(session, tk.id)
    attempt = prior + 1
    terminal = attempt >= MAX_PAYLOAD_VALIDATION_ATTEMPTS
    note = (
        f"{PAYLOAD_VALIDATION_FAILURE_SIGNATURE}: the ticket's stored typed "
        f"payload failed schema validation before the agent ran "
        f"(attempt {attempt}/{MAX_PAYLOAD_VALIDATION_ATTEMPTS}): {detail}"
    )
    tk.error_message = note[:2000]
    # The work order/upstream lineage must change; re-running the specialist
    # unchanged is wasteful. This is an orchestrator-owned failure.
    tk.repair_route = "orchestrator"
    if terminal:
        tk.status = "failed"
        tk.summary = (
            f"stored payload failed validation {attempt}× — "
            "failed terminally instead of looping"
        )[:400]
    else:
        tk.summary = (
            f"stored payload failed validation "
            f"(attempt {attempt}/{MAX_PAYLOAD_VALIDATION_ATTEMPTS}); "
            "escalated to orchestrator"
        )[:400]
    await session.commit()

    # Escalate to the orchestrator with the validation error as feedback. A
    # ticket the orchestrator itself owns has nobody upstream to hand to, and a
    # run with no supervisor (e.g. a standalone ticket) has nowhere to escalate;
    # in both cases the recorded error + terminal cap still break the loop.
    if tk.agent_id != "orchestrator" and run.supervisor_ticket_id:
        from zevo.engine.run.wakeup import queue_wakeup
        await queue_wakeup(
            session,
            agent_id="orchestrator",
            ticket_id=run.supervisor_ticket_id,
            source="handoff",
            trigger_detail=f"payload_validation_failed:{tk.id}"[:128],
            reason=(
                f"child ticket {tk.id} could not start: its stored payload "
                f"failed schema validation (attempt {attempt}/"
                f"{MAX_PAYLOAD_VALIDATION_ATTEMPTS}"
                f"{'; failed terminally' if terminal else ''}). "
                f"Re-emit a corrected ticket. Error: {detail}"
            ),
        )
    raise StoredPayloadValidationError(note) from exc


async def run_ticket(
    session: AsyncSession,
    *,
    ticket_id: str,
    work_dir_root: str = "",
    driver_name: str = "",
    model_override: str = "",
) -> Ticket:
    """Run one ticket through the appropriate Driver. Persists everything."""
    tk = (
        await session.execute(select(Ticket).where(Ticket.id == ticket_id))
    ).scalar_one()
    run = (await session.execute(select(Run).where(Run.id == tk.run_id))).scalar_one()
    # Run-terminal guard. A wakeup can outlive its run: reconciler-queued
    # supervisor wakeups don't flip the ticket status, and cancel_run/delete_run
    # mark the run terminal without clearing queued wakeups. Draining such a
    # wakeup would run a full orchestrator LLM heartbeat against a cancelled/
    # finished run — burning tokens and producing confusing post-terminal
    # activity (the pipeline can't advance because children re-check run status,
    # but the heartbeat still executes). The run is over; there is nothing to do.
    if run.status in TERMINAL_RUN_STATUSES:
        return tk
    # Background transcript and progress writers need independent sessions, but
    # they must use the SAME database binding as the caller. This keeps manual
    # runs and isolated test databases from leaking writes into the process-wide
    # default database.
    BackgroundSession = async_sessionmaker(session.bind, expire_on_commit=False)

    # Resolve the Ticket envelope's role-addressed inputs against WorkProducts.
    # If an upstream Ticket failed without producing a WorkProduct, the
    # resolver raises ValueError with a clear message. Don't let that
    # leave the Ticket stuck in `queued` forever (= silent stall, next
    # wakeup retries the same broken state). Mark the ticket FAILED
    # immediately so the orchestrator sees a terminal status and can
    # decide whether to retry the upstream or mark the run failed.
    try:
        resolved_inputs = await resolve_input_bindings(
            session, run_id=tk.run_id, inputs=tk.inputs or {}
        )
    except ValueError as binding_err:
        tk.status = "failed"
        tk.repair_route = "orchestrator"
        tk.error_message = (f"inputs unresolvable: {binding_err}")[:2000]
        tk.summary = (f"inputs unresolvable: {binding_err}")[:400]
        await session.commit()
        if tk.agent_id != "orchestrator":
            await _maybe_wake_supervisor(session, tk)
        raise
    tk.inputs = resolved_inputs
    resolved_payload = _resolve_heldout_payload_assets(tk, tk.payload or {})

    # Per-ticket work dir, ARCHIVED UNDER THE RUN so each run's artifacts are
    # grouped: <work_dir_root>/<run-id>/<ticket-id>/ . Standalone tickets (no
    # run) go under a "standalone" bucket. The per-heartbeat stdout logs land in
    # <run-id>/_logs so they're archived with the run too.
    # Empty = the standard root. Resolved here rather than baked into the
    # signature, so a container and a host checkout each get a path that
    # exists (see zevo.paths).
    work_dir_root = work_dir_root or paths_work_dir_root()
    run_dir = Path(work_dir_root) / (tk.run_id or "standalone")
    work_dir = str(run_dir / tk.id)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    if tk.agent_id == "train":
        telemetry_source = REPO_ROOT / "playbook" / "runners" / "train_telemetry.py"
        if not telemetry_source.is_file():
            raise ValueError(f"system training telemetry helper is missing: {telemetry_source}")
        shutil.copyfile(
            telemetry_source,
            Path(work_dir) / "zevo_train_telemetry.py",
        )
    if tk.agent_id == "inference":
        memory_source = REPO_ROOT / "playbook" / "runners" / "inference_memory.py"
        if not memory_source.is_file():
            raise ValueError(
                f"system inference memory helper is missing: {memory_source}"
            )
        shutil.copyfile(
            memory_source,
            Path(work_dir) / "zevo_inference_memory.py",
        )
        option_scoring_source = (
            REPO_ROOT / "playbook" / "runners" / "option_scoring.py"
        )
        if not option_scoring_source.is_file():
            raise ValueError(
                f"system option-scoring helper is missing: {option_scoring_source}"
            )
        shutil.copyfile(
            option_scoring_source,
            Path(work_dir) / "zevo_option_scoring.py",
        )
    log_dir = str(run_dir / "_logs")

    is_evaluation_runner = tk.agent_id == "evaluation"

    # Ticket input format selects the matching invocation contract. Run.mode is
    # a separate, run-level axis.
    blueprint = (
        AgentBlueprint(
            id="evaluation",
            name="Evaluation Runner",
            title="Quality Evaluator",
            reports_to="system",
            default_driver="evaluation_runner",
            default_model="no-llm",
            output_schema=EvaluationResult,
            tools=[],
            instructions="",
            identity_path=REPO_ROOT / "playbook" / "runners" / "evaluation.md",
        )
        if is_evaluation_runner
        else load_agent(tk.agent_id, input_format=tk.input_format)
    )
    specialist_context = (
        {} if is_evaluation_runner
        else await _build_specialist_context(
            session, run=run, ticket=tk, payload=resolved_payload,
        )
    )
    # Building the per-agent typed input enforces the same schema contract as
    # validate_stored_payload -- this is the typed-payload validation that runs
    # BEFORE the agent executes. A ValidationError here means the ticket's stored
    # payload (or a resolved typed input) is malformed: deterministic, not
    # transient. Left to propagate it would leave the ticket `queued` and the
    # cron would re-drive the identical payload forever. Escalate + bound it.
    try:
        inp = await _build_input(
            agent_id=tk.agent_id,
            ticket=tk,
            payload=resolved_payload,
            inputs=resolved_inputs,
            work_dir=work_dir,
            generation_backend=run.generation_backend,
            run=run,
            session=session,
            specialist_context=specialist_context,
        )
    except ValidationError as payload_err:
        # The session may be mid-statement from the failed build; clear it so the
        # escalation writes commit cleanly.
        await session.rollback()
        await session.refresh(tk)
        await session.refresh(run)
        await _escalate_stored_payload_validation_failure(
            session, tk, run, payload_err
        )
    # Provider sessions are not durable state.  Inject only the bounded lessons
    # that belong to this Run, Agent, lane, and current semantic identity.
    # Transient runtime state is deliberately excluded; a new Run always
    # begins with an empty context.
    if hasattr(inp, "memory") and not is_evaluation_runner:
        inp = inp.model_copy(update={
            "memory": await load_memory_context(
                session,
                run=run,
                ticket=tk,
                current_applicability=applicability_for(run, tk),
            ),
        })

    # Driver + model resolution priority:
    #   1. explicit driver_name / model_override args (agent run --driver X)
    #   2. live Agent DB row (mutable via PATCH /agents/{id} + the UI
    #      Configuration card)
    #   3. identity.md frontmatter (the ship default; what `seed-agents`
    #      pulls into the DB on first install)
    from zevo.db import Agent as _AgentRow
    db_agent = (await session.execute(
        select(_AgentRow).where(_AgentRow.id == tk.agent_id)
    )).scalar_one_or_none()
    if is_evaluation_runner and tk.input_format != "typed":
        raise ValueError(
            "evaluation is a deterministic system stage and accepts typed tickets only"
        )
    # Evaluation is deliberately not configurable through LLM Agent settings. It
    # must never drift back to an LLM provider because a stale DB override or a
    # command-line model override survived an upgrade.
    effective_driver = (
        "evaluation_runner"
        if is_evaluation_runner
        else driver_name
        or (db_agent.default_driver if db_agent else "")
        or blueprint.default_driver
    )
    effective_model = (
        "no-llm"
        if is_evaluation_runner
        else model_override
        or (db_agent.default_model if db_agent else "")
        or blueprint.default_model
    )
    # Per-agent terminal choice (Agent Config page toggle): 'none' | 'openshell'.
    # OpenShell is a claude_cli capability.  Reject an impossible stored/global
    # combination instead of silently claiming that an unsandboxed run was
    # isolated.
    effective_sandbox = (
        "none" if is_evaluation_runner
        else getattr(db_agent, "sandbox", "none") if db_agent else "none"
    )
    if effective_sandbox == "openshell" \
            and (effective_driver != "claude_cli" or tk.agent_id != "orchestrator"):
        raise ValueError(
            "OpenShell is supported only for the claude_cli orchestrator; "
            f"got agent={tk.agent_id!r}, driver={effective_driver!r}"
        )
    # No fallback model: if neither an explicit override, the agent's DB row,
    # nor its frontmatter sets driver+model, fail loudly instead of guessing.
    missing = [
        field for field, val in (("driver", effective_driver), ("model", effective_model))
        if not (val or "").strip()
    ]
    if missing:
        raise ValueError(
            f"agent {tk.agent_id!r} has no {' and '.join(missing)} set — "
            f"configure it at runtime, e.g. "
            f"`agent set {tk.agent_id} --driver bedrock --model moonshotai.kimi-k2.5`"
        )
    driver: Driver = get_driver(effective_driver)

    # Heartbeat row + stdout log file (per-heartbeat, not per-ticket --
    # repeated heartbeats for the same ticket each get their own file so
    # the transcript history is preserved + the UI can stream the LATEST
    # one independently).
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    hb = HeartbeatRun(
        ticket_id=tk.id,
        agent_id=tk.agent_id,
        driver=driver.name,
        model=effective_model,
        operation=str(getattr(inp, "operation", "") or ""),
        activation_phase="",
        stdout_path="",  # filled below now that hb.id is known
    )
    # If the ticket was already cancelled before the runner picks it up
    # (e.g. user clicked Cancel while the wakeup was still queued), abort
    # immediately instead of overwriting the cancelled status.
    await session.refresh(tk)
    if tk.status == "cancelled":
        return tk
    # What the ticket was BEFORE this activation. Once the status flips to
    # `running` below there is no way back to it, and the handoff at the end
    # needs it: a ticket that had already finished is being LOOKED at, not done,
    # and re-announcing it would restart the pipeline behind it.
    status_at_pickup = tk.status
    slurm_contract = getattr(inp, "slurm_job", None)
    if status_at_pickup == "repairing":
        hb.activation_phase = "repair"
    elif isinstance(slurm_contract, SlurmStageJobContract) and slurm_contract.enabled:
        hb.activation_phase = str(slurm_contract.phase or "")

    session.add(hb)
    await session.flush()  # populates hb.id without committing
    stdout_path = str(Path(log_dir) / f"{tk.id}__{hb.id}.stdout.log")
    hb.stdout_path = stdout_path
    tk.status = "running"
    await session.commit()
    heartbeat_id = hb.id

    # Marker -> DB sink
    pending_phase_inserts: list[dict] = []

    # Every row carries the moment its marker was SEEN. The rows are flushed in
    # one batch when the heartbeat ends, so without an explicit ts the column's
    # `server_default=now()` stamped them all with the flush time — every phase
    # of a heartbeat landing on the same instant, which made the timestamps
    # useless for ordering, for locating a phase in the transcript, and for
    # plotting progress over time.
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _at(emitted: float | None) -> datetime:
        """When the marker was PRINTED, falling back to when we saw it.

        A remote stage's markers all reach us in one instant — the agent waits
        for the job, then reads the whole log at once — so `_now()` stamps every
        row in a minutes-long run with the same time and the order is lost. The
        emitter puts its own clock in the marker; prefer that. Guarded because
        the value crosses a machine boundary: a wrong clock, or a bare integer
        in milliseconds, must not produce a row dated 1970 or 55000 AD.
        """
        if emitted is None:
            return _now()
        try:
            ts = datetime.fromtimestamp(float(emitted), timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            return _now()
        now = _now()
        # A remote clock may drift, but not by a day. Anything further out is
        # not drift, it is a unit or format mistake -- use our own time.
        if abs((now - ts).total_seconds()) > 86400:
            return now
        return ts

    # Markers can arrive more than once. A stage that runs remotely writes them
    # to a log, and the agent reads that log whenever it wants to check on the
    # job -- so every read replays the WHOLE log, and one inference stage
    # recorded its `generating` sequence 0->510 twice because the agent looked
    # at the log twice. A replayed marker describes no new progress.
    seen_progress: set[tuple] = set()
    progress_by_step: dict[tuple[str, str, int], dict[str, Any]] = {}

    # A heartbeat is one agent activation, but that activation may launch a
    # concrete Trainer process more than once after repairing an implementation
    # failure.  Keep process identity explicit: otherwise step 20 from the
    # restarted process overwrites step 20 from the failed process and leaves a
    # synthetic curve made from two different executions.  The heartbeat UUID
    # is only the fallback for marker producers that do not launch Trainer.
    marker_attempt = {"current": heartbeat_id}
    marker_phase = {"current": ""}
    seen_attempts: set[str] = set()

    def on_attempt(attempt_id: str, emitted: float | None = None) -> None:
        attempt_id = str(attempt_id or "").strip()
        if not attempt_id:
            return
        marker_attempt["current"] = attempt_id
        if attempt_id in seen_attempts:
            return
        seen_attempts.add(attempt_id)
        pending_phase_inserts.append({
            "attempt_id": attempt_id,
            "event_type": "attempt",
            "phase": marker_phase["current"] or "train",
            "current_step": 0,
            "total_steps": 0,
            "loss": -1.0,
            "extras": {},
            "ts": _at(emitted),
        })

    seen_timed_phases: set[tuple[str, str, float]] = set()

    def on_phase(name: str, emitted: float | None = None) -> None:
        # A live log row is swept again at terminal cleanup. Timestamped phase
        # markers have a stable identity, so drop that replay even when progress
        # rows between phases mean the repeat is not consecutive.
        if emitted is not None:
            try:
                timed_key = (marker_attempt["current"], name, float(emitted))
            except (TypeError, ValueError):
                timed_key = None
            if timed_key is not None:
                if timed_key in seen_timed_phases:
                    return
                seen_timed_phases.add(timed_key)
        # Only consecutive repeats are dropped. An agent legitimately re-reports
        # the same phase as it works (one `reformatting` per file), and phases
        # that alternate (a->b->a) are a real sequence, so a global set would
        # flatten it.
        if (
            pending_phase_inserts
            and pending_phase_inserts[-1].get("event_type") == "phase"
            and pending_phase_inserts[-1].get("attempt_id") == marker_attempt["current"]
            and pending_phase_inserts[-1].get("phase") == name
        ):
            return
        pending_phase_inserts.append({
            "attempt_id": marker_attempt["current"],
            "event_type": "phase",
            "phase": name,
            "current_step": 0,
            "total_steps": 0,
            "loss": -1.0,
            "extras": {},
            "ts": _at(emitted),
        })

    def on_progress(phase: str, data: dict) -> None:
        # Per-step HF logs use `loss`; the final summary uses `train_loss` — and
        # some scripts only emit the latter. Prefer a real per-step loss, else
        # fall back to train_loss/eval_loss so the chart still gets the point.
        loss = data.get("loss", -1.0)
        try:
            loss = float(loss)
        except (TypeError, ValueError):
            loss = -1.0
        if loss < 0:
            for alt in ("train_loss", "eval_loss"):
                try:
                    v = float(data.get(alt))
                    if v >= 0:
                        loss = v
                        break
                except (TypeError, ValueError):
                    pass
        attempt_id = str(data.get("attempt_id") or marker_attempt["current"] or heartbeat_id)
        marker_attempt["current"] = attempt_id
        cur = int(data.get("step", data.get("current_step", 0)))
        tot = int(data.get("total", data.get("total_steps", 0)))
        name = phase or data.get("phase", "")
        # Exact-tuple, not just consecutive: a replay re-emits the whole run of
        # steps, so the duplicates are not adjacent to their originals. Training
        # steps carry a distinct loss each, so real progress is never collapsed.
        dedup_payload = {
            key: value for key, value in data.items()
            if key not in {"t"}
        }
        key = (
            attempt_id, name, cur, tot,
            json.dumps(dedup_payload, sort_keys=True, default=str),
        )
        if key in seen_progress:
            return
        seen_progress.add(key)
        step_key = (attempt_id, str(name), cur)
        extras = {k: v for k, v in data.items()
                  if k not in {"attempt_id", "step", "current_step", "total", "total_steps", "t"}}
        existing = progress_by_step.get(step_key)
        if existing is not None:
            existing["total_steps"] = max(int(existing.get("total_steps") or 0), tot)
            if "loss" in data or float(existing.get("loss", -1.0)) < 0:
                existing["loss"] = loss
            existing["extras"] = {**dict(existing.get("extras") or {}), **extras}
            existing["ts"] = max(existing.get("ts") or _at(None), _at(data.get("t")))
            # A trainer commonly reports two records for one step: the training
            # metrics first and eval_loss a few seconds later. The first record
            # may already be in the DB when the second one is merged here. Mark
            # the row dirty so live persistence performs an UPDATE instead of
            # treating the earlier, incomplete version as final.
            existing["_live_revision"] = int(existing.get("_live_revision") or 0) + 1
            existing.pop("_live_persisted", None)
            return
        row = {
            "attempt_id": attempt_id,
            "event_type": "progress",
            "phase": name,
            "current_step": cur,
            "total_steps": tot,
            "loss": loss,
            "extras": extras,
            "ts": _at(data.get("t")),
            "_live_revision": 1,
        }
        progress_by_step[step_key] = row
        pending_phase_inserts.append(row)

    seen_config: set[str] = set()
    config_rows: list[dict] = []

    def on_config(data: dict) -> None:
        # The full resolved hyper-parameter set the script ran with. It is
        # persisted on HeartbeatRun, not disguised as an execution phase.
        cfg = dict(data) if isinstance(data, dict) else {}
        emitted = cfg.pop("t", None)
        # One config per distinct config. A stage reports it once, but the same
        # report reaches us by more than one route — printed to stdout, and again
        # out of the log swept below — and two identical parameter panels are not
        # two facts. Keyed on the content, so a script that genuinely reconfigures
        # mid-run still records the change.
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key in seen_config:
            return
        seen_config.add(key)
        config_rows.append({
            "phase": "__config__", "current_step": 0, "total_steps": 0,
            "loss": -1.0, "extras": cfg,
            "ts": _at(emitted),
        })

    # MarkerStream consumes RAW stdout lines (bare-stdout drivers). The
    # claude driver's stdout is JSON, so its markers arrive as structured events
    # instead — routed to the SAME callbacks from event_sink below. We keep
    # current phase/attempt tracking here but route inserts through event_sink only, so
    # a marker that appears in BOTH places isn't double-counted.

    def feed_marker_event(ev_type: str, payload: dict) -> None:
        payload = payload if isinstance(payload, dict) else {}
        # Markers carry the ticket that emitted them. An agent routinely READS
        # files that contain other stages' markers -- it cats a train.log to see
        # how the last step went, or reads a sibling's script -- and those are
        # byte-for-byte identical to the ones its own program prints. One
        # inference ticket picked up `connecting`, `downloading_data` and
        # `saving_model` this way, from the train log it inspected; a train
        # ticket recorded `saving_model 492/492` before it had connected,
        # from a stale log it tailed first. Ownership is the only thing that
        # separates them.
        #
        # An UNSTAMPED marker is accepted: older templates and any script we
        # don't control emit bare markers, and dropping those would report
        # nothing at all for them.
        owner = str(payload.pop("owner", "") or "")
        if owner and owner != tk.id:
            return
        if ev_type == "attempt":
            on_attempt(str(payload.get("attempt_id") or ""), payload.get("t"))
        elif ev_type == "phase":
            if payload.get("attempt_id"):
                marker_attempt["current"] = str(payload["attempt_id"])
            marker_phase["current"] = str(payload.get("phase", ""))
            on_phase(marker_phase["current"], payload.get("t"))
        elif ev_type == "progress":
            on_progress(marker_phase["current"], payload)
        elif ev_type == "config":
            on_config(payload)

    log_fh = open(stdout_path, "w", encoding="utf-8")

    def stdout_sink(text: str) -> None:
        log_fh.write(text)
        log_fh.flush()
        # Raw-stdout drivers carry the same stamped marker protocol as
        # structured drivers. Route both through `feed_marker_event` so owner
        # filtering and persistence are identical. The old empty MarkerStream
        # had no callbacks attached, so it parsed these lines and discarded
        # every attempt/phase/progress/config event.
        for parsed in scan_text(text):
            if parsed is None:
                continue
            kind, payload = parsed
            feed_marker_event(kind, payload)

    # Per-heartbeat structured event collector. Each event gets a fresh
    # `seq` (monotonic across the heartbeat) and:
    #   (a) is published to in-process WS subscribers (if any share the
    #       process) via transcript_bus, AND
    #   (b) is queued for incremental DB persistence by a background
    #       flusher task -- so the backend (which lives in a separate
    #       container) can stream events live by tailing the DB.
    event_queue: asyncio.Queue = asyncio.Queue()
    pending_events: list[dict] = []  # kept for the final terminal commit

    # Per-heartbeat usage accumulator. Filled from the driver's `turn_completed`
    # events; folded into hb.* columns on close.
    usage_running: dict[str, int] = {
        "input_tokens": 0, "output_tokens": 0,
        "cached_input_tokens": 0, "reasoning_output_tokens": 0,
    }

    def event_sink(ev: dict) -> None:
        ev_type = str(ev.get("type") or "raw")
        payload = ev.get("payload") or {}
        # __ATTEMPT__/__PHASE__/__PROGRESS__/__CONFIG__ markers (from tool_result or
        # agent stdout) feed the execution_events sink — the ONLY insert path, so
        # nothing is double-counted. They aren't kept as transcript rows.
        if ev_type in ("attempt", "phase", "progress", "config"):
            feed_marker_event(ev_type, payload if isinstance(payload, dict) else {})
            return
        # Accumulate token usage as turns complete. The driver's JSON stream emits
        # one `turn_completed` per assistant turn with a usage subobject.
        if ev_type in ("turn_completed", "turn.completed"):
            usage = payload.get("usage") if isinstance(payload, dict) else None
            if isinstance(usage, dict):
                accumulate_usage(usage_running, usage)
        envelope = transcript_bus.make_event(
            seq=transcript_bus.next_seq(heartbeat_id),
            ev_type=ev_type,
            payload=payload,
        )
        transcript_bus.publish(heartbeat_id, envelope)
        pending_events.append(envelope)
        # Fire-and-forget: queue up for the flusher.
        try:
            event_queue.put_nowait(envelope)
        except Exception as e:
            _warn("event_sink.queue_put", e)

    async def _flush_events_periodically() -> None:
        """Drain event_queue and batch-insert every ~750ms.

        Uses its own session so we never contend with the main runner's
        transaction. Stops when it receives a None sentinel.

        When no real events arrive for ALIVE_INTERVAL_S, emits a
        synthetic 'heartbeat_alive' event so the UI shows a pulse
        instead of going silent during long subprocess executions.
        """
        ALIVE_INTERVAL_S = 30.0   # a meaningful progress pulse every 30s, not spammy noise
        CANCEL_CHECK_S = 3.0
        Session = BackgroundSession
        batch: list[dict] = []
        stopped = False
        last_real_event_time = asyncio.get_event_loop().time()
        # Closure-scoped: each heartbeat gets its own counter so
        # concurrent flushers don't race on a shared function attribute.
        last_cancel_check = 0.0
        try:
            while not stopped:
                try:
                    ev = await asyncio.wait_for(event_queue.get(), timeout=0.75)
                    if ev is None:
                        stopped = True
                    else:
                        batch.append(ev)
                        last_real_event_time = asyncio.get_event_loop().time()
                    # Drain anything else already waiting.
                    while True:
                        try:
                            more = event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if more is None:
                            stopped = True
                        else:
                            batch.append(more)
                            last_real_event_time = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    pass  # idle tick -- still flush whatever we have

                # Check cancel flag every ~3s. The cancel endpoint (in
                # the backend container) sets ticket.status='cancelled'
                # in the DB; we poll here because the flusher is in the
                # same process as the agent subprocess and can SIGTERM it.
                now = asyncio.get_event_loop().time()
                if not stopped and (now - last_cancel_check) >= CANCEL_CHECK_S:
                    last_cancel_check = now
                    try:
                        from zevo.engine.run import process_registry
                        async with Session() as cs:
                            tk_row = (await cs.execute(
                                select(Ticket).where(Ticket.id == tk.id)
                            )).scalar_one_or_none()
                            if tk_row is not None and tk_row.status == "cancelled":
                                # Both keys: a CLI driver registers its own
                                # subprocess under the heartbeat, an in-process
                                # driver registers each shell it runs under the
                                # ticket. Cancelling only the first left the
                                # second running against a released GPU.
                                process_registry.cancel(heartbeat_id)
                                process_registry.cancel(tk.id)
                                # The API performs an eager remote cleanup for
                                # fast feedback. Repeat it only after this
                                # scheduler process has stopped the local
                                # Agent/tool process group: that closes the race
                                # where an SSH command could start a remote
                                # trainer between the API scan and this poll.
                                try:
                                    from zevo.engine.run.remote_jobs import (
                                        cancel_ticket_remote_job,
                                    )
                                    await cancel_ticket_remote_job(cs, tk_row)
                                except Exception as cleanup_exc:
                                    _warn("cancel_remote_cleanup", cleanup_exc)
                                cancel_ev = transcript_bus.make_event(
                                    seq=transcript_bus.next_seq(heartbeat_id),
                                    ev_type="cancelled",
                                    payload={"message": "Cancelled by user"},
                                )
                                transcript_bus.publish(heartbeat_id, cancel_ev)
                                batch.append(cancel_ev)
                                stopped = True
                    except Exception as e:
                        _warn("flusher.cancel_check", e)

                # Emit a synthetic pulse when the subprocess is silent (e.g. a long
                # blocking training command). Instead of a bare "still running",
                # carry the LATEST progress for this ticket — train/inference (and
                # the 30s monitor loop) post __PROGRESS__ into ExecutionEvent, so the
                # pulse can report "training · step N/M · loss L".
                now = asyncio.get_event_loop().time()
                if not stopped and (now - last_real_event_time) >= ALIVE_INTERVAL_S:
                    msg = "Agent is still running…"
                    prog: dict = {}
                    try:
                        async with Session() as ps:
                            # By TIME, not by id: ExecutionEvent.id is a uuid, so
                            # ordering on it returns whichever row happens to sort
                            # highest and never moves — every pulse reported the
                            # same step for the whole of a 20-minute training run.
                            row = (await ps.execute(
                                select(ExecutionEvent)
                                .where(ExecutionEvent.ticket_id == tk.id)
                                .order_by(ExecutionEvent.ts.desc(),
                                          ExecutionEvent.current_step.desc())
                                .limit(1)
                            )).scalar_one_or_none()
                        if row is not None:
                            ph = (row.phase or "").strip()
                            if row.current_step and row.total_steps:
                                pct = f" ({row.current_step * 100 // row.total_steps}%)"
                                loss = f" · loss {row.loss:.4f}" if (row.loss and row.loss > 0) else ""
                                msg = f"{ph or 'running'} · step {row.current_step}/{row.total_steps}{pct}{loss}"
                                prog = {"phase": ph, "step": row.current_step,
                                        "total": row.total_steps, "loss": row.loss}
                            elif ph and ph != "__config__":
                                msg = f"phase: {ph}"
                                prog = {"phase": ph}
                    except Exception as e:
                        _warn("flusher.alive_progress", e)
                    alive_ev = transcript_bus.make_event(
                        seq=transcript_bus.next_seq(heartbeat_id),
                        ev_type="heartbeat_alive",
                        payload={
                            "message": msg,
                            "silent_seconds": round(now - last_real_event_time, 1),
                            **prog,
                        },
                    )
                    transcript_bus.publish(heartbeat_id, alive_ev)
                    batch.append(alive_ev)
                    last_real_event_time = now

                if not batch:
                    continue
                # Persist this batch.
                try:
                    async with Session() as s:
                        for e in batch:
                            ts_raw = e.get("ts")
                            try:
                                ts_dt = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else None
                            except ValueError:
                                ts_dt = None
                            kwargs = {
                                "heartbeat_id": heartbeat_id,
                                "seq": int(e.get("seq") or 0),
                                "type": str(e.get("type") or "raw"),
                                "payload": e.get("payload") or {},
                            }
                            if ts_dt is not None:
                                kwargs["ts"] = ts_dt
                            s.add(TranscriptEvent(**kwargs))
                        await s.commit()
                    batch.clear()
                except Exception as e:
                    # Don't lose events on a transient DB blip -- keep
                    # them in `batch` and retry next iteration. But DO
                    # warn so a permanent DB outage shows up in logs.
                    _warn("flusher.batch_insert", e)
        finally:
            # Final drain on shutdown
            if batch:
                try:
                    async with Session() as s:
                        for e in batch:
                            s.add(TranscriptEvent(
                                heartbeat_id=heartbeat_id,
                                seq=int(e.get("seq") or 0),
                                type=str(e.get("type") or "raw"),
                                payload=e.get("payload") or {},
                            ))
                        await s.commit()
                except Exception as e:
                    _warn("flusher.final_drain", e)

    marker_reader = LiveMarkerReader(Path(work_dir), tk.id, start_at_end=True)
    marker_watch_stop = asyncio.Event()

    async def _persist_new_marker_rows() -> None:
        """Commit newly observed marker rows while the activation is running."""
        rows = [
            row for row in pending_phase_inserts
            if not row.get("_live_persisted") and not row.get("_live_inflight")
        ]
        config = dict(config_rows[-1]["extras"]) if config_rows else None
        config_key = json.dumps(config, sort_keys=True, default=str) if config is not None else ""
        last_config_key = getattr(_persist_new_marker_rows, "last_config_key", "")
        if not rows and (not config_key or config_key == last_config_key):
            return

        for row in rows:
            row["_live_inflight"] = True
            row["_live_inflight_revision"] = int(row.get("_live_revision") or 0)
        Session = BackgroundSession
        try:
            async with Session() as live_session:
                for row in rows:
                    values = {k: v for k, v in row.items() if not k.startswith("_")}
                    if values.get("event_type") == "progress":
                        existing = (await live_session.execute(select(ExecutionEvent).where(
                            ExecutionEvent.heartbeat_id == heartbeat_id,
                            ExecutionEvent.attempt_id == str(values.get("attempt_id") or heartbeat_id),
                            ExecutionEvent.event_type == "progress",
                            ExecutionEvent.phase == str(values.get("phase") or ""),
                            ExecutionEvent.current_step == int(values.get("current_step") or 0),
                        ).limit(1))).scalar_one_or_none()
                        if existing is not None:
                            existing.total_steps = max(
                                int(existing.total_steps or 0),
                                int(values.get("total_steps") or 0),
                            )
                            if float(existing.loss or -1.0) < 0:
                                existing.loss = float(values.get("loss", -1.0))
                            existing.extras = {
                                **dict(existing.extras or {}),
                                **dict(values.get("extras") or {}),
                            }
                            existing.ts = (
                                max(existing.ts, values["ts"])
                                if existing.ts else values["ts"]
                            )
                            continue
                    live_session.add(ExecutionEvent(
                        ticket_id=tk.id, heartbeat_id=heartbeat_id, **values,
                    ))
                if config is not None and config_key != last_config_key:
                    live_hb = (await live_session.execute(
                        select(HeartbeatRun).where(HeartbeatRun.id == heartbeat_id)
                    )).scalar_one_or_none()
                    if live_hb is not None:
                        live_hb.resolved_config = config
                await live_session.commit()
            for row in rows:
                row.pop("_live_inflight", None)
                persisted_revision = int(row.pop("_live_inflight_revision", 0) or 0)
                if int(row.get("_live_revision") or 0) == persisted_revision:
                    row["_live_persisted"] = True
                else:
                    # New metrics arrived while the DB operation was in flight;
                    # leave the row dirty so the next watcher pass writes them.
                    row.pop("_live_persisted", None)
            if config is not None:
                setattr(_persist_new_marker_rows, "last_config_key", config_key)
        except Exception:
            for row in rows:
                row.pop("_live_inflight", None)
                row.pop("_live_inflight_revision", None)
            raise

    async def _watch_local_marker_logs() -> None:
        """Turn markers in a growing local ``*.log`` into live DB rows."""
        try:
            while not marker_watch_stop.is_set():
                for kind, payload in marker_reader.poll():
                    feed_marker_event(kind, payload)
                await _persist_new_marker_rows()
                try:
                    await asyncio.wait_for(marker_watch_stop.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
            for kind, payload in marker_reader.poll(finish=True):
                feed_marker_event(kind, payload)
            await _persist_new_marker_rows()
        except Exception as e:
            _warn("live_marker_watch", e)

    flusher_task = asyncio.create_task(_flush_events_periodically())
    marker_watch_task = asyncio.create_task(_watch_local_marker_logs())

    # Emit a synthetic "started" event so subscribers connecting right
    # at boot get a clear first frame.
    event_sink({"type": "heartbeat_started", "payload": {
        "agent_id": tk.agent_id,
        "ticket_id": tk.id,
        "driver": driver.name,
        "model": effective_model,
    }})

    # Load existing conversation messages for this ticket so the agent sees any
    # prior conversation (e.g. its own previous summary + a follow-up
    # user message that just woke it). Most-recent N, oldest first.
    prior_messages = (await session.execute(
        select(TicketMessage).where(TicketMessage.ticket_id == tk.id).order_by(TicketMessage.created_at)
    )).scalars().all()
    conversation_for_prompt = [
        {"author": c.author, "body": (c.body or "")[:4000]}
        for c in prior_messages[-12:]
    ]

    # Export HEARTBEAT_ID so the driver's subprocess env includes it
    # (the cancel endpoint uses it to look up the process in the registry).
    import os as _os
    _os.environ["HEARTBEAT_ID"] = heartbeat_id

    error_message = ""
    output: BaseModel | None = None
    exit_code = 0
    try:
        result = await driver.run_agent(
            blueprint=blueprint,
            input_payload=inp,
            workspace_dir=work_dir,
            stdout_sink=stdout_sink,
            event_sink=event_sink,
            conversation=conversation_for_prompt,
            model_override=effective_model,
            sandbox_mode=effective_sandbox,
        )
        output = result.output
        exit_code = result.exit_code
    except Exception as e:
        exit_code = 1
        error_message = f"{type(e).__name__}: {e}"
    finally:
        log_fh.close()
        _os.environ.pop("HEARTBEAT_ID", None)

    marker_watch_stop.set()
    try:
        await asyncio.wait_for(marker_watch_task, timeout=5.0)
    except asyncio.TimeoutError:
        marker_watch_task.cancel()

    # Emit terminal transcript event so subscribers see a clean close.
    event_sink({"type": "finished", "payload": {
        "exit_code": exit_code,
        "error_message": error_message[:500],
    }})

    # Signal the flusher to drain its tail + exit, and wait for it
    # (bounded) so we don't lose the terminal `finished` event on
    # subscriber reconnects after this function returns.
    try:
        event_queue.put_nowait(None)
    except Exception as e:
        _warn("flusher.stop_signal", e)
    try:
        await asyncio.wait_for(flusher_task, timeout=5.0)
    except asyncio.TimeoutError:
        flusher_task.cancel()

    transcript_bus.cleanup(heartbeat_id)

    runtime_contract_mismatches: dict[str, object] = {}

    # Persist phase markers in one go, each stamped with the activation that
    # emitted it — the orchestrator writes into one ticket across many wakes,
    # and without this the rows are only separable by guessing from `ts`.
    persisted_progress = (await session.execute(select(ExecutionEvent).where(
        ExecutionEvent.heartbeat_id == heartbeat_id,
        ExecutionEvent.event_type == "progress",
    ))).scalars().all()
    persisted_by_step = {
        (str(row.attempt_id or heartbeat_id), str(row.phase or ""), int(row.current_step or 0)): row
        for row in persisted_progress
    }
    for row in pending_phase_inserts:
        values = {k: v for k, v in row.items() if not k.startswith("_")}
        if values.get("event_type") == "progress":
            key = (
                str(values.get("attempt_id") or heartbeat_id),
                str(values.get("phase") or ""),
                int(values.get("current_step") or 0),
            )
            existing = persisted_by_step.get(key)
            if existing is not None:
                existing.total_steps = max(
                    int(existing.total_steps or 0), int(values.get("total_steps") or 0)
                )
                if float(existing.loss or -1.0) < 0:
                    existing.loss = float(values.get("loss", -1.0))
                existing.extras = {
                    **dict(existing.extras or {}), **dict(values.get("extras") or {})
                }
                existing.ts = max(existing.ts, values["ts"]) if existing.ts else values["ts"]
                continue
        elif row.get("_live_persisted"):
            # Phase rows are immutable markers. Progress rows above are always
            # reconciled because a later eval record may have enriched an
            # already-persisted training step.
            continue
        session.add(ExecutionEvent(ticket_id=tk.id, heartbeat_id=heartbeat_id, **values))
    if config_rows:
        hb.resolved_config = dict(config_rows[-1]["extras"])

    # Did the measured stage go looking at its grader?
    #
    # Inference is handed the questions and the output format and nothing else,
    # so that the score it produces is about the model. It cannot be PREVENTED
    # from reading the scorer — the task directory is a shared mount and the
    # stage has a shell — so the next best thing is that doing so cannot pass
    # unnoticed. Agents were reverse-engineering the output columns straight out
    # of the scorer (`grep -nE "response|prediction|read_csv|columns" eval.py`)
    # while `sample_submission.csv`, which exists to answer exactly that, sat
    # beside it unread. A number produced that way measures the stage and its
    # grader together.
    heldout_access_hits: list[str] = []
    validation_selection_hits: list[str] = []

    if tk.lane == "optimization" and tk.agent_id == "data":
        try:
            holdout = dict(run.holdout or {})
            guarded_validation = [
                str(holdout.get(key) or "")
                for key in (
                    "validation_set", "validation_public",
                    "validation_sample_submission",
                    "validation_evaluation_script",
                )
                if str(holdout.get(key) or "")
            ]
            validation_selection_hits = _referenced_paths(
                guarded_validation, pending_events,
            )
            if validation_selection_hits:
                session.add(TicketNotice(
                    ticket_id=tk.id,
                    code="validation.selection_access",
                    severity="error",
                    body=(
                        "**Training-data selection touched hidden Validation "
                        "artifacts.** Referenced: "
                        + ", ".join(f"`{name}`" for name in validation_selection_hits)
                        + ". The Ticket is rejected so Validation content cannot "
                        "shape the training corpus."
                    )[:8000],
                ))
        except Exception as e:
            _warn("validation_selection_check", e)

    if tk.agent_id == "inference":
        try:
            # The eval script of THIS run, whichever ticket carries it — an
            # inference payload does not name it, which is the whole design.
            wanted = []
            sib = (await session.execute(
                select(Ticket.payload).where(
                    Ticket.run_id == tk.run_id, Ticket.agent_id == "evaluation")
            )).scalars().all()
            for pay in sib:
                s = str((pay or {}).get("evaluation_script") or "")
                if s:
                    wanted.append(s)
            hits = _referenced_paths(wanted, pending_events)
            if hits:
                session.add(TicketNotice(
                    ticket_id=tk.id, code="inference.scorer_access", severity="warning",
                    body=(
                        "**This ticket touched files it is not given.** "
                        f"Referenced: {', '.join('`%s`' % h for h in hits)}.\n\n"
                        "Inference is the measured party: it receives the questions "
                        "and `sample_submission.csv`, and the scorer and answers go "
                        "to the evaluation ticket instead, so the score is about the "
                        "model. Output format comes from `sample_submission.csv`. "
                        "Treat this run's number as unattributable until you have "
                        "checked what was read and why."
                    )[:8000],
                ))
        except Exception as e:
            _warn("leak_check", e)

    # Did anything reach for the held-out test set?
    #
    # The run tunes on validation and is judged on test, and the test files are
    # kept off every payload an agent receives so that no stage has a path to
    # them. A shell running on the shared scheduler mount can still go looking;
    # detect that access and reject the result below. Merely recording a warning
    # allowed a contaminated artifact to flow into Train, which made the final
    # score unusable while the pipeline still looked healthy.
    if tk.lane != "held_out_test":
        try:
            holdout = dict(run.holdout or {})
            # BOTH halves of the test lane. The questions-only copy carries no
            # answers, but a loop ticket reading it is still the loop touching
            # the set it is not allowed to see — and it is the half an agent
            # looking for "the test data" is most likely to find.
            guarded = {
                str(holdout.get(k) or "")
                for k in ("test_set", "test_public")
                if str(holdout.get(k) or "")
            }
            for item in holdout.get("test_sets") or []:
                if not isinstance(item, dict):
                    continue
                guarded.update(
                    str(item.get(key) or "")
                    for key in ("test_set", "public")
                    if str(item.get(key) or "")
                )
            heldout_access_hits = _referenced_paths(
                sorted(guarded), pending_events,
            )
            if heldout_access_hits:
                session.add(TicketNotice(
                    ticket_id=tk.id, code="held_out_test.access", severity="error",
                    body=(
                        "**This ticket touched the held-out test set.** "
                        f"Referenced: {', '.join('`%s`' % h for h in heldout_access_hits)}.\n\n"
                        "The test set is the goal this run is measured against, and "
                        "nothing inside the loop is given a path to it — the harness "
                        "scores it separately. This Ticket is failed and none of its "
                        "artifacts may enter the optimization graph. Tune on the "
                        "validation set instead."
                    )[:8000],
                ))
        except Exception as e:
            _warn("holdout_check", e)

    hb.finished_at = datetime.now(timezone.utc)
    hb.exit_code = exit_code
    hb.error_message = error_message[:2000]
    # Fold token usage + estimated cost onto the heartbeat row.
    hb.input_tokens = int(usage_running.get("input_tokens", 0))
    hb.output_tokens = int(usage_running.get("output_tokens", 0))
    hb.cached_input_tokens = int(usage_running.get("cached_input_tokens", 0))
    hb.reasoning_output_tokens = int(usage_running.get("reasoning_output_tokens", 0))
    hb.estimated_cost_usd = estimate_cost(hb.model, usage_running)

    # Set before the branch below: only one of the two paths assigns it, and the
    # handoff at the end of this function reads it either way.
    cancelled_by_user = False
    repair_scheduled = False
    training_diagnostics_path = ""
    # Same reason, and it was NOT set: only the success branch binds
    # `artifact_meta`, while the measurements handoff at the end of this
    # function reads it unconditionally. Every FAILED ticket therefore raised
    # `UnboundLocalError` there, and the broad `except` around that call turned
    # it into a WARN line — so the failure was invisible except as log noise,
    # and it fired on exactly the tickets a reader is already looking at.
    #
    # Nothing was lost when it fired (a failed ticket has no score to record
    # and no mirror to raise), but a real error inside `_advance_measurements`
    # was indistinguishable from this one in the log.
    artifact_meta: dict = {}
    deferred_stage_row: InfraInstance | None = None
    await session.refresh(tk)
    cancelled_by_user = tk.status == "cancelled"
    if output is None or error_message:
        summary = error_message or "driver produced no output"
        deferred_stage_row = await _latest_slurm_stage_job(session, tk)
        if (
            not cancelled_by_user
            and tk.agent_id in {"train", "inference"}
            and _slurm_job_is_live(deferred_stage_row)
        ):
            # Submission is already a durable side effect. A malformed
            # non-essential Result (for example an invalid memory update) must
            # not cancel a healthy, registered GPU job and pay to run it again.
            # Keep the exact job, let the watcher observe it, and validate the
            # real artifacts on the normal collect activation.
            status = "deferred"
            tk.status = "waiting_external"
            tk.repair_route = ""
            tk.error_message = ""
            tk.summary = (
                f"Slurm job {deferred_stage_row.instance_id} remains active; "
                "waiting for collect despite an invalid submission Result"
            )[:400]
            session.add(TicketNotice(
                ticket_id=tk.id,
                code="slurm.submission_result_discarded",
                severity="warning",
                body=(
                    "The submission activation registered a live finite Slurm "
                    "job, but its typed Result was rejected. Zevo preserved the "
                    "job instead of cancelling/re-submitting it. The later "
                    f"collect activation must report a valid Result. Details: {summary}"
                )[:8000],
            ))
            session.add(HeartbeatResult(
                ticket_id=tk.id,
                heartbeat_id=heartbeat_id,
                agent_id=tk.agent_id,
                status="deferred",
                output={
                    "status": "deferred",
                    "ticket_id": tk.id,
                    "job_id": deferred_stage_row.instance_id,
                    "submission_result_error": summary,
                },
            ))
        else:
            status = "failed"
            tk.error_message = summary[:2000]
            tk.summary = summary[:400]
            repair_scheduled = await _apply_ticket_failure_policy(
                session, tk, error_message=summary, cancelled=cancelled_by_user,
            )
            session.add(HeartbeatResult(
                ticket_id=tk.id,
                heartbeat_id=heartbeat_id,
                agent_id=tk.agent_id,
                status="failed",
                output={
                    "status": "failed",
                    "ticket_id": tk.id,
                    "error_message": summary,
                    "repair_route": tk.repair_route,
                    "repair_attempts": int(tk.repair_attempts or 0),
                },
            ))
    else:
        if (
            isinstance(output, DataResult)
            and output.status == "succeeded"
            and output.operation == "prepare_run_data"
        ):
            try:
                output = _bind_system_validation_artifacts(
                    output, run=run, work_dir=work_dir,
                )
            except (OSError, ValueError) as exc:
                message = f"engine could not prepare frozen Validation artifacts: {exc}"
                output = output.model_copy(update={
                    "status": "failed",
                    "error_message": message,
                    "notes": message,
                })
        status, summary, artifact, artifact_meta = _extract_summary_artifact_meta(
            output, inp=inp, agent_id=tk.agent_id, work_dir=work_dir,
            run_id=tk.run_id,
        )
        if (
            hb.activation_phase == "collect"
            and tk.agent_id == "train"
            and status == "deferred"
        ):
            # A terminal Train collection may submit a finite continuation from
            # a full trainer checkpoint. Its activation is more accurately
            # described as continuation than as ordinary result collection.
            hb.activation_phase = "continue"
        reported_ticket_id = str(getattr(output, "ticket_id", "") or "")
        if reported_ticket_id != tk.id:
            runtime_contract_mismatches["ticket_id"] = {
                "declared": tk.id,
                "reported": reported_ticket_id or "<missing>",
            }
        if status in ("succeeded", "degraded") and isinstance(
            output, (InferenceResult, TrainResult),
        ):
            advisory_warnings: list[str] = []
            try:
                realized_configuration = _validate_specialist_yaml(
                    output, inp, advisory_warnings=advisory_warnings,
                )
                artifact_meta["configuration"] = realized_configuration
                if isinstance(output, TrainResult) and isinstance(inp, TrainTaskInput):
                    artifact_meta.update({
                        "base_model": inp.base_model,
                        "training_method": realized_configuration["training_method"],
                    })
                elif isinstance(output, InferenceResult):
                    artifact_meta["base_model"] = realized_configuration["base_model"]
                    diagnostics = load_generation_diagnostics(
                        output.generation_diagnostics_path
                    )
                    artifact_meta["generation_termination"] = (
                        summarize_generation_diagnostics(diagnostics).model_dump(
                            mode="json"
                        )
                    )
            except (OSError, ValueError) as exc:
                runtime_contract_mismatches["configuration_yaml"] = {
                    "declared": "valid, lineage-consistent YAML",
                    "reported": str(exc),
                }
            for warning in advisory_warnings:
                session.add(TicketNotice(
                    ticket_id=tk.id,
                    code="configuration.advisory_drift",
                    severity="warning",
                    body=warning[:8000],
                ))
        if "operation" in resolved_payload and hasattr(output, "operation"):
            expected_operation = str(resolved_payload.get("operation") or "")
            reported_operation = str(getattr(output, "operation", "") or "")
            if reported_operation != expected_operation:
                runtime_contract_mismatches["operation"] = {
                    "declared": expected_operation,
                    "reported": reported_operation or "<missing>",
                }
        if status == "deferred":
            if not isinstance(output, (TrainResult, InferenceResult)):
                runtime_contract_mismatches["deferred_status"] = {
                    "declared": "Train or Inference cluster stage",
                    "reported": type(output).__name__,
                }
            elif not isinstance(
                getattr(inp, "slurm_job", None), SlurmStageJobContract
            ) or not inp.slurm_job.enabled:
                runtime_contract_mismatches["deferred_status"] = {
                    "declared": "enabled cluster Slurm contract",
                    "reported": "non-cluster execution",
                }
            else:
                stage_row = await _latest_slurm_stage_job(session, tk)
                deferred_stage_row = stage_row
                if stage_row is None:
                    runtime_contract_mismatches["deferred_status"] = {
                        "declared": "registered finite Slurm job",
                        "reported": "no cluster InfraInstance row for this Ticket",
                    }
                elif str((stage_row.meta or {}).get("status_path") or "") != (
                    inp.slurm_job.status_path
                ):
                    runtime_contract_mismatches["slurm_status_path"] = {
                        "declared": inp.slurm_job.status_path,
                        "reported": str(
                            (stage_row.meta or {}).get("status_path") or "<missing>"
                        ),
                    }
                elif (
                    inp.slurm_job.phase == "collect"
                    and stage_row.id == inp.slurm_job.bookkeeping_row_id
                ):
                    runtime_contract_mismatches["deferred_status"] = {
                        "declared": "a newly registered continuation Slurm job",
                        "reported": (
                            "deferred collect result still points at the terminal "
                            f"job {inp.slurm_job.job_id}"
                        ),
                    }
        if heldout_access_hits and status in ("succeeded", "degraded"):
            status = "failed"
            summary = (
                "held-out test isolation violated: "
                + ", ".join(heldout_access_hits)
            )
            error_message = summary
            artifact = ""
            artifact_meta = {}
            if hasattr(output, "status"):
                output.status = "failed"
            if hasattr(output, "error_message"):
                output.error_message = summary
        if validation_selection_hits and status in ("succeeded", "degraded"):
            status = "failed"
            summary = (
                "Validation-blind data selection violated: "
                + ", ".join(validation_selection_hits)
            )
            error_message = summary
            artifact = ""
            artifact_meta = {}
            if hasattr(output, "status"):
                output.status = "failed"
            if hasattr(output, "error_message"):
                output.error_message = summary
        if (
            isinstance(output, DataResult)
            and output.status == "succeeded"
            and output.operation == "prepare_run_data"
        ):
            try:
                if not isinstance(inp, DataTaskInput):
                    raise ValueError("DataResult received a non-Data input")
                stored_recipe = load_data_recipe(output.data_recipe_path)
                if stored_recipe.source_identity != inp.expected_source_identity:
                    raise ValueError(
                        "data_recipe.source_identity must exactly copy "
                        "expected_source_identity"
                    )
                if stored_recipe.training_method != inp.training_method:
                    raise ValueError("data_recipe.training_method differs from the Ticket")
                if inp.expected_source_fingerprint:
                    current_source_fingerprint = file_sha256(inp.dataset)
                    if current_source_fingerprint != inp.expected_source_fingerprint:
                        raise ValueError(
                            "source file bytes changed after the Data work order "
                            "was created"
                        )
                    if (
                        stored_recipe.source_fingerprint
                        != inp.expected_source_fingerprint
                    ):
                        raise ValueError(
                            "data_recipe.source_fingerprint must exactly copy "
                            "expected_source_fingerprint (bare 64-character "
                            "lowercase SHA-256 with no prefix)"
                        )
                pinned_methods = inp.configuration_pins.get("method_ids")
                if pinned_methods is not None and list(pinned_methods) != stored_recipe.method_ids:
                    raise ValueError("data_recipe.method_ids violates pinned method_ids")
                pinned_size = inp.configuration_pins.get("target_size")
                if pinned_size is not None and int(pinned_size) != output.n_rows_out:
                    raise ValueError("DataResult.n_rows_out violates pinned target_size")
                if stored_recipe.direction != inp.recipe_intent.direction:
                    # `direction` is explanatory prose, not an execution
                    # control. Surface drift for review, but do not discard a
                    # byte-identifiable dataset whose operative recipe fields
                    # still match exactly.
                    session.add(TicketNotice(
                        ticket_id=tk.id,
                        code="data.recipe_direction_drift",
                        severity="warning",
                        body=(
                            "Data's recipe rationale differs from the requested "
                            "direction. Operative recipe fields and artifact "
                            "identity still matched, so the result was accepted."
                        ),
                    ))
                for field in (
                    "subset", "filters", "sampling", "weighting",
                    "transformations", "field_mapping", "seed",
                ):
                    requested = getattr(inp.recipe_intent, field)
                    if requested not in (None, "", 0, [], {}) and (
                        getattr(stored_recipe, field) != requested
                    ):
                        raise ValueError(
                            f"data_recipe.{field} differs from the requested recipe intent"
                        )
                realized_signature = data_recipe_signature(
                    stored_recipe, output.training_dataset_path,
                )
                artifact_meta.update({
                    "dataset_name": stored_recipe.dataset_name,
                    "dataset_source": stored_recipe.source_identity,
                    "system_scoring_duplicates_removed": int(
                        output.system_scoring_duplicates_removed
                    ),
                    "method_ids": list(stored_recipe.method_ids),
                    "audit_steps": list(stored_recipe.audit_steps),
                    "data_intent_signature": inp.data_intent_signature,
                    "data_signature": realized_signature,
                    "data_recipe": stored_recipe.model_dump(mode="json"),
                })
                if output.validation_source_path:
                    validation_source = str(
                        (run.holdout or {}).get("validation_set") or ""
                    )
                    if not _same_local_file(
                        output.validation_source_path, validation_source,
                    ):
                        raise ValueError(
                            "engine-bound Validation source differs from Run state"
                        )
                    if output.validation_answer_fields != _answer_fields(
                        dict(run.holdout or {}), "validation",
                    ):
                        raise ValueError(
                            "engine-bound Validation answer fields differ from Run state"
                        )
            except (OSError, ValueError) as exc:
                runtime_contract_mismatches["data_recipe"] = {
                    "declared": "exact recipe identity with frozen Validation artifacts",
                    "reported": str(exc),
                }
            try:
                profile = InferenceDataProfile.model_validate_json(
                    Path(output.inference_data_profile_path).read_text(encoding="utf-8")
                )
                expected_removed = set(_answer_fields(
                    dict(run.holdout or {}), "validation",
                ))
                if set(profile.answer_fields_removed) != expected_removed:
                    raise ValueError(
                        "profile answer_fields_removed differs from Validation contract"
                    )
                shape = _tabular_shape(output.scoring_public_path)
                if shape is not None:
                    n_rows, record_fields = shape
                    if profile.n_rows != n_rows:
                        raise ValueError(
                            "profile n_rows differs from questions-only Validation "
                            f"data ({profile.n_rows} != {n_rows})"
                        )
                    if set(profile.record_fields) != record_fields:
                        raise ValueError(
                            "profile record_fields differ from questions-only Validation data"
                        )
                    missing_inputs = sorted(set(profile.input_fields) - record_fields)
                    if missing_inputs:
                        raise ValueError(
                            f"profile input_fields are absent from Validation data: {missing_inputs}"
                        )
                else:
                    session.add(TicketNotice(
                        ticket_id=tk.id,
                        code="data.profile_shape_unverified",
                        severity="warning",
                        body=(
                            "The inference profile schema is valid, but row/field "
                            "cross-checking is unavailable for this Validation file format."
                        ),
                    ))
                submission_path = (
                    output.sample_submission_path
                    or str((run.holdout or {}).get("validation_sample_submission") or "")
                )
                if not submission_path:
                    raise ValueError(
                        "Data did not preserve or create the required sample submission"
                    )
                with Path(submission_path).open(
                    "r", encoding="utf-8-sig", newline=""
                ) as handle:
                    columns = list(next(csv.reader(handle), []))
                if columns != profile.submission_columns:
                    raise ValueError(
                        "profile submission_columns differ from sample submission"
                    )
            except (OSError, ValueError) as exc:
                runtime_contract_mismatches["inference_data_profile"] = {
                    "declared": "valid closed answer-free profile",
                    "reported": str(exc),
                }
        if (
            isinstance(output, DataResult)
            and output.status == "succeeded"
            and output.operation == "scope_problem"
        ):
            # Auto mode: the contract is the artifact. Validate it here, at the
            # runtime-contract boundary, so a malformed ScopingResult fails the
            # Ticket (and enters ordinary repair) before settlement is attempted.
            try:
                from zevo.contracts.scoping import load_scoping_result
                scoping = load_scoping_result(output.scoping_result_path)
                for name in ("test_set_path", "test_sample_submission_path"):
                    if not Path(getattr(scoping, name)).is_file():
                        raise ValueError(f"{name} does not exist: {getattr(scoping, name)}")
                if scoping.metric_type == "custom" and not Path(scoping.evaluation_script).is_file():
                    raise ValueError(
                        f"evaluation_script does not exist: {scoping.evaluation_script}"
                    )
                artifact_meta.update({
                    "eval_source": scoping.eval_source,
                    "metric": scoping.metric,
                    "metric_direction": scoping.metric_direction,
                    "test_rows": scoping.test_rows,
                })
            except (OSError, ValueError) as exc:
                runtime_contract_mismatches["scoping_result"] = {
                    "declared": "valid ScopingResult with existing held-out assets",
                    "reported": str(exc),
                }
        elif isinstance(output, DataResult) and output.status == "succeeded":
            try:
                if not isinstance(inp, DataTaskInput):
                    raise ValueError("DataResult received a non-Data input")
                holdout = dict(run.holdout or {})
                is_private = output.operation == "prepare_holdout_data"
                scoring_source = (
                    inp.scoring_set
                    if is_private
                    else str(holdout.get("validation_set") or "")
                )
                answer_fields = (
                    inp.answer_fields
                    if is_private
                    else _answer_fields(holdout, "validation")
                )
                sample_submission = (
                    output.sample_submission_path
                    or (
                        inp.sample_submission
                        if is_private
                        else str(holdout.get("validation_sample_submission") or "")
                    )
                )
                report = validate_data_artifacts(
                    operation=output.operation,
                    scoring_source=scoring_source,
                    questions=output.scoring_public_path,
                    sample_submission=sample_submission,
                    answer_fields=answer_fields,
                    training_dataset=output.training_dataset_path,
                    validation_dataset=output.validation_dataset_path,
                    profile_path=output.inference_data_profile_path,
                )
                artifact_meta.update({
                    "training_rows": report.training_rows,
                    "validation_rows": report.validation_rows,
                    "question_rows": report.question_rows,
                })
            except (OSError, ValueError) as exc:
                runtime_contract_mismatches["data_artifacts"] = {
                    "declared": "valid, source-faithful Data artifacts",
                    "reported": str(exc),
                }
        if isinstance(output, InferenceResult) and output.status == "succeeded":
            try:
                if not isinstance(inp, InferenceTaskInput):
                    raise ValueError("InferenceResult received a non-Inference input")
                report = validate_prediction_artifacts(
                    predictions=output.predictions_path,
                    questions=inp.scoring_set,
                    sample_submission=inp.sample_submission,
                )
                # Measured artifact shape, rather than an Agent-authored echo,
                # owns the persisted row count and columns.
                artifact_meta["n_rows"] = report.rows
                artifact_meta["prediction_columns"] = list(report.columns)
            except (OSError, ValueError) as exc:
                runtime_contract_mismatches["prediction_artifacts"] = {
                    "declared": "valid predictions aligned to assigned inputs",
                    "reported": str(exc),
                }
        if runtime_contract_mismatches and status in ("succeeded", "degraded", "deferred"):
            status = "failed"
            details = "; ".join(
                f"{key}: {value.get('reported', value) if isinstance(value, dict) else value}"
                for key, value in sorted(runtime_contract_mismatches.items())
            )
            summary = f"runtime experiment contract violated: {details}"[:2000]
            error_message = summary
            artifact = ""
            artifact_meta = {}
            if hasattr(output, "status"):
                output.status = "failed"
            if hasattr(output, "error_message"):
                output.error_message = summary
            if hasattr(output, "summary"):
                output.summary = summary
        if isinstance(output, TrainResult) and status in ("succeeded", "degraded"):
            # Loss evidence is derived from the runner's persisted marker
            # protocol, never copied from Agent prose. Keep it as a small local
            # artifact so the Orchestrator can compare training/Validation loss
            # trends after the Validation metric arrives.
            try:
                # Cluster telemetry normally arrives under the submit
                # heartbeat; Collect may only replay its tail. Summarize the
                # complete Ticket ledger so the history describes the actual
                # Trainer attempt rather than whichever lines Collect printed.
                diagnostic_events = (await session.execute(
                    select(ExecutionEvent).where(
                        ExecutionEvent.ticket_id == tk.id,
                        ExecutionEvent.event_type == "progress",
                    ).order_by(ExecutionEvent.ts, ExecutionEvent.id)
                )).scalars().all()
                diagnostics = _training_diagnostics_from_events(diagnostic_events)
                diagnostics_file = Path(work_dir) / "training_diagnostics.json"
                diagnostics_file.write_text(
                    json.dumps(
                        diagnostics.model_dump(mode="json"), indent=2, sort_keys=True,
                    ) + "\n",
                    encoding="utf-8",
                )
                training_diagnostics_path = str(diagnostics_file.resolve())
                artifact_meta["training_diagnostics"] = diagnostics.model_dump(
                    mode="json"
                )
            except (OSError, ValueError) as exc:
                session.add(TicketNotice(
                    ticket_id=tk.id,
                    code="training.diagnostics_unavailable",
                    severity="warning",
                    body=(
                        "The checkpoint remains usable, but the runner could not "
                        f"materialize its loss summary: {exc}"
                    )[:8000],
                ))
        if status in ("succeeded", "degraded"):
            essential_problems: list[str] = []
            for field in _essential_artifact_fields(output):
                claimed = str(getattr(output, field, "") or "")
                if not claimed:
                    essential_problems.append(f"{field} is empty")
                    continue
                found = _relocate_artifact(claimed, work_dir)
                if found != claimed:
                    setattr(output, field, found)
                problem = _artifact_problem(found)
                if problem:
                    essential_problems.append(f"{field} is {problem}: {found}")
            if essential_problems:
                status = "failed"
                summary = (
                    f"{tk.agent_id} reported usable success without required "
                    "artifacts: " + "; ".join(essential_problems)
                )[:2000]
                error_message = summary
                artifact = ""
                artifact_meta = {}
                if hasattr(output, "status"):
                    output.status = "failed"
                if hasattr(output, "error_message"):
                    output.error_message = summary
        # GenerationBackend-level guard: if the agent claimed success WITH an
        # artifact path, that file/dir MUST really exist and be non-empty.
        # This catches agents that hallucinate success without producing
        # their output (the lazy data agent that never wrote
        # dataset.jsonl). Demote to failure so the run fails here instead
        # of letting a downstream agent crash on a phantom path.
        # A finetuned model left on the REMOTE GPU box is intentionally NOT on
        # this host (inference reads it over ssh; registry pulls only the best one
        # back). Skip the local-existence guard for it — the path is remote.
        remote_model = (
            tk.agent_id == "train"
            and bool(getattr(output, "checkpoint_is_remote", False))
        )
        if artifact and status in ("succeeded", "degraded") and not remote_model:
            # A path that is merely mis-written is not a failed ticket: look for
            # the artifact under this ticket's work dir before judging it.
            found = _relocate_artifact(artifact, work_dir)
            if found != artifact:
                print(f"[runner] artifact path corrected: {artifact} -> {found}",
                      file=sys.stderr, flush=True)
                artifact = found
            problem = _artifact_problem(artifact)
            if problem:
                status = "failed"
                summary = (
                    f"agent reported {tk.agent_id} success but its claimed "
                    f"artifact is {problem}: {artifact}"
                )
                error_message = summary
                artifact = ""  # never register a phantom work_product
        # A cancelled ticket stays cancelled. Cancelling SIGTERMs the driver,
        # which surfaces here as a non-zero exit — indistinguishable from a
        # genuine failure unless we look. Overwriting the status lost the fact
        # that a human stopped this, and made the handoff below treat a
        # deliberate stop as a failure worth reacting to.
        await session.refresh(tk)
        if tk.status == "cancelled":
            cancelled_by_user = True
            tk.repair_route = "terminal"
        else:
            cancelled_by_user = False
            if status == "failed":
                repair_scheduled = await _apply_ticket_failure_policy(
                    session,
                    tk,
                    error_message=error_message or summary,
                    cancelled=False,
                )
            elif status == "deferred":
                tk.status = "waiting_external"
                tk.repair_route = ""
            else:
                tk.status = (
                    "succeeded"
                    if status == "succeeded"
                    else "degraded"
                    if status == "degraded"
                    else "skipped"
                )
                tk.repair_route = ""
        _set_current_ticket_outcome_text(tk, status=status, summary=summary)
        if status in ("succeeded", "degraded"):
            _apply_run_score_monitor(run, tk, output, artifact_meta)
        session.add(HeartbeatResult(
            ticket_id=tk.id, heartbeat_id=heartbeat_id, agent_id=tk.agent_id,
            status=status, output=output.model_dump(),
        ))
        # Work products — register EVERY local artifact this agent produced
        # (script + output), so a ticket's artifact list is complete + uniform
        # across agents, not just one file. The rich meta rides on the primary
        # OUTPUT; scripts get bare meta.
        if artifact and status in ("succeeded", "degraded"):
            if isinstance(output, EvaluationResult):
                # Evaluation's metrics are materialized to `artifact` already.
                # A custom Task scorer also leaves the exact Bash wrapper and
                # captured diagnostics that ran it, so the deterministic call
                # is inspectable without an LLM transcript.
                items = [(artifact, "metrics", dict(artifact_meta or {}))]
                for name, role in (
                    ("evaluate.sh", "evaluation_command"),
                    ("evaluate.log", "evaluation_log"),
                ):
                    candidate = str(Path(work_dir) / name)
                    if not _artifact_problem(candidate):
                        items.append((candidate, role, {}))
            elif isinstance(output, RegisterResult):
                # The manifest owns the selected version, score, retention
                # decision, and model path. RegisterResult deliberately reports
                # only paths, so derive the retained-model WorkProduct from the
                # validated manifest metadata instead of a second agent echo.
                items = [
                    (output.registry_path, "registry_entry", dict(artifact_meta or {})),
                ]
                retained_model = str(artifact_meta.get("model_path") or "")
                if artifact_meta.get("retained") and retained_model:
                    items.append(
                        (retained_model, "retained_model", dict(artifact_meta or {}))
                    )
                if output.register_script_path:
                    items.append((output.register_script_path, "script", {}))
            else:
                items = []
                for field, role in _TICKET_ARTIFACTS.get(tk.agent_id, []):
                    v = getattr(output, field, "")
                    if isinstance(v, str) and v:
                        # The first artifact ACTUALLY PRESENT is the primary
                        # output and carries measured metadata. Held-out Data
                        # has no training_dataset_path, so using the manifest
                        # index here used to leave its scoring_questions row
                        # without the measured Test row count.
                        items.append((
                            v, role,
                            dict(artifact_meta or {}) if not items else {},
                        ))
                if isinstance(output, TrainResult):
                    # The primary final model is the default checkpoint for
                    # downstream bindings. Intermediate models share the role
                    # so an Orchestrator may select one by exact WorkProduct id.
                    if items and items[0][1] == "checkpoint":
                        items[0][2]["checkpoint_kind"] = "final"
                    for checkpoint in output.intermediate_checkpoints:
                        items.append((
                            checkpoint.path,
                            "checkpoint",
                            {
                                "checkpoint_kind": "intermediate",
                                "step": checkpoint.step,
                                "epoch": checkpoint.epoch,
                                "retention_reason": checkpoint.retention_reason,
                                "save_only_model": True,
                                "base_model": artifact_meta.get("base_model", ""),
                                "training_method": artifact_meta.get("training_method", ""),
                            },
                        ))
                    if training_diagnostics_path:
                        items.append((
                            training_diagnostics_path,
                            "training_diagnostics",
                            {"source": "engine_execution_events"},
                        ))
                if not items:  # fallback: at least the primary artifact
                    items = [(artifact, _role_for(tk.agent_id), dict(artifact_meta or {}))]
            for path, role, wp_meta in _unique_work_products(items):
                is_remote = tk.agent_id == "train" and role == "checkpoint" and remote_model
                if is_remote:
                    # UI shows "on remote (transient)" instead of a scary "missing".
                    wp_meta = {**wp_meta, "location": "remote"}
                elif path != artifact:
                    # The PRIMARY artifact is existence-checked above, and a
                    # miss there fails the ticket. The secondary ones were not
                    # checked at all, so a mistyped `log_path` or an eval file
                    # the agent claimed but never wrote became a row in the
                    # Artifacts panel that opens onto nothing. Correct the path
                    # the same way, and skip the row if there is really no file
                    # — a missing script is not worth failing a good ticket for,
                    # but it is not worth listing either.
                    found = _relocate_artifact(path, work_dir)
                    if _artifact_problem(found):
                        print(f"[runner] skipping unregistered artifact {path!r} "
                              f"({_artifact_problem(found)})", file=sys.stderr, flush=True)
                        continue
                    path = found
                session.add(WorkProduct(
                    ticket_id=tk.id, role=role, path=path, meta=wp_meta,
                ))

        # Successful workers may report small durable lessons for their own
        # later Tickets in this Run.  The savepoint keeps a memory persistence
        # bug from rolling back an otherwise valid Ticket and its artifacts.
        # Failed/cancelled outputs never teach the next iteration.
        if status in ("succeeded", "degraded") and not cancelled_by_user:
            memory_updates = list(getattr(output, "memory_updates", []) or [])
            if memory_updates:
                try:
                    async with session.begin_nested():
                        await persist_memory_updates(
                            session,
                            run=run,
                            ticket=tk,
                            updates=memory_updates,
                            current_applicability=applicability_for(run, tk),
                        )
                except Exception as exc:  # noqa: BLE001
                    _warn("memory_persist", exc)
                    session.add(TicketNotice(
                        ticket_id=tk.id,
                        code="memory.persistence_failed",
                        severity="warning",
                        body=(
                            "**Agent memory could not be persisted.** The Ticket "
                            "result and artifacts remain valid, but later Tickets "
                            "will not receive these reported lessons."
                        ),
                    ))

    if not cancelled_by_user and status == "deferred" and deferred_stage_row is not None:
        # This is the ownership boundary: all local configuration/script
        # checks and the durable JOBID/status-path registration have passed.
        # Only now may the deterministic watcher stream or poll this job.
        await session.refresh(deferred_stage_row)
        state = _commit_slurm_submission(
            deferred_stage_row,
            heartbeat_id=heartbeat_id,
            stdout_path=inp.slurm_job.stdout_path,
            stderr_path=inp.slurm_job.stderr_path,
        )
        prefix = "Running" if state == "RUNNING" else "Waiting"
        if prefix == "Running":
            tk.status = "running"
            body = (
                f"Running: Slurm job {deferred_stage_row.instance_id} is running; "
                "Zevo will collect and validate its outputs after it finishes."
            )
        else:
            body = (
                f"Waiting: Slurm job {deferred_stage_row.instance_id} was submitted "
                "and registered; Zevo is waiting for Slurm to start it."
            )
        await _add_slurm_stage_message_once(
            session,
            ticket=tk,
            job=deferred_stage_row,
            prefix=prefix,
            body=body,
        )
    elif (
        not cancelled_by_user
        and status == "succeeded"
        and isinstance(getattr(inp, "slurm_job", None), SlurmStageJobContract)
        and inp.slurm_job.enabled
    ):
        completed_stage_row = await _latest_slurm_stage_job(session, tk)
        if completed_stage_row is not None:
            await _add_slurm_stage_message_once(
                session,
                ticket=tk,
                job=completed_stage_row,
                prefix="Done",
                body=(
                    f"Done: Slurm job {completed_stage_row.instance_id} completed; "
                    f"the {tk.agent_id} Result and all required artifacts passed "
                    "Zevo validation."
                ),
            )

    await session.commit()

    if status == "deferred" and not cancelled_by_user:
        # The LLM activation ends here. A deterministic backend watcher owns
        # Slurm polling and will re-queue this same Ticket after the exact job
        # reaches a terminal state. Do not wake the Orchestrator in between.
        return tk

    if repair_scheduled:
        # Repair remains inside the same Ticket and must not wake the
        # Orchestrator between attempts. The new activation sees the exact
        # system repair message above plus every existing artifact in work_dir.
        # Reap an orphaned remote process before starting the next activation;
        # otherwise a failed local Agent/SSH session could leave the old
        # trainer consuming the same GPU beside its repair attempt.
        try:
            from zevo.engine.run.remote_jobs import cancel_ticket_remote_job
            await cancel_ticket_remote_job(session, tk)
        except Exception as exc:  # cleanup is best-effort; repair remains queued
            _warn("repair_remote_cleanup", exc)
        from zevo.engine.run.wakeup import queue_wakeup
        await queue_wakeup(
            session,
            agent_id=tk.agent_id,
            ticket_id=tk.id,
            source="retry",
            trigger_detail=f"repair_{int(tk.repair_attempts or 0)}",
            reason=(
                f"automatic self-repair {int(tk.repair_attempts or 0)}/"
                f"{MAX_REPAIR_ATTEMPTS}: {tk.error_message[:500]}"
            ),
            payload={"repair_attempt": int(tk.repair_attempts or 0)},
        )
        return tk

    # Mirror only this Run's committed Registry entry. Keep the DB bridge in a
    # separate transaction: a malformed Registry document is reported without
    # poisoning the Ticket session needed for the supervisor handoff.
    if isinstance(output, RegisterResult) and output.status == "succeeded" and output.registry_path:
        try:
            from zevo.engine.persistence import upsert_registry_from_manifest
            from zevo.db import get_session_factory
            MirrorSession = get_session_factory()
            async with MirrorSession() as mirror_session:
                await upsert_registry_from_manifest(
                    mirror_session,
                    registry_manifest_path=output.registry_path,
                    run_id=tk.run_id or "",
                )
        except Exception as e:
            _warn("registry_manifest_mirror", e)
            # Mirroring uses its own session, so a malformed Registry row can
            # never expire the Ticket objects needed for the supervisor handoff.

    # The child POST made during an Orchestrator activation only persists the
    # Ticket.  Release its execution signal now, after this heartbeat's result,
    # final status, and finished_at have all been validated and committed.
    # Consequently an in-progress "starting …" message cannot run the next
    # stage beside the Orchestrator that authored it.
    if tk.agent_id == "orchestrator" and not cancelled_by_user:
        await _release_emitted_child(
            session,
            supervisor=tk,
            output=output,
            heartbeat_id=heartbeat_id,
        )

    # Reactive supervisor: after any non-orchestrator ticket
    # completes, refresh and wake the run's one supervisor ticket. This is the
    # "reactive loop" that
    # turns the orchestrator from a one-shot planner into a supervisor.
    # A ticket that was ALREADY finished before this wake does not announce
    # itself again. Re-opening a completed ticket is a normal thing to do — to
    # inspect a checkpoint, to answer a question about it — and the agent knows
    # it: told to re-run one with no new instruction, it verified the artifact
    # and reported success, which is the correct outcome for what it was asked.
    # But the supervisor reads "child finished" as "advance the pipeline", so
    # that success re-fired a step the run had already taken.
    if (
        tk.agent_id != "orchestrator"
        and not cancelled_by_user
        and status_at_pickup not in TERMINAL_TICKET_STATUSES
    ):
        # Auto mode: the scoping Ticket is Run Setup, not a pipeline stage. Its
        # completion settles the scoring contract onto the Run and creates the
        # supervisor; a terminal failure fails the Run. There is no
        # Orchestrator to wake before that, and no measurement to record.
        from zevo.engine.run.scoping import complete_scoping, is_scoping_ticket
        if is_scoping_ticket(tk):
            try:
                await complete_scoping(session, tk, output)
            except Exception as e:  # noqa: BLE001
                _warn("scoping_settlement", e)
            return tk
        # Both score series are written here. The held-out mirror is deliberately
        # not raised from ordinary Inference completion: Orchestrator first
        # creates Validation Evaluation, and only its completion starts the
        # private Test chain. Validation therefore always precedes Test.
        try:
            await _advance_measurements(
                session, tk, extract_score(artifact_meta or {}), output,
            )
        except Exception as e:
            _warn("measurements", e)
        # A held-out ticket is the harness measuring the run, not the run
        # making progress. Waking the supervisor for it would put a test-set
        # completion into `last_event` and hand the loop the one signal the
        # whole arrangement exists to keep from it.
        #
        # With one exception, and it is a barrier rather than a signal. An
        # EVALUATION is the end of a round's measurement, and the round is not
        # measured until both lanes have reported — so if the held-out lane is
        # still working, the wake is deferred and the lane's own final eval
        # issues it instead. Nothing about the test set crosses over: the
        # supervisor is told only that its own child finished, which is what it
        # was going to be told anyway, just later.
        #
        # Only evaluations wait. Inference wakes Orchestrator so it can create
        # Validation Evaluation; completion of that Evaluation starts the
        # private mirror branch.
        held_out_ticket = tk.lane == "held_out_test"
        is_eval = tk.agent_id == "evaluation"
        if held_out_ticket:
            # ANY mirror completion can be the one that settles the lane, not
            # just its final eval. A copy job that fails never reaches an eval
            # at all, so releasing only there left the deferred wake held by a
            # lane that had already stopped, and the run simply never advanced.
            if await _holdout_settled(session, tk.run_id):
                await _release_deferred_wake(session, tk.run_id)
        elif is_eval and not await _holdout_settled(session, tk.run_id):
            # Record WHICH wake is being held, rather than have the release step
            # guess: a lane that settles with nothing waiting must issue no wake,
            # and "the newest finished evaluation" is not the same question.
            tk.supervisor_wake_deferred = True
            await session.commit()
        else:
            await _maybe_wake_supervisor(session, tk)

    return tk


# ─────────────────────────── the held-out lane ───────────────────────────────
#
# The orchestrator optimizes the validation set; nobody optimizes the test set.
# Every time the loop measures a model on validation, the harness quietly
# measures the SAME model on the test set and files the result where no agent
# reads. That is what makes the run's headline number worth anything: it was
# never available to the process that chose the model, so it could not have
# been chosen for.
#
# The lane is three hops, each triggered by the completion of the last:
#   1. validation Evaluation is created   -> mirror its inference on test
#   2. the mirror succeeds                 -> score it with the user's eval.py
#   3. that eval succeeds                  -> record the number, tell no one
#
# Held-out tickets never wake the supervisor, so the loop does not even see
# them finish.


def _current_iteration(run: Run) -> int:
    """Which iteration the run is in: 0 until the baseline is on the board.

    The baseline probe IS iteration 0, so the phase that produces it cannot
    also be iteration 1. `iterations_completed` counts Validation-measured
    trained candidates; the baseline's own entry in `history` tells setup from
    the first training loop.
    """
    has_baseline = any(
        isinstance(h, dict) and h.get("source") == "baseline"
        for h in (run.history or [])
    )
    return int(run.iterations_completed or 0) + 1 if has_baseline else 0


async def _record_prepared_scoring_data(
    session: AsyncSession, run: Run, ticket: Ticket, result: BaseModel,
) -> None:
    """Record the engine-bound scoring package once and for good.

    Optimization Data returns before the engine creates these artifacts and
    never receives their paths. Private held-out Data still owns only exact
    answer removal for Test. Freezing the resulting paths here keeps every
    iteration on one population and one output schema.
    """
    path = str(getattr(result, "scoring_public_path", "") or "")
    is_held_out_test = ticket.lane == "held_out_test"
    holdout = dict(run.holdout or {})
    changed = False

    if is_held_out_test:
        test_set_name = str((ticket.payload or {}).get("test_set_name") or "test")
        suite = _heldout_test_sets(run)
        for item in suite:
            if item["name"] != test_set_name:
                continue
            if path and not item.get("public"):
                item["public"] = path
                changed = True
            profile_path = str(
                getattr(result, "inference_data_profile_path", "") or ""
            )
            if profile_path and not item.get("inference_data_profile"):
                item["inference_data_profile"] = profile_path
                changed = True
            break
        if changed:
            holdout["test_sets"] = suite
            # The scalar projection keeps old dashboard/run readers useful.
            if suite and suite[0]["name"] == test_set_name:
                holdout["test_public"] = suite[0].get("public", "")
    else:
        if path and not holdout.get("validation_public"):
            # The first prepared copy is the one the run is scored on.
            holdout["validation_public"] = path
            changed = True

    # The engine supplies the Validation binding after optimization Data has
    # finished. The Setting-owned evaluator remains fixed independently from
    # Test, and the binding remains fixed across iterations.
    if not is_held_out_test:
        profile_path = str(
            getattr(result, "inference_data_profile_path", "") or ""
        )
        if profile_path and not holdout.get("inference_data_profile"):
            try:
                profile = InferenceDataProfile.model_validate_json(
                    Path(profile_path).read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "The engine produced an invalid inference_data_profile: "
                    f"{exc}"
                ) from exc
            expected_removed = set(_answer_fields(holdout, "validation"))
            if set(profile.answer_fields_removed) != expected_removed:
                raise ValueError(
                    "inference_data_profile.answer_fields_removed must exactly "
                    "match the recorded Validation answer fields"
                )
            holdout["inference_data_profile"] = profile_path
            changed = True

        for field, attr in (
            # The frozen raw Validation population. Train may render a
            # temporary method-specific eval view, but Data never receives or
            # authors this artifact.
            ("validation_dataset", "validation_dataset_path"),
            ("validation_sample_submission", "sample_submission_path"),
        ):
            written = str(getattr(result, attr, "") or "")
            if written and not holdout.get(field):
                holdout[field] = written
                changed = True

    if not changed:
        return
    run.holdout = holdout
    await session.commit()


def _heldout_test_sets(run: Run) -> list[dict[str, Any]]:
    """Return mutable copies of this Run's named held-out Test contracts."""
    holdout = dict(run.holdout or {})
    raw = list(holdout.get("test_sets") or [])
    if raw:
        return [dict(item) for item in raw]
    test_set = str(holdout.get("test_set") or "")
    if not test_set:
        return []
    return [{
        "name": "test",
        "test_set": test_set,
        "inference_query": str(
            holdout.get("validation_inference_query")
            or run.task_objective
            or "Produce the requested prediction for this input."
        ),
        "sample_submission": str(holdout.get("test_sample_submission") or ""),
        "metric": run.metric,
        "metric_direction": run.metric_direction,
        "answer_fields": list(_answer_fields(holdout, "test")),
        "metric_type": str(holdout.get("test_metric_type") or "builtin"),
        "evaluation_script": str(holdout.get("test_evaluation_script") or ""),
        "evaluator_sha256": str(holdout.get("test_evaluator_sha256") or ""),
        "public": str(holdout.get("test_public") or ""),
        "inference_data_profile": "",
    }]


def _heldout_test_set(run: Run, name: str) -> dict[str, Any] | None:
    return next(
        (item for item in _heldout_test_sets(run) if item["name"] == name),
        None,
    )


async def _spawn_holdout_data(
    session: AsyncSession, run: Run, source: Ticket, *, test_set_name: str = "",
    enqueue: bool = True,
) -> Ticket | None:
    """Create answer-stripping tickets for missing members of the Test suite."""
    from zevo.engine.run.wakeup import queue_wakeup

    suite = _heldout_test_sets(run)
    selected = [
        item for item in suite
        if not test_set_name or item["name"] == test_set_name
    ]
    existing = (await session.execute(
        select(Ticket).where(
            Ticket.run_id == run.id,
            Ticket.lane == "held_out_test",
            Ticket.agent_id == "data",
        )
    )).scalars().all()
    created: list[Ticket] = []
    for item in selected:
        if item.get("public"):
            continue
        active = next((
            ticket for ticket in existing
            if (ticket.payload or {}).get("test_set_name") == item["name"]
            and ticket.status not in ("failed", "cancelled", "skipped")
        ), None)
        if active is not None:
            created.append(active)
            continue
        tid = f"holdout-data-{run.id[:8]}-{len(existing) + len(created) + 1:03d}"
        payload = validate_stored_payload(
            agent_id="data", input_format="typed", payload={
                "operation": "prepare_holdout_data",
                "test_set_name": item["name"],
                "dataset_source": "", "dataset": "",
                "dataset_split": "", "dataset_config": "",
                "data_query": "", "training_method": "",
                "scoring_set": item["test_set"],
                "answer_fields": list(item["answer_fields"]),
                "metric_type": str(item.get("metric_type") or "builtin"),
                "metric": item["metric"],
                "evaluation_script": str(item.get("evaluation_script") or ""),
                "evaluator_sha256": str(item.get("evaluator_sha256") or ""),
                "sample_submission": item["sample_submission"],
                "configuration_suggestions": {}, "configuration_pins": {},
            },
        )
        ticket = Ticket(
            id=tid, run_id=run.id, agent_id="data", status="queued",
            input_format="typed", lane="held_out_test",
            iteration=int(source.iteration or 0), payload=payload,
            customization={}, inputs={}, summary="",
        )
        session.add(ticket)
        created.append(ticket)
    if not created:
        return None
    await session.commit()
    if enqueue:
        for ticket in created:
            if ticket.status == "queued":
                await queue_wakeup(
                    session, agent_id="data", ticket_id=ticket.id,
                    source="handoff",
                    reason=f"preparing held-out Test set {ticket.payload['test_set_name']}",
                )
    return created[0]


async def _spawn_holdout_infer(
    session: AsyncSession, run: Run, source: Ticket, *, test_set_name: str = "",
    enqueue: bool = True,
) -> Ticket | None:
    """Measure one candidate independently on every named Test contract."""
    from zevo.engine.run.wakeup import queue_wakeup

    suite = _heldout_test_sets(run)
    selected = [
        item for item in suite
        if not test_set_name or item["name"] == test_set_name
    ]
    iteration = int(source.iteration or 0)
    already = (await session.execute(
        select(Ticket).where(
            Ticket.run_id == run.id,
            Ticket.lane == "held_out_test",
            Ticket.agent_id == "inference",
        )
    )).scalars().all()
    created: list[Ticket] = []
    for item in selected:
        if not item.get("public"):
            await _spawn_holdout_data(
                session, run, source, test_set_name=item["name"], enqueue=enqueue,
            )
            continue
        duplicate = next((
            ticket for ticket in already
            if int(ticket.iteration or 0) == iteration
            and (ticket.payload or {}).get("model_source")
                == (source.payload or {}).get("model_source")
            and (ticket.payload or {}).get("base_model")
                == (source.payload or {}).get("base_model")
            and (ticket.payload or {}).get("test_set_name") == item["name"]
        ), None)
        if duplicate is not None:
            created.append(duplicate)
            continue

        payload = dict(source.payload or {})
        configuration_pins = dict(payload.get("configuration_pins") or {})
        inference_mapping = dict(configuration_pins.get("inference_config") or {})
        inference_mapping["inference_query"] = item["inference_query"]
        configuration_pins["inference_config"] = inference_mapping
        payload.update({
            "test_set_name": item["name"],
            "scoring_set": item["public"],
            "sample_submission": item["sample_submission"],
            "configuration_suggestions": {},
            "configuration_pins": configuration_pins,
        })
        payload = validate_stored_payload(
            agent_id="inference", input_format="typed", payload=payload,
        )
        inputs = dict(source.inputs or {})
        inputs.pop("inference_config", None)
        inputs.pop("predict_script", None)
        data_rows = (await session.execute(
            select(Ticket).where(
                Ticket.run_id == run.id,
                Ticket.lane == "held_out_test",
                Ticket.agent_id == "data",
                Ticket.status.in_(("succeeded", "degraded")),
            ).order_by(Ticket.created_at.desc())
        )).scalars().all()
        data_ticket = next((
            ticket for ticket in data_rows
            if (ticket.payload or {}).get("test_set_name") == item["name"]
        ), None)
        if data_ticket is None:
            raise ValueError(f"Test set {item['name']!r} has no prepared Data ticket")
        if item.get("inference_data_profile"):
            inputs["inference_data_profile"] = _artifact_binding(
                "inference_data_profile", data_ticket,
            )

        if payload.get("model_source") == "checkpoint":
            baseline = next((
                ticket for ticket in reversed(already)
                if (ticket.payload or {}).get("model_source") == "base_model"
                and (ticket.payload or {}).get("base_model") == payload.get("base_model")
                and (ticket.payload or {}).get("test_set_name") == item["name"]
                and ticket.status in ("succeeded", "degraded")
            ), None)
            configuration_source = baseline or source
            inputs["inference_config"] = _artifact_binding(
                "inference_config", configuration_source,
            )
            inputs["predict_script"] = _artifact_binding(
                "script", configuration_source,
            )

        tid = f"holdout-infer-{run.id[:8]}-{len(already) + len(created) + 1:03d}"
        ticket = Ticket(
            id=tid, run_id=run.id, agent_id="inference", status="queued",
            input_format="typed", lane="held_out_test", iteration=iteration,
            payload=payload, customization=dict(source.customization or {}),
            inputs=inputs, summary="",
        )
        session.add(ticket)
        created.append(ticket)
    if not created:
        return None
    await session.commit()
    if enqueue:
        for ticket in created:
            if ticket.status == "queued":
                await queue_wakeup(
                    session, agent_id="inference", ticket_id=ticket.id,
                    source="handoff",
                    reason=(
                        f"held-out {ticket.payload['test_set_name']} measurement "
                        f"mirroring {source.id}"
                    ),
                )
    return created[0]


async def _spawn_holdout_eval(
    session: AsyncSession, run: Run, source: Ticket,
) -> None:
    """Score held-out predictions with the Task's custom or built-in metric."""
    from zevo.engine.run.wakeup import queue_wakeup

    test_set_name = str((source.payload or {}).get("test_set_name") or "")
    item = _heldout_test_set(run, test_set_name)
    if item is None:
        return

    iteration = int(source.iteration or 0)
    n = (await session.execute(
        select(func.count()).select_from(Ticket).where(
            Ticket.run_id == run.id,
            Ticket.lane == "held_out_test",
            Ticket.agent_id == "evaluation",
        )
    )).scalar_one()

    tid = f"holdout-eval-{run.id[:8]}-{int(n) + 1:03d}"
    payload = validate_stored_payload(
        agent_id="evaluation",
        input_format="typed",
        payload={
            "test_set_name": test_set_name,
            "metric": item["metric"], "evaluation_config": {},
            "scoring_set": item["test_set"],
            "evaluation_script": str(item.get("evaluation_script") or ""),
            "evaluator_sha256": str(item.get("evaluator_sha256") or ""),
            "answer_fields": list(item["answer_fields"]),
            "sample_submission": item["sample_submission"],
        },
    )
    session.add(Ticket(
        id=tid, run_id=run.id, agent_id="evaluation",
        status="queued", input_format="typed", lane="held_out_test",
        iteration=iteration,
        payload=payload,
        customization={},
        inputs={"predictions": {"source_ticket_id": source.id,
                                "artifact_role": "predictions",
                                "work_product_id": "", "path": ""}},
        summary="",
    ))
    await session.commit()
    await queue_wakeup(
        session, agent_id="evaluation", ticket_id=tid,
        source="handoff",
        reason=f"scoring held-out {test_set_name} predictions from {source.id}",
    )


async def _record_holdout_score(
    session: AsyncSession, run: Run, ticket: Ticket, score: float,
) -> None:
    """File one Test member, then publish the suite average once complete.

    `champion_test_score` deliberately tracks the iteration that wins on
    VALIDATION rather than the best test score seen. Reporting the maximum over
    test would be selecting on the test set one level up — the same mistake as
    letting the orchestrator see it, just made by the harness instead.
    """
    from zevo.db import ScoreEvent

    pred_binding = (ticket.inputs or {}).get("predictions") or {}
    infer_id = str(pred_binding.get("source_ticket_id") or "")
    heldout_infer = (await session.execute(
        select(Ticket).where(Ticket.id == infer_id)
    )).scalar_one_or_none() if infer_id else None
    iteration = int(heldout_infer.iteration if heldout_infer else ticket.iteration or 0)
    infer_payload = dict(heldout_infer.payload or {}) if heldout_infer else {}
    source = "baseline" if infer_payload.get("model_source") == "base_model" else "trained"
    base_model = str(infer_payload.get("base_model") or "")
    test_set_name = str((ticket.payload or {}).get("test_set_name") or "test")
    suite = _heldout_test_sets(run)
    suite_names = [str(item["name"]) for item in suite]
    if test_set_name not in suite_names:
        raise ValueError(f"Unknown held-out Test set {test_set_name!r}")

    # Evaluation tickets finish independently. Keep their private component
    # scores in the holdout snapshot and expose one factual Test point only
    # after the entire suite for this candidate has arrived.
    holdout = dict(run.holdout or {})
    all_results = dict(holdout.get("suite_results") or {})
    result_key = f"{source}|{iteration}|{base_model}"
    candidate_results = dict(all_results.get(result_key) or {})
    suite_item = next(item for item in suite if item["name"] == test_set_name)
    candidate_results[test_set_name] = {
        "score": float(score),
        "metric": str((ticket.payload or {}).get("metric") or ""),
        "metric_direction": str(suite_item.get("metric_direction") or "max"),
        "evaluation_ticket_id": ticket.id,
    }
    all_results[result_key] = candidate_results
    holdout["suite_results"] = all_results
    recorded = list(holdout.get("suite_recorded") or [])
    run.holdout = holdout
    if result_key in recorded:
        await session.commit()
        return
    if any(name not in candidate_results for name in suite_names):
        await session.commit()
        return

    component_scores = {
        name: float(candidate_results[name]["score"])
        for name in suite_names
    }
    component_metrics = {
        name: str(candidate_results[name]["metric"])
        for name in suite_names
    }
    component_directions = {
        name: str(candidate_results[name]["metric_direction"])
        for name in suite_names
    }
    score = sum(component_scores.values()) / len(component_scores)
    recorded.append(result_key)
    holdout["suite_recorded"] = recorded
    run.holdout = holdout
    session.add(ScoreEvent(
        run_id=run.id, iteration=iteration, split="test", source=source,
        score=float(score), metric_name=run.metric,
        extras={
            "aggregation": "unweighted_mean",
            "test_sets": {
                name: {
                    "score": component_scores[name],
                    "metric": candidate_results[name]["metric"],
                    "metric_direction": component_directions[name],
                }
                for name in suite_names
            },
        },
        notes=f"held-out Test suite completed by {ticket.id}",
    ))
    # Stamp it onto the matching history entry so the run's journal can show
    # the pair side by side. The orchestrator's copy of history is stripped of
    # this key on its way out (see _agent_history).
    # Copied, not mutated in place. `run.history` is a plain JSON column, so
    # SQLAlchemy decides whether to emit an UPDATE by comparing the new value
    # against the loaded one — and editing the loaded dicts changes both sides
    # at once, so the comparison finds them equal and the write is skipped.
    history = [dict(e) if isinstance(e, dict) else e for e in (run.history or [])]
    best_val: float | None = None
    best_test: float | None = None
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if (
            int(entry.get("iteration", -1)) == iteration
            and entry.get("source") == source
        ):
            entry["test_score"] = float(score)
            entry["test_scores"] = dict(component_scores)
            entry["test_metrics"] = dict(component_metrics)
            entry["test_metric_directions"] = dict(component_directions)
        # The baseline gets its test score stamped but does not compete for
        # champion — same rule as run_metrics.baseline_and_best_test, or the
        # column and the boards it sorts disagree the moment training never
        # beats the untuned probe.
        if entry.get("source") == "baseline":
            continue
        raw_val = entry.get("score")
        raw_test = entry.get("test_score")
        if not isinstance(raw_val, (int, float)) or isinstance(raw_val, bool):
            continue
        if not isinstance(raw_test, (int, float)) or isinstance(raw_test, bool):
            continue
        val, test = float(raw_val), float(raw_test)
        if is_better(val, best_val, run.validation_metric_direction):
            best_val, best_test = val, test
    run.history = history
    _sync_best_validation_score(run)
    if best_test is not None:
        run.champion_test_score = best_test
    elif run.champion_test_score is None:
        # No validation score has been paired with a test score yet — the
        # baseline, usually. Show the one number we have rather than nothing.
        run.champion_test_score = float(score)
    await session.commit()


async def _record_validation_score(
    session: AsyncSession, run: Run, ticket: Ticket, score: float,
) -> None:
    """File the score from an ordinary eval ticket as this run's validation point.

    The engine records both series. The two curves are comparable only if they
    are written from completed Evaluation outputs through one path, and a loop
    that could file its own scores could choose the ones it likes.
    """
    from zevo.db import ScoreEvent

    # Whether this was the untuned probe is a fact about the inference ticket
    # that produced the predictions, not about the eval ticket.
    pred_binding = (ticket.inputs or {}).get("predictions") or {}
    predictions_ticket_id = str(pred_binding.get("source_ticket_id") or "")
    source_infer = None
    if predictions_ticket_id:
        source_infer = (await session.execute(
            select(Ticket).where(Ticket.id == predictions_ticket_id)
        )).scalar_one_or_none()
    iteration = int(source_infer.iteration if source_infer else ticket.iteration or 0)
    infer_payload = dict(source_infer.payload or {}) if source_infer else {}
    source = "baseline" if infer_payload.get("model_source") == "base_model" else "trained"
    session.add(ScoreEvent(
        run_id=run.id, iteration=iteration, split="validation",
        source=source,
        score=float(score), metric_name=run.validation_metric,
        notes=f"validation score from {ticket.id}",
    ))

    # The chart and journal are factual views of measurements the engine has
    # already recorded. Build their row here instead of relying on the
    # supervisor to repeat the score in a separate PATCH.
    base_model = str((source_infer.payload or {}).get("base_model") or "") if source_infer else ""
    training_method = ""
    method_ids: list[str] = []
    training_diagnostics: dict[str, Any] | None = None
    generation_termination: dict[str, Any] | None = None
    if source_infer is not None:
        inference_product = (await session.execute(
            select(WorkProduct).where(
                WorkProduct.ticket_id == source_infer.id,
                WorkProduct.role == "predictions",
            ).order_by(WorkProduct.created_at.desc()).limit(1)
        )).scalar_one_or_none()
        raw_termination = (
            (inference_product.meta or {}).get("generation_termination")
            if inference_product is not None
            else None
        )
        if isinstance(raw_termination, dict):
            generation_termination = dict(raw_termination)
    if source_infer is not None and source == "trained":
        checkpoint = (source_infer.inputs or {}).get("checkpoint") or {}
        train_id = str(checkpoint.get("source_ticket_id") or "")
        train_ticket = None
        if train_id:
            train_ticket = (await session.execute(
                select(Ticket).where(Ticket.id == train_id)
            )).scalar_one_or_none()
        if train_ticket is not None:
            checkpoint_products = (await session.execute(
                select(WorkProduct).where(
                    WorkProduct.ticket_id == train_ticket.id,
                    WorkProduct.role == "checkpoint",
                ).order_by(WorkProduct.created_at.desc())
            )).scalars().all()
            selected_product_id = str(checkpoint.get("work_product_id") or "")
            checkpoint_product = next((
                product for product in checkpoint_products
                if selected_product_id and product.id == selected_product_id
            ), None)
            if checkpoint_product is None:
                checkpoint_product = next((
                    product for product in checkpoint_products
                    if (product.meta or {}).get("checkpoint_kind") == "final"
                ), checkpoint_products[0] if checkpoint_products else None)
            checkpoint_meta = (
                dict(checkpoint_product.meta or {}) if checkpoint_product else {}
            )
            raw_diagnostics = checkpoint_meta.get("training_diagnostics")
            if isinstance(raw_diagnostics, dict):
                training_diagnostics = dict(raw_diagnostics)
            training_method = str(
                checkpoint_meta.get("training_method")
                or (train_ticket.payload or {}).get("training_method_pin")
                or ""
            )
            training_data = (train_ticket.inputs or {}).get("training_dataset") or {}
            data_id = str(training_data.get("source_ticket_id") or "")
            if data_id:
                data_ticket = (await session.execute(
                    select(Ticket).where(Ticket.id == data_id)
                )).scalar_one_or_none()
                if data_ticket is not None:
                    # Journal the validated recipe metadata attached to the
                    # actual dataset artifact, never a second Result echo.
                    data_product = (await session.execute(
                        select(WorkProduct).where(
                            WorkProduct.ticket_id == data_id,
                            WorkProduct.role == "training_dataset",
                        ).order_by(WorkProduct.created_at.desc()).limit(1)
                    )).scalar_one_or_none()
                    if data_product is not None:
                        method_ids = list(
                            (data_product.meta or {}).get("method_ids") or []
                        )

    test_score = (await session.execute(
        select(ScoreEvent.score).where(
            ScoreEvent.run_id == run.id,
            ScoreEvent.iteration == iteration,
            ScoreEvent.split == "test",
            ScoreEvent.source == source,
        ).order_by(ScoreEvent.ts.desc()).limit(1)
    )).scalar_one_or_none()
    run.history = upsert_history_fact(
        run.history,
        iteration=iteration,
        source=source,
        score=float(score),
        base_model=base_model,
        training_method=training_method,
        method_ids=method_ids,
        test_score=float(test_score) if test_score is not None else None,
        training_diagnostics=training_diagnostics,
        generation_termination=generation_termination,
    )
    _sync_best_validation_score(run)
    if source == "baseline" and base_model:
        lineages = dict(run.model_lineages or {})
        lineage = dict(lineages.get(base_model) or {"base_model": base_model})
        lineage["baseline_validation_score"] = float(score)
        lineages[base_model] = lineage
        run.model_lineages = lineages
    elif source == "trained" and iteration > 0:
        # A trained iteration is complete when its authoritative Validation
        # measurement lands. Registry is a separate once-per-Run finalization
        # stage and therefore cannot be the loop counter.
        run.iterations_completed = max(
            int(run.iterations_completed or 0), iteration,
        )
    _baseline_test, champion_test = baseline_and_best_test(
        run.history, run.validation_metric_direction
    )
    if champion_test is not None:
        run.champion_test_score = champion_test
    await session.commit()


async def _advance_measurements(
    session: AsyncSession, tk: Ticket, score: float | None,
    result: BaseModel | None = None,
) -> None:
    """Record what this ticket measured, and move the held-out lane one hop."""
    from zevo.db import Run as _Run

    # Validation Evaluation, held-out Evaluation, and the Orchestrator Journal
    # can finish almost together. Run.history is one JSON value, so ordinary
    # read/modify/write transactions can overwrite each other and leave a
    # ScoreEvent whose Journal row vanished. Serialize those writers on the Run
    # row; each one then merges against the latest committed history.
    run = (await session.execute(
        select(_Run).where(_Run.id == tk.run_id).with_for_update()
    )).scalar_one_or_none()
    if run is None:
        return
    await session.refresh(run)
    if run.status in TERMINAL_RUN_STATUSES:
        return
    if tk.status not in ("succeeded", "degraded"):
        return  # a failed measurement leaves a gap in the curve, not a wrong point

    is_held_out_test = tk.lane == "held_out_test"
    if tk.agent_id == "data":
        # Both lanes start here: the data agent is what turns a scoring set into
        # something inference can be handed.
        if result is not None:
            await _record_prepared_scoring_data(session, run, tk, result)
        if is_held_out_test:
            # The copy job finished after the inference it was meant to mirror
            # — pick that one back up, or the lane stalls until the next
            # iteration produces another.
            latest = (await session.execute(
                select(Ticket).where(
                    Ticket.run_id == run.id,
                    Ticket.agent_id == "inference",
                    Ticket.lane == "optimization",
                    Ticket.iteration == int(tk.iteration or 0),
                    Ticket.status.in_(("succeeded", "degraded")),
                ).order_by(Ticket.created_at.desc()).limit(1)
            )).scalar_one_or_none()
            if latest is not None:
                await _spawn_holdout_infer(
                    session,
                    run,
                    latest,
                    test_set_name=str((tk.payload or {}).get("test_set_name") or ""),
                )
    elif score is not None and not is_held_out_test and tk.agent_id == "evaluation":
        await _record_validation_score(session, run, tk, score)
        # Strict order: Validation Evaluation finishes first. Only then does
        # the engine start the private held-out mirror for the exact Inference
        # that produced these predictions.
        pred = dict((tk.inputs or {}).get("predictions") or {})
        source = await session.get(Ticket, str(pred.get("source_ticket_id") or ""))
        if source is not None and source.agent_id == "inference":
            await _spawn_holdout_infer(session, run, source)
    elif is_held_out_test and tk.agent_id == "inference":
        await _spawn_holdout_eval(session, run, tk)
    elif score is not None and is_held_out_test and tk.agent_id == "evaluation":
        await _record_holdout_score(session, run, tk, score)


def _agent_history(history: list) -> list:
    """The run's history with the held-out numbers taken back out.

    The orchestrator reads its own past iterations to decide the next one. It
    may read what it scored on validation; it may not read what those choices
    turned out to be worth on the test set, or the loop would be selecting on
    the test set through its own journal.
    """
    out = []
    for entry in history or []:
        if isinstance(entry, dict):
            out.append({
                k: v for k, v in entry.items()
                if k not in (
                    "test_score", "test_scores", "test_metrics",
                    "test_metric_directions", "notes",
                )
            })
        else:
            out.append(entry)
    return out


async def _holdout_settled(session: AsyncSession, run_id: str) -> bool:
    """Has the held-out lane finished whatever it had in flight?

    An iteration is not over when the validation number lands — it is over when
    BOTH numbers do. Letting the loop start the next round on the validation
    score alone leaves the test measurement running against a model the run has
    already moved past, and the two series drift apart: iteration 3's test point
    arrives while iteration 4 is training, or not at all if the run ends first.

    Failed and cancelled count as settled. A lane that cannot finish must not
    hold the loop forever — a missing test point is a gap in one series, an
    unwakeable supervisor is a dead run.
    """
    pending = (await session.execute(
        select(func.count()).select_from(Ticket).where(
            Ticket.run_id == run_id,
            Ticket.lane == "held_out_test",
            Ticket.status.in_(("queued", "running", "repairing", "awaiting_input", "waiting_external")),
        )
    )).scalar_one()
    return int(pending or 0) == 0


async def _release_deferred_wake(session: AsyncSession, run_id: str) -> None:
    """Issue the supervisor wake the held-out lane was holding, if any.

    The flag lives on the evaluation that deferred, so a lane settling with
    nothing waiting does nothing at all. That matters: the mirror must never
    manufacture a wake of its own — only hand back the one it borrowed. The
    held-out lane also runs during iteration 0, where the loop's own evaluation
    has usually already woken the supervisor on its own, and a second wake there
    would have the orchestrator plan the same round twice.
    """
    rows = (await session.execute(
        select(Ticket).where(
            Ticket.run_id == run_id,
            Ticket.lane == "optimization",
            Ticket.agent_id == "evaluation",
        ).order_by(Ticket.created_at.desc())
    )).scalars().all()
    held = next((t for t in rows if t.supervisor_wake_deferred), None)
    if held is None:
        return
    held.supervisor_wake_deferred = False
    await session.commit()
    await _maybe_wake_supervisor(session, held)


async def _release_emitted_child(
    session: AsyncSession,
    *,
    supervisor: Ticket,
    output: BaseModel | None,
    heartbeat_id: str,
) -> None:
    """Wake an emitted child only after its supervisor heartbeat is complete."""
    if (
        supervisor.status not in ("succeeded", "degraded")
        or not isinstance(output, SupervisorAction)
        or output.action != "emit_ticket"
    ):
        return

    owning_run = await session.get(Run, supervisor.run_id)
    if owning_run is None:
        return
    await session.refresh(owning_run)
    if owning_run.status in TERMINAL_RUN_STATUSES:
        return

    child = await session.get(Ticket, output.child_ticket_id)
    if (
        child is None
        or child.run_id != supervisor.run_id
        or child.agent_id == "orchestrator"
    ):
        session.add(TicketNotice(
            ticket_id=supervisor.id,
            code="supervisor.child_handoff_invalid",
            severity="error",
            body=(
                "The validated SupervisorAction did not identify a specialist "
                f"Ticket in this Run: `{output.child_ticket_id}`."
            ),
        ))
        await session.commit()
        return

    wait_notice = (await session.execute(
        select(TicketNotice).where(
            TicketNotice.ticket_id == child.id,
            TicketNotice.code == "ticket.awaiting_supervisor_completion",
        ).order_by(TicketNotice.created_at.desc()).limit(1)
    )).scalar_one_or_none()
    if wait_notice is not None:
        # queue_wakeup commits this notice mutation and the wakeup together.
        # The scheduler's orphan recovery therefore sees either "still
        # waiting" or a real wakeup, never an unprotected gap between them.
        wait_notice.code = "ticket.supervisor_handoff_released"
        wait_notice.body = (
            f"Orchestrator heartbeat `{heartbeat_id}` completed and released "
            "this Ticket for execution."
        )

    # A child may already be terminal after manual intervention, or running
    # when a mixed-version deployment is being drained.  Never manufacture a
    # second activation in either case.  queue_wakeup itself coalesces any
    # already-queued signal for the normal case.
    if child.status != "queued":
        if wait_notice is not None:
            await session.commit()
        return

    from zevo.engine.run.wakeup import queue_wakeup
    await queue_wakeup(
        session,
        agent_id=child.agent_id,
        ticket_id=child.id,
        source="handoff",
        trigger_detail=f"supervisor_heartbeat:{heartbeat_id}",
        reason=(
            f"orchestrator heartbeat {heartbeat_id} completed and emitted "
            f"child ticket {child.id}"
        ),
    )


async def _maybe_wake_supervisor(session: AsyncSession, child: Ticket) -> None:
    """Wake the supervisor only when the completed DAG frontier is settled."""
    from zevo.db import Run
    from zevo.engine.run.wakeup import queue_wakeup
    run = (await session.execute(select(Run).where(Run.id == child.run_id))).scalar_one_or_none()
    if run is None or not run.supervisor_ticket_id:
        return
    # Re-read before trusting the status. This session loaded `run` when the
    # heartbeat STARTED, and a long stage runs for many minutes — a cancel in
    # the meantime is invisible to the in-memory copy. That is how a cancelled
    # run kept going: the stage was killed, the guard below read a stale
    # "running", and the supervisor was woken to plan the next step.
    await session.refresh(run)
    if run.status in TERMINAL_RUN_STATUSES:
        return  # run is terminal (incl. user-cancelled) — don't keep poking
    # Same for the child: a ticket the user cancelled is not a completion to
    # react to, whatever exit code killing it produced.
    if child.status == "cancelled":
        return
    # An auto Run has no scoring contract until scoping settles it. Nothing the
    # Orchestrator plans can be stamped before then (Data needs the settled
    # Validation set), so it is never woken early; settlement creates and
    # queues the supervisor itself (zevo.engine.run.scoping).
    if not bool(getattr(run, "scoring_settled", True)):
        return
    # The orchestrator is ONE agent with ONE ticket for the whole run. Every
    # supervisor wake — across every Zevo model-improvement loop — re-runs that single
    # `orchestrate-<run>-001` ticket (a fresh heartbeat each child completion),
    # so all its heartbeats and emitted children live under one work order.
    # `iteration` = the loop we're in. It bumps when Validation Evaluation
    # records a trained candidate. Final Registry runs once after the loop and
    # is not an iteration counter.
    iteration = _current_iteration(run)
    sup_id = f"orchestrate-{run.id[:8]}-001"
    trigger = {
        "type": "ticket_completed",
        "ticket_id": child.id,
        "agent_id": child.agent_id,
        "status": child.status,
    }
    runtime = {
        "gpu_provider": run.gpu_provider or "instance",
        "num_gpus": _num_gpus(run),
        "generation_backend": run.generation_backend,
        "max_queue_wait_hours": float(run.max_queue_wait_hours or 0.0),
    }
    existing = (await session.execute(select(Ticket).where(Ticket.id == sup_id))).scalar_one_or_none()
    if existing is None:
        # Defensive repair: recreate the stable supervisor if it is absent.
        # The rebuilt payload goes through the same stored-payload contract as
        # an API-created one — an engine-written supervisor ticket that a later
        # rerun would 422 on is worse than failing loudly here.
        from zevo.contracts.tickets import validate_stored_payload
        from zevo.engine.cost.budget import snapshot_for_run

        prev = (await session.execute(
            select(Ticket).where(Ticket.id == run.supervisor_ticket_id)
        )).scalar_one_or_none()
        prev_payload = prev.payload if prev is not None and isinstance(prev.payload, dict) else {}
        user_request = prev_payload.get("user_request") or {}
        if not user_request:
            # The original request is gone with the old ticket; rebuild the
            # minimum valid one from what the Run itself records. '' means
            # "not specified" throughout UserRequest. The test_* fields stay
            # BLANK on purpose — run creation blanks them before the
            # orchestrator ever sees the request (see _settle_splits), and
            # that blanking preserves the Agent API boundary. A repair path
            # must not hand the loop the private path through its payload.
            user_request = {
                "task_objective": run.task_objective,
                "metric": run.validation_metric,
                "metric_direction": run.validation_metric_direction,
                "validation_metric_type": "builtin",
                "validation_metric": run.validation_metric,
                "validation_metric_direction": run.validation_metric_direction,
                "validation_evaluation_script": "",
                "validation_evaluator_sha256": "",
                "training_method": "", "dataset": "", "data_query": "",
                "base_model": "",
                "test_set": "",
                "test_sample_submission": "",
                "metric_type": "builtin",
                "evaluation_script": "",
                "evaluator_sha256": "",
                "constraints": [],
            }
        snap = await snapshot_for_run(session, run.id)
        budget = {
            "max_cost_usd": max(0.0, snap.max_cost_usd),
            "spent_usd": max(0.0, snap.spent_usd),
            "remaining_usd": max(0.0, snap.remaining_usd),
            "projected_next_iteration_usd": max(
                0.0, snap.projected_next_iteration_usd,
            ),
            "can_afford_next_iteration": snap.can_afford_next_iteration,
            "projection_basis": snap.projection_basis,
            "max_runtime_hours": max(0.0, snap.max_runtime_hours),
            "max_queue_wait_hours": max(0.0, snap.max_queue_wait_hours),
            "queue_wait_hours": max(0.0, snap.queue_wait_hours),
            "elapsed_runtime_hours": max(0.0, snap.elapsed_runtime_hours),
            "remaining_runtime_hours": max(0.0, snap.remaining_runtime_hours),
            "over_time_limit": snap.over_time_limit,
            "projected_next_iteration_hours": max(
                0.0, snap.projected_next_iteration_hours,
            ),
            "can_finish_next_iteration_in_time": snap.can_finish_next_iteration_in_time,
            "runtime_projection_basis": snap.runtime_projection_basis,
        }
        payload = validate_stored_payload(
            agent_id="orchestrator", input_format="typed",
            payload={
                "task_objective": run.task_objective,
                "agent_objective": run.agent_objective,
                "user_request": user_request,
                "trigger": trigger,
                # Snapshot the perf log of past iterations at this iteration's
                # start, minus the held-out numbers it must not tune on.
                "history": _agent_history(run.history or []),
                # Carry the Run mode and per-Agent customizations on every wake.
                "mode": run.mode,
                "customizations": run.customizations or {},
                "runtime": runtime,
                "validation_rows": int((run.holdout or {}).get("validation_rows", 0) or 0),
                "budget": budget,
                "dataset_profile": prev_payload.get("dataset_profile", {}) or {},
            },
        )
        session.add(Ticket(
            id=sup_id, run_id=run.id, agent_id="orchestrator",
            status="queued", input_format="typed", lane="optimization",
            iteration=iteration,
            payload=payload,
            customization={}, inputs={},
            summary=(
                f"supervisor · {'baseline (iteration 0)' if iteration == 0 else f'iteration {iteration}'}"
                f" (after {child.id})"
            ),
        ))
    else:
        # Re-run the single supervisor ticket (fresh heartbeat). Refresh the
        # history snapshot so this wake sees current run state. The loop number
        # stays on the Ticket envelope.
        p = dict(existing.payload or {})
        p["trigger"] = trigger
        p["history"] = _agent_history(run.history or [])
        p["task_objective"] = run.task_objective
        p["agent_objective"] = run.agent_objective
        p["mode"] = run.mode
        p["customizations"] = run.customizations or {}
        p["runtime"] = runtime
        p["validation_rows"] = int((run.holdout or {}).get("validation_rows", 0) or 0)
        from zevo.engine.cost.budget import snapshot_for_run
        snap = await snapshot_for_run(session, run.id)
        p["budget"] = {
            "max_cost_usd": max(0.0, snap.max_cost_usd),
            "spent_usd": max(0.0, snap.spent_usd),
            "remaining_usd": max(0.0, snap.remaining_usd),
            "projected_next_iteration_usd": max(
                0.0, snap.projected_next_iteration_usd,
            ),
            "can_afford_next_iteration": snap.can_afford_next_iteration,
            "projection_basis": snap.projection_basis,
            "max_runtime_hours": max(0.0, snap.max_runtime_hours),
            "max_queue_wait_hours": max(0.0, snap.max_queue_wait_hours),
            "queue_wait_hours": max(0.0, snap.queue_wait_hours),
            "elapsed_runtime_hours": max(0.0, snap.elapsed_runtime_hours),
            "remaining_runtime_hours": max(0.0, snap.remaining_runtime_hours),
            "over_time_limit": snap.over_time_limit,
            "projected_next_iteration_hours": max(
                0.0, snap.projected_next_iteration_hours,
            ),
            "can_finish_next_iteration_in_time": snap.can_finish_next_iteration_in_time,
            "runtime_projection_basis": snap.runtime_projection_basis,
        }
        existing.payload = p
        existing.status = "queued"
        existing.iteration = iteration
        existing.summary = (
            f"supervisor · {'baseline (iteration 0)' if iteration == 0 else f'iteration {iteration}'}"
            f" (re-woken after {child.id})"
        )
    run.supervisor_ticket_id = sup_id
    await session.commit()
    await queue_wakeup(
        session, agent_id="orchestrator", ticket_id=sup_id,
        source="handoff",
        reason=f"child ticket {child.id} finished ({child.status})",
    )


def _unique_work_products(
    items: list[tuple[str, str, dict]],
) -> list[tuple[str, str, dict]]:
    """Drop repeats of the same (path, role), keeping first-seen order.

    One file may legitimately carry two roles. Since validation isolation the
    Data result names the frozen validation population as both
    `validation_source` and `validation_dataset` (runner: "the frozen raw
    scoring population"); de-duplicating by path alone dropped whichever role
    came second, so no Train Ticket could ever bind `validation_dataset` and
    every run halted at Train 1 (run 77b4ad61).
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, dict]] = []
    for path, role, meta in items:
        if not path or (path, role) in seen:
            continue
        seen.add((path, role))
        out.append((path, role, meta))
    return out


def _role_for(agent_id: str) -> str:
    return {
        "data":            "training_dataset",
        "infrastructure":  "device_info",
        "train":      "checkpoint",
        "inference":       "predictions",
        "evaluation":      "metrics",
        "registry":  "registry_entry",
    }.get(agent_id, "artifact")
