"""
Input/output contracts for the Infrastructure Agent.

Three providers, and the mode follows from `release`:
  - cloud     -- rent a box (Vast.ai / Lambda), wait for SSH-ready, probe CUDA.
                 `release=true` destroys it and stops billing.
  - cluster   -- validate a Slurm login route. Train and Inference submit finite
                 stage jobs; the backend Scheduler, not an LLM Agent, streams
                 job-local lifecycle events and keeps polite polling as backup.
  - instance  -- use a fixed, directly reachable GPU host. There is no Slurm
                 allocation or queue; Zevo leases idle device indices in its DB
                 so concurrent Runs do not select the same cards.

Provision writes `device_info.json` in the resolved output directory; release
writes no device artifact. For cluster this is an access-and-resource-plan
contract, not an allocation: it deliberately contains no Slurm job id, node,
or measured GPU. Train and Inference turn that route into finite `train.sbatch`
or `predict.sbatch` jobs. Instance executes directly over SSH.
The exact device artifact is ``InfrastructureDeviceInfo`` below. The runner
injects that JSON Schema and a validator command into every provision work
order, then independently revalidates the reopened file against Run id, Ticket
id, provider, Run GPU limit, resource plan, measured hardware, route, lifecycle,
and cost facts. Release produces no device artifact.
"""

from __future__ import annotations

import json
import math
import shlex
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from zevo.contracts._base import AgentResult, AgentTaskInput, StrictBody


SLURM_STATUS_FILENAME = ".zevo-slurm-status"
SLURM_STATUS_EVENT_PREFIX = "ZEVO_SLURM_EVENT_V1"


def slurm_lifecycle_prologue(status_path: str) -> str:
    """Exact shell block embedded in every finite Zevo Slurm stage job.

    The append-only file is streamed over SSH by the backend. ``RUNNING`` is a
    direct start event; ``EXITED`` asks the fallback accounting watcher to
    confirm Slurm's authoritative terminal state (including timeout/preemption).
    Monitoring must never change the workload's exit status.
    """
    quoted_path = shlex.quote(status_path)
    return f'''ZEVO_SLURM_STATUS_FILE={quoted_path}
zevo_slurm_event() {{
  printf '{SLURM_STATUS_EVENT_PREFIX}|%s|%s|%s|%s\\n' "${{SLURM_JOB_ID}}" "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ZEVO_SLURM_STATUS_FILE" || true
}}
zevo_slurm_finish() {{
  zevo_slurm_rc=$?
  trap - EXIT
  zevo_slurm_event EXITED "$zevo_slurm_rc"
  exit "$zevo_slurm_rc"
}}
trap zevo_slurm_finish EXIT
zevo_slurm_event RUNNING ""'''


class GpuAllocationCandidate(BaseModel):
    """One fixed GPU host plus a fresh physical-device idle probe.

    ``jobid`` is retained as the storage/API key name, but for instance mode it
    contains the stable SSH connection identity rather than a Slurm job id.
    """

    model_config = ConfigDict(extra="forbid")

    jobid: str = Field(min_length=1)
    node: str = ""
    gpu_count: int = Field(ge=0)
    idle_gpu_indices: list[int] = Field(
        ...,
        description=(
            "Physical indices verified at zero utilization with no foreign "
            "compute process or material memory use immediately before POST."
        ),
    )
    gpu_name: str = ""
    vram_gb: int = Field(0, ge=0)

    @model_validator(mode="after")
    def validate_idle_indices(self) -> "GpuAllocationCandidate":
        if len(self.idle_gpu_indices) != len(set(self.idle_gpu_indices)):
            raise ValueError("idle_gpu_indices must not contain duplicates")
        invalid = [
            index for index in self.idle_gpu_indices
            if index < 0 or index >= self.gpu_count
        ]
        if invalid:
            raise ValueError(
                f"idle_gpu_indices {invalid} outside allocation GPU range "
                f"0..{max(self.gpu_count - 1, 0)}"
            )
        return self


class GpuLeaseRequest(StrictBody):
    """The one accepted POST /api/gpu/leases request body."""

    run_id: str = Field(min_length=1)
    ticket_id: str | None = None
    num_gpus: int = Field(default=0, ge=0)
    min_vram_gb: int = Field(default=0, ge=0)
    allocations: list[GpuAllocationCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_allocations(self) -> "GpuLeaseRequest":
        jobids = [allocation.jobid.strip() for allocation in self.allocations]
        if len(jobids) != len(set(jobids)):
            raise ValueError("allocations must contain unique jobids")
        return self


class GpuLeaseGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobid: str
    node: str
    gpu_indices: list[int]
    gpu_count: int
    visible_devices: str
    reused: bool = False


class GpuLeaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobid: str
    node: str
    gpu_index: int
    run_id: str | None
    gpu_name: str
    acquired_at: datetime | None


class InfraInstanceDTO(BaseModel):
    """Exact response shape for Infrastructure bookkeeping endpoints."""

    model_config = ConfigDict(extra="forbid")
    id: str
    instance_id: str
    provider: str
    status: str
    run_id: str | None
    ticket_id: str | None
    gpu_name: str
    gpu_count: int
    vram_gb: int
    dph: float
    ssh_host: str
    ssh_port: int
    ssh_user: str
    meta: dict[str, Any]
    created_at: str
    ready_at: str | None = None
    released_at: str | None = None
    release_reason: str = ""
    uptime_seconds: int = 0
    estimated_cost_usd: float = 0.0


class CreateInfraInstanceBody(StrictBody):
    """The one accepted POST /infra/instances body."""

    instance_id: str = ""
    provider: Literal["cluster", "cloud", "instance"]
    status: Literal["provisioning", "ready", "failed"] = "provisioning"
    run_id: str | None = None
    ticket_id: str | None = None
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    dph: float = 0.0
    ssh_host: str = ""
    ssh_port: int = 0
    ssh_user: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class PatchInfraInstanceBody(StrictBody):
    """The one accepted PATCH /infra/instances/{row_id} body."""

    instance_id: str | None = None
    status: Literal["provisioning", "ready", "released", "failed"] | None = None
    gpu_name: str | None = None
    gpu_count: int | None = None
    vram_gb: int | None = None
    dph: float | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_user: str | None = None
    meta: dict[str, Any] | None = None
    release_reason: str | None = None


class SlurmStageJobContract(BaseModel):
    """Engine-owned contract for a finite Train or Inference Slurm job.

    Cluster stages render this contract as a local ``.sbatch`` artifact, copy
    it to the verified login route, and submit that file. Other providers keep
    ``enabled=false`` and use direct SSH execution.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    phase: Literal["submit", "collect"] = "submit"
    script_path: str = ""
    job_name: str = ""
    status_path: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    lifecycle_prologue: str = ""
    bookkeeping_row_id: str = ""
    job_id: str = ""
    scheduler_state: str = ""
    scheduler_exit_code: str = ""
    scheduler_reason: str = ""
    num_gpus: int = Field(
        1,
        ge=1,
        description=(
            "Total GPU count selected in the bound Infrastructure resource plan "
            "for this finite Slurm stage, summed across nodes. It renders as "
            "#SBATCH --gpus (or nodes * --gpus-per-node) in the stage script."
        ),
    )
    nodes: int = Field(
        1,
        ge=1,
        description=(
            "Number of nodes for this finite Slurm stage. Default 1 (single "
            "node, unchanged). With nodes>1 the stage script renders "
            "#SBATCH --nodes, --ntasks-per-node, and --gpus-per-node "
            "(num_gpus // nodes) and launches with torchrun/deepspeed "
            "rendezvous. num_gpus must be an exact multiple of nodes."
        ),
    )
    max_queue_wait_hours: float = Field(24, gt=0, le=168)
    infra_instances_endpoint: Literal["/api/infra/instances"] = "/api/infra/instances"
    openapi_endpoint: Literal["/api/openapi.json"] = "/api/openapi.json"
    infra_instance_create_schema: dict[str, Any] = Field(default_factory=dict)
    infra_instance_patch_schema: dict[str, Any] = Field(default_factory=dict)
    infra_instance_response_schema: dict[str, Any] = Field(default_factory=dict)

    @property
    def gpus_per_node(self) -> int:
        """Per-node GPU count rendered as #SBATCH --gpus-per-node."""
        return self.num_gpus // self.nodes

    @model_validator(mode="after")
    def require_enabled_shape(self) -> "SlurmStageJobContract":
        if self.num_gpus % self.nodes != 0:
            raise ValueError(
                "num_gpus must be an exact multiple of nodes so every node "
                "receives the same gpus_per_node"
            )
        if not self.enabled:
            if any((
                self.script_path, self.job_name, self.bookkeeping_row_id,
                self.job_id, self.scheduler_state, self.scheduler_exit_code,
                self.scheduler_reason, self.status_path, self.stdout_path,
                self.stderr_path, self.lifecycle_prologue,
            )) or self.phase != "submit":
                raise ValueError("disabled Slurm stage contract cannot carry job state")
            return self
        if not self.script_path.startswith("/") or not self.script_path.endswith(".sbatch"):
            raise ValueError("enabled Slurm stage contract requires an absolute .sbatch path")
        if not self.job_name.startswith("zevo-"):
            raise ValueError("enabled Slurm stage contract requires a zevo-* job name")
        if not self.status_path.startswith("/") or not self.status_path.endswith(
            "/" + SLURM_STATUS_FILENAME
        ):
            raise ValueError("enabled Slurm stage contract requires an absolute status path")
        ticket_dir = str(PurePosixPath(self.status_path).parent)
        expected_stdout = str(PurePosixPath(ticket_dir) / "slurm-%j.out")
        expected_stderr = str(PurePosixPath(ticket_dir) / "slurm-%j.err")
        if self.stdout_path != expected_stdout or self.stderr_path != expected_stderr:
            raise ValueError(
                "enabled Slurm stage contract requires job-id-specific "
                "slurm-%j.out and slurm-%j.err paths beside the status file"
            )
        if self.lifecycle_prologue != slurm_lifecycle_prologue(self.status_path):
            raise ValueError("enabled Slurm stage contract requires the exact lifecycle prologue")
        if not (
            self.infra_instance_create_schema
            and self.infra_instance_patch_schema
            and self.infra_instance_response_schema
        ):
            raise ValueError("enabled Slurm stage contract requires bookkeeping schemas")
        if self.phase == "submit" and any((
            self.bookkeeping_row_id, self.job_id, self.scheduler_state,
            self.scheduler_exit_code, self.scheduler_reason,
        )):
            raise ValueError("a new Slurm submission cannot carry prior job state")
        if self.phase == "collect" and not (
            self.bookkeeping_row_id and self.job_id and self.scheduler_state
        ):
            raise ValueError("Slurm collect phase requires row, job, and terminal state")
        return self


# ---------- Input ----------


class HostMemoryPlanningContract(BaseModel):
    """Engine-owned safety margin for CPU RAM requested by a stage."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["model_and_stage_sized"] = "model_and_stage_sized"
    headroom_multiplier: Literal[1.5] = 1.5
    additive_headroom_gib: Literal[16] = 16
    round_to_gib: Literal[8] = 8
    inference_floor_gib: Literal[16] = 16
    train_floor_gib: Literal[32] = 32
    request_formula: Literal[
        "round_up_8(max(stage_floor_gib, required_working_set_gib * 1.5, required_working_set_gib + 16))"
    ] = (
        "round_up_8(max(stage_floor_gib, required_working_set_gib * 1.5, "
        "required_working_set_gib + 16))"
    )


def planned_host_ram_gib(
    required_working_set_gib: float,
    purpose: Literal["train", "inference"],
    contract: HostMemoryPlanningContract | None = None,
) -> int:
    """Return the contract-compliant host-RAM request for one stage."""
    if required_working_set_gib <= 0 or not math.isfinite(required_working_set_gib):
        raise ValueError("required_working_set_gib must be a positive finite value")
    memory = contract or HostMemoryPlanningContract()
    floor = (
        memory.train_floor_gib
        if purpose == "train"
        else memory.inference_floor_gib
    )
    unrounded = max(
        float(floor),
        required_working_set_gib * memory.headroom_multiplier,
        required_working_set_gib + memory.additive_headroom_gib,
    )
    return math.ceil(unrounded / memory.round_to_gib) * memory.round_to_gib


class InfrastructureResourcePlan(BaseModel):
    """Infrastructure-owned choices made before acquisition.

    ``provider`` is an immutable Run input. The Run's ``num_gpus`` value is an
    upper bound; this plan records the concrete count Infrastructure selected
    within that bound, together with every other capacity/provider choice.
    """

    model_config = ConfigDict(extra="forbid")

    purpose: Literal["train", "inference"]
    required_working_set_gib: float = Field(
        gt=0,
        allow_inf_nan=False,
        description=(
            "Estimated peak host-memory working set for the immediate stage, "
            "before Zevo's safety margin is added."
        ),
    )
    host_memory_components_gib: dict[str, float] = Field(
        min_length=1,
        description=(
            "Auditable non-negative component estimates supporting the peak "
            "working-set value, such as model, optimizer, checkpoint, data, "
            "and runtime memory."
        ),
    )
    host_memory_formula_gib: int = Field(
        ge=1,
        description="Exact result of Zevo's host-memory planning formula.",
    )
    site_min_ram_gib: int = Field(
        default=0,
        ge=0,
        description="Documented site or partition minimum, or zero when absent.",
    )

    num_gpus: int = Field(
        ge=1,
        description=(
            "Total GPU count selected by Infrastructure for this resource plan, "
            "summed across all nodes. It must be no greater than a positive Run "
            "num_gpus maximum; Run num_gpus=0 means no upper bound. With nodes>1 "
            "it must be an exact multiple of nodes so each node gets the same "
            "gpus_per_node."
        ),
    )
    nodes: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of physical nodes requested. Default 1 (single node, "
            "unchanged). Multi-node plans set nodes>1 for distributed training "
            "(FSDP HYBRID_SHARD / ZeRO-3 across nodes); num_gpus is then the "
            "cluster-wide total and gpus_per_node = num_gpus // nodes."
        ),
    )
    min_vram_gb: int = Field(ge=1)
    min_ram_gb: int = Field(
        ge=1,
        description=(
            "Minimum total host RAM required by the immediate stage. Right-size "
            "this from model/data/runtime evidence; never request a proportional "
            "share of a node's installed RAM merely because GPUs are shared."
        ),
    )
    min_cpus: int = Field(
        ge=1,
        description=(
            "Minimum CPU cores required by the immediate stage. Right-size this "
            "from tokenizer, data-loader, preprocessing, and runtime evidence; "
            "never scale it from the node's total CPU count alone."
        ),
    )
    time_limit_hours: int = Field(ge=0)
    cloud_backend: Literal["", "vastai", "lambda"] = ""
    gpu_type: str = ""
    docker_image: str = ""
    disk_gb: int = Field(ge=0)
    slurm_partition: str = ""
    slurm_account: str = ""
    slurm_qos: str = ""
    rationale: str = Field(min_length=1)

    @property
    def gpus_per_node(self) -> int:
        """Per-node GPU count: num_gpus // nodes (equal across nodes)."""
        return self.num_gpus // self.nodes

    @model_validator(mode="after")
    def require_evidence_based_host_memory(self) -> "InfrastructureResourcePlan":
        if self.num_gpus % self.nodes != 0:
            raise ValueError(
                "num_gpus must be an exact multiple of nodes so every node "
                "receives the same gpus_per_node"
            )
        components: dict[str, float] = {}
        for raw_name, raw_value in self.host_memory_components_gib.items():
            name = raw_name.strip()
            value = float(raw_value)
            if not name:
                raise ValueError("host-memory component names must not be empty")
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "host-memory component values must be finite and non-negative"
                )
            components[name] = value
        if not any(value > 0 for value in components.values()):
            raise ValueError("at least one host-memory component must be positive")
        if self.required_working_set_gib < max(components.values()):
            raise ValueError(
                "required_working_set_gib cannot be smaller than an individual "
                "host-memory component"
            )

        formula_value = planned_host_ram_gib(
            self.required_working_set_gib, self.purpose
        )
        if self.host_memory_formula_gib != formula_value:
            raise ValueError(
                "host_memory_formula_gib must equal the Zevo planning formula "
                f"result ({formula_value} GiB)"
            )
        requested = max(formula_value, self.site_min_ram_gib)
        if self.min_ram_gb != requested:
            raise ValueError(
                "min_ram_gb must equal max(host_memory_formula_gib, "
                f"site_min_ram_gib) ({requested} GiB)"
            )
        self.host_memory_components_gib = components
        return self


class DeviceSshRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    user: str = Field(min_length=1)
    key_path: str = ""
    password_path: str = ""

    @model_validator(mode="after")
    def require_one_authentication_method(self) -> "DeviceSshRoute":
        if bool(self.key_path.strip()) == bool(self.password_path.strip()):
            raise ValueError(
                "ssh route requires exactly one of key_path or password_path"
            )
        return self


class DeviceGpuRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    name: str = Field(min_length=1)
    vram_mb: int = Field(ge=1)


class DeviceGpuInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    has_gpu: Literal[True] = True
    gpu_count: int = Field(ge=1)
    gpu_name: str = Field(min_length=1)
    vram_gb: int = Field(ge=1)
    vram_mb: int = Field(ge=1)
    devices: list[DeviceGpuRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_devices(self) -> "DeviceGpuInfo":
        if len(self.devices) != self.gpu_count:
            raise ValueError("gpu.devices length must equal gpu.gpu_count")
        indices = [device.index for device in self.devices]
        if len(indices) != len(set(indices)):
            raise ValueError("gpu.devices indices must be unique")
        if self.vram_mb != min(device.vram_mb for device in self.devices):
            raise ValueError("gpu.vram_mb must equal the minimum assigned-device VRAM")
        if self.vram_gb != self.vram_mb // 1024:
            raise ValueError("gpu.vram_gb must be floor(gpu.vram_mb / 1024)")
        return self


class DeviceCudaInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    driver_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    recommended_torch_index: str = ""


class DeviceCostInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dph_total: float = Field(ge=0, allow_inf_nan=False)


class DeviceClusterRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jobid: str = ""
    node: str = ""
    requested_gpus: int = Field(
        ge=1,
        description="Total GPUs requested across all nodes for the stage job.",
    )
    nodes: int = Field(
        default=1,
        ge=1,
        description=(
            "Node count for the stage job. Default 1 (single node). Must equal "
            "resource_plan.nodes; requested_gpus must be a multiple of it."
        ),
    )
    partition: str = ""
    account: str = ""
    qos: str = ""
    container_image: str = ""
    env_setup: str = ""
    workdir: str = Field(min_length=1)
    hf_cache: str = Field(min_length=1)


class DeviceInstanceRoute(BaseModel):
    """Execution route for a fixed GPU host reached directly over SSH."""

    model_config = ConfigDict(extra="forbid")
    env_setup: str = ""
    workdir: str = Field(min_length=1)
    hf_cache: str = Field(min_length=1)
    visible_devices: str = Field(min_length=1)


class InfrastructureDeviceInfo(BaseModel):
    """Exact cross-Agent device_info.json produced by a usable provision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    ticket_id: str = Field(min_length=1)
    purpose: Literal["train", "inference"]
    provider: Literal["cluster", "cloud", "instance"]
    cloud_backend: Literal["", "vastai", "lambda"] = ""
    instance_id: str = ""
    auto_release: bool
    host: str = Field(min_length=1)
    # Every provider ultimately exposes one SSH route. Its credential is a
    # filesystem path, never inline key/password material.
    ssh: DeviceSshRoute | None = None
    cluster: DeviceClusterRoute | None = None
    instance: DeviceInstanceRoute | None = None
    gpu: DeviceGpuInfo | None = None
    cuda: DeviceCudaInfo | None = None
    cost: DeviceCostInfo
    resource_plan: InfrastructureResourcePlan
    probe_source: Literal[
        "nvidia-smi", "torch", "remote_nvidia-smi", "ssh-environment", "none"
    ]
    probed_at: datetime = Field(description="UTC ISO-8601 probe timestamp.")

    @model_validator(mode="after")
    def require_provider_shape(self) -> "InfrastructureDeviceInfo":
        if self.resource_plan.purpose != self.purpose:
            raise ValueError(
                "resource_plan.purpose must match device_info purpose"
            )
        if self.provider == "cloud":
            if not self.cloud_backend:
                raise ValueError("cloud device info requires cloud_backend")
            if self.cluster is not None:
                raise ValueError("cloud device info must not contain cluster routing")
            if self.instance is not None:
                raise ValueError("cloud device info must not contain instance routing")
            if not self.instance_id or self.gpu is None or self.cuda is None:
                raise ValueError("cloud device info requires instance and measured GPU/CUDA")
        elif self.provider == "cluster":
            if self.cloud_backend:
                raise ValueError("non-cloud device info must use empty cloud_backend")
            if self.cluster is None:
                raise ValueError("cluster device info requires Slurm access routing")
            if self.instance is not None:
                raise ValueError("cluster device info must not contain instance routing")
            if self.instance_id or self.cluster.jobid or self.cluster.node:
                raise ValueError(
                    "cluster access must not contain a holder job, allocation id, or node"
                )
            if self.gpu is not None or self.cuda is not None:
                raise ValueError(
                    "cluster access must not claim GPU/CUDA measurements before a stage job runs"
                )
            if self.cluster.nodes != self.resource_plan.nodes:
                raise ValueError(
                    "cluster route nodes must equal resource_plan.nodes"
                )
            if self.cluster.requested_gpus % self.cluster.nodes != 0:
                raise ValueError(
                    "cluster requested_gpus must be an exact multiple of nodes"
                )
            if self.auto_release:
                raise ValueError("cluster access has no resource to auto-release")
        else:
            if self.cloud_backend:
                raise ValueError("non-cloud device info must use empty cloud_backend")
            if self.cluster is not None:
                raise ValueError("instance device info must not contain Slurm routing")
            if self.instance is None or not self.instance_id:
                raise ValueError("instance device info requires a fixed-host route")
            if self.gpu is None or self.cuda is None:
                raise ValueError("instance device info requires measured GPU/CUDA")
        # All three providers ultimately expose a remote SSH route.
        if self.ssh is None:
            raise ValueError(f"{self.provider} device info requires an ssh route")
        if self.resource_plan.cloud_backend != self.cloud_backend:
            raise ValueError(
                "device cloud_backend must equal resource_plan.cloud_backend"
            )
        if self.provider == "instance":
            if self.auto_release:
                raise ValueError("instance device info cannot enable auto_release")
            if not self.instance or not self.instance.visible_devices.strip():
                raise ValueError("instance device info requires visible_devices")
            try:
                visible = [
                    int(value.strip())
                    for value in self.instance.visible_devices.split(",")
                    if value.strip()
                ]
            except ValueError as exc:
                raise ValueError("instance visible_devices must be comma-separated indices") from exc
            measured = [device.index for device in self.gpu.devices] if self.gpu else []
            if not visible or len(visible) != len(set(visible)) or visible != measured:
                raise ValueError(
                    "instance visible_devices must exactly match measured GPU device indices"
                )
        actual_gpu_count = (
            self.cluster.requested_gpus
            if self.provider == "cluster" and self.cluster is not None
            else self.gpu.gpu_count if self.gpu is not None else 0
        )
        if actual_gpu_count != self.resource_plan.num_gpus:
            raise ValueError(
                "actual/requested GPU count must equal resource_plan.num_gpus"
            )
        return self


def validate_device_info(
    path: str | Path, *, run_id: str, ticket_id: str,
    provider: str, num_gpus: int, purpose: str = "",
) -> InfrastructureDeviceInfo:
    """Validate a device artifact against the Run's maximum GPU count."""
    source = Path(path)
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read device info {source}: {exc}") from exc
    info = InfrastructureDeviceInfo.model_validate(body)
    expected = {
        "run_id": run_id,
        "ticket_id": ticket_id,
        "provider": provider,
    }
    actual = {
        "run_id": info.run_id,
        "ticket_id": info.ticket_id,
        "provider": info.provider,
    }
    if actual != expected:
        raise ValueError(f"device identity differs: expected {expected}, got {actual}")
    actual_gpu_count = (
        info.cluster.requested_gpus
        if info.provider == "cluster" and info.cluster is not None
        else info.gpu.gpu_count if info.gpu is not None else 0
    )
    if actual_gpu_count < 1 or (num_gpus and actual_gpu_count > num_gpus):
        limit = str(num_gpus) if num_gpus else "unlimited"
        raise ValueError(
            "device GPU count exceeds Run limit: "
            f"expected 1..{limit}, got {actual_gpu_count}"
        )
    if purpose and info.purpose != purpose:
        raise ValueError(
            f"device purpose differs: expected {purpose!r}, got {info.purpose!r}"
        )
    return info


def _main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate a Zevo device_info.json")
    parser.add_argument("command", choices=["validate-device"])
    parser.add_argument("path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ticket-id", required=True)
    parser.add_argument("--provider", required=True, choices=["cluster", "cloud", "instance"])
    parser.add_argument("--num-gpus", required=True, type=int)
    parser.add_argument("--purpose", required=True, choices=["train", "inference"])
    args = parser.parse_args(argv)
    try:
        validate_device_info(
            args.path,
            run_id=args.run_id,
            ticket_id=args.ticket_id,
            provider=args.provider,
            num_gpus=args.num_gpus,
            purpose=args.purpose,
        )
    except ValueError as exc:
        print(f"INVALID InfrastructureDeviceInfo: {exc}", file=sys.stderr)
        return 1
    print("VALID InfrastructureDeviceInfo")
    return 0


class InfraTaskInput(AgentTaskInput):
    """Resolved runtime/deployment context for one Infrastructure operation."""

    run_id: str = Field(
        "",
        description=(
            "Parent run id supplied by the runner. Use it for remote-directory "
            "scope, lease ownership, and infrastructure bookkeeping; never "
            "derive it from ticket_id. Empty only for standalone work."
        ),
    )
    purpose: Literal["", "train", "inference"] = Field(
        "",
        description=(
            "The immediate GPU consumer whose resource plan this route serves; "
            "empty only for release."
        ),
    )
    # Which backend to provision on.
    provider: Literal["cluster", "cloud", "instance"] = Field(
        ...,
        description=(
            "'cluster' = validate our Slurm login route; Train/Inference submit "
            "their own finite sbatch jobs. "
            "'cloud' = rent a GPU on an available platform selected by "
            "Infrastructure from capability, preference, price, and budget evidence. "
            "'instance' = use a fixed GPU host directly over SSH, without Slurm."
        ),
    )
    cloud_backend: Literal["", "vastai", "lambda"] = Field(
        "",
        description=(
            "Release-only lifecycle identity. For release, identify the cloud "
            "backend that created instance_id. Empty for provision and for "
            "non-cloud providers; Infrastructure chooses a provision backend "
            "from available_cloud_backends."
        ),
    )
    available_cloud_backends: list[Literal["vastai", "lambda"]] = Field(
        default_factory=list,
        description=(
            "Credentialed cloud backends available to Infrastructure. Empty "
            "for non-cloud providers. This is capability context, not a user "
            "selection; Infrastructure chooses and reports one for provision."
        ),
    )
    preferred_cloud_backend: Literal["", "vastai", "lambda"] = Field(
        "",
        description=(
            "Optional deployment preference from ZEVO_CLOUD_BACKEND. It is "
            "context for Infrastructure, not a Ticket/user resource value; it "
            "must also appear in available_cloud_backends before selection."
        ),
    )
    release: bool = Field(
        False,
        description=(
            "False = prepare usable compute access (the normal case). "
            "True = release the exact scope named by instance_id: destroy the "
            "cloud instance, or cancel only this run's steps and return its "
            "lease under instance. Cluster access owns no releasable job."
        ),
    )

    num_gpus: int = Field(
        0,
        ge=0,
        description=(
            "Maximum GPUs this Run permits Infrastructure to use. Select a "
            "concrete count of at least one and record it in "
            "resource_plan.num_gpus. Cloud rents an offer with that concrete "
            "count; cluster passes it to the downstream stage-owned sbatch job; "
            "instance leases that many idle cards from one fixed host. "
            "A positive value is a hard upper bound; zero means the user left "
            "the field blank and imposed no GPU-count limit."
        ),
    )

    max_queue_wait_hours: float = Field(
        24,
        gt=0,
        le=168,
        description=(
            "Maximum Slurm PENDING time for an automatic cluster submission. "
            "Cancel only the exact Zevo job when this expires. Queue time is "
            "excluded from the Run runtime cap."
        ),
    )

    resource_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Engine-owned run context used to derive the resource plan: task, "
            "pins, dataset profile, iteration horizon, runtime, and live budget. "
            "It contains no held-out Test paths or scores. Infrastructure must "
            "not ask the Orchestrator to turn it into sizing values."
        ),
    )
    host_memory_planning_contract: HostMemoryPlanningContract = Field(
        default_factory=HostMemoryPlanningContract,
        description=(
            "Engine-owned CPU RAM sizing rule. Estimate the stage's real peak "
            "working set from model loading/checkpoint saving, data, offload, "
            "and runtime behavior; then apply this headroom and record both "
            "the estimate and requested RAM in resource_plan.rationale."
        ),
    )
    gpu_lease_endpoint: Literal["/api/gpu/leases"] = "/api/gpu/leases"
    infra_instances_endpoint: Literal["/api/infra/instances"] = "/api/infra/instances"
    openapi_endpoint: Literal["/api/openapi.json"] = "/api/openapi.json"
    gpu_lease_request_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact JSON Schema for the instance-mode lease POST. It is the sole "
            "request-key authority. Validate locally; never probe a mutation "
            "endpoint with guessed bodies."
        ),
    )
    gpu_lease_grant_schema: dict[str, Any] = Field(
        ...,
        description="Exact JSON Schema for a successful lease response.",
    )
    infra_instance_create_schema: dict[str, Any] = Field(
        ...,
        description="Exact body schema for POST /api/infra/instances.",
    )
    infra_instance_patch_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact body schema for PATCH /api/infra/instances/{row_id}; "
            "row_id is the bookkeeping response id, not provider instance_id."
        ),
    )
    infra_instance_response_schema: dict[str, Any] = Field(
        ...,
        description="Exact successful bookkeeping response schema.",
    )
    device_info_schema: dict[str, Any] = Field(
        ...,
        description=(
            "Exact JSON Schema for a successful provision's device_info.json. "
            "This is the sole cross-Agent artifact key/type authority."
        ),
    )
    device_info_validation_command: str = Field(
        ...,
        min_length=1,
        description=(
            "Side-effect-free validator for device_info.json. Replace only "
            "<absolute-device-info-json-path>; identity/provider/GPU-count "
            "arguments are already shell-quoted and inserted by the engine."
        ),
    )

    # ----- verified Cluster/Instance SSH connection -----
    ssh_host: str = Field(..., description="Remote SSH host selected for cluster/instance.")
    ssh_port: int = Field(..., ge=0, le=65535, description="Remote SSH port for cluster/instance.")
    ssh_user: str = Field(..., description="Remote SSH user for cluster/instance.")
    ssh_key_path: str = Field("", description="Path to the SSH private key; mutually exclusive with ssh_password_path.")
    ssh_password_path: str = Field("", description="Path to a mode-0600 SSH password file; mutually exclusive with ssh_key_path.")
    slurm_partition: str = Field(
        "",
        description=(
            "Slurm partition for a system-owned cluster submission. Empty lets "
            "the selected site skill use a verified site/scheduler default. "
            "Ignored for instance mode."
        ),
    )
    slurm_account: str = Field(
        "",
        description=(
            "Optional Slurm account for a cluster submission. Site skills may "
            "require it to be paired with slurm_qos. Never infer another "
            "user/group's allocation."
        ),
    )
    slurm_qos: str = Field(
        "",
        description="Optional Slurm QOS for a cluster submission; validated by the selected site skill.",
    )
    # Resolved by the runner from the typed payload or the correct provider's
    # ZEVO_* variables. They used to live only in the agent's markdown as
    # `${VAR:-default}`, which meant an agent that copied the fallback silently
    # ignored the user's configuration — and one did.
    remote_dir: str = Field(
        "",
        description=(
            "Remote work ROOT on the cluster, used AS-IS (no /zevo suffix). "
            "Per-run dir is <remote_dir>/<run_id>; the default HF cache is "
            "<remote_dir>/hf_cache unless a matched site skill overrides it. "
            "For cluster/instance provisioning it must "
            "be a non-empty absolute path outside the remote user's HOME."
        ),
    )
    env_setup: str = Field(
        "",
        description=(
            "Shell line that activates the remote Python env before any "
            "command, e.g. 'source ~/miniconda3/bin/activate zevo'. Empty = the "
            "agent's own default."
        ),
    )
    container_image: str = Field(
        "",
        description=(
            "Optional site container image for cluster sbatch jobs. Use this exact "
            "value when set; otherwise the selected site skill may provide a "
            "documented default. Ignored for cloud and instance providers."
        ),
    )

    # ----- release params -----
    instance_id: str = Field(
        ...,
        description="Cloud instance id to release. Empty for cluster/instance and new provision.",
    )

    # ----- common -----
    auto_release: bool = Field(
        ...,
        description=(
            "If true, normal run finalization frees the system-owned cloud "
            "resource. Cluster access has no held resource and is always false. "
            "False is honored on ordinary completion; "
            "cancellation/deletion may force cleanup. Always false for instance "
            "and for a release ticket."
        ),
    )
    work_dir: str = Field(..., description="Directory where device_info.json will be written.")

    @model_validator(mode="after")
    def validate_remote_root(self) -> "InfraTaskInput":
        if self.release and self.purpose:
            raise ValueError("release must not carry a stage purpose")
        if not self.release and not self.purpose:
            raise ValueError("provision requires purpose=train|inference")
        if self.provider in ("cluster", "instance"):
            if bool(self.ssh_key_path.strip()) == bool(self.ssh_password_path.strip()):
                raise ValueError(
                    "cluster/instance requires exactly one SSH credential path"
                )
        # Cloud rentals have no preconfigured remote root; SSH-backed Cluster
        # and Instance connections always do.
        if self.release or self.provider == "cloud":
            return self
        root = self.remote_dir.strip()
        if not root:
            raise ValueError("remote_dir is required for cluster/instance provisioning")
        if root.startswith("~") or not root.startswith("/"):
            raise ValueError("remote_dir must be an absolute remote path")
        return self


# ---------- Output ----------

class InfraResult(AgentResult):
    """Infrastructure handoff without duplicating device_info.json facts."""

    status: Literal["succeeded", "degraded", "failed"] = Field(
        ...,
        description=(
            "'succeeded' = cloud/instance GPU is usable, or cluster SSH, "
            "scheduler, remote directory, and environment are validated. "
            "'degraded' = an explicitly accepted non-strict shortfall that is still usable. "
            "'failed' = probe/provision/release errored."
        ),
    )
    operation: Literal["provision", "release"] = Field(
        ...,
        description="The operation actually attempted; never infer it from empty artifact fields.",
    )

    device_info_path: str = Field("", description="Absolute path to device_info.json after a usable provision; empty for release/failure.")
    provider: Literal["", "cluster", "cloud", "instance"] = Field(
        "",
        description=(
            "Release/failure lifecycle identity. Empty on usable provision; "
            "the validated device artifact owns successful provision facts."
        ),
    )
    instance_id: str = Field(
        "",
        description=(
            "Rented box id for cloud failure/release bookkeeping. Preserve it "
            "when cloud cleanup fails so leak detection can find the resource. "
            "Cluster and instance results leave it empty."
        ),
    )
    cloud_backend: Literal["", "vastai", "lambda"] = Field(
        "",
        description=(
            "Release/failure cloud lifecycle identity. Empty on usable provision "
            "and for non-cloud providers."
        ),
    )
    auto_release: bool = Field(
        False,
        description=(
            "Failure-only cleanup evidence. Always false on usable provision "
            "and release; successful lifecycle policy lives in device_info.json."
        ),
    )

    @model_validator(mode="after")
    def validate_operation_result(self) -> "InfraResult":
        usable = self.status in {"succeeded", "degraded"}
        if self.operation == "provision" and usable:
            if not self.device_info_path:
                raise ValueError("usable provision requires device_info_path")
            if self.provider or self.instance_id or self.cloud_backend or self.auto_release:
                raise ValueError(
                    "usable provision lifecycle/device facts belong only in device_info.json"
                )
        elif self.device_info_path:
            raise ValueError("release/failure must not report device_info_path")
        if self.operation == "release" and usable:
            if not self.provider or not self.instance_id:
                raise ValueError("successful release requires provider and instance_id")
            if self.provider == "cloud" and not self.cloud_backend:
                raise ValueError("cloud release requires cloud_backend")
            if self.provider != "cloud" and self.cloud_backend:
                raise ValueError("non-cloud release must use empty cloud_backend")
            if self.auto_release:
                raise ValueError("release result cannot request later auto_release")
        return self


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(_main())
