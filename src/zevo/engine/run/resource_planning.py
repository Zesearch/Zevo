"""Coarse GPU planning for cluster stages using current scheduler capacity.

Infrastructure discovers what a cluster permits.  This module turns that
capability envelope, current free capacity, and a rough workload estimate into
the exact GPU shape stamped on a stage work order. It intentionally avoids
false precision: estimates are rounded up to a small set of roomy tiers.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from zevo.contracts.infrastructure import (
    ClusterGpuConstraints,
    InfrastructureDeviceInfo,
)


StageKind = Literal["data", "train", "inference"]

# The common range stays easy to understand in the UI and in scheduler logs.
# Larger jobs continue the same coarse progression instead of pretending that
# a model estimate can distinguish, for example, 31 GPUs from 32.
GPU_RESOURCE_TIERS: tuple[int, ...] = (
    1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
)

_PARAMETER_COUNT = re.compile(r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+)?)\s*[Bb](?![A-Za-z])")


@dataclass(frozen=True)
class StageResourceSelection:
    """One auditable stage estimate and its scheduler-compatible realization."""

    stage: StageKind
    estimated_gpus: int
    num_gpus: int
    nodes: int
    gpus_per_node: int
    source: str
    rationale: str


@dataclass(frozen=True)
class SlurmCapacitySnapshot:
    """Immediately free GPUs on eligible nodes and within the selected QoS."""

    free_gpus_by_node: tuple[int, ...]
    qos_free_gpus: int | None = None

    @property
    def immediately_free_gpus(self) -> int:
        physical = sum(self.free_gpus_by_node)
        return min(physical, self.qos_free_gpus) if self.qos_free_gpus is not None else physical


def model_parameter_billions(model_name: str) -> float | None:
    """Read a conventional ``7B``/``32B`` size hint without network access."""
    matches = [float(value) for value in _PARAMETER_COUNT.findall(model_name)]
    return max(matches) if matches else None


def _inference_estimate(
    *, base_model: str, info: InfrastructureDeviceInfo,
    constraints: ClusterGpuConstraints,
) -> tuple[int, str]:
    """Estimate inference GPUs with roomy BF16 defaults, not exact simulation."""
    parameters = model_parameter_billions(base_model)
    if parameters is None:
        # Infrastructure's Train envelope is the best closed evidence when a
        # model id has no conventional size hint.  Inference usually needs less
        # memory, but retaining half of that envelope leaves useful slack.
        estimate = max(1, math.ceil(info.resource_plan.num_gpus / 2))
        return estimate, (
            "model id has no parameter-count hint; retained half of the "
            "Infrastructure training envelope"
        )

    # BF16 weights plus 25% runtime headroom and a coarse 8-GiB KV/runtime
    # allowance.  Only an order-of-magnitude decision is needed because the
    # result is rounded upward again below.
    weight_gib = parameters * 1_000_000_000 * 2 / (1024 ** 3)
    required_gib = weight_gib * 1.25 + 8.0
    reported_vram = float(constraints.gpu_vram_gib or 0)
    if reported_vram > 0:
        usable_per_gpu = reported_vram * 0.80
        capacity_source = f"80% of the discovered {reported_vram:g}-GiB GPU"
    else:
        # 64 GiB usable is deliberately conservative across common 80-GiB
        # accelerators.  The execution-time memory preflight remains the final
        # authority and can reject a genuinely unfit model.
        usable_per_gpu = 64.0
        capacity_source = "a conservative 64-GiB usable-GPU fallback"
    estimate = max(1, math.ceil(required_gib / usable_per_gpu))
    return estimate, (
        f"parsed about {parameters:g}B parameters; allowed BF16/runtime headroom "
        f"against {capacity_source}"
    )


def _constraints_for(info: InfrastructureDeviceInfo) -> ClusterGpuConstraints:
    if info.cluster is None:
        raise ValueError("cluster stage has no cluster execution route")
    return info.cluster.gpu_constraints


def _shape_for(
    total_gpus: int, constraints: ClusterGpuConstraints, *, single_node: bool,
) -> tuple[int, int] | None:
    if total_gpus < constraints.min_gpus_per_job:
        return None
    if total_gpus % constraints.allocation_step:
        return None
    capacity = constraints.gpus_per_node
    if constraints.whole_node:
        if total_gpus % capacity:
            return None
        nodes = total_gpus // capacity
        if single_node and nodes != 1:
            return None
        return nodes, capacity
    if single_node:
        return (1, total_gpus) if total_gpus <= capacity else None

    minimum_nodes = max(1, math.ceil(total_gpus / capacity))
    for nodes in range(minimum_nodes, total_gpus + 1):
        if total_gpus % nodes == 0:
            return nodes, total_gpus // nodes
    return None


def _tier_candidates(required: int, maximum: int) -> list[int]:
    candidates = list(GPU_RESOURCE_TIERS)
    if required > candidates[-1]:
        rounded = math.ceil(required / 32) * 32
        candidates.extend(range(candidates[-1] + 32, rounded + 1, 32))
    return [value for value in candidates if value >= required and (not maximum or value <= maximum)]


def _live_shape_for(
    total_gpus: int,
    constraints: ClusterGpuConstraints,
    *,
    single_node: bool,
    capacity: SlurmCapacitySnapshot,
) -> tuple[int, int] | None:
    """Choose an equal-per-node shape that fits the current node snapshot."""
    if total_gpus > capacity.immediately_free_gpus:
        return None
    for nodes in range(1, total_gpus + 1):
        if single_node and nodes > 1:
            break
        if total_gpus % nodes:
            continue
        per_node = total_gpus // nodes
        if per_node > constraints.gpus_per_node:
            continue
        if constraints.whole_node and per_node != constraints.gpus_per_node:
            continue
        if sum(free >= per_node for free in capacity.free_gpus_by_node) >= nodes:
            return nodes, per_node
    return None


def plan_stage_resources(
    *,
    stage: StageKind,
    base_model: str,
    info: InfrastructureDeviceInfo,
    training_method: str = "",
    live_capacity: SlurmCapacitySnapshot | None = None,
    capacity_error: str = "",
    maximum_gpus: int = 0,
    inference_rows: int = 0,
    inference_members: int = 1,
    inference_gpus_per_replica: int = 0,
    registered_gpus: int = 0,
    registered_nodes: int = 0,
) -> StageResourceSelection:
    """Choose a coarse tier, then conform it to the discovered cluster rules.

    ``registered_*`` is used during collect.  A submitted scheduler job is a
    historical fact, so collection preserves its exact shape and never runs a
    fresh estimate.
    """
    if info.provider != "cluster":
        raise ValueError("Slurm stage resource planning requires provider='cluster'")
    constraints = _constraints_for(info)

    if registered_gpus or registered_nodes:
        if registered_gpus < 1 or registered_nodes < 1:
            raise ValueError("registered Slurm job requires positive GPUs and nodes")
        if maximum_gpus and registered_gpus > maximum_gpus:
            raise ValueError(
                f"registered Slurm job uses {registered_gpus} GPUs above Run maximum {maximum_gpus}"
            )
        if registered_gpus % registered_nodes:
            raise ValueError("registered Slurm GPU count must be divisible by its node count")
        return StageResourceSelection(
            stage=stage,
            estimated_gpus=registered_gpus,
            num_gpus=registered_gpus,
            nodes=registered_nodes,
            gpus_per_node=registered_gpus // registered_nodes,
            source="registered scheduler job",
            rationale="collection preserves the exact already-submitted Slurm job shape",
        )

    if stage == "train":
        estimated = int(info.resource_plan.num_gpus)
        estimate_reason = "used Infrastructure's model/method-aware Train minimum"
    elif stage == "data":
        # Ordinary preparation is CPU-heavy. Some sites nevertheless require a
        # GPU GRES to enter their container/QoS path; start at the smallest tier
        # and let the discovered scheduler minimum raise it (Beta: 1 -> 4).
        estimated = 1
        estimate_reason = "Data preparation starts at the smallest resource tier"
    else:
        estimated, estimate_reason = _inference_estimate(
            base_model=base_model, info=info, constraints=constraints,
        )
        estimated = max(estimated, inference_gpus_per_replica)

    single_node = stage in {"data", "inference"}
    candidates = _tier_candidates(estimated, maximum_gpus)
    baseline = next(
        ((tier, shape) for tier in candidates
         if (shape := _shape_for(tier, constraints, single_node=single_node)) is not None),
        None,
    )
    if baseline is None:
        cap = str(maximum_gpus) if maximum_gpus else "unlimited"
        raise ValueError(
            f"no valid {stage} GPU tier can cover estimate {estimated} within Run maximum "
            f"{cap}; cluster constraints are minimum={constraints.min_gpus_per_job}, "
            f"step={constraints.allocation_step}, gpus_per_node={constraints.gpus_per_node}, "
            f"whole_node={constraints.whole_node}"
        )

    # Extra Train GPUs can shorten a large-model/full-parameter job. Inference
    # scales only for a sizeable scoring workload: each replica gets the full
    # minimum model envelope, and a single ticket still owns the whole job.
    model_size = model_parameter_billions(base_model)
    can_scale_train = stage == "train" and (
        training_method == "full_sft"
        or (not training_method and model_size is not None and model_size >= 30)
    )
    useful_ceiling = baseline[0] * 2 if can_scale_train else baseline[0]
    replica_gpus = max(1, estimated)
    if stage == "inference":
        by_members = math.ceil(max(1, inference_members) / 3)
        by_rows = math.ceil(max(0, inference_rows) / 2000)
        useful_replicas = min(4, max(by_members, by_rows, 1))
        if inference_members < 2 and inference_rows < 2000:
            useful_replicas = 1
        useful_ceiling = max(baseline[0], replica_gpus * useful_replicas)
    selected_tier, selected_shape = baseline
    if live_capacity is not None:
        for tier in reversed(candidates):
            if tier < baseline[0] or tier > useful_ceiling:
                continue
            if stage == "inference" and tier > baseline[0] and tier % replica_gpus:
                continue
            if _shape_for(tier, constraints, single_node=single_node) is None:
                continue
            live_shape = _live_shape_for(
                tier, constraints, single_node=single_node, capacity=live_capacity,
            )
            if live_shape is not None:
                selected_tier, selected_shape = tier, live_shape
                break

    nodes, per_node = selected_shape
    if live_capacity is None:
        capacity_reason = (
            f"live Slurm probe unavailable ({capacity_error}); requested the safe minimum"
            if capacity_error else "live Slurm probe not requested; requested the safe minimum"
        )
    elif _live_shape_for(
        selected_tier, constraints, single_node=single_node, capacity=live_capacity,
    ) is None:
        capacity_reason = (
            f"only {live_capacity.immediately_free_gpus} GPUs free now; "
            "submitted the safe minimum to wait in Slurm"
        )
    else:
        capacity_reason = (
            f"{live_capacity.immediately_free_gpus} GPUs free now; "
            f"selected {selected_tier} across {nodes} node(s)"
        )
    return StageResourceSelection(
        stage=stage,
        estimated_gpus=estimated,
        num_gpus=selected_tier,
        nodes=nodes,
        gpus_per_node=per_node,
        source="live Slurm capacity" if live_capacity is not None else "static cluster constraints",
        rationale=(
            f"{estimate_reason}; rounded minimum to tier {baseline[0]} using "
            f"{constraints.source}; "
            + (
                f"inference workload {inference_members} member(s), {inference_rows} "
                f"known row(s), at most {max(1, selected_tier // replica_gpus)} "
                "model replica(s); "
                if stage == "inference" else ""
            )
            + capacity_reason
        ),
    )
