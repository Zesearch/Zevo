"""System-owned vLLM device-memory calculations.

Inference records an absolute per-GPU target based on model weights and the
planned live KV-cache workload.  The target remains stable for the model
lineage; this helper converts it to the fraction required by the actual GPU.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


def gpu_memory_utilization(
    memory_plan: Mapping[str, Any], total_gpu_memory_gib: float,
) -> float:
    """Return the rounded vLLM fraction for one concrete GPU."""
    if total_gpu_memory_gib <= 0:
        raise ValueError("total_gpu_memory_gib must be positive")
    target = float(memory_plan["target_gpu_memory_gib"])
    step = float(memory_plan["utilization_step"])
    minimum = float(memory_plan["min_utilization"])
    maximum = float(memory_plan["max_utilization"])
    if target <= 0 or step <= 0 or not 0 < minimum <= maximum <= 1:
        raise ValueError("invalid inference memory plan")
    utilization = math.ceil((target / total_gpu_memory_gib) / step) * step
    utilization = max(minimum, utilization)
    if utilization > maximum + 1e-9:
        raise ValueError(
            "the planned inference workload does not fit this GPU; increase "
            "tensor parallelism or reduce concurrency/context"
        )
    return round(utilization, 10)


def required_free_memory_gib(
    memory_plan: Mapping[str, Any], total_gpu_memory_gib: float,
) -> float:
    """Return free memory required before constructing the vLLM engine."""
    utilization = gpu_memory_utilization(memory_plan, total_gpu_memory_gib)
    margin = float(memory_plan["free_memory_margin_gib"])
    return utilization * total_gpu_memory_gib + margin
