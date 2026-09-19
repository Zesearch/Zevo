"""Stable namespace for progress steps emitted by a combined benchmark job."""
from __future__ import annotations

from hashlib import blake2s
from typing import Any, Mapping


def benchmark_id(suite: str, index: int) -> str:
    """Identity of a member in one Run's frozen, ordered scoring suite."""
    if suite not in {"validation", "test"} or index < 0:
        raise ValueError("invalid benchmark suite or index")
    return f"{suite}:{index}"


def benchmark_names(holdout: Mapping[str, Any], suite: str) -> list[str]:
    key = "validation_sets" if suite == "validation" else "test_sets"
    fallback = "Validation" if suite == "validation" else "Test"
    names = [
        str(item.get("name") or f"{fallback} {index + 1}")
        for index, item in enumerate(holdout.get(key) or [])
    ]
    singular = "validation_set" if suite == "validation" else "test_set"
    return names or ([fallback] if holdout.get(singular) else [])


def resolve_benchmark_id(
    marker: Mapping[str, Any], names: list[str], suite: str,
) -> str | None:
    """Resolve telemetry by ID, exact Task name, or ordered slot.

    A missing ID can use an exact Task name or the one-based slot emitted by
    the system inference helper. The slot also makes already-running jobs
    readable after a backend update. An explicitly invalid ID is never
    silently replaced by a plausible name or index.
    """
    raw_id = str(marker.get("benchmark_id") or "").strip()
    known = {benchmark_id(suite, index) for index in range(len(names))}
    if raw_id:
        return raw_id if raw_id in known else None
    name = str(marker.get("benchmark_name") or "").strip()
    if name in names:
        return benchmark_id(suite, names.index(name))
    try:
        index = int(marker.get("benchmark_index") or 0)
        total = int(marker.get("benchmark_total") or 0)
    except (TypeError, ValueError):
        index = total = 0
    if total == len(names) and 1 <= index <= len(names):
        return benchmark_id(suite, index - 1)
    return None


def canonical_benchmark_marker(
    marker: Mapping[str, Any], holdout: Mapping[str, Any], lane: str,
) -> dict[str, Any]:
    """Keep the Task's display name while preserving the stable progress ID."""
    suite = "test" if lane == "held_out_test" else "validation"
    names = benchmark_names(holdout, suite)
    identity = resolve_benchmark_id(marker, names, suite)
    result = dict(marker)
    if identity is not None:
        index = int(identity.rsplit(":", 1)[1])
        result["benchmark_id"] = identity
        result["benchmark_name"] = names[index]
    active_ids = marker.get("active_benchmark_ids")
    if isinstance(active_ids, list):
        known = {benchmark_id(suite, index): name
                 for index, name in enumerate(names)}
        result["active_benchmarks"] = [
            known.get(str(item), str(item)) for item in active_ids
        ]
    return result


def benchmark_progress_phase(phase: str, benchmark_key: str) -> str:
    """Keep equal row-step numbers from different benchmarks distinct.

    ExecutionEvent's unique progress key includes phase but not JSON extras.
    The human-readable name remains in extras. A short digest of the stable
    benchmark ID (or the name when no ID exists) prevents collisions without
    changing the database schema.
    """
    key = benchmark_key.strip()
    if not key:
        return phase
    digest = blake2s(key.encode("utf-8"), digest_size=6).hexdigest()
    return f"{(phase or 'inference')[:50]}:{digest}"
