"""Canonical comparison helpers for task score direction."""
from __future__ import annotations

from typing import Iterable, Literal


MetricDirection = Literal["max", "min"]


def is_better(candidate: float, incumbent: float | None, direction: MetricDirection) -> bool:
    """Whether candidate replaces incumbent; ``None`` means absent."""
    if incumbent is None:
        return True
    return candidate < incumbent if direction == "min" else candidate > incumbent


def best_score(values: Iterable[float], direction: MetricDirection) -> float | None:
    valid = [float(v) for v in values]
    if not valid:
        return None
    return min(valid) if direction == "min" else max(valid)


def improvement(candidate: float, baseline: float, direction: MetricDirection) -> float:
    """Positive means improvement in either direction."""
    return baseline - candidate if direction == "min" else candidate - baseline
