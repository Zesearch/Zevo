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


def champion_index(scores: Iterable[float], direction: MetricDirection) -> int | None:
    """Index of the fine-tuned candidate the final Registry keeps.

    Policy (decided 2026-10-02): the Run keeps its best fine-tuned model
    however small its margin. There is no minimum improvement, no tolerance
    band and no comparison with the baseline here: a candidate that beats the
    base model by one Validation row is still the model the Run produced. The
    comparison is strict, so among equal scores the earliest iteration wins
    and a later tie never displaces it. ``None`` only when nothing was
    measured.
    """
    best: int | None = None
    best_score: float | None = None
    for index, value in enumerate(scores):
        score = float(value)
        if is_better(score, best_score, direction):
            best, best_score = index, score
    return best
