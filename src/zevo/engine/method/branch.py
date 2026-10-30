"""Method-search identity is independent of the checkpoint chosen as weights."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def active_training_method(
    history: Sequence[dict[str, Any]], *, before_iteration: int | None = None,
) -> str:
    """Use the latest measured training round, never the champion's lineage.

    Exclude this and later rounds when collecting/retrying an existing Ticket.
    """
    for row in reversed(history):
        if not isinstance(row, dict) or row.get("source") != "trained":
            continue
        if before_iteration is not None and int(row.get("iteration") or 0) >= before_iteration:
            continue
        method = str(row.get("training_method") or "").strip().lower()
        if method:
            return method
    return ""


def method_branch_problem(
    active: str, selected: str, *, pin: str = "",
    transition: Mapping[str, Any] | None = None,
) -> str:
    """The same work-order check at creation, activation and result validation."""
    if pin or not active or not selected or active == selected:
        return ""
    transition = transition or {}
    if transition.get("level") not in {"method", "base_model"} or not all(
        str(transition.get(key) or "").strip()
        for key in ("next_branch", "exhausted_branch", "validation_evidence")
    ):
        return (
            f"upstream method branch conflict: {active} -> {selected} requires "
            "a complete method/base_model branch_transition with Validation evidence"
        )
    return ""
