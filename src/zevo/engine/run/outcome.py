"""Deterministic Run outcome facts shared by the API, scheduler, and UI.

A Run can stop because it reached a configured limit without anything going
wrong.  Keep that normal stop trigger separate from unresolved execution
issues; only the latter makes an otherwise usable Run ``degraded``.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable


_USABLE_TICKET_STATUSES = {"succeeded", "skipped"}
_ISSUE_TICKET_STATUSES = {"failed", "degraded", "cancelled"}


def _aware(value: datetime | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _ticket_slot(ticket: Any) -> tuple[str, str, int, str, str]:
    """Return the semantic work slot whose newest Ticket is authoritative.

    The pipeline owns at most one work order per Agent/lane/iteration/operation
    and benchmark name.  A new Ticket in that slot is a replacement work order,
    so a successful replacement resolves an older failed attempt.  Including
    operation and test_set_name keeps distinct deterministic jobs separate.
    """
    payload = dict(getattr(ticket, "payload", None) or {})
    return (
        str(getattr(ticket, "agent_id", "") or ""),
        str(getattr(ticket, "lane", "optimization") or "optimization"),
        int(getattr(ticket, "iteration", 0) or 0),
        str(payload.get("operation") or ""),
        str(payload.get("test_set_name") or ""),
    )


def _stage_name(ticket: Any) -> str:
    agent = str(getattr(ticket, "agent_id", "") or "stage")
    names = {
        "data": "Data preparation",
        "inference": "Inference",
        "evaluation": "Evaluation",
        "train": "Training",
        "registry": "Model registration",
        "infrastructure": "Infrastructure setup",
    }
    label = names.get(agent, agent.replace("_", " ").title())
    if str(getattr(ticket, "lane", "") or "") == "held_out_test":
        label = f"Held-out Test {label.lower()}"
    return label


def _limit_reaped(ticket: Any) -> bool:
    text = " ".join((
        str(getattr(ticket, "error_message", "") or ""),
        str(getattr(ticket, "summary", "") or ""),
    )).strip()
    return bool(re.search(r"(?i)\brun limit reached\b", text))


def unresolved_ticket_issues(tickets: Iterable[Any]) -> list[dict[str, Any]]:
    """Describe current unresolved worker failures, excluding old attempts.

    A historical failure does not stain the final Run when a newer Ticket for
    the same semantic work slot succeeded.  Tickets reaped solely because the
    Run entered bounded finalization are a consequence of the normal stop
    trigger, not an execution issue of their own.
    """
    slots: dict[tuple[str, str, int, str, str], list[Any]] = defaultdict(list)
    for ticket in tickets:
        if str(getattr(ticket, "agent_id", "") or "") == "orchestrator":
            continue
        slots[_ticket_slot(ticket)].append(ticket)

    issues: list[dict[str, Any]] = []
    for rows in slots.values():
        latest = max(
            rows,
            key=lambda item: (
                _aware(getattr(item, "created_at", None)),
                _aware(getattr(item, "updated_at", None)),
                str(getattr(item, "id", "") or ""),
            ),
        )
        status = str(getattr(latest, "status", "") or "")
        if status in _USABLE_TICKET_STATUSES or status not in _ISSUE_TICKET_STATUSES:
            continue
        if status == "failed" and _limit_reaped(latest):
            continue
        ticket_id = str(getattr(latest, "id", "") or "")
        label = _stage_name(latest)
        if status == "degraded":
            message = f"{label} completed with a usable shortfall ({ticket_id})."
        elif status == "cancelled":
            message = f"{label} was cancelled before completion ({ticket_id})."
        else:
            message = f"{label} did not complete ({ticket_id})."
        issues.append({
            "code": (
                "ticket_degraded" if status == "degraded"
                else "ticket_cancelled" if status == "cancelled"
                else "ticket_failed"
            ),
            "ticket_id": ticket_id,
            "agent_id": str(getattr(latest, "agent_id", "") or ""),
            "lane": str(getattr(latest, "lane", "") or ""),
            "iteration": int(getattr(latest, "iteration", 0) or 0),
            "message": message,
        })
    return sorted(issues, key=lambda item: (item["iteration"], item["ticket_id"]))


def _legacy_stop_trigger(reason: str) -> dict[str, Any]:
    iteration = re.search(r"iterations\s+(\d+)\s*>=\s*budget\s+(\d+)", reason, re.I)
    if iteration:
        current, limit = (int(iteration.group(1)), int(iteration.group(2)))
        return {
            "code": "iteration_limit",
            "current": current,
            "limit": limit,
            "message": f"Stopped after completing {current} of {limit} iterations.",
        }
    runtime = re.search(r"runtime\s+([0-9.]+)h\s*>=\s*cap\s+([0-9.]+)h", reason, re.I)
    if runtime:
        return {
            "code": "runtime_limit",
            "current": float(runtime.group(1)),
            "limit": float(runtime.group(2)),
            "message": f"Stopped after reaching the {float(runtime.group(2)):g}-hour runtime limit.",
        }
    cost = re.search(r"cost\s+\$([0-9.]+)\s*>=\s*cap\s+\$([0-9.]+)", reason, re.I)
    if cost:
        return {
            "code": "cost_limit",
            "current": float(cost.group(1)),
            "limit": float(cost.group(2)),
            "message": f"Stopped after reaching the ${float(cost.group(2)):g} cost limit.",
        }
    return {
        "code": "configured_limit",
        "message": "Stopped after reaching a configured Run limit.",
    }


def stop_trigger(run: Any) -> dict[str, Any] | None:
    lifecycle = dict(getattr(run, "lifecycle", None) or {})
    state = dict(lifecycle.get("finalization") or {})
    structured = state.get("stop_trigger")
    if isinstance(structured, dict) and structured.get("message"):
        return dict(structured)
    reason = str(state.get("reason") or "").strip()
    return _legacy_stop_trigger(reason) if reason else None


def terminal_outcome(
    run: Any,
    tickets: Iterable[Any],
    *,
    extra_issues: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build the structured facts rendered for a terminal Run."""
    lifecycle = dict(getattr(run, "lifecycle", None) or {})
    finalization = dict(lifecycle.get("finalization") or {})
    issues = (
        []
        if str(getattr(run, "status", "") or "") == "cancelled"
        else list(unresolved_ticket_issues(tickets))
    )
    # The legacy value is retained here so old in-flight rescue rows still
    # receive the correct issue after an application-first rolling deploy.
    if finalization.get("expired_at") or lifecycle.get("rescue_terminal_status") in {"failed", "halted"}:
        issues.append({
            "code": "finalization_timeout",
            "ticket_id": "",
            "agent_id": "",
            "lane": "",
            "iteration": int(getattr(run, "iterations_completed", 0) or 0),
            "message": "Finalization did not complete before its deadline.",
        })
    issues.extend(dict(item) for item in extra_issues)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        key = (str(issue.get("code") or ""), str(issue.get("ticket_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)

    return {
        "stop_trigger": stop_trigger(run),
        "issues": deduped,
        "registered_model": str(getattr(run, "registry_version_tag", "") or ""),
    }


def lifecycle_with_terminal_outcome(
    run: Any,
    tickets: Iterable[Any],
    *,
    extra_issues: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    lifecycle = dict(getattr(run, "lifecycle", None) or {})
    if lifecycle.get("rescue_terminal_status") == "halted":
        lifecycle["rescue_terminal_status"] = "failed"
    lifecycle["terminal_outcome"] = terminal_outcome(
        run, tickets, extra_issues=extra_issues,
    )
    return lifecycle


def issue_summary(outcome: dict[str, Any]) -> str:
    messages = [
        str(item.get("message") or "").strip()
        for item in list(outcome.get("issues") or [])
        if str(item.get("message") or "").strip()
    ]
    return " ".join(messages)
