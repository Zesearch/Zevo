"""Read-only recovery of legacy checkpoint associations from recorded API calls."""
from __future__ import annotations

import re

from sqlalchemy import String, cast, select
from zevo.db import TranscriptEvent


# The response can be truncated in the middle of its reason; these identity
# fields precede it. Never infer a decision from natural-language summaries.
_ACCEPTED = re.compile(r'\{\s*"status"\s*:\s*"decided"\s*,\s*"decision"\s*:\s*\{([^{}]*)')
_REQUEST = re.compile(r'[\"\']request_id[\"\']\s*:\s*[\"\']([a-f0-9]{32})[\"\']')


def legacy_decision_context(command: str, output: str, requests: dict, *, is_error=False):
    """Require a recorded POST and its paired accepted response + stored decision."""
    if is_error or not re.search(r'-X\s+[\"\']?POST\b|--request[=\s]+[\"\']?POST\b', command):
        return None
    matches = []
    for response in _ACCEPTED.finditer(output):
        fields = response.group(1).split('"reason"', 1)[0]
        identity = _REQUEST.search(fields)
        if not identity:
            continue
        request_id = identity.group(1)
        request = requests.get(request_id)
        if not request or request.get("status") != "decided":
            continue
        if f'/adaptive-training/{request["ticket_id"]}/decision' not in command:
            continue
        action = re.search(r'"action"\s*:\s*"(continue|stop)"', fields)
        next_step = re.search(r'"next_step"\s*:\s*(\d+)', fields)
        decision = request.get("decision") or {}
        if not action or action.group(1) != decision.get("action"):
            continue
        step = int(next_step.group(1)) if next_step else None
        if action.group(1) == "continue" and step != decision.get("next_step"):
            continue
        matches.append({"request_id": request_id, "train_ticket_id": request["ticket_id"],
                        "step": request["step"], "action": action.group(1), "next_step": step})
    unique = {row["request_id"]: row for row in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


async def legacy_contexts(db, rows, lifecycles):
    candidates = [row for row in rows if row.agent_id == "orchestrator"
                  and (lifecycles.get(row.ticket_id) or {}).get("adaptive_training_requests")]
    if not candidates:
        return {}
    # Fetch only calls mentioning the decision route, not every transcript.
    calls = (await db.execute(select(TranscriptEvent).where(
        TranscriptEvent.heartbeat_id.in_([row.id for row in candidates]),
        TranscriptEvent.type == "tool_call",
        cast(TranscriptEvent.payload, String).contains("/adaptive-training/"),
    ))).scalars().all()
    call_map = {(call.heartbeat_id, (call.payload or {}).get("tool_use_id")): call for call in calls
                if (call.payload or {}).get("tool_use_id")}
    if not call_map:
        return {}
    results = (await db.execute(select(TranscriptEvent).where(
        TranscriptEvent.heartbeat_id.in_([key[0] for key in call_map]),
        TranscriptEvent.type == "tool_result",
        TranscriptEvent.payload["tool_use_id"].as_string().in_([key[1] for key in call_map]),
    ).order_by(TranscriptEvent.seq))).scalars().all()
    tickets = {row.id: row.ticket_id for row in candidates}
    found = {}
    ambiguous = set()
    for result in results:
        payload = result.payload or {}
        call = call_map.get((result.heartbeat_id, payload.get("tool_use_id")))
        if call is None:
            continue
        arguments = call.payload.get("input")
        if not isinstance(arguments, dict):
            continue
        command = arguments.get("command", "")
        if not isinstance(command, str) or not isinstance(payload.get("output"), str):
            continue
        requests = (lifecycles.get(tickets[result.heartbeat_id]) or {}).get("adaptive_training_requests", {})
        context = legacy_decision_context(command, payload["output"], requests, is_error=payload.get("is_error", False))
        if context:
            prior = found.get(result.heartbeat_id)
            if prior and prior["request_id"] != context["request_id"]:
                ambiguous.add(result.heartbeat_id)
            found[result.heartbeat_id] = context
    return {key: value for key, value in found.items() if key not in ambiguous}
