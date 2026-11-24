"""Explicit upstream repair dependencies and reviewed Train configuration versions."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import select

from zevo.db import HeartbeatRun, Run, Ticket, TicketNotice, WorkProduct
from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS

WAIT_CODE = "repair.review_required"
DEPENDENCY_CODE = "repair.upstream_dependency"


def _utc(value):
    # SQLite returns naive UTC; PostgreSQL preserves the timezone.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def waiting_for_review(ticket):
    return ticket.status == "awaiting_input" and ticket.repair_route == "orchestrator"


async def latest_request(session, ticket):
    notice = await session.scalar(select(TicketNotice).where(
        TicketNotice.ticket_id == ticket.id, TicketNotice.code == WAIT_CODE,
    ).order_by(TicketNotice.created_at.desc(), TicketNotice.id.desc()).limit(1))
    return json.loads(notice.body) if notice else {}


async def wait_for_review(session, ticket, recovery, heartbeat_id):
    if (ticket.lane != "optimization" or ticket.agent_id == "orchestrator"
            or recovery is None or int(ticket.repair_attempts or 0) >= MAX_REPAIR_ATTEMPTS
            or not (recovery.scope == "upstream_input"
                    or ticket.agent_id == "train" and recovery.scope == "fixed_configuration")):
        return False
    ticket.status = "awaiting_input"
    ticket.repair_route = "orchestrator"
    session.add(TicketNotice(ticket_id=ticket.id, code=WAIT_CODE, severity="info",
        body=json.dumps({"heartbeat_id": heartbeat_id, "request": recovery.model_dump()})))
    return True


async def validate_dependency(session, run, target_id, *, child_agent, iteration):
    target = await session.get(Ticket, target_id)
    if (target is None or target.run_id != run.id or target.lane != "optimization"
            or not waiting_for_review(target) or target.iteration != iteration
            or child_agent == "orchestrator"):
        raise HTTPException(409, "Upstream repair requires a waiting Ticket in the same Run and iteration")
    request = await latest_request(session, target)
    if request.get("request", {}).get("scope") != "upstream_input":
        raise HTTPException(409, "Ticket is not waiting for upstream inputs")
    if int(target.repair_attempts or 0) >= MAX_REPAIR_ATTEMPTS:
        raise HTTPException(409, "Ticket repair limit reached")
    heartbeat = await session.scalar(select(HeartbeatRun.id).where(
        HeartbeatRun.ticket_id == run.supervisor_ticket_id,
        HeartbeatRun.agent_id == "orchestrator", HeartbeatRun.finished_at.is_(None),
    ).limit(1))
    supervisor = await session.get(Ticket, run.supervisor_ticket_id)
    if not heartbeat or not supervisor or supervisor.status != "running":
        raise HTTPException(409, "Upstream repair requires a live supervisor decision")
    return target


async def waiting_ancestors(session, ticket):
    """Permit a repair chain, not unrelated parallel optimization work."""
    ids = set()
    current = ticket
    while current is not None and current.id not in ids:
        ids.add(current.id)
        parent_id = await session.scalar(select(TicketNotice.body).where(
            TicketNotice.ticket_id == current.id, TicketNotice.code == DEPENDENCY_CODE,
        ).limit(1))
        parent = await session.get(Ticket, parent_id) if parent_id else None
        current = parent if (parent and parent.run_id == ticket.run_id
            and parent.lane == ticket.lane and waiting_for_review(parent)) else None
    return ids


async def restore_inputs(session, run, ticket, request, row):
    """Only reviewed products may replace bindings; never mutate a trained lineage."""
    from zevo.contracts.tickets import CreateTicketBody, validate_stored_payload
    from zevo.engine.run.scheduler.bindings import resolve_input_bindings
    from zevo.api.routers.shared.tickets import validate_pipeline_bindings, _train_allows_input_repair
    review = await latest_request(session, ticket)
    if review.get("request", {}).get("scope") != "upstream_input":
        raise HTTPException(409, "Resume inputs requires an upstream-input review")
    if request.input_signature:
        raise HTTPException(422, "Resume inputs uses reviewed products, not a preparation signature")
    if not request.input_products:
        # Environment/connectivity repairs can leave artifact bindings unchanged.
        source = await session.scalar(select(Ticket.id).join(TicketNotice, TicketNotice.ticket_id == Ticket.id).where(
            Ticket.run_id == run.id, Ticket.lane == ticket.lane, Ticket.status == "succeeded",
            TicketNotice.code == DEPENDENCY_CODE, TicketNotice.body == ticket.id,
        ).limit(1))
        if source is None:
            raise HTTPException(409, "Resume requires reviewed products or a succeeded explicit repair dependency")
    if ticket.agent_id == "train" and not await _train_allows_input_repair(session, ticket):
        raise HTTPException(409, "Train inputs cannot change after execution or a checkpoint")
    bindings = dict(ticket.inputs or {})
    config_changes = None
    for name, product_id in request.input_products.items():
        old = bindings.get(name)
        product = await session.get(WorkProduct, product_id)
        source = await session.get(Ticket, product.ticket_id) if product else None
        if (not isinstance(old, dict) or not source or source.run_id != run.id
                or source.lane != ticket.lane or source.id == ticket.id
                or source.status != "succeeded" or product.role != old.get("artifact_role")):
            raise HTTPException(409, "Replacement must be a succeeded same-lane upstream product of the original role")
        if source.id != old.get("source_ticket_id"):
            linked = await session.scalar(select(TicketNotice.id).where(
                TicketNotice.ticket_id == source.id, TicketNotice.code == DEPENDENCY_CODE,
                TicketNotice.body == ticket.id,
            ).limit(1))
            if not linked:
                raise HTTPException(409, "New upstream Ticket must explicitly repair this waiting Ticket")
        # Frozen model/measurement identity is not a data repair permission.
        if name == "inference_config":
            previous = await session.get(WorkProduct, old.get("work_product_id"))
            if (ticket.agent_id != "train" or previous is None
                    or previous.ticket_id != source.id or source.agent_id != "inference"
                    or (source.payload or {}).get("model_source") != "base_model"
                    or (source.payload or {}).get("base_model") != (ticket.payload or {}).get("base_model")
                    or _utc(previous.created_at) > _utc(ticket.created_at)
                    or _utc(product.created_at) > _utc(ticket.created_at)):
                raise HTTPException(409, "Train may only rebind existing members of its own frozen baseline before submission")
            config_changes = compatible_training_member(previous.path, product.path,
                pins=[run.decision_pins or {}, (ticket.payload or {}).get("configuration_pins", {})])
        elif name in {"parent_checkpoint", "parent_train_config", "train_config"}:
            raise HTTPException(409, "Frozen model and measurement bindings require a new experiment")
        bindings[name] = {**old, "source_ticket_id": source.id,
                          "work_product_id": product.id, "path": product.path}
    try:
        bindings = await resolve_input_bindings(session, run_id=run.id, inputs=bindings)
        body = CreateTicketBody(agent_id=ticket.agent_id, iteration=ticket.iteration,
                                run_id=run.id, payload=ticket.payload, inputs=bindings)
        payload, bindings = await validate_pipeline_bindings(session, owning_run=run,
            body=body, payload=dict(ticket.payload), inputs=bindings, current_ticket_id=ticket.id)
        payload = validate_stored_payload(agent_id=ticket.agent_id,
            input_format=ticket.input_format, payload=payload)
    except ValueError as exc:
        raise HTTPException(422, f"Repaired inputs are invalid: {exc}") from exc
    session.add(TicketNotice(ticket_id=ticket.id, code="repair.inputs_rebound", severity="info",
        body=json.dumps({"previous_inputs": ticket.inputs, "inputs": bindings,
            "previous_data_signature": (ticket.payload or {}).get("data_signature"),
            "data_signature": payload.get("data_signature"),
            "inference_member_change": config_changes})))
    ticket.inputs, ticket.payload = bindings, payload


def compatible_training_member(old_path, new_path, *, pins):
    """A narrow baseline-member selection, never permission to edit evaluation."""
    from zevo.contracts.configuration import load_inference_config
    try:
        before, after = [load_inference_config(p).model_dump() for p in (old_path, new_path)]
        changes = {}
        def compare(a, b, path=""):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in a.keys() | b.keys():
                    compare(a.get(key), b.get(key), f"{path}.{key}" if path else key)
            elif a != b:
                changes[path] = {"before": a, "after": b}
        compare(before, after)
        allowed = {"measurement.inference_config.answer_regex", "measurement.max_new_tokens"}
        if set(changes) - allowed:
            raise ValueError("baseline member changes protected model, prompt or decoding fields")
        regex = changes.get("measurement.inference_config.answer_regex")
        if regex:
            a, b = regex['before'], regex['after']
            pattern = r"\[A-([A-Z])\]"
            old_ranges, new_ranges = re.findall(pattern, a), re.findall(pattern, b)
            if (not old_ranges or len(old_ranges) != len(new_ranges)
                    or re.sub(pattern, "CHOICE_RANGE", a) != re.sub(pattern, "CHOICE_RANGE", b)
                    or any(x > y for x, y in zip(old_ranges, new_ranges))):
                raise ValueError("only widening existing A-Z choice capture ranges is permitted")
        def check_pins(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if any(path.rsplit('.', 1)[-1] == key for path in changes):
                        raise ValueError(f"cannot change pinned {key}")
                    check_pins(item)
        for pin in pins:
            check_pins(pin)
        return {"changes": changes,
                "previous_sha256": hashlib.sha256(Path(old_path).read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(Path(new_path).read_bytes()).hexdigest()}
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(409, f"Incompatible frozen baseline member: {exc}") from exc


async def approve_configuration(session, ticket, row, request, heartbeat_id):
    from zevo.contracts.configuration import load_train_config
    proposal = request.training_configuration_change
    review = await latest_request(session, ticket)
    offered = review.get("request", {}).get("training_configuration_change")
    if ticket.agent_id != "train" or proposal is None or offered != proposal.model_dump():
        raise HTTPException(409, "Approval must match the Train's exact submitted configuration proposal")
    if proposal.original_sha256 != (row.meta or {}).get("training_config_sha256"):
        raise HTTPException(409, "Configuration proposal refers to a stale execution")
    configs = []
    try:
        for path, digest in [(proposal.original_config_path, proposal.original_sha256),
                             (proposal.proposed_config_path, proposal.proposed_sha256)]:
            if not Path(path).is_absolute() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError("configuration bytes do not match the proposal")
            configs.append(load_train_config(path).model_dump())
    except (OSError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    before, after = configs
    # Apply user pins first. Review may approve objective hyperparameters that
    # Train cannot change autonomously, but not model/data/method identity.
    from types import SimpleNamespace
    from zevo.engine.run.training_config_retry import compare_registered
    payload = ticket.payload or {}
    try:
        audit = compare_registered({
            "training_config_sha256": proposal.original_sha256,
            "training_config_snapshot": Path(proposal.original_config_path).read_text(),
        }, proposal.proposed_config_path, train_input=SimpleNamespace(**{
            key: payload.get(key, {}) for key in
            ("configuration_pins", "method_config_pins", "loss_objective_pins")
        }))
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    blocked = [c["path"] for c in audit["changes"]
               if not c.get("train_owned")
               and not c["path"].startswith("loss_contract.objective_config.")]
    if blocked:
        raise HTTPException(409, "Configuration repair changes protected task boundaries: " + ", ".join(blocked))
    from zevo.contracts.training_resume import resume_mode
    try:
        if resume_mode(after.get("training", {}).get("implementation_config", {})) != "none":
            raise ValueError("Reviewed configuration changes must restart from the original parent, not a checkpoint continuation")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    changes = {c["path"]: c for c in audit["changes"]}
    if not changes or proposal.original_sha256 == proposal.proposed_sha256:
        raise HTTPException(422, "Configuration repair must contain an actual change")
    row.meta = {**(row.meta or {}), "training_configuration_approval": {
        **proposal.model_dump(), "heartbeat_id": heartbeat_id,
        "execution_attempt": request.execution_attempt, "released": False,
        "configuration_diff": changes, "original_configuration": before, "proposed_configuration": after,
    }}
