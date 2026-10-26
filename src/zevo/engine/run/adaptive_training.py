"""Validation-only checkpoint feedback and live Orchestrator decisions."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil

from sqlalchemy import select

from zevo.db import Run, Ticket, ExecutionEvent, WorkProduct


def activation_validation_context(lifecycle, reason, payload, previous=None):
    """Identify a checkpoint by its request ID, never by activation timing."""
    request_id = (payload or {}).get("adaptive_training_request_id")
    prefix = "mid-training Validation decision "
    if not request_id and reason.startswith(prefix):
        request_id = reason[len(prefix):].strip()
    if not request_id and previous:
        request_id = previous.get("request_id")
    request = lifecycle.get("adaptive_training_requests", {}).get(request_id)
    if not request or (request.get("status") != "pending" and not previous):
        return None
    return {"request_id": request_id, "train_ticket_id": request["ticket_id"], "step": request["step"]}


async def build_validation_contract(session, run, ticket, work_dir):
    if not (run.lifecycle or {}).get("adaptive_mid_training_evaluation"):
        return {}
    fraction = (ticket.payload or {}).get("first_validation_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("Orchestrator must specify first_validation_fraction between zero and one")
    from zevo.engine.run.runner import _validation_sets, _baseline_suite_config_path
    from zevo.engine.run.scheduler.bindings import input_path
    root = Path(work_dir) / "adaptive-validation"
    root.mkdir(parents=True, exist_ok=True)
    # Reuse the exact deterministic built-in implementation on the GPU host.
    from zevo.paths import REPO_ROOT
    metrics_file = root / "zevo_eval_metrics.py"
    shutil.copyfile(REPO_ROOT / "src/zevo/engine/method/eval_metrics.py", metrics_file)
    helpers = {"metrics": {"path": metrics_file.name,
                          "sha256": hashlib.sha256(metrics_file.read_bytes()).hexdigest()}}
    for name in ("inference_memory", "option_scoring", "parallel_inference"):
        target = root / f"zevo_{name}.py"
        shutil.copyfile(REPO_ROOT / f"playbook/runners/{name}.py", target)
        helpers[name] = {"path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    baselines = (await session.execute(
        select(WorkProduct, Ticket).join(Ticket, Ticket.id == WorkProduct.ticket_id)
        .where(Ticket.run_id == run.id, Ticket.agent_id == "inference",
               Ticket.lane == "optimization", Ticket.status.in_(("succeeded", "degraded")),
               WorkProduct.role == "script").order_by(WorkProduct.created_at.desc())
    )).all()
    baseline_script = next((product.path for product, source in baselines
        if (source.payload or {}).get("model_source") == "base_model"
        and (source.payload or {}).get("base_model") == (ticket.payload or {}).get("base_model")), "")
    if not baseline_script:
        raise ValueError("mid-training evaluation requires the baseline prediction script")
    members = []
    for index, item in enumerate(_validation_sets(run)):
        config = await _baseline_suite_config_path(
            session, run=run, base_model=str((ticket.payload or {}).get("base_model", "")),
            member_name=item.get("name", ""))
        if not config and index == 0:
            config = input_path(ticket.inputs or {}, "inference_config")
        files = {}
        sources = {"questions": item.get("public"), "labels": item.get("validation_set"),
                   "sample_submission": item.get("sample_submission"),
                   "inference_config": config, "predict_script": baseline_script,
                   "evaluation_script": item.get("evaluation_script")}
        for role, source in sources.items():
            if not source and role == "evaluation_script" and item.get("metric_type") == "builtin":
                continue
            if not source or not Path(source).is_file():
                raise ValueError(f"mid-training Validation requires a frozen {role} for member {index}")
            target = root / f"{index}-{role}{Path(source).suffix}"
            shutil.copyfile(source, target)
            files[role] = {"path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        members.append({"name": item.get("name") or f"validation-{index}",
                        "metric": item.get("metric") or run.validation_metric,
                        "metric_type": item.get("metric_type") or "builtin",
                        "direction": item.get("metric_direction") or run.validation_metric_direction,
                        "answer_fields": list(item.get("answer_fields") or []),
                        "evaluation_config": {**dict(item.get("evaluation_config") or {}),
                            **({"prediction_column": item["prediction_column"]} if item.get("prediction_column") else {})},
                        "files": files})
    if not members:
        raise ValueError("mid-training evaluation requires prepared Validation members")
    if len({m["direction"] for m in members}) != 1:
        raise ValueError("mid-training suite members must share a metric direction")
    contract = {"first_validation_fraction": fraction, "split": "validation", "direction": members[0]["direction"], "members": members, "helpers": helpers}
    path = root / "contract.json"
    path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    contracts = dict((run.lifecycle or {}).get("adaptive_validation_contracts") or {})
    contracts[ticket.id] = digest
    run.lifecycle = {**(run.lifecycle or {}), "adaptive_validation_contracts": contracts}
    await session.flush()
    return {"path": str(path), "sha256": digest}


def validate_request(request, ticket):
    """Reject malformed or cross-ticket telemetry before waking an agent."""
    if request.get("ticket_id") != ticket.id or ticket.agent_id != "train" or ticket.lane != "optimization":
        raise ValueError("adaptive feedback must belong to the optimization Train")
    request_id = str(request.get("request_id", ""))
    if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
        raise ValueError("invalid adaptive request id")
    path = PurePosixPath(str(request.get("decision_path", "")))
    if (not path.is_absolute() or ".." in path.parts or ticket.id not in path.parts
            or path.name != f"decision-{request_id}.json"):
        raise ValueError("decision path must belong to this Train ticket")
    if request.get("direction") not in {"min", "max"}:
        raise ValueError("invalid Validation metric direction")
    step, target = request.get("step"), request.get("planned_steps")
    if type(step) is not int or type(target) is not int or not 0 < step < target:
        raise ValueError("invalid checkpoint position")
    rows = request.get("measurements", [])
    if (not isinstance(rows, list) or not rows or len(rows) > 100
            or not all(isinstance(row, dict) for row in rows) or rows[-1].get("step") != step):
        raise ValueError("invalid Validation history")
    import math
    previous = 0
    for row in rows:
        score = row.get("score")
        if (type(row.get("step")) is not int or not previous < row["step"] <= step
                or isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score)):
            raise ValueError("invalid Validation measurement")
        previous = row["step"]


async def reconcile_adaptive_training(session):
    """Consume ordinary progress events for both Slurm and direct GPU workers."""
    events = (await session.execute(
        select(ExecutionEvent, Ticket).join(Ticket, Ticket.id == ExecutionEvent.ticket_id)
        .join(Run, Run.id == Ticket.run_id)
        .where(ExecutionEvent.phase == "adaptive_validation", Run.status == "running",
               Ticket.status.in_(("running", "waiting_external")))
        .order_by(ExecutionEvent.ts.desc()).limit(200)
    )).all()
    count = 0
    for event, ticket in events:
        run = (await session.execute(select(Run).where(Run.id == ticket.run_id).with_for_update().execution_options(populate_existing=True))).scalar_one()
        if not (run.lifecycle or {}).get("adaptive_mid_training_evaluation"):
            continue
        raw_request = (event.extras or {}).get("adaptive_request")
        if not isinstance(raw_request, dict):
            continue
        request = dict(raw_request)
        try:
            validate_request(request, ticket)
        except (ValueError, TypeError):
            continue
        requests = dict((run.lifecycle or {}).get("adaptive_training_requests") or {})
        key = request["request_id"]
        if key in requests:
            continue
        if any(row.get("ticket_id") == ticket.id and row.get("step", 0) > request["step"]
               for row in requests.values()):
            continue
        # Match the engine-built frozen contract, not a worker-selected evaluation.
        contracts = (run.lifecycle or {}).get("adaptive_validation_contracts", {})
        if request.get("contract_sha256") != contracts.get(ticket.id):
            continue
        requests[key] = {**request, "iteration": ticket.iteration, "status": "pending"}
        run.lifecycle = {**(run.lifecycle or {}), "adaptive_training_requests": requests}
        if run.supervisor_ticket_id:
            from zevo.engine.run.wakeup import queue_wakeup
            await queue_wakeup(session, agent_id="orchestrator", ticket_id=run.supervisor_ticket_id,
                               source="on_demand", reason=f"mid-training Validation decision {key}",
                               payload={"adaptive_training_request_id": key})
        else:
            await session.commit()
        count += 1
    return count


async def decide_adaptive_training(session, run_id, ticket_id, body):
    from fastapi import HTTPException
    from zevo.engine.run.remote_jobs import _device_info_for_ticket, _ssh
    run = (await session.execute(select(Run).where(Run.id == run_id).with_for_update().execution_options(populate_existing=True))).scalar_one_or_none()
    ticket = await session.get(Ticket, ticket_id)
    if run is None or ticket is None or ticket.run_id != run_id:
        raise HTTPException(404, "Run or Train ticket not found")
    if run.status != "running" or not (run.lifecycle or {}).get("adaptive_mid_training_evaluation"):
        raise HTTPException(409, "adaptive training is not active")
    if ticket.status not in {"running", "waiting_external"}:
        raise HTTPException(409, "Train is no longer active")
    from zevo.engine.run.steering import instruction_gate_active
    if await instruction_gate_active(session, run_id):
        raise HTTPException(409, "resolve the user's pause/instruction before resuming training")
    requests = dict((run.lifecycle or {}).get("adaptive_training_requests") or {})
    request = requests.get(body.request_id)
    if not request or request.get("ticket_id") != ticket_id:
        raise HTTPException(404, "checkpoint request not found")
    if any(row.get("ticket_id") == ticket_id and row.get("step", 0) > request["step"]
           for row in requests.values()):
        raise HTTPException(409, "a newer checkpoint has superseded this request")
    from zevo.paths import REPO_ROOT
    import runpy
    helper = runpy.run_path(str(REPO_ROOT / "playbook/runners/adaptive_training.py"))
    decision = body.model_dump()
    try:
        validate_request(request, ticket)
        helper["validate_decision"](decision, request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if request.get("status") == "decided":
        if request.get("decision") != decision:
            raise HTTPException(409, "checkpoint already has a different decision")
        return {"status": "decided", "decision": decision}
    # Bound path + argv quoting; no credentials or general remote shell in the request.
    info = await _device_info_for_ticket(session, ticket)
    if not info:
        raise HTTPException(409, "Train device route unavailable")
    program = (
        "import json,os;from pathlib import Path;"
        f"p=Path({request['decision_path']!r});v={decision!r};"
        "old=json.loads(p.read_text()) if p.exists() else None;"
        "assert old is None or old==v, 'conflicting decision';"
        "q=p.with_suffix('.tmp');q.write_text(json.dumps(v));os.replace(q,p);print('written')"
    )
    encoded = base64.b64encode(program.encode()).decode()
    code = f"import base64;exec(base64.b64decode({encoded!r}))"
    response = await _ssh(info, "python3 -c " + shlex.quote(code), timeout_seconds=40)
    if not response.get("ok"):
        raise HTTPException(503, "could not deliver decision; checkpoint remains resumable")
    # Attribute the accepted decision to the actual live activation. Manual
    # submissions without an unambiguous activation stay unattributed.
    from zevo.db import HeartbeatRun
    live = (await session.execute(select(HeartbeatRun).where(
        HeartbeatRun.ticket_id == run.supervisor_ticket_id,
        HeartbeatRun.finished_at.is_(None),
    ))).scalars().all()
    matches = [h for h in live if (h.resolved_config or {}).get("mid_training", {}).get("request_id") == body.request_id]
    if not matches and len(live) == 1 and live[0].activation_phase != "instruction":
        matches = live
    requests[body.request_id] = {**request, "status": "decided", "decision": decision,
                               "decision_heartbeat_id": matches[0].id if len(matches) == 1 else ""}
    run.lifecycle = {**(run.lifecycle or {}), "adaptive_training_requests": requests}
    await session.commit()
    return {"status": "decided", "decision": decision}


def validation_summary(ticket, lifecycle, events):
    """Project this Train's Validation telemetry and server-approved decisions."""
    if ticket.agent_id != "train" or ticket.lane != "optimization":
        return None
    digest = lifecycle.get("adaptive_validation_contracts", {}).get(ticket.id)
    if not digest:
        return None
    requests = [r for r in lifecycle.get("adaptive_training_requests", {}).values()
                if r.get("ticket_id") == ticket.id and r.get("contract_sha256") == digest]
    snapshots = list(requests)
    final_report = lifecycle.get("adaptive_training_reports", {}).get(ticket.id)
    if isinstance(final_report, dict) and final_report.get("contract_sha256") == digest:
        snapshots.append(final_report)
    for event in events:
        for key in ("adaptive_request", "adaptive_report"):
            row = (event.extras or {}).get(key)
            if (isinstance(row, dict) and row.get("ticket_id") == ticket.id
                    and row.get("contract_sha256") == digest):
                snapshots.append(row)
    import math
    valid = []
    for row in snapshots:
        measures, total = row.get("measurements"), row.get("planned_steps")
        if (not isinstance(measures, list) or not measures or type(total) is not int or total <= 0
                or row.get("direction") not in ("min", "max")):
            continue
        previous = 0
        for m in measures:
            if (not isinstance(m, dict) or type(m.get("step")) is not int
                    or not previous < m["step"] <= total
                    or type(m.get("score")) not in (int, float) or not math.isfinite(m["score"])
                    or not isinstance(m.get("metrics"), dict)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in m["metrics"].values())):
                break
            previous = m["step"]
        else:
            valid.append(row)
    if not valid:
        return None
    latest = max(valid, key=lambda r: (r["measurements"][-1]["step"], bool(r.get("complete"))))
    decisions = {r["step"]: r["decision"] for r in requests
                 if r.get("status") == "decided" and isinstance(r.get("decision"), dict)}
    best = (max if latest["direction"] == "max" else min)(latest["measurements"], key=lambda m: m["score"])
    complete = bool(latest.get("complete"))
    rows = []
    for m in latest["measurements"]:
        decision = decisions.get(m["step"])
        rows.append({"step": m["step"], "score": m["score"], "metrics": m["metrics"],
                     "decision": decision,
                     "plan_completed": complete and m["step"] == latest["planned_steps"]})
    last_decision = decisions.get(rows[-1]["step"])
    return {"planned_steps": latest["planned_steps"], "direction": latest["direction"],
            "measurements": rows, "best_step": best["step"], "complete": complete,
            "next_step": last_decision.get("next_step") if last_decision and last_decision.get("action") == "continue" and not complete else None,
            "awaiting_decision": not complete and not last_decision and ticket.status in ("running", "waiting_external")}
