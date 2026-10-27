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
    if (run.lifecycle or {}).get("adaptive_validation_contracts", {}).get(ticket.id):
        path, _ = checkpoint_contract(run, ticket)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
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
    run.lifecycle = {**(run.lifecycle or {}), "adaptive_validation_contracts": contracts,
                     "adaptive_validation_contract_paths": {**(run.lifecycle or {}).get("adaptive_validation_contract_paths", {}), ticket.id: str(path.resolve())}}
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
    await reconcile_checkpoint_stages(session)
    return 0


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


def is_checkpoint_stage(ticket):
    return ticket.agent_id in {"inference", "evaluation"} and bool((ticket.payload or {}).get("mid_training_checkpoint"))


def is_checkpoint_evaluation(ticket):
    return ticket.agent_id == "evaluation" and is_checkpoint_stage(ticket)


def checkpoint_resume_authorized(run, ticket, job):
    if job is None or getattr(ticket, "agent_id", "") != "train":
        return False
    meta = job.meta or {}
    key = meta.get("checkpoint_resume")
    if not key or key != meta.get("checkpoint_yield"):
        return False
    stage = (run.lifecycle or {}).get("adaptive_checkpoints", {}).get(key, {})
    request = stage.get("request", {})
    decision = (run.lifecycle or {}).get("adaptive_training_requests", {}).get(key, {})
    return bool(stage.get("resume_queued") and stage.get("metrics_delivered")
        and stage.get("job_row_id") == job.id and request.get("ticket_id") == ticket.id
        and (request.get("step") == request.get("planned_steps") or decision.get("status") == "decided"))


def checkpoint_contract(run, train):
    path = Path((run.lifecycle or {}).get("adaptive_validation_contract_paths", {}).get(train.id, ""))
    digest = (run.lifecycle or {}).get("adaptive_validation_contracts", {}).get(train.id)
    if not digest or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("frozen checkpoint Validation contract unavailable or changed")
    contract = json.loads(path.read_text())
    for member in contract["members"]:
        for artifact in member["files"].values():
            file = (path.parent / artifact["path"]).resolve()
            if not file.is_relative_to(path.parent.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("frozen checkpoint Validation artifact changed")
    return path, contract


def validate_checkpoint_request(request, train, digest):
    import re
    if request.get("ticket_id") != train.id or train.agent_id != "train" or train.lane != "optimization":
        raise ValueError("checkpoint belongs to another Train")
    key = str(request.get("request_id", ""))
    if request.get("contract_sha256") != digest or not re.fullmatch(r"[0-9a-f]{32}", key):
        raise ValueError("invalid checkpoint identity")
    step, total = request.get("step"), request.get("planned_steps")
    if type(step) is not int or type(total) is not int or not 0 < step <= total:
        raise ValueError("invalid checkpoint step")
    checkpoint = PurePosixPath(str(request.get("checkpoint", "")))
    if (not checkpoint.is_absolute() or ".." in checkpoint.parts or train.id not in checkpoint.parts
            or checkpoint.name != f"checkpoint-{step}"):
        raise ValueError("checkpoint path outside Train")
    if request.get("metrics_path") != str(checkpoint.parent / f"validation-{step}.json") or request.get("decision_path") != str(checkpoint.parent / f"decision-{key}.json"):
        raise ValueError("checkpoint response path outside Train")
    if type(request.get("world_size")) is not int or request["world_size"] < 1:
        raise ValueError("invalid checkpoint world size")
    if not isinstance(request.get("measurements"), list) or not isinstance(request.get("decisions"), list):
        raise ValueError("invalid checkpoint history")


async def handle_checkpoint_yield(session, run, train, job, meta):
    """Intercept successful *segment* exit before ordinary final-model collect."""
    if train.agent_id != "train" or not (run.lifecycle or {}).get("adaptive_mid_training_evaluation"):
        return False
    if meta.get("checkpoint_yield"):
        return True
    from zevo.engine.run.remote_jobs import _device_info_for_ticket, _ssh
    from zevo.engine.run.wakeup import queue_wakeup
    from zevo.paths import REPO_ROOT
    events = (await session.execute(select(ExecutionEvent).where(
        ExecutionEvent.ticket_id == train.id, ExecutionEvent.phase == "adaptive_checkpoint")
        .order_by(ExecutionEvent.ts.desc()).limit(20))).scalars().all()
    event = next((e for e in events if (e.extras or {}).get("execution_id") == job.id
                  and (e.extras or {}).get("execution_purpose") == "workload"), None)
    if event is None:
        # EXITED can arrive before the telemetry stream drains its last marker.
        # Read the registered workload log before treating it as final Train.
        from zevo.engine.run.scheduler.reconciler import _slurm_telemetry_path
        from zevo.engine.observe.markers import scan_text
        log_path = _slurm_telemetry_path(meta)
        info = await _device_info_for_ticket(session, train)
        if not log_path or not info:
            raise ValueError("cannot verify adaptive Train exit without its workload log")
        response = await _ssh(info, "tail -n 2000 -- " + shlex.quote(log_path), timeout_seconds=40)
        if not response.get("ok"):
            raise ValueError("cannot verify adaptive Train exit; retry log collection")
        markers = [payload for kind, payload in scan_text(response.get("stdout", ""))
                   if kind == "progress" and payload.get("owner") == train.id
                   and payload.get("phase") == "adaptive_checkpoint"]
        if not markers:
            return False
        request = markers[-1].get("adaptive_checkpoint_request")
    else:
        request = (event.extras or {}).get("adaptive_checkpoint_request")
    if not isinstance(request, dict):
        raise ValueError("checkpoint handoff missing request")
    # Serialize handoff creation against decision/scoring callbacks.
    run = (await session.execute(select(Run).where(Run.id == run.id).with_for_update()
           .execution_options(populate_existing=True))).scalar_one()
    path, contract = checkpoint_contract(run, train)
    validate_checkpoint_request(request, train, hashlib.sha256(path.read_bytes()).hexdigest())
    stages = dict((run.lifecycle or {}).get("adaptive_checkpoints", {}))
    key = request["request_id"]
    previous = stages.get(key)
    if request.get("direction") != contract["direction"]:
        raise ValueError("checkpoint changed Validation direction")
    if previous and previous["request"] != request:
        raise ValueError("conflicting checkpoint handoff")
    if not previous:
        # Use the system verifier, never trust only the worker's claimed step.
        info = await _device_info_for_ticket(session, train)
        if not info:
            raise ValueError("checkpoint device route unavailable")
        program = (REPO_ROOT / "playbook/runners/adaptive_training.py").read_text() + (
            f"\nverify_resume_checkpoint({request['checkpoint']!r},step={request['step']},planned_steps={request['planned_steps']},world_size={request['world_size']})\n"
            f"r=json.loads((Path({request['checkpoint']!r}).parent/'adaptive-training.json').read_text())\n"
            f"assert r.get('pending_checkpoint')=={request!r}\nprint('checkpoint-verified')\n")
        code = "import base64;exec(base64.b64decode(" + repr(base64.b64encode(program.encode()).decode()) + "))"
        proof = await _ssh(info, "python3 -c " + shlex.quote(code), timeout_seconds=60)
        if not proof.get("ok") or "checkpoint-verified" not in proof.get("stdout", ""):
            raise ValueError("checkpoint state verification failed; segment remains uncollected")
        infer_id = f"mid-infer-{key}"
        primary = contract["members"][0]["files"]
        infer = Ticket(id=infer_id, run_id=run.id, agent_id="inference", lane="optimization", iteration=train.iteration,
            status="queued", payload={"operation": "run_inference", "model_source": "checkpoint",
                "base_model": train.payload["base_model"], "scoring_set": str(path.parent / primary["questions"]["path"]),
                "sample_submission": str(path.parent / primary["sample_submission"]["path"]),
                "test_set_name": "", "mid_training_checkpoint": request},
            inputs={"device_info": dict(train.inputs["device_info"]),
                    "checkpoint": {"path": request["checkpoint"]},
                    "inference_config": {"path": str(path.parent / primary["inference_config"]["path"])}},
            customization={}, summary=f"Checkpoint {request['step']} Validation inference")
        session.add(infer)
        stages[key] = {"request": request, "inference_ticket_id": infer_id, "evaluation_ticket_id": "",
                       "job_row_id": job.id, "metrics_delivered": False, "resume_queued": False}
        run.lifecycle = {**(run.lifecycle or {}), "adaptive_checkpoints": stages}
    train.status = "waiting_external"
    train.summary = f"Checkpoint {request['step']} saved; waiting for Inference and Evaluation"
    train.error_message = ""
    meta.pop("checkpoint_handoff_error", None)
    meta.update(checkpoint_yield=key, collect_wakeup_queued=True)
    job.meta = meta
    await session.flush()
    if not previous:
        await queue_wakeup(session, agent_id="inference", ticket_id=stages[key]["inference_ticket_id"],
                           source="on_demand", reason=f"Validate checkpoint {request['step']}")
    else:
        await session.commit()
    return True


async def build_checkpoint_inference_input(session, run, ticket, work_dir, generation_backend, specialist_context):
    from zevo.contracts.inference import InferenceSuiteMemberInput
    from zevo.engine.run.runner import _build_inference_input
    request = ticket.payload["mid_training_checkpoint"]
    train = await session.get(Ticket, request["ticket_id"])
    if train is None or train.run_id != run.id:
        raise ValueError("checkpoint Train unavailable")
    path, contract = checkpoint_contract(run, train)
    validate_checkpoint_request(request, train, hashlib.sha256(path.read_bytes()).hexdigest())
    members = []
    for index, member in enumerate(contract["members"]):
        files = member["files"]
        if index:
            members.append(InferenceSuiteMemberInput(name=member["name"], benchmark_id=f"validation:{index}",
                test_set_name=f"validation:{member['name']}", scoring_set=str(path.parent / files["questions"]["path"]),
                sample_submission=str(path.parent / files["sample_submission"]["path"]), configuration_mode="reuse",
                inference_config_path=str(path.parent / files["inference_config"]["path"]),
                predictions_validation_command=("python -m zevo.engine.artifact_validation validate-predictions "
                    "--predictions <absolute-predictions-csv-path> "
                    f"--questions {shlex.quote(str(path.parent / files['questions']['path']))} "
                    f"--sample-submission {shlex.quote(str(path.parent / files['sample_submission']['path']))}"),
                work_dir=str(Path(work_dir) / "suite" / f"{index:03d}")))
    inp = await _build_inference_input(ticket, ticket.payload, ticket.inputs, work_dir,
        generation_backend, run, session, specialist_context, checkpoint_members=members)
    return inp.model_copy(update={"suite_members": members,
        "reusable_predict_script_path": str(path.parent / contract["members"][0]["files"]["predict_script"]["path"]),
        "run_context": {**inp.run_context, "mid_training_checkpoint": request,
            "instruction": "Run frozen baseline inference on this resumable checkpoint only. For sharded ZeRO/FSDP, use its framework export to an inference-owned directory; preserve all original resume state. Do not train, score, change budgets, or publish it as a final model."}})


async def build_checkpoint_evaluation_input(session, run, ticket, work_dir):
    from zevo.contracts.evaluation import EvaluationTaskInput, EvaluationSuiteMemberInput
    request = ticket.payload["mid_training_checkpoint"]
    train = await session.get(Ticket, request["ticket_id"])
    if train is None or train.run_id != run.id:
        raise ValueError("checkpoint Train unavailable")
    path, contract = checkpoint_contract(run, train)
    stage = (run.lifecycle or {}).get("adaptive_checkpoints", {}).get(request["request_id"], {})
    infer = await session.get(Ticket, stage.get("inference_ticket_id", ""))
    if infer is None or infer.status != "succeeded":
        raise ValueError("checkpoint Inference has not succeeded")
    products = (await session.execute(select(WorkProduct).where(WorkProduct.ticket_id == infer.id,
        WorkProduct.role == "predictions").order_by(WorkProduct.created_at.desc()))).scalars().all()
    members = []
    for index, member in enumerate(contract["members"]):
        product = next((p for p in products if (p.meta or {}).get("suite_member_name") == member["name"]), None)
        if product is None and index == 0:
            product = next((p for p in products if (p.meta or {}).get("suite_primary") or not (p.meta or {}).get("suite_member_name")), None)
        if product is None or not Path(product.path).is_file():
            raise ValueError(f"checkpoint predictions missing for {member['name']}")
        files = member["files"]
        script = files.get("evaluation_script")
        members.append(EvaluationSuiteMemberInput(name=member["name"], benchmark_id=f"validation:{index}",
            predictions_path=product.path, scoring_set=str(path.parent / files["labels"]["path"]),
            sample_submission=str(path.parent / files["sample_submission"]["path"]), metric=member["metric"],
            answer_fields=member["answer_fields"], evaluation_config=member.get("evaluation_config", {}),
            evaluation_script=str(path.parent / script["path"]) if script else "",
            evaluator_sha256=script["sha256"] if script else ""))
    primary = members[0].model_dump()
    primary["test_set_name"] = primary.pop("name")
    return EvaluationTaskInput(ticket_id=ticket.id, **primary, suite_members=members[1:])


async def _queue_checkpoint_resume(session, run, train, stages, key):
    from zevo.db import InfraInstance
    from zevo.engine.run.wakeup import queue_wakeup
    stages = dict(stages)
    stage = stages[key]
    if stage.get("resume_queued"):
        return
    job = await session.get(InfraInstance, stage["job_row_id"])
    if not job or (job.meta or {}).get("checkpoint_yield") != key:
        raise ValueError("Train resume lost its checkpoint execution")
    job.meta = {**job.meta, "checkpoint_resume": key}
    train.status = "queued"
    train.summary = f"Resume checkpoint {stage['request']['step']} after Validation"
    stages[key] = {**stage, "resume_queued": True}
    run.lifecycle = {**run.lifecycle, "adaptive_checkpoints": stages}
    await session.flush()
    await queue_wakeup(session, agent_id="train", ticket_id=train.id, source="on_demand",
        reason=f"resume checkpoint {key}", payload={"adaptive_resume_request_id": key})


async def reconcile_checkpoint_stages(session):
    from zevo.engine.run.wakeup import queue_wakeup
    from zevo.engine.run.remote_jobs import _device_info_for_ticket, _ssh
    import runpy
    from zevo.paths import REPO_ROOT
    helper = runpy.run_path(str(REPO_ROOT / "playbook/runners/adaptive_training.py"))
    ids = (await session.execute(select(Run.id).where(Run.status == "running"))).scalars().all()
    for run_id in ids:
        run = (await session.execute(select(Run).where(Run.id == run_id).with_for_update()
            .execution_options(populate_existing=True))).scalar_one()
        stages = dict((run.lifecycle or {}).get("adaptive_checkpoints", {}))
        for key, stage in list(stages.items()):
            request = stage["request"]
            train = await session.get(Ticket, request["ticket_id"])
            if not train or train.status not in {"waiting_external", "running"} or stage.get("resume_queued"):
                continue
            from zevo.engine.run.steering import instruction_gate_active
            if await instruction_gate_active(session, run.id):
                continue
            path, contract = checkpoint_contract(run, train)
            infer = await session.get(Ticket, stage["inference_ticket_id"])
            if not infer or infer.status != "succeeded":
                continue
            eval_id = stage.get("evaluation_ticket_id") or f"mid-eval-{key}"
            evaluation = await session.get(Ticket, eval_id)
            if evaluation is None:
                evaluation = Ticket(id=eval_id, run_id=run.id, agent_id="evaluation", status="queued", lane="optimization",
                    iteration=train.iteration, inputs={}, customization={}, payload={"mid_training_checkpoint": request},
                    summary=f"Checkpoint {request['step']} Validation evaluation")
                session.add(evaluation)
                stage = {**stage, "evaluation_ticket_id": eval_id}
                stages[key] = stage
                run.lifecycle = {**run.lifecycle, "adaptive_checkpoints": stages}
                await session.flush()
                await queue_wakeup(session, agent_id="evaluation", ticket_id=eval_id, source="on_demand",
                                   reason=f"Evaluate checkpoint {request['step']}")
                continue
            if evaluation.status != "succeeded":
                continue
            if not stage.get("metrics_delivered"):
                products = (await session.execute(select(WorkProduct).where(WorkProduct.ticket_id == evaluation.id,
                    WorkProduct.role == "metrics").order_by(WorkProduct.created_at.desc()))).scalars().all()
                aggregate = next((p for p in products if not (p.meta or {}).get("suite_member_name")), None)
                if aggregate is None:
                    continue
                metrics = json.loads(Path(aggregate.path).read_text())
                values = metrics.get("components") or ({contract["members"][0]["name"]: metrics.get("score")} if len(contract["members"]) == 1 else {})
                measurement = {"request_id": key, "split": "validation", "step": request["step"],
                               "contract_sha256": request["contract_sha256"], "metrics": values}
                score = helper["validate_measurement"](measurement, contract, request["contract_sha256"], request["step"])
                info = await _device_info_for_ticket(session, train)
                if not info:
                    continue
                program = ("import json,os;from pathlib import Path;"
                    f"p=Path({request['metrics_path']!r});v={measurement!r};"
                    "old=json.loads(p.read_text()) if p.exists() else None;"
                    "assert old is None or old==v, 'conflicting checkpoint score';"
                    "q=p.with_suffix('.tmp');q.write_text(json.dumps(v));os.replace(q,p)")
                response = await _ssh(info, "python3 -c " + shlex.quote(program), timeout_seconds=40)
                if not response.get("ok"):
                    continue
                stage = {**stage, "metrics_delivered": True, "measurement": measurement}
                stages[key] = stage
                lifecycle = {**run.lifecycle, "adaptive_checkpoints": stages}
                if request["step"] < request["planned_steps"]:
                    decisions = dict(lifecycle.get("adaptive_training_requests", {}))
                    decisions[key] = {**request, "status": "pending", "iteration": train.iteration,
                        "measurements": [*request["measurements"], {"step": request["step"], "score": score,
                            "metrics": values, "checkpoint": request["checkpoint"]}]}
                    lifecycle["adaptive_training_requests"] = decisions
                run.lifecycle = lifecycle
                if request["step"] < request["planned_steps"] and run.supervisor_ticket_id:
                    await queue_wakeup(session, agent_id="orchestrator", ticket_id=run.supervisor_ticket_id,
                        source="on_demand", reason=f"mid-training Validation decision {key}",
                        payload={"adaptive_training_request_id": key})
                else:
                    await session.commit()
            decision = (run.lifecycle or {}).get("adaptive_training_requests", {}).get(key, {})
            if request["step"] == request["planned_steps"] or decision.get("status") == "decided":
                from zevo.db import HeartbeatRun
                live = (await session.execute(select(HeartbeatRun.id).where(
                    HeartbeatRun.ticket_id == run.supervisor_ticket_id,
                    HeartbeatRun.finished_at.is_(None)))).first()
                if live:
                    continue
                await _queue_checkpoint_resume(session, run, train, stages, key)
