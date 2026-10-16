"""Exercise segmented recovery, live decisions and final checkpoint selection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location("adaptive_helper", ROOT / "playbook/runners/adaptive_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_checkpoint(root, step, total):
    root.mkdir(parents=True, exist_ok=True)
    (root / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": total}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (root / name).write_bytes(b"state")
    (root / "config.json").write_text("{}")


def test_loss_only_or_test_measurements_cannot_drive_decisions(helper):
    contract = {"members": [{"name": "math"}]}
    for row in [
        {"split": "test", "step": 20, "contract_sha256": "abc", "metrics": {"math": 1}},
        {"split": "validation", "step": 20, "contract_sha256": "abc", "metrics": {"eval_loss": 1}},
        {"split": "validation", "step": 20, "contract_sha256": "abc", "metrics": {"math": float("nan")}},
    ]:
        with pytest.raises(ValueError):
            helper.validate_measurement(row, contract, "abc", 20)


def test_single_flat_observation_cannot_stop_and_steps_must_advance(helper):
    request = {"request_id": "r", "step": 20, "planned_steps": 100, "measurements": [{"step": 10}, {"step": 20}]}
    with pytest.raises(ValueError, match="three"):
        helper.validate_decision({"request_id": "r", "action": "stop", "reason": "flat"}, request)
    for step in (20, 101):
        with pytest.raises(ValueError, match="advance"):
            helper.validate_decision({"request_id": "r", "action": "continue", "next_step": step, "reason": "continue"}, request)


def test_weights_only_checkpoint_is_not_resumable(helper, tmp_path):
    write_checkpoint(tmp_path, 20, 100)
    (tmp_path / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="optimizer"):
        helper.verify_resume_checkpoint(tmp_path, step=20, planned_steps=100)


def test_live_decisions_stop_and_keep_best_checkpoint(helper, tmp_path, monkeypatch):
    contract = {"first_validation_fraction": 0.2, "split": "validation", "direction": "max", "members": [{"name": "math", "files": {}}]}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    root = tmp_path / "train-1"
    calls = []
    scores = {20: .6, 40: .59, 60: .58}
    def command(argv, *, env, check):
        step = int(env["ZEVO_ADAPTIVE_TARGET_STEP"])
        calls.append((argv[0], step, env["ZEVO_ADAPTIVE_RESUME_CHECKPOINT"]))
        if argv == ["train"]:
            write_checkpoint(Path(env["ZEVO_ADAPTIVE_CHECKPOINT"]), step, 100)
        else:
            Path(env["ZEVO_ADAPTIVE_METRICS_PATH"]).write_text(json.dumps({
                "split": "validation", "contract_sha256": digest, "step": step, "metrics": {"math": scores[step]}}))
    def decide(_):
        ledger = json.loads((root / "adaptive-training.json").read_text())
        request = ledger["pending"]
        decision = {"request_id": request["request_id"], "reason": "repeated plateau",
                    "action": "stop" if request["step"] == 60 else "continue",
                    "next_step": None if request["step"] == 60 else request["step"] + 20}
        helper.atomic_json(request["decision_path"], decision)
    monkeypatch.setattr(helper.subprocess, "run", command)
    monkeypatch.setattr(helper, "score_validation_predictions", lambda *a, **kw: {
        "split": "validation", "contract_sha256": digest, "step": kw["step"],
        "metrics": {"math": scores[kw["step"]]}})
    monkeypatch.setattr(helper.time, "sleep", decide)
    arguments = dict(ticket_id="train-1", contract_path=contract_path, contract_sha256=digest,
                     planned_steps=100, first_step=20, train_command=["train"], evaluate_command=["eval"], work_dir=root)
    report = helper.run_adaptive_training(**arguments)
    assert report["reason"] == "adaptive_early_stopping"
    assert report["selected_step"] == 20
    assert report["completed_steps"] == 60
    assert [row[:2] for row in calls] == [("train", 20), ("eval", 20), ("train", 40), ("eval", 40), ("train", 60), ("eval", 60)]
    assert calls[2][2] == str(root / "checkpoint-20")
    assert helper.run_adaptive_training(**arguments) == report
    assert len(calls) == 6  # Completed invocation is idempotent.


def test_evaluation_failure_preserves_segment_without_negative_score(helper, tmp_path, monkeypatch):
    contract = {"first_validation_fraction": 0.99, "split": "validation", "direction": "max", "members": [{"name": "math", "files": {}}]}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    root = tmp_path / "train-1"
    trained = []
    failed = [False]
    def command(argv, *, env, check):
        if argv == ["train"]:
            trained.append(20)
            write_checkpoint(Path(env["ZEVO_ADAPTIVE_CHECKPOINT"]), 20, 20)
        elif not failed[0]:
            failed[0] = True
            raise RuntimeError("GPU unavailable")
        else:
            Path(env["ZEVO_ADAPTIVE_METRICS_PATH"]).write_text(json.dumps({
                "split": "validation", "contract_sha256": digest, "step": 20, "metrics": {"math": .5}}))
    monkeypatch.setattr(helper.subprocess, "run", command)
    monkeypatch.setattr(helper, "score_validation_predictions", lambda *a, **kw: {
        "split": "validation", "contract_sha256": digest, "step": kw["step"], "metrics": {"math": .5}})
    args = dict(ticket_id="train-1", contract_path=path, contract_sha256=digest, planned_steps=20,
                first_step=20, train_command=["train"], evaluate_command=["eval"], work_dir=root)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        helper.run_adaptive_training(**args)
    assert json.loads((root / "adaptive-training.json").read_text())["measurements"] == []
    assert helper.run_adaptive_training(**args)["reason"] == "plan_completed"
    assert trained == [20]


def test_adaptive_completion_validates_retained_best(helper, tmp_path, monkeypatch):
    import runpy
    checkpoint = runpy.run_path(str(ROOT / "playbook/runners/train_checkpoint.py"))
    plan = {"ticket_id": "train-1", "planned_steps": 100, "adaptive_evaluation": True}
    state = {"global_step": 60, "max_steps": 100}
    report = {"ticket_id": "train-1", "planned_steps": 100, "complete": True,
              "completed_steps": 60, "direction": "max", "selected_step": 20, "selected_checkpoint": "c20",
              "measurements": [{"step": s, "score": score, "checkpoint": f"c{s}"} for s, score in [(20, .6), (40, .5), (60, .4)]],
              "decisions": [{"action": "stop", "reason": "plateau"}]}
    assert checkpoint["_validate_training_progress"](plan, state, report) == "adaptive_early_stopping"
    report["selected_step"] = 60
    with pytest.raises(ValueError, match="best"):
        checkpoint["_validate_training_progress"](plan, state, report)


@pytest.mark.asyncio
async def test_progress_wakes_orchestrator_and_decision_is_delivered(tmp_path, monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import select
    from zevo.db import Base, Run, Ticket, HeartbeatRun, ExecutionEvent, AgentWakeupRequest
    from zevo.engine.run.adaptive_training import reconcile_adaptive_training, decide_adaptive_training
    from zevo.api.routers.shared.runs import AdaptiveTrainingDecision
    from zevo.engine.run import remote_jobs
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    request_id = "a" * 32
    path = tmp_path / "train-1" / f"decision-{request_id}.json"
    path.parent.mkdir()
    request = {"request_id": request_id, "ticket_id": "train-1", "step": 20, "planned_steps": 100,
               "decision_path": str(path), "contract_sha256": "frozen", "direction": "max",
               "measurements": [{"step": 20, "score": .6, "metrics": {"math": .6}, "checkpoint": "/train-1/checkpoint-20"}]}
    async with Session() as db:
        db.add(Run(id="r", task_name="t", metric="accuracy", status="running", supervisor_ticket_id="orch-1",
                   lifecycle={"adaptive_mid_training_evaluation": True,
                              "adaptive_validation_contracts": {"train-1": "frozen"}}))
        db.add_all([Ticket(id="train-1", run_id="r", agent_id="train", status="waiting_external", iteration=1),
                    Ticket(id="orch-1", run_id="r", agent_id="orchestrator", status="succeeded")])
        db.add(HeartbeatRun(id="hb", ticket_id="train-1", agent_id="train", driver="stub", model="stub"))
        await db.commit()
        db.add(ExecutionEvent(ticket_id="train-1", heartbeat_id="hb", attempt_id="attempt",
                              event_type="progress", phase="adaptive_validation", current_step=20,
                              extras={"adaptive_request": request}))
        await db.commit()
        assert await reconcile_adaptive_training(db) == 1
        assert await reconcile_adaptive_training(db) == 0
        wakes = (await db.execute(select(AgentWakeupRequest))).scalars().all()
        assert len(wakes) == 1
        assert wakes[0].ticket_id == "orch-1"
        async def route(*args):
            return object()
        async def execute(info, command, **kwargs):
            import shlex, subprocess
            argv = shlex.split(command)
            result = subprocess.run([sys.executable, *argv[1:]], text=True, capture_output=True)
            return {"ok": result.returncode == 0, "stdout": result.stdout, "error": result.stderr}
        monkeypatch.setattr(remote_jobs, "_device_info_for_ticket", route)
        monkeypatch.setattr(remote_jobs, "_ssh", execute)
        decision = AdaptiveTrainingDecision(request_id=request_id, action="continue", next_step=40, reason="check progress")
        result = await decide_adaptive_training(db, "r", "train-1", decision)
        assert result["status"] == "decided"
        assert json.loads(path.read_text()) == decision.model_dump()
        assert await decide_adaptive_training(db, "r", "train-1", decision) == result
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as conflict:
            await decide_adaptive_training(db, "r", "train-1",
                decision.model_copy(update={"next_step": 60}))
        assert conflict.value.status_code == 409
        # The persisted decision must match the evidence in the final receipt.
        from zevo.engine.run.training_completion import _verify_adaptive_decisions
        completion = {"adaptive_report": {"contract_sha256": "frozen",
                       "measurements": request["measurements"], "decisions": [decision.model_dump()]}}
        await _verify_adaptive_decisions(db, await db.get(Ticket, "train-1"), completion)
        completion["adaptive_report"]["decisions"][0]["next_step"] = 70
        with pytest.raises(ValueError, match="not approved"):
            await _verify_adaptive_decisions(db, await db.get(Ticket, "train-1"), completion)
    await engine.dispose()


def test_request_rejects_other_ticket_paths_and_test_lane():
    from types import SimpleNamespace
    from zevo.engine.run.adaptive_training import validate_request
    ticket = SimpleNamespace(id="train-1", agent_id="train", lane="optimization")
    request = {"ticket_id": "train-1", "request_id": "a" * 32,
               "decision_path": "/train-other/decision-" + "a" * 32 + ".json"}
    with pytest.raises(ValueError, match="path"):
        validate_request(request, ticket)
    ticket.lane = "held_out_test"
    with pytest.raises(ValueError, match="optimization"):
        validate_request(request, ticket)


def test_system_scores_predictions_with_frozen_builtin_and_checks_alignment(helper, tmp_path):
    import shutil
    source = ROOT / "src/zevo/engine/method/eval_metrics.py"
    shutil.copyfile(source, tmp_path / "metrics.py")
    (tmp_path / "gold.csv").write_text("id,answer\n1,A\n2,B\n")
    (tmp_path / "sample.csv").write_text("id,prediction\n")
    prediction_dir = tmp_path / "predictions"
    (prediction_dir / "000").mkdir(parents=True)
    prediction = prediction_dir / "000/predictions.csv"
    prediction.write_text("id,prediction\n1,A\n2,C\n")
    contract = {"split": "validation", "direction": "max",
                "helpers": {"metrics": {"path": "metrics.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}},
                "members": [{"name": "math", "metric": "accuracy", "answer_fields": ["answer"],
                             "files": {"labels": {"path": "gold.csv"}, "sample_submission": {"path": "sample.csv"}}}]}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    result = helper.score_validation_predictions(path, predictions_dir=prediction_dir, step=20, contract_sha256="frozen")
    assert result["metrics"] == {"math": .5}
    prediction.write_text("id,prediction\n2,B\n1,A\n")
    with pytest.raises(ValueError, match="identity"):
        helper.score_validation_predictions(path, predictions_dir=prediction_dir, step=20, contract_sha256="frozen")


@pytest.mark.asyncio
async def test_contract_copies_validation_only_and_default_off_is_inert(tmp_path):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from zevo.db import Base, Run, Ticket, WorkProduct
    from zevo.engine.run.adaptive_training import build_validation_contract
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    files = {}
    for role, suffix in [("public", ".csv"), ("validation_set", ".csv"), ("sample_submission", ".csv"),
                         ("inference_config", ".yaml"), ("script", ".py")]:
        file = tmp_path / (role + suffix)
        file.write_text("frozen " + role)
        files[role] = str(file)
    async with Session() as db:
        run = Run(id="r", task_name="t", metric="accuracy", validation_metric="accuracy", status="running",
                  holdout={"validation_sets": [{"name": "math", "metric_type": "builtin",
                           "answer_fields": ["answer"], **files}],
                           "test_sets": [{"name": "private", "public": "/must/not/read/test.csv"}]})
        train = Ticket(id="train-1", run_id="r", agent_id="train", status="queued",
                       inputs={"inference_config": {"path": files["inference_config"]}},
                       payload={"base_model": "model", "first_validation_fraction": 0.2})
        baseline = Ticket(id="infer-0", run_id="r", agent_id="inference", status="succeeded",
                          payload={"base_model": "model", "model_source": "base_model"})
        db.add_all([run, train, baseline])
        await db.commit()
        assert await build_validation_contract(db, run, train, str(tmp_path / "off")) == {}
        assert not (tmp_path / "off").exists()
        run.lifecycle = {"adaptive_mid_training_evaluation": True}
        db.add(WorkProduct(ticket_id=baseline.id, role="script", path=files["script"]))
        await db.commit()
        result = await build_validation_contract(db, run, train, str(tmp_path / "train-1"))
        contract = json.loads(Path(result["path"]).read_text())
        assert contract["split"] == "validation"
        assert [m["name"] for m in contract["members"]] == ["math"]
        assert "private" not in json.dumps(contract)
        assert result["sha256"] == run.lifecycle["adaptive_validation_contracts"][train.id]
        for artifact in contract["members"][0]["files"].values():
            copied = Path(result["path"]).parent / artifact["path"]
            assert hashlib.sha256(copied.read_bytes()).hexdigest() == artifact["sha256"]
    await engine.dispose()


@pytest.mark.parametrize("layout", ["deepspeed", "fsdp"])
def test_sharded_resume_requires_optimizer_and_all_rank_rng(helper, tmp_path, layout):
    (tmp_path / "trainer_state.json").write_text(json.dumps({"global_step": 20, "max_steps": 100}))
    for rank in range(2):
        (tmp_path / f"rng_state_{rank}.pth").write_bytes(b"state")
    if layout == "deepspeed":
        (tmp_path / "latest").write_text("global_step20")
        (tmp_path / "global_step20").mkdir()
        (tmp_path / "global_step20/rank0_optim_states.pt").write_bytes(b"optimizer")
        (tmp_path / "global_step20/rank0_model_states.pt").write_bytes(b"model and scheduler")
    else:
        (tmp_path / "optimizer_0").mkdir()
        (tmp_path / "optimizer_0/0.distcp").write_bytes(b"optimizer")
        (tmp_path / "pytorch_model_fsdp_0").mkdir()
        (tmp_path / "pytorch_model_fsdp_0/0.distcp").write_bytes(b"model")
        (tmp_path / "scheduler.pt").write_bytes(b"scheduler")
    assert helper.verify_resume_checkpoint(tmp_path, step=20, planned_steps=100, world_size=2)["global_step"] == 20
    (tmp_path / "rng_state_1.pth").unlink()
    with pytest.raises(ValueError, match="RNG"):
        helper.verify_resume_checkpoint(tmp_path, step=20, planned_steps=100, world_size=2)


@pytest.mark.parametrize("fraction", [None, 0, 1, -0.2, float("nan")])
def test_first_boundary_must_be_supplied_by_orchestrator(helper, tmp_path, fraction):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"split": "validation", "direction": "max",
                               "members": [{"name": "math", "files": {}}],
                               "first_validation_fraction": fraction}))
    with pytest.raises(ValueError, match="Orchestrator"):
        helper.run_adaptive_training(ticket_id="train-1", contract_path=path,
            contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            planned_steps=100, first_step=20, train_command=["must-not-run"],
            evaluate_command=["must-not-run"], work_dir=tmp_path / "train-1")


def test_train_cannot_override_first_boundary(helper, tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"split": "validation", "direction": "max",
                               "members": [{"name": "math", "files": {}}],
                               "first_validation_fraction": 0.2}))
    with pytest.raises(ValueError, match="Orchestrator"):
        helper.run_adaptive_training(ticket_id="train-1", contract_path=path,
            contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            planned_steps=101, first_step=20, train_command=["must-not-run"],
            evaluate_command=["must-not-run"], work_dir=tmp_path / "train-1")


def test_pipeline_requires_initial_plan_only_when_enabled():
    from zevo.api.routers.shared.tickets import _stamp_pipeline_payload
    from zevo.db import Run
    run = Run(id="r", mode="full_pipeline", lifecycle={"adaptive_mid_training_evaluation": True})
    with pytest.raises(ValueError, match="first_validation_fraction"):
        _stamp_pipeline_payload(run=run, agent_id="train", payload={})
    assert _stamp_pipeline_payload(run=run, agent_id="train", payload={"first_validation_fraction": .2})["first_validation_fraction"] == .2
    run.lifecycle = {}
    _stamp_pipeline_payload(run=run, agent_id="train", payload={})


def test_validation_panel_projects_live_decisions_and_final_measurement():
    from types import SimpleNamespace
    from zevo.engine.run.adaptive_training import validation_summary
    ticket = SimpleNamespace(id="train-1", agent_id="train", lane="optimization", status="running")
    first = {"step": 20, "score": .6, "metrics": {"math": .6}}
    final = {"step": 100, "score": .5, "metrics": {"math": .5}}
    request = {"ticket_id": ticket.id, "contract_sha256": "hash", "step": 20,
               "planned_steps": 100, "direction": "max", "measurements": [first], "status": "pending"}
    lifecycle = {"adaptive_validation_contracts": {ticket.id: "hash"},
                 "adaptive_training_requests": {"req": request}}
    assert validation_summary(ticket, lifecycle, [])['awaiting_decision']
    request.update(status="decided", decision={"action": "continue", "next_step": 100, "reason": "Check full plan"})
    live = validation_summary(ticket, lifecycle, [])
    assert live['next_step'] == 100 and not live['awaiting_decision']
    report = {**request, "measurements": [first, final], "complete": True}
    events = [SimpleNamespace(extras={"adaptive_report": report})] * 3
    summary = validation_summary(ticket, lifecycle, events)
    assert len(summary['measurements']) == 2
    assert summary['best_step'] == 20 and summary['complete'] and summary['next_step'] is None
    assert summary['measurements'][-1]['plan_completed']
    assert summary['measurements'][0]['decision']['reason'] == "Check full plan"
    lifecycle['adaptive_training_reports'] = {ticket.id: report}
    assert validation_summary(ticket, lifecycle, []) == summary  # Missing final log marker.
    ticket.lane = "held_out_test"
    assert validation_summary(ticket, lifecycle, events) is None


def test_validation_panel_min_metric_stop_and_untrusted_history():
    from types import SimpleNamespace
    from zevo.engine.run.adaptive_training import validation_summary
    ticket = SimpleNamespace(id="t", agent_id="train", lane="optimization", status="succeeded")
    rows = [{"step": s, "score": v, "metrics": {"error": v}} for s, v in [(10, .4), (20, .3), (30, .5)]]
    report = {"ticket_id": "t", "contract_sha256": "h", "planned_steps": 100, "direction": "min",
              "measurements": rows, "complete": True}
    request = {**report, "step": 30, "status": "decided", "decision": {"action": "stop", "reason": "Plateau"}}
    life = {"adaptive_validation_contracts": {"t": "h"}, "adaptive_training_requests": {"r": request}}
    summary = validation_summary(ticket, life, [])
    assert summary['best_step'] == 20 and summary['measurements'][-1]['decision']['action'] == "stop"
    assert not summary['measurements'][-1]['plan_completed']
    life['adaptive_training_requests'] = {}
    for bad in [{**report, "contract_sha256": "other"}, {**report, "ticket_id": "other"},
                {**report, "measurements": [{"step": 10, "score": float('nan'), "metrics": {}}]}]:
        assert validation_summary(ticket, life, [SimpleNamespace(extras={"adaptive_report": bad})]) is None
