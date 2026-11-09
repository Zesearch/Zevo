from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run, Ticket, TicketMessage
from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS, MAX_UNCHANGED_FAILURES, classify_failure
from zevo.engine.run.runner import _apply_ticket_failure_policy
from zevo.contracts._base import AgentResult
from zevo.contracts.failure import FailureRecovery


def recovery(scope="fixed_configuration"):
    return FailureRecovery(
        scope=scope,
        reason="The next action needs a change outside the current work order.",
        attempted_actions=["Verified input integrity and reproduced the failure."],
        evidence=["workload.err: failure during state loading before the first step"],
        next_action="Review whether to change the fixed memory strategy or select another feasible action.",
    )


@pytest.mark.parametrize("scope", ["upstream_input", "resource_plan", "fixed_configuration", "run_decision"])
@pytest.mark.parametrize("agent", ["data", "train", "inference", "evaluation", "infrastructure", "registry"])
def test_structured_review_does_not_require_trigger_words(scope, agent):
    disposition = classify_failure("The operation could not complete.", agent_id=agent,
                                   recovery=recovery(scope))
    assert disposition.route == "orchestrator"
    assert disposition.code == "recovery_" + scope


def test_recovery_keeps_repairable_oom_local_and_respects_boundaries():
    local = recovery("current_task")
    assert classify_failure("CUDA out of memory", agent_id="train", recovery=local).route == "self"
    assert classify_failure("CUDA out of memory", agent_id="train", recovery=recovery()).route == "orchestrator"
    assert classify_failure("authentication failed", agent_id="train", recovery=recovery()).route == "orchestrator"
    assert classify_failure("OOM", agent_id="train", recovery=recovery(), cancelled=True).route == "terminal"
    assert classify_failure("upstream ticket is missing", agent_id="train", recovery=local).route == "self"


def test_recovery_contract_requires_actionable_evidence_and_failed_status():
    from pydantic import ValidationError
    base = dict(status="failed", ticket_id="stage", error_message="failed", notes="")
    assert AgentResult(**base).failure_recovery is None  # Existing results remain valid.
    for change in [{"evidence": []}, {"evidence": [" "]}, {"next_action": " "}, {"scope": "stop_run"}]:
        with pytest.raises(ValidationError):
            AgentResult(**base, failure_recovery={**recovery().model_dump(), **change})
    for status in ["succeeded", "degraded", "deferred"]:
        with pytest.raises(ValidationError, match="only valid for failed"):
            AgentResult(**{**base, "status": status, "error_message": ""}, failure_recovery=recovery())


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["data", "train", "inference", "evaluation", "infrastructure", "registry"])
async def test_review_request_hands_off_without_spending_another_repair(session, agent):
    session.add(Run(id="r", metric="score", status="running"))
    ticket = Ticket(id="stage", run_id="r", agent_id=agent, status="running", repair_attempts=7)
    session.add(ticket)
    await session.flush()
    result = AgentResult(status="failed", ticket_id=ticket.id, error_message="State load failed.",
                         notes="", failure_recovery=recovery())
    assert not await _apply_ticket_failure_policy(session, ticket, error_message=result.error_message, result=result)
    assert ticket.status == "failed" and ticket.repair_route == "orchestrator"
    assert ticket.repair_attempts == 7
    assert not (await session.scalars(select(TicketMessage))).all()


def test_train_failure_uses_current_error_not_historical_notes(tmp_path):
    from types import SimpleNamespace
    from zevo.contracts.train import TrainResult
    from zevo.engine.run.runner import _extract_summary_artifact_meta
    result = TrainResult(status="failed", ticket_id="stage", error_message="State load failed.",
                         notes="An earlier authentication failed; it was resolved.", failure_recovery=recovery())
    status, message, artifact, meta = _extract_summary_artifact_meta(
        result, inp=SimpleNamespace(), agent_id="train", work_dir=str(tmp_path),
    )
    assert (status, message, artifact, meta) == ("failed", result.error_message, "", {})
    assert classify_failure(message, agent_id="train", recovery=result.failure_recovery).route == "orchestrator"


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as value:
        yield value
    await engine.dispose()


def test_failure_ownership_is_deterministic() -> None:
    assert classify_failure(
        "ValidationError: malformed TrainResult", agent_id="train"
    ).route == "self"
    assert classify_failure(
        "inputs unresolvable: upstream ticket has no Work Product", agent_id="train"
    ).route == "self"
    assert classify_failure(
        "held-out test isolation violated", agent_id="inference", boundary="held_out_isolation"
    ).route == "terminal"
    assert classify_failure(
        "engine could not prepare frozen Validation artifacts: "
        "Validation sample submission must be an absolute path",
        agent_id="data",
    ).route == "self"


@pytest.mark.asyncio
async def test_deterministic_executor_requests_diagnosis_instead_of_blind_retry(session: AsyncSession) -> None:
    session.add(Run(
        id="r", task_name="t", agent_objective="t", metric="score", status="running",
    ))
    ticket = Ticket(
        id="eval-r-001", run_id="r", agent_id="evaluation", status="running",
        input_format="typed", lane="optimization", iteration=0,
    )
    session.add(ticket)
    await session.flush()
    retry = await _apply_ticket_failure_policy(
        session, ticket,
        error_message="Coding: pass_at_1 requires a registered code execution adapter",
        can_diagnose=False,
    )
    assert not retry
    assert ticket.status == "failed"
    assert ticket.repair_route == "orchestrator"
    assert not ticket.repair_attempts
    assert not (await session.execute(select(TicketMessage))).scalars().all()


@pytest.mark.asyncio
async def test_upstream_method_change_requires_a_structured_review_request(session: AsyncSession) -> None:
    session.add(Run(id="r", task_name="t", metric="score", status="running"))
    ticket = Ticket(id="train-r-013", run_id="r", agent_id="train", status="running")
    session.add(ticket)
    await session.flush()
    retry = await _apply_ticket_failure_policy(
        session, ticket,
        error_message="runtime experiment contract violated: configuration_yaml: "
        "upstream method branch conflict: sft -> rft requires transition evidence",
        result=AgentResult(status="failed", ticket_id=ticket.id, error_message="Method change needed",
                           notes="", failure_recovery=recovery("run_decision")),
    )
    assert not retry
    assert ticket.status == "failed"
    assert ticket.repair_route == "orchestrator"
    assert not ticket.repair_attempts
    assert not (await session.execute(select(TicketMessage))).scalars().all()


@pytest.mark.asyncio
async def test_twenty_repairs_stay_on_one_ticket_then_escalate(session: AsyncSession) -> None:
    session.add(Run(
        id="r", task_name="t", agent_objective="t", metric="score", status="running",
    ))
    ticket = Ticket(
        id="train-r-001", run_id="r", agent_id="train", status="running",
        input_format="typed", lane="optimization", iteration=1,
        payload={}, customization={}, inputs={},
    )
    session.add(ticket)
    await session.commit()

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        scheduled = await _apply_ticket_failure_policy(
            session, ticket, error_message=f"generated train_config schema mismatch {attempt}",
        )
        assert scheduled is True
        assert ticket.status == "repairing"
        assert ticket.repair_attempts == attempt

    scheduled = await _apply_ticket_failure_policy(
        session, ticket, error_message="generated train_config schema mismatch",
    )
    await session.commit()
    assert scheduled is False
    assert ticket.status == "failed"
    assert ticket.repair_route == "orchestrator"
    messages = (await session.execute(select(TicketMessage).where(
        TicketMessage.ticket_id == ticket.id,
    ).order_by(TicketMessage.created_at))).scalars().all()
    assert len(messages) == 20
    assert "Repair attempt 1/20" in messages[0].body
    assert "Repair attempt 20/20" in messages[-1].body


@pytest.mark.asyncio
async def test_terminal_failure_never_enters_repairing(session: AsyncSession) -> None:
    ticket = Ticket(id="infer-r-001", run_id="r", agent_id="inference", status="cancelled")
    scheduled = await _apply_ticket_failure_policy(
        session, ticket, error_message="cancelled by user", cancelled=True,
    )
    assert scheduled is False
    assert ticket.status == "cancelled"
    assert ticket.repair_attempts == 0
    assert ticket.repair_route == "terminal"


@pytest.mark.asyncio
async def test_engine_postprocessing_failure_returns_to_the_same_data_ticket(session):
    import json
    from zevo.db import TicketNotice
    session.add(Run(id="r", metric="accuracy"))
    ticket = Ticket(id="data-r-001", run_id="r", agent_id="data", status="running", repair_attempts=2)
    session.add(ticket)
    await session.flush()
    message = "engine could not prepare frozen Validation artifacts: ModuleNotFoundError: No module named 'pydantic'"
    assert await _apply_ticket_failure_policy(session, ticket, error_message=message,
                                             origin="engine_postprocessing")
    assert ticket.status == "repairing" and ticket.repair_attempts == 3
    assert ticket.repair_route == "self"
    repair = (await session.scalars(select(TicketMessage))).one()
    assert repair.ticket_id == "data-r-001" and message in repair.body
    assert "preserve the already prepared artifacts" in repair.body
    assert "Do not change engine code" in repair.body
    context = (await session.scalars(select(TicketNotice).where(TicketNotice.code == "failure.context"))).one()
    assert json.loads(context.body)["origin"] == "engine_postprocessing"


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['data', 'train', 'inference'])
async def test_failed_workload_enters_one_repair_without_collect(session, agent):
    from zevo.db import InfraInstance
    from zevo.engine.run.runner import _activate_failed_workload_repair
    ticket = Ticket(id='stage', run_id='r', agent_id=agent, status='queued')
    session.add_all([Run(id='r', metric='accuracy', status='running'), ticket])
    row = InfraInstance(id='job-row', run_id='r', ticket_id='stage', provider='cluster',
                        instance_id='123', status='failed', meta={
        'resource_request': True, 'scheduler_state': 'FAILED',
        'log_path': '/work/slurm-123.out', 'stderr_path': '/work/slurm-123.err',
    })
    session.add(row)
    await session.commit()
    assert await _activate_failed_workload_repair(session, ticket)
    assert ticket.status == 'repairing'
    assert ticket.repair_attempts == 1
    assert await _activate_failed_workload_repair(session, ticket)
    assert ticket.repair_attempts == 1  # A repeated watcher observation is not another repair.
    messages = (await session.execute(select(TicketMessage))).scalars().all()
    assert len(messages) == 1
    assert '/work/slurm-123.err' in messages[0].body


@pytest.mark.asyncio
@pytest.mark.parametrize('state,used,allowed,expected', [
    ('COMPLETED', 0, True, 'queued'),
    ('FAILED', 9, True, 'repairing'),
    ('FAILED', 10, True, 'repairing'),
    ('FAILED', 19, True, 'repairing'),
    ('FAILED', 20, False, 'failed'),
])
async def test_external_repair_preserves_collection_and_enforces_budget(session, state, used, allowed, expected):
    from zevo.db import InfraInstance
    from zevo.engine.run.runner import _activate_failed_workload_repair
    ticket = Ticket(id='stage', run_id='r', agent_id='inference', status='queued', repair_attempts=used)
    session.add_all([Run(id='r', metric='accuracy', status='running'), ticket,
        InfraInstance(id='job-row', run_id='r', ticket_id='stage', provider='cluster',
                      instance_id='123', status='released', meta={
                          'resource_request': True, 'scheduler_state': state,
                      })])
    await session.commit()
    assert await _activate_failed_workload_repair(session, ticket) is allowed
    assert ticket.status == expected
    assert ticket.repair_attempts == used + (1 if expected == "repairing" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["data", "train", "inference", "evaluation", "infrastructure", "registry"])
@pytest.mark.parametrize("lane,route", [("held_out_test", "terminal"), ("optimization", "orchestrator")])
async def test_unchanged_failures_stop_and_artifact_change_resets(session, tmp_path, agent, lane, route):
    from zevo.db import WorkProduct
    session.add(Run(id="r", task_name="t", metric="score", status="running"))
    ticket = Ticket(id="infer-r", run_id="r", agent_id=agent, lane=lane, status="running")
    artifact = tmp_path / "config.yaml"
    artifact.write_text("v1")
    session.add(ticket)
    session.add(WorkProduct(ticket_id=ticket.id, role="inference_config", path=str(artifact)))
    await session.commit()
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message="invalid calibration")
    artifact.write_text("v2")
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message="invalid calibration")
    assert not await _apply_ticket_failure_policy(session, ticket, error_message="invalid calibration")
    assert ticket.status == "failed" and ticket.repair_route == route
    assert ticket.repair_attempts == 2 * (MAX_UNCHANGED_FAILURES - 1)
    assert "no new execution" in ticket.summary


def test_execution_retry_permission_is_consumed_by_new_attempt():
    from zevo.engine.run.repair_progress import execution_retry_authorized
    meta = {'execution_attempt': 3, 'inference_execution_retry': {'execution_attempt': 3, 'reason': 'missing outputs'}}
    assert execution_retry_authorized(meta)
    assert not execution_retry_authorized({**meta, 'execution_attempt': 4})
    assert not execution_retry_authorized({'execution_attempt': 3})


@pytest.mark.asyncio
@pytest.mark.parametrize('agent,field', [('data', 'script_path'), ('evaluation', 'evaluation_script'), ('infrastructure', 'script_path'), ('registry', 'model_path')])
async def test_input_artifact_change_resets_without_work_products(session, tmp_path, agent, field):
    script = tmp_path / 'script.py'
    script.write_text('version 1')
    session.add(Run(id='r', metric='score', status='running'))
    ticket = Ticket(id='stage', run_id='r', agent_id=agent, lane='optimization', status='running',
                    payload={'suite_members': [{field: str(script)}]})
    session.add(ticket)
    await session.commit()
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message='same scorer failure')
    script.write_text('version 2')
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message='same scorer failure')
    assert not await _apply_ticket_failure_policy(session, ticket, error_message='same scorer failure')
    assert ticket.status == 'failed'
    assert ticket.repair_route == 'orchestrator'


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['data', 'train', 'inference', 'evaluation', 'infrastructure', 'registry', 'orchestrator'])
@pytest.mark.parametrize('lane', ['optimization', 'held_out_test'])
async def test_five_identical_failures_only_schedule_four_repairs(session, agent, lane):
    session.add(Run(id='r', metric='score', status='running'))
    ticket = Ticket(id='stage', run_id='r', agent_id=agent, lane=lane, status='running')
    session.add(ticket)
    await session.commit()
    for _ in range(4):
        assert await _apply_ticket_failure_policy(session, ticket, error_message='deterministic failure')
    assert not await _apply_ticket_failure_policy(session, ticket, error_message='deterministic failure')
    assert ticket.repair_attempts == 4
    assert ticket.repair_route == ('terminal' if lane == 'held_out_test' or agent == 'orchestrator' else 'orchestrator')
    messages = (await session.scalars(select(TicketMessage).where(TicketMessage.ticket_id == ticket.id))).all()
    assert len(messages) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['data', 'evaluation'])
@pytest.mark.parametrize('change', ['error', 'completed_job_error', 'execution'])
async def test_changed_error_or_execution_restarts_unchanged_count(session, agent, change):
    from zevo.db import InfraInstance
    session.add(Run(id='r', metric='score', status='running'))
    ticket = Ticket(id='stage', run_id='r', agent_id=agent, lane='optimization', status='running')
    job = InfraInstance(id='job', run_id='r', ticket_id='stage', provider='cluster',
                        instance_id='123', status='released', meta={
                            'scheduler_state': 'COMPLETED' if change == 'completed_job_error' else 'FAILED',
                            'execution_attempt': 1,
                        })
    session.add_all([ticket, job])
    await session.commit()
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message='first error')
    error = 'different error' if 'error' in change else 'first error'
    if change == 'execution':
        job.meta = {**job.meta, 'execution_attempt': 2}
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message=error)
    assert not await _apply_ticket_failure_policy(session, ticket, error_message=error)
    assert ticket.repair_route == 'orchestrator'
    assert ticket.repair_attempts == 2 * (MAX_UNCHANGED_FAILURES - 1)


@pytest.mark.asyncio
async def test_report_and_recovery_prose_changes_do_not_reset_progress(session, tmp_path):
    import json
    from zevo.contracts.train import TrainResult
    from zevo.db import InfraInstance, TicketNotice, WorkProduct
    session.add(Run(id="r", metric="score", status="running"))
    ticket = Ticket(id="stage", run_id="r", agent_id="train", lane="optimization", status="running")
    report = tmp_path / "adaptive-training-report.json"
    script = tmp_path / "train.py"
    script.write_text("print('unchanged implementation')")
    session.add_all([ticket,
        InfraInstance(id="job", run_id="r", ticket_id=ticket.id, provider="cluster",
                      instance_id="123", status="released", meta={"scheduler_state": "COMPLETED"}),
        WorkProduct(ticket_id=ticket.id, role="adaptive_training_report", path=str(report)),
    ])
    await session.flush()
    for attempt in range(MAX_UNCHANGED_FAILURES):
        report.write_text(json.dumps({"repair_observation": {"attempt": attempt}, "next_step": 100}))
        result = TrainResult(status="failed", ticket_id=ticket.id,
                             error_message=f"Different description {attempt}", notes=f"Report revised {attempt}",
                             failure_recovery=recovery("current_task"),
                             train_script_path=str(script), adaptive_training_report_path=str(report))
        result.failure_recovery.reason = f"Diagnostic explanation {attempt}"
        scheduled = await _apply_ticket_failure_policy(session, ticket,
            error_message=result.error_message, result=result)
        assert scheduled is (attempt < MAX_UNCHANGED_FAILURES - 1)
    assert ticket.repair_route == "orchestrator" and ticket.repair_attempts == MAX_UNCHANGED_FAILURES - 1
    notices = (await session.scalars(select(TicketNotice).where(
        TicketNotice.code == "repair.progress").order_by(TicketNotice.created_at))).all()
    assert [json.loads(n.body)["count"] for n in notices] == list(range(1, MAX_UNCHANGED_FAILURES + 1))


@pytest.mark.asyncio
async def test_current_result_artifacts_are_used_before_result_is_persisted(session, tmp_path):
    from zevo.contracts.train import TrainResult
    from zevo.db import HeartbeatResult
    session.add(Run(id="r", metric="score", status="running"))
    ticket = Ticket(id="stage", run_id="r", agent_id="train", lane="optimization", status="running")
    stale = tmp_path / "old.py"
    stale.write_text("old")
    script = tmp_path / "current.py"
    script.write_text("v1")
    session.add_all([ticket, HeartbeatResult(ticket_id=ticket.id, agent_id="train", status="failed",
                                            output={"train_script_path": str(stale)})])
    await session.flush()
    result = TrainResult(status="failed", ticket_id=ticket.id, error_message="same error", notes="",
                         train_script_path=str(script))
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message="same error", result=result)
    script.write_text("v2")
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message="same error", result=result)
    assert not await _apply_ticket_failure_policy(session, ticket, error_message="same error", result=result)
    assert ticket.repair_attempts == 2 * (MAX_UNCHANGED_FAILURES - 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["workload", "diagnostic", "unverified"])
async def test_only_measured_workload_progress_resets_stall_counter(session, source):
    from zevo.db import ExecutionEvent, HeartbeatRun, WorkloadExecution
    session.add(Run(id="r", metric="score", status="running"))
    ticket = Ticket(id="stage", run_id="r", agent_id="train", lane="optimization", status="running")
    session.add_all([ticket,
        HeartbeatRun(id="h", ticket_id=ticket.id, agent_id="train", driver="test", model="test"),
        WorkloadExecution(id="work", ticket_id=ticket.id, heartbeat_id="h", runtime_key="job", log_path="/job.out"),
    ])
    await session.flush()
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message="same error")
    session.add(ExecutionEvent(ticket_id=ticket.id, heartbeat_id="h", attempt_id="a",
        event_type="progress", phase="training", current_step=10, total_steps=100,
        extras={"execution_id": "unknown" if source == "unverified" else "work",
                "progress_scope": "diagnostic" if source == "diagnostic" else "workload"}))
    await session.flush()
    if source == "workload":
        for _ in range(MAX_UNCHANGED_FAILURES - 1):
            assert await _apply_ticket_failure_policy(session, ticket, error_message="same error")
    assert not await _apply_ticket_failure_policy(session, ticket, error_message="same error")
    assert ticket.repair_attempts == (MAX_UNCHANGED_FAILURES - 1) * (2 if source == "workload" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('allocated', [False, True])
async def test_infrastructure_only_real_allocation_resets_streak(session, allocated):
    from zevo.db import InfraInstance
    session.add(Run(id='r', metric='score', status='running'))
    ticket = Ticket(id='infra', run_id='r', agent_id='infrastructure', status='running')
    session.add(ticket)
    await session.flush()
    error = 'MaxSpotInstanceCountExceeded'
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message=error)
    session.add(InfraInstance(id='cloud', run_id='r', ticket_id=ticket.id, provider='cloud',
                             instance_id='i-real' if allocated else '',
                             status='ready' if allocated else 'failed'))
    await session.flush()
    if allocated:
        for _ in range(MAX_UNCHANGED_FAILURES - 1):
            assert await _apply_ticket_failure_policy(session, ticket, error_message=error)
    assert not await _apply_ticket_failure_policy(session, ticket, error_message=error)
    assert ticket.repair_route == 'orchestrator'
    assert ticket.repair_attempts == (MAX_UNCHANGED_FAILURES - 1) * (2 if allocated else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['succeeded', 'degraded', 'deferred'])
async def test_success_breaks_failure_streak_but_waiting_does_not(session, status):
    from zevo.db import HeartbeatResult
    session.add(Run(id='r', metric='score', status='running'))
    ticket = Ticket(id='stage', run_id='r', agent_id='infrastructure', status='running')
    session.add(ticket)
    await session.flush()
    for _ in range(MAX_UNCHANGED_FAILURES - 1):
        assert await _apply_ticket_failure_policy(session, ticket, error_message='same error')
    await session.flush()
    session.add(HeartbeatResult(ticket_id=ticket.id, agent_id=ticket.agent_id,
                                status=status, output={}))
    await session.flush()
    if status != 'deferred':
        for _ in range(MAX_UNCHANGED_FAILURES - 1):
            assert await _apply_ticket_failure_policy(session, ticket, error_message='same error')
    assert not await _apply_ticket_failure_policy(session, ticket, error_message='same error')
    assert ticket.repair_route == 'orchestrator'


@pytest.mark.parametrize("agent", ["data", "train", "inference", "evaluation", "infrastructure", "registry", "orchestrator"])
@pytest.mark.parametrize("message", [
    "engine could not prepare frozen Validation artifacts: missing pydantic",
    "unauthorized in an old log; the current failure is a missing local file",
    "budget exceeded is mentioned in the test fixture",
    "upstream ticket is missing", "GPU hardware defect: Xid 48",
    "sample_submission parser failed", "pass_at_1 requires a registered code execution adapter",
])
def test_error_prose_cannot_override_semantic_recovery(agent, message):
    assert classify_failure(message, agent_id=agent).route == "self"
    assert classify_failure(message, agent_id=agent, recovery=recovery("current_task")).route == "self"
    disposition = classify_failure(message, agent_id=agent, recovery=recovery("resource_plan"))
    assert disposition.route == ("terminal" if agent == "orchestrator" else "orchestrator")


@pytest.mark.parametrize("boundary", ["cancelled", "held_out_isolation", "budget"])
@pytest.mark.parametrize("scope", ["current_task", "resource_plan"])
def test_engine_boundaries_override_every_proposed_recovery(boundary, scope):
    assert classify_failure("any message", agent_id="data", boundary=boundary,
                            recovery=recovery(scope)).route == "terminal"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["agent_execution", "result_validation", "engine_postprocessing", "external_workload", "runner"])
@pytest.mark.parametrize("agent", ["data", "train", "inference", "evaluation", "infrastructure", "registry"])
async def test_all_failure_origins_share_the_twenty_repair_limit(session, origin, agent):
    session.add(Run(id="r", metric="accuracy"))
    ticket = Ticket(id="stage", run_id="r", agent_id=agent, status="running", repair_attempts=19)
    session.add(ticket)
    await session.flush()
    assert await _apply_ticket_failure_policy(session, ticket, error_message="diagnose before retry", origin=origin)
    assert ticket.repair_attempts == 20
    assert not await _apply_ticket_failure_policy(session, ticket, error_message="still failing", origin=origin)
    assert ticket.repair_attempts == 20 and ticket.repair_route == "orchestrator"
    assert len((await session.scalars(select(TicketMessage))).all()) == 1
