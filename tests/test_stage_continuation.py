from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from tests.test_cloud_resource_recovery import setup
from zevo.db import AgentWakeupRequest, TicketMessage
from zevo.engine.run import runner, resource_recovery, stage_continuation


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['data', 'train', 'inference', 'evaluation', 'registry'])
@pytest.mark.parametrize('state,code', [('TIMEOUT', '0:0'), ('FAILED', '75:0'), ('DEADLINE', '0:0')])
async def test_normal_expiry_does_not_consume_repair_budget(setup, agent, state, code):
    s = setup
    s.ticket.agent_id = agent
    s.ticket.repair_attempts = 20
    s.ticket.status = 'waiting_external'
    s.job.meta = {**s.job.meta, 'scheduler_state': state, 'scheduler_exit_code': code,
                  'monitor_terminal': True}
    s.job.released_at = datetime.now(timezone.utc)
    await s.db.commit()
    assert await runner._activate_failed_workload_repair(s.db, s.ticket) is (agent == 'train')
    assert s.ticket.repair_attempts == 20 and not s.ticket.repair_route
    assert 'execution_repair' not in s.job.meta
    assert s.owner.released_at is None  # a process window does not destroy its host
    assert not (await s.db.scalars(select(TicketMessage))).all()
    wakes = (await s.db.scalars(select(AgentWakeupRequest))).all()
    if agent == 'train':
        assert not wakes and not stage_continuation.authorized(s.ticket, s.job)
    else:
        assert len(wakes) == 1 and wakes[0].source == 'stage_continuation'
        assert wakes[0].payload['continuation_resource_id'] == s.job.id
        assert stage_continuation.authorized(s.ticket, s.job)
        # A delivery after the first continuation was already consumed is inert.
        wakes[0].status = 'completed'
        await s.db.commit()
        assert not await runner._activate_failed_workload_repair(s.db, s.ticket)
        assert len((await s.db.scalars(select(AgentWakeupRequest))).all()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['data', 'inference', 'evaluation', 'registry'])
async def test_cluster_expiry_waits_for_confirmed_release(setup, monkeypatch, agent):
    s = setup
    s.ticket.agent_id = agent
    s.ticket.repair_attempts = 20
    s.run.gpu_provider = s.job.provider = s.owner.provider = 'cluster'
    s.job.meta = {**s.job.meta, 'scheduler_state': 'TIMEOUT'}
    await s.db.commit()
    assert not await runner._activate_failed_workload_repair(s.db, s.ticket)
    event = s.run.lifecycle[resource_recovery.KEY][s.ticket.id]
    assert event['kind'] == 'continuation' and event['owner_id'] == s.owner.id
    stop = AsyncMock(return_value=False)
    monkeypatch.setattr(resource_recovery, '_stop_owner', stop)
    await resource_recovery._advance(s.db, s.run, s.ticket, event)
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()
    stop.return_value = True
    await resource_recovery._advance(s.db, s.run, s.ticket, event)
    assert s.ticket.status == 'queued' and s.ticket.repair_attempts == 20
    assert len((await s.db.scalars(select(AgentWakeupRequest))).all()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('state,code', [('FAILED', '1:0'), ('OUT_OF_MEMORY', '1:0'), ('NODE_FAIL', '0:0'), ('CANCELLED', '0:15')])
async def test_failures_are_not_reclassified_as_normal_expiry(setup, state, code):
    s = setup
    s.job.meta = {**s.job.meta, 'scheduler_state': state, 'scheduler_exit_code': code}
    assert not stage_continuation.expired(s.ticket, s.job)
    await runner._activate_failed_workload_repair(s.db, s.ticket)
    assert not stage_continuation.authorized(s.ticket, s.job)
    assert s.ticket.repair_route


@pytest.mark.asyncio
async def test_expiry_cannot_authorize_foreign_or_superseded_execution(setup):
    s = setup
    s.job.meta = {**s.job.meta, 'scheduler_state': 'TIMEOUT'}
    s.job.ticket_id = 'other'
    assert not stage_continuation.expired(s.ticket, s.job)
    s.job.ticket_id = s.ticket.id
    s.job.meta = {**s.job.meta, 'superseded_by_instruction_id': 'restart'}
    assert not stage_continuation.expired(s.ticket, s.job)
