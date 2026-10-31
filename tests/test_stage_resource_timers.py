from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from zevo.engine.run.scheduler.reconciler import _mark_slurm_job_running


def test_reused_allocation_does_not_become_stage_start():
    now = datetime.now(timezone.utc)
    allocation = now - timedelta(hours=12)
    row = SimpleNamespace(status='ready', ready_at=allocation)
    ticket = SimpleNamespace(status='running')
    meta = {'running_announced': True, 'allocation_started_at': allocation.isoformat()}
    _mark_slurm_job_running(session=None, row=row, ticket=ticket, meta=meta, observed_at=now)
    assert meta['stage_started_at'] == now.isoformat()
    assert meta['allocation_started_at'] == allocation.isoformat()
    _mark_slurm_job_running(session=None, row=row, ticket=ticket, meta=meta, observed_at=now+timedelta(minutes=2))
    assert meta['stage_started_at'] == now.isoformat()
    assert row.ready_at == allocation


def test_stage_event_corrects_poll_estimate_without_resetting_each_poll():
    now = datetime.now(timezone.utc)
    actual = now - timedelta(seconds=20)
    row = SimpleNamespace(status='ready', ready_at=now-timedelta(hours=12))
    ticket = SimpleNamespace(status='running')
    meta = {'running_announced': True, 'stage_started_at': now.isoformat(), 'status_stream_observed': True}
    _mark_slurm_job_running(session=None, row=row, ticket=ticket, meta=meta, observed_at=actual, stage_event=True)
    _mark_slurm_job_running(session=None, row=row, ticket=ticket, meta=meta, observed_at=now)
    assert meta['stage_started_at'] == actual.isoformat()
    assert row.ready_at == actual
