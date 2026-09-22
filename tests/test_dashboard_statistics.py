from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, InfraInstance, Run
from zevo.engine.observe.dashboard import aggregate_runs
from zevo.api.routers.shared.runs import _queue_wait_state_for_runs


@pytest.mark.asyncio
async def test_dashboard_all_runs_windows_queue_wait_and_visibility():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    now = datetime.now(timezone.utc)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        # A historical active run and old results must survive >500 newer rows.
        db.add_all([Run(id=f"recent-{i:04}", task_name="visible", metric="accuracy", status="cancelled",
                        started_at=now-timedelta(seconds=60), finished_at=now)
                    for i in range(501)])
        db.add(Run(id="old-active", task_name="visible", metric="accuracy", status="running", started_at=now-timedelta(days=60)))
        db.add(Run(id="private", task_name="other", metric="accuracy", status="running"))
        for name, age, direction, baseline, evolved in [
            ('old', 40, 'max', .2, .8), ('week', 3, 'min', .8, .3),
            ('month', 10, 'max', .4, .6),
        ]:
            end = now - timedelta(days=age)
            db.add(Run(id=name, task_name="visible", status="success", metric="accuracy",
                       metric_direction=direction, validation_metric_direction=direction,
                       started_at=end-timedelta(seconds=100), finished_at=end,
                       registry_version_tag=name, champion_test_score=evolved,
                       history=[{'source': 'baseline', 'test_score': baseline}]))
        db.add(InfraInstance(id='queue', run_id='old', provider='cluster',
                             created_at=now-timedelta(days=40, seconds=100),
                             ready_at=now-timedelta(days=40, seconds=80)))
        await db.commit()
        stats = await aggregate_runs(db, select(Run).where(Run.task_name == 'visible'), _queue_wait_state_for_runs)
        assert stats['total'] == 505
        assert stats['active'] == 1
        assert stats['succeeded'] == 3
        assert stats['runtime_seconds'] == {'1d': 501*60, '1w': 501*60+100, '1m': 501*60+200, 'all': 501*60+280}
        groups = {g['metricDirection']: g for g in stats['improvements']}
        assert groups['max']['runCount'] == 2
        assert groups['max']['averageImprovement'] == pytest.approx(.4)
        assert groups['max']['bestScore'] == .8
        assert groups['min']['averageImprovement'] == pytest.approx(.5)
        assert groups['min']['bestScore'] == .3
        empty = await aggregate_runs(db, select(Run).where(Run.task_name == 'absent'), _queue_wait_state_for_runs)
        assert empty['total'] == 0
        assert empty['runtime_seconds']['all'] == 0
        assert empty['improvements'] == []
    await engine.dispose()
