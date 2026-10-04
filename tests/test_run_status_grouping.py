"""Status groups stay contiguous across server-side sorting and pagination."""
import pytest
from fastapi import Response
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run
from zevo.api.routers.shared.runs import list_runs


@pytest.mark.asyncio
@pytest.mark.parametrize('sort', ['started', 'cost', 'improvement'])
@pytest.mark.parametrize('order', ['asc', 'desc'])
async def test_status_grouping_before_pagination(sort, order):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        db.add_all([Run(id=str(i), metric='accuracy', task_name='group-test', status=status)
                    for i, status in enumerate(['cancelled', 'running', 'failed', 'success',
                                                'degraded', 'planning', 'running', 'success'])])
        await db.commit()
        response = Response()
        rows = []
        for offset in range(0, 8, 3):
            rows += await list_runs(response, db, limit=3, offset=offset,
                                    group='status', sort=sort, order=order)
        assert [r.status for r in rows] == ['success', 'success', 'running', 'running',
                                           'planning', 'degraded', 'failed', 'cancelled']
        assert len({r.id for r in rows}) == 8
        assert response.headers['X-Total-Count'] == '8'
        filtered = await list_runs(Response(), db, q='no-match', group='status', sort=sort)
        assert filtered == []
    await engine.dispose()
