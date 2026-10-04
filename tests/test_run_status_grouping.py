"""Status groups stay contiguous across server-side sorting and pagination."""
import pytest
from fastapi import Response
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Base, Run
from zevo.api.routers.shared.runs import list_runs


@pytest.mark.asyncio
@pytest.mark.parametrize('sort', ['started', 'cost', 'improvement'])
async def test_status_grouping_before_pagination(sort):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        db.add_all([Run(id=str(i), metric='accuracy', task_name='group-test', status=status)
                    for i, status in enumerate(['running', 'failed', 'running'])])
        await db.commit()
        response = Response()
        first = await list_runs(response, db, limit=1, group='status', sort=sort)
        rest = await list_runs(Response(), db, limit=2, offset=1, group='status', sort=sort)
        assert [r.status for r in first + rest] == ['failed', 'running', 'running']
        assert response.headers['X-Total-Count'] == '3'
        filtered = await list_runs(Response(), db, q='no-match', group='status', sort=sort)
        assert filtered == []
    await engine.dispose()
