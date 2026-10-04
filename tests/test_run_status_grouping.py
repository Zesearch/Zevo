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


@pytest.mark.asyncio
@pytest.mark.parametrize('sort', ['started', 'cost', 'improvement'])
async def test_multi_filters_apply_before_pagination_and_count(sort):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from zevo.api.database import get_db
    from zevo.api.routers.shared.runs import router

    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        for i, (task, status) in enumerate([
            ('Alpha, β', 'success'), ('Alpha, β', 'failed'),
            ('Beta', 'running'), ('Beta-extra', 'running'), ('Beta', 'success'),
        ]):
            db.add(Run(id=f'filter-{i}', task_name=task, status=status, metric='accuracy'))
        await db.commit()
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: db
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            params = [('task', 'Alpha, β'), ('task', 'Beta'), ('status', 'success'),
                      ('status', 'running'), ('group', 'status'), ('sort', sort), ('limit', '1')]
            rows = []
            for offset in range(3):
                response = await client.get('/runs', params=params + [('offset', str(offset))])
                assert response.status_code == 200, response.text
                assert response.headers['X-Total-Count'] == '3'
                rows += response.json()
            assert {row['id'] for row in rows} == {'filter-0', 'filter-2', 'filter-4'}
            assert [row['status'] for row in rows] == ['success', 'success', 'running']
            searched = await client.get('/runs', params=params + [('q', 'Alpha')])
            assert searched.headers['X-Total-Count'] == '1'
            empty = await client.get('/runs', params=[('task', 'absent')])
            assert empty.json() == []
            assert empty.headers['X-Total-Count'] == '0'
            all_rows = await client.get('/runs')
            assert all_rows.headers['X-Total-Count'] == '5'
            counts = await client.get('/runs/status-counts')
            assert counts.status_code == 200
            assert counts.json() == {'success': 2, 'running': 2, 'failed': 1}
            assert list(counts.json()) == ['success', 'running', 'failed']
    await engine.dispose()
