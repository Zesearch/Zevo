"""PATCH /agents/{id} can raise max_concurrent_runs, the per-agent cap the
wakeup daemon enforces across all runs; the DTO reports the stored value."""
import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.ui.agents import PatchAgentBody, get_agent, patch_agent
from zevo.db import Agent, Base


@pytest.mark.asyncio
async def test_patch_agent_sets_concurrency_and_dto_reports_it():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        db.add(Agent(id="train", name="Train", title="Training", identity_path=""))
        await db.commit()
        dto = await patch_agent("train", PatchAgentBody(max_concurrent_runs=5), db)
        assert dto.max_concurrent_runs == 5
        row = (await db.execute(select(Agent).where(Agent.id == "train"))).scalar_one()
        assert row.max_concurrent_runs == 5


def test_concurrency_bounds():
    with pytest.raises(ValidationError):
        PatchAgentBody(max_concurrent_runs=0)
    with pytest.raises(ValidationError):
        PatchAgentBody(max_concurrent_runs=33)
