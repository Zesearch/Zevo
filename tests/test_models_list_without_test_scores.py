"""GET /api/models must list a model whose run has no held-out score yet.

A checkpoint rescued on cancel is registered before the run's first held-out
eval, so its run has neither a baseline nor a champion test score. The list
endpoint computed `improvement(None, None)` for it and returned a 500, taking
the whole Models page down.
"""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.ui.models import _run_outcomes
from zevo.db.models import Base, Run


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.asyncio
async def test_run_outcomes_tolerate_missing_test_scores() -> None:
    Session = await _session()
    async with Session() as db:
        db.add(Run(
            id="r1", task_name="t", task_objective="o", agent_objective="ao",
            metric="accuracy", metric_direction="max",
            validation_metric="accuracy", validation_metric_direction="max",
            status="cancelled", registry_version_tag="M-r1",
            history=[{"iteration": 1, "score": 0.55, "source": "evaluation"}],
        ))
        await db.commit()
        out = await _run_outcomes(db, ["r1"])
        assert out["r1"] == {
            "champion_test_score": None, "baseline_test_score": None, "improvement": None,
        }
