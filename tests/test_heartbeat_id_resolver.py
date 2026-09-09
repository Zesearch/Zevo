"""_resolve_heartbeat_id must not 500 on an ambiguous prefix or LIKE-wildcard
characters in the path (previously MultipleResultsFound / broad match → 500).
"""
from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.ui.heartbeat_events import _resolve_heartbeat_id
from zevo.db.models import Base, HeartbeatRun


async def _session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


def _hb(hb_id: str, started: datetime) -> HeartbeatRun:
    return HeartbeatRun(
        id=hb_id, ticket_id="tk", agent_id="train", driver="claude_cli",
        model="m", exit_code=0, stdout_path="", stderr_path="",
        error_message="", started_at=started,
    )


@pytest.mark.asyncio
async def test_ambiguous_prefix_returns_newest_not_500() -> None:
    Session = await _session()
    now = datetime.now(timezone.utc)
    async with Session() as db:
        db.add(_hb("abc11111", now - _dt.timedelta(hours=1)))
        db.add(_hb("abc22222", now))  # newest
        await db.commit()
        # Two rows share prefix "abc" — must return the newest, not raise.
        assert await _resolve_heartbeat_id(db, "abc") == "abc22222"


@pytest.mark.asyncio
async def test_wildcard_char_is_literal_and_404s() -> None:
    Session = await _session()
    async with Session() as db:
        db.add(_hb("abc11111", datetime.now(timezone.utc)))
        await db.commit()
        # "_" must match literally (no row is literally "_..."), so 404 not 500.
        with pytest.raises(HTTPException) as ei:
            await _resolve_heartbeat_id(db, "_")
        assert ei.value.status_code == 404
