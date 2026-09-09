"""make_engine must build a working engine for a sqlite URL: the async sqlite
pool doesn't accept pool_size/max_overflow, so passing them (as the postgres
path does) makes `alembic upgrade` against sqlite raise TypeError before any
migration runs.
"""
from __future__ import annotations

import pytest

from zevo.db.session import make_engine


def test_sqlite_url_builds_engine(monkeypatch) -> None:
    monkeypatch.setenv("ZEVO_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    engine = make_engine()  # must not raise TypeError on pool kwargs
    assert engine.url.get_backend_name() == "sqlite"


def test_postgres_url_keeps_pool_tuning(monkeypatch) -> None:
    monkeypatch.setenv(
        "ZEVO_DATABASE_URL", "postgresql+asyncpg://u@localhost:5432/db"
    )
    engine = make_engine()
    assert engine.url.get_backend_name() == "postgresql"
    # QueuePool honours the tuned size.
    assert engine.pool.size() == 10
