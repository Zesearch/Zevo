"""Alembic env -- async-aware, reads URL from zevo.db.session.

Run:
    alembic upgrade head
    alembic revision --autogenerate -m "msg"
"""
from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from alembic import context

# Ensure the engine packages (under src/) are importable so zevo.engine.* resolves,
# even without the editable install (e.g. a local `alembic` run).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from zevo.db import Base, make_engine  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Render SQL without a DB connection."""
    from zevo.db.session import _resolve_url
    url = _resolve_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# One arbitrary constant, shared by every process that migrates this database.
# Postgres advisory locks are namespaced by the number alone, so the only
# requirement is that nothing else in the system picks the same one.
_MIGRATION_LOCK_KEY = 8_154_113_207_641_002


def do_run_migrations(connection: Connection) -> None:
    """Migrate, but never concurrently with another process doing the same.

    `backend` and `scheduler` both run `alembic upgrade head` at boot, and
    docker starts them together. Adding a migration therefore raced: both read
    the same current revision, both ran `CREATE TABLE`, and the loser died with
    DuplicateTableError before its service ever started. The database ended up
    correct — which is why this stayed hidden — but one container was simply
    gone until someone restarted it by hand.

    A session-level advisory lock serializes them. The waiter re-reads
    `alembic_version` only after the holder has committed, finds itself already
    at head, and does nothing. Postgres drops the lock if the connection dies,
    so a crash mid-migration cannot wedge the next boot.
    """
    is_pg = connection.dialect.name == "postgresql"
    if is_pg:
        # Outside a transaction on purpose: a session-level lock has to outlive
        # the migration's own transaction, and `begin_transaction()` below owns
        # that one.
        connection.exec_driver_sql(f"SELECT pg_advisory_lock({_MIGRATION_LOCK_KEY})")
        connection.commit()
    try:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    finally:
        if is_pg:
            connection.exec_driver_sql(
                f"SELECT pg_advisory_unlock({_MIGRATION_LOCK_KEY})"
            )
            connection.commit()


async def run_async_migrations() -> None:
    engine: AsyncEngine = make_engine()
    async with engine.connect() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
