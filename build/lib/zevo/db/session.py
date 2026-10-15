"""Async SQLAlchemy session factory.

Reads ZEVO_DATABASE_URL. Auto-loads `.env.local` at repo root if present
(simple KEY=VALUE per line; comments with '#' allowed).

Defaults to local Postgres on the current user (`postgresql+asyncpg://
$USER@localhost:5432/zevo_dev`); SQLite is supported only as a last
resort.
"""
from __future__ import annotations

import os
import getpass
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = REPO_ROOT / ".env.local"


def _load_dotenv_once() -> None:
    """Read .env.local and merge into os.environ (does not overwrite)."""
    if not DOTENV_PATH.exists():
        return
    for raw in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv_once()


def _resolve_url() -> str:
    url = os.environ.get("ZEVO_DATABASE_URL", "").strip()
    if url:
        return url
    user = getpass.getuser()
    return f"postgresql+asyncpg://{user}@localhost:5432/zevo_dev"


def make_engine(echo: bool = False) -> AsyncEngine:
    url = _resolve_url()
    # SQLite's async driver runs on SingletonThreadPool / StaticPool, which do
    # NOT accept pool_size / max_overflow (create_async_engine raises TypeError).
    # Only the server-backed pools take those knobs, so apply them only for
    # non-sqlite URLs -- otherwise `alembic upgrade` against a sqlite URL blows
    # up before it can run a single migration.
    if url.startswith("sqlite"):
        return create_async_engine(url, echo=echo, future=True)
    # Room for every agent to be in flight at once. Each one holds a
    # connection of its own for its advisory lock (see
    # `advisory_lock`) on top of the connections its work uses, and
    # agents run concurrently, so the 5 + 10 default is too tight to be
    # comfortable.
    return create_async_engine(
        url, echo=echo, future=True,
        pool_size=10, max_overflow=20, pool_pre_ping=True,
    )


_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _SessionLocal
