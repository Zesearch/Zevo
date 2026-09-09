"""allow gpu_provider='local' (true same-machine execution)

Widen ck_runs_gpu_provider to include 'local' — running the pipeline directly
on the host the runner process runs on (no SSH, no rental).

Idempotent; Postgres only (sqlite builds tables from the models, which already
carry the widened constraint).

Revision ID: c3f1a2b40d56
Revises: b7e2c1a90d34
Create Date: 2026-08-29
"""
from __future__ import annotations

from alembic import op


revision = "c3f1a2b40d56"
down_revision = "b7e2c1a90d34"
branch_labels = None
depends_on = None

_ALLOWED_NEW = "('instance', 'cluster', 'cloud', 'ssh', 'local')"
_ALLOWED_OLD = "('instance', 'cluster', 'cloud', 'ssh')"


def _set_check(values: str) -> None:
    op.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS ck_runs_gpu_provider")
    op.create_check_constraint(
        "ck_runs_gpu_provider", "runs", f"gpu_provider IN {values}"
    )


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _set_check(_ALLOWED_NEW)


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _set_check(_ALLOWED_OLD)
