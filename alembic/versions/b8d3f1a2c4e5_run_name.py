"""add runs.run_name (what the user calls this execution)

A run's `task_name` names the WORK, and many runs share one, so a list of runs
of the same task had nothing per-run to read but the uuid. `run_name` is the
per-run label the launch dialog now asks for. Existing rows backfill to '', and
the UI falls back to the run id for those.

Revision ID: b8d3f1a2c4e5
Revises: a7c1e2d4b8f6
Create Date: 2026-08-09
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "b8d3f1a2c4e5"
down_revision = "a7c1e2d4b8f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("run_name", sa.String(length=128), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("runs", "run_name")
