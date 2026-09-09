"""runs.cloud_backend — per-run Vast.ai/Lambda pin

Lets a run pin the cloud backend (Vast.ai vs Lambda) at launch instead of
using only the ZEVO_CLOUD_BACKEND deployment default. Empty = default.

Idempotent.

Revision ID: d4a2b1c50e67
Revises: c3f1a2b40d56
Create Date: 2026-08-29
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "d4a2b1c50e67"
down_revision = "c3f1a2b40d56"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in inspect(op.get_bind()).get_columns("runs")}
    if "cloud_backend" not in cols:
        op.add_column(
            "runs",
            sa.Column("cloud_backend", sa.String(length=16), nullable=False, server_default=""),
        )


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("cloud_backend")
