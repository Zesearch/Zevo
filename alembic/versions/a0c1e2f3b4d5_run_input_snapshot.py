"""store immutable Task and File identities on Runs

Revision ID: a0c1e2f3b4d5
Revises: 9b7c6d5e4f3a
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "a0c1e2f3b4d5"
down_revision = "9b7c6d5e4f3a"
branch_labels = None
depends_on = None


_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("input_snapshot", _JSON, nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("runs", "input_snapshot")
