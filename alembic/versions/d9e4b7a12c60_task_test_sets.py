"""Replace the Task protocol experiment with a list of Test contracts.

Revision ID: d9e4b7a12c60
Revises: c8a71d2e4f90
Create Date: 2026-09-11
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "d9e4b7a12c60"
down_revision = "c8a71d2e4f90"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("test_sets", _JSON, nullable=False, server_default="[]"),
    )
    op.drop_column("tasks", "inference_protocol")


def downgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column(
            "inference_protocol", _JSON, nullable=False, server_default="{}",
        ),
    )
    op.drop_column("tasks", "test_sets")
