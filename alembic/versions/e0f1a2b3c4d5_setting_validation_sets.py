"""Allow a Setting to carry an independent Validation suite.

Revision ID: e0f1a2b3c4d5
Revises: d9e4b7a12c60
Create Date: 2026-09-13
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "e0f1a2b3c4d5"
down_revision = "d9e4b7a12c60"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "task_settings",
        sa.Column("validation_sets", _JSON, nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("task_settings", "validation_sets")
