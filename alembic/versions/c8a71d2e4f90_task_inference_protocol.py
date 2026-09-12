"""Historical Task inference-protocol migration.

Revision ID: c8a71d2e4f90
Revises: b2c3d4e5f6a9
Create Date: 2026-09-11
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "c8a71d2e4f90"
down_revision = "b2c3d4e5f6a9"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column(
            "inference_protocol", _JSON, nullable=False, server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("tasks", "inference_protocol")
