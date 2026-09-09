"""persist method-specific setting configuration

Revision ID: f9a0b1c2d3e4
Revises: e6f7a8b9c0d1
Create Date: 2026-08-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f9a0b1c2d3e4"
down_revision = "e6f7a8b9c0d1"
branch_labels = None
depends_on = None


_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "task_settings",
        sa.Column("method_config", _JSON, nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("task_settings", "method_config")
