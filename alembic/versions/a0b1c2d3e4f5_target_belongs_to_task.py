"""remove target direction from settings

Revision ID: a0b1c2d3e4f5
Revises: f9a0b1c2d3e4
Create Date: 2026-08-16
"""
from alembic import op
import sqlalchemy as sa


revision = "a0b1c2d3e4f5"
down_revision = "f9a0b1c2d3e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_task_settings_metric_direction", "task_settings", type_="check",
    )
    op.drop_column("task_settings", "metric_direction")
    # The previous migration needed a temporary default to backfill existing
    # Tasks. New Tasks must now state evaluator direction explicitly.
    op.alter_column("tasks", "metric_direction", server_default=None)


def downgrade() -> None:
    op.alter_column("tasks", "metric_direction", server_default="max")
    op.add_column(
        "task_settings",
        sa.Column("metric_direction", sa.String(length=3), nullable=False, server_default="max"),
    )
    op.create_check_constraint(
        "ck_task_settings_metric_direction",
        "task_settings",
        "metric_direction IN ('max', 'min')",
    )
