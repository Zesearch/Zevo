"""add canonical task and run metric direction

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
"""
from alembic import op
import sqlalchemy as sa


revision = "d5e6f7a8b9c0"
down_revision = "c4d5e6f7a8b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("metric_direction", sa.String(length=3), nullable=False, server_default="max"),
    )
    op.add_column(
        "runs",
        sa.Column("metric_direction", sa.String(length=3), nullable=False, server_default="max"),
    )
    op.create_check_constraint(
        "ck_tasks_metric_direction", "tasks", "metric_direction IN ('max', 'min')"
    )
    op.create_check_constraint(
        "ck_runs_metric_direction", "runs", "metric_direction IN ('max', 'min')"
    )


def downgrade() -> None:
    op.drop_constraint("ck_runs_metric_direction", "runs", type_="check")
    op.drop_constraint("ck_tasks_metric_direction", "tasks", type_="check")
    op.drop_column("runs", "metric_direction")
    op.drop_column("tasks", "metric_direction")
