"""replace stale identity/summary names and bind runs to settings

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-08-15
"""
from alembic import op
import sqlalchemy as sa


revision = "e6f7a8b9c0d1"
down_revision = "d5e6f7a8b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("agents", "agents_md_path", new_column_name="identity_path")
    op.alter_column("runs", "plan_summary", new_column_name="summary")
    op.add_column(
        "task_settings",
        sa.Column("metric_direction", sa.String(length=3), nullable=False, server_default="max"),
    )
    op.create_check_constraint(
        "ck_task_settings_metric_direction",
        "task_settings",
        "metric_direction IN ('max', 'min')",
    )
    op.add_column("runs", sa.Column("setting_id", sa.String(length=36), nullable=True))
    op.create_foreign_key(
        "fk_runs_setting_id_task_settings",
        "runs", "task_settings", ["setting_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_runs_setting_id", "runs", ["setting_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_setting_id", table_name="runs")
    op.drop_constraint("fk_runs_setting_id_task_settings", "runs", type_="foreignkey")
    op.drop_column("runs", "setting_id")
    op.drop_constraint("ck_task_settings_metric_direction", "task_settings", type_="check")
    op.drop_column("task_settings", "metric_direction")
    op.alter_column("runs", "summary", new_column_name="plan_summary")
    op.alter_column("agents", "identity_path", new_column_name="agents_md_path")
