"""Replace lane-specific scorers with one Task metric contract.

Revision ID: b0c1d2e3f4a5
Revises: af0a1b2c3d4e
Create Date: 2026-08-29
"""
from alembic import op
import sqlalchemy as sa


revision = "b0c1d2e3f4a5"
down_revision = "af0a1b2c3d4e"
branch_labels = None
depends_on = None


_BUILTINS = ("accuracy", "exact_match", "f1", "token_f1", "bleu", "rouge_l")


def upgrade() -> None:
    op.alter_column("tasks", "test_eval_script", new_column_name="evaluation_script")
    op.add_column(
        "tasks",
        sa.Column("metric_type", sa.String(length=16), nullable=False, server_default="builtin"),
    )
    op.add_column(
        "tasks",
        sa.Column("evaluator_sha256", sa.String(length=64), nullable=False, server_default=""),
    )
    op.create_check_constraint(
        "ck_tasks_metric_type", "tasks", "metric_type IN ('builtin', 'custom')",
    )
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE tasks SET metric_type = CASE "
            "WHEN lower(metric) IN :builtins THEN 'builtin' ELSE 'custom' END"
        ).bindparams(sa.bindparam("builtins", expanding=True)),
        {"builtins": list(_BUILTINS)},
    )
    # A built-in is engine code now. Old task scripts must not shadow it.
    bind.execute(sa.text(
        "UPDATE tasks SET evaluation_script = '', evaluator_sha256 = '' "
        "WHERE metric_type = 'builtin'"
    ))
    op.drop_column("task_settings", "validation_eval_script")


def downgrade() -> None:
    op.add_column(
        "task_settings",
        sa.Column("validation_eval_script", sa.Text(), nullable=False, server_default=""),
    )
    op.drop_constraint("ck_tasks_metric_type", "tasks", type_="check")
    op.drop_column("tasks", "evaluator_sha256")
    op.drop_column("tasks", "metric_type")
    op.alter_column("tasks", "evaluation_script", new_column_name="test_eval_script")
