"""separate task/agent objectives and remove Task experiment defaults

Revision ID: f7a8b9c0d1e2
Revises: e4f5a6b7c8d9
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f7a8b9c0d1e2"
down_revision: Union[str, None] = "e4f5a6b7c8d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("tasks", "objective", new_column_name="task_objective")
    for column in (
        "training_method", "dataset", "data_query", "base_model", "constraints",
        "gpu_provider", "framework", "iteration_budget", "max_cost_usd",
        "target_accuracy", "domain",
    ):
        op.drop_column("tasks", column)

    op.alter_column("runs", "root_objective", new_column_name="agent_objective")
    op.add_column(
        "runs",
        sa.Column("task_objective", sa.Text(), nullable=False, server_default=""),
    )
    # Keep the readable problem for rows created before the two meanings were
    # stored independently; old field names are not exposed by the application.
    op.execute("UPDATE runs SET task_objective = agent_objective")
    op.alter_column(
        "runs", "iteration_budget", server_default="0", existing_type=sa.Integer()
    )
    # contract_v2 changed the values but retained VARCHAR(16) and default
    # 'auto'; customized_pipeline needs 19 characters and all new rows use the
    # three canonical mode values.
    op.alter_column(
        "runs", "mode", existing_type=sa.String(16), type_=sa.String(32),
        server_default="full_pipeline",
    )

    op.alter_column(
        "registry_models", "dataset_path", new_column_name="dataset_source"
    )


def downgrade() -> None:
    op.alter_column(
        "registry_models", "dataset_source", new_column_name="dataset_path"
    )
    op.execute("""
        UPDATE runs SET mode = CASE mode
            WHEN 'customized_pipeline' THEN 'directed'
            WHEN 'single_stage' THEN 'individual'
            ELSE 'auto'
        END
    """)
    op.alter_column(
        "runs", "mode", existing_type=sa.String(32), type_=sa.String(16),
        server_default="auto",
    )
    op.alter_column(
        "runs", "iteration_budget", server_default="5", existing_type=sa.Integer()
    )
    op.drop_column("runs", "task_objective")
    op.alter_column("runs", "agent_objective", new_column_name="root_objective")

    op.add_column("tasks", sa.Column("training_method", sa.String(64), nullable=False, server_default=""))
    op.add_column("tasks", sa.Column("dataset", sa.Text(), nullable=False, server_default=""))
    op.add_column("tasks", sa.Column("data_query", sa.Text(), nullable=False, server_default=""))
    op.add_column("tasks", sa.Column("base_model", sa.String(128), nullable=False, server_default=""))
    op.add_column("tasks", sa.Column("constraints", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")))
    op.add_column("tasks", sa.Column("gpu_provider", sa.String(16), nullable=False, server_default=""))
    op.add_column("tasks", sa.Column("framework", sa.String(16), nullable=False, server_default="vllm"))
    op.add_column("tasks", sa.Column("iteration_budget", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("tasks", sa.Column("max_cost_usd", sa.Float(), nullable=False, server_default="0"))
    op.add_column("tasks", sa.Column("target_accuracy", sa.Float(), nullable=False, server_default="0"))
    op.add_column("tasks", sa.Column("domain", sa.String(32), nullable=False, server_default=""))
    op.alter_column("tasks", "task_objective", new_column_name="objective")
