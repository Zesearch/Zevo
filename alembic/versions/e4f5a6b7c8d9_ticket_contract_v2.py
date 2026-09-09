"""canonical run, ticket, event, result, and artifact vocabulary

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "e4f5a6b7c8d9"
down_revision: Union[str, None] = "d3e4f5a6b7c8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.alter_column("tasks", "method_hint", new_column_name="training_method")
    op.alter_column("task_settings", "method_hint", new_column_name="training_method")

    # Run mode and per-Agent guidance have one product/API vocabulary.
    op.alter_column("runs", "run_mode", new_column_name="mode")
    op.alter_column("runs", "directives", new_column_name="customizations")
    op.add_column(
        "runs",
        sa.Column("gpu_provider", sa.String(length=16), nullable=False,
                  server_default="instance"),
    )
    op.execute("""
        UPDATE runs SET mode = CASE mode
            WHEN 'directed' THEN 'customized_pipeline'
            WHEN 'individual' THEN 'single_stage'
            ELSE 'full_pipeline'
        END
    """)
    op.execute("UPDATE runs SET num_gpus = 1 WHERE num_gpus IS NULL OR num_gpus < 1")
    op.alter_column("runs", "num_gpus", server_default="1")
    op.alter_column("runs", "framework", server_default="vllm")

    # Ticket dependencies and resolved paths now share one role-addressed map.
    op.add_column("tickets", sa.Column("input_format", sa.String(16), nullable=False,
                                        server_default="typed"))
    op.add_column("tickets", sa.Column("lane", sa.String(24), nullable=False,
                                        server_default="optimization"))
    op.add_column("tickets", sa.Column("iteration", sa.Integer(), nullable=False,
                                        server_default="0"))
    op.add_column("tickets", sa.Column("customization", JSONB, nullable=False,
                                        server_default=sa.text("'{}'::jsonb")))
    op.add_column("tickets", sa.Column("inputs", JSONB, nullable=False,
                                        server_default=sa.text("'{}'::jsonb")))
    op.add_column(
        "tickets",
        sa.Column("supervisor_wake_deferred", sa.Boolean(), nullable=False,
                  server_default=sa.text("false")),
    )
    op.execute("UPDATE tickets SET lane='held_out_test' WHERE held_out=true")
    op.execute("""
        UPDATE tickets SET status = CASE status
            WHEN 'todo' THEN 'queued'
            WHEN 'ready' THEN 'queued'
            WHEN 'in_progress' THEN 'running'
            WHEN 'done' THEN 'succeeded'
            WHEN 'completed' THEN 'succeeded'
            WHEN 'clarification_needed' THEN 'awaiting_input'
            ELSE status
        END
    """)
    op.drop_column("tickets", "parent_id")
    op.drop_column("tickets", "stage")
    op.drop_column("tickets", "mode")
    op.drop_column("tickets", "held_out")
    op.drop_column("tickets", "depends_on")
    op.drop_column("tickets", "refs")
    op.create_index("ix_tickets_lane", "tickets", ["lane"])

    # Conversation, system notices, structured outputs, and execution telemetry
    # are different records instead of overloading comments/phases.
    op.drop_table("comments")
    op.create_table(
        "ticket_messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("ticket_id", sa.String(64),
                  sa.ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("author", sa.String(64), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("triggered_wakeup_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_ticket_messages_ticket_id", "ticket_messages", ["ticket_id"])
    op.create_table(
        "ticket_notices",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("ticket_id", sa.String(64),
                  sa.ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("code", sa.String(64), nullable=False, server_default=""),
        sa.Column("severity", sa.String(16), nullable=False, server_default="info"),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_ticket_notices_ticket_id", "ticket_notices", ["ticket_id"])
    op.create_table(
        "heartbeat_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("ticket_id", sa.String(64),
                  sa.ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("heartbeat_id", sa.String(36), nullable=False, server_default=""),
        sa.Column("agent_id", sa.String(64), nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="failed"),
        sa.Column("output", JSONB, nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_heartbeat_results_ticket_id", "heartbeat_results", ["ticket_id"])
    op.create_index("ix_heartbeat_results_heartbeat_id", "heartbeat_results", ["heartbeat_id"])

    op.rename_table("pipeline_phases", "execution_events")
    op.add_column("execution_events", sa.Column("event_type", sa.String(16),
                                                 nullable=False, server_default="phase"))
    op.drop_index("ix_pipeline_phases_heartbeat_id", table_name="execution_events")
    op.create_index("ix_execution_events_heartbeat_id", "execution_events", ["heartbeat_id"])
    op.create_index("ix_execution_events_ticket_id", "execution_events", ["ticket_id"])

    op.alter_column("work_products", "kind", new_column_name="role",
                    existing_type=sa.String(32), type_=sa.String(48))
    op.create_index("ix_work_products_role", "work_products", ["role"])
    op.alter_column("registry_models", "method", new_column_name="training_method",
                    existing_type=sa.String(64))
    op.alter_column("registry_models", "dataset", new_column_name="dataset_path",
                    existing_type=sa.Text())
    op.alter_column("registry_models", "objective", new_column_name="task_objective",
                    existing_type=sa.Text())
    op.add_column("heartbeat_runs", sa.Column("resolved_config", JSONB,
                                               nullable=False,
                                               server_default=sa.text("'{}'::jsonb")))
    op.add_column("task_settings", sa.Column("target_accuracy", sa.Float(),
                                              nullable=False, server_default="0.0"))


def downgrade() -> None:
    op.alter_column("runs", "framework", server_default="hf")
    op.alter_column("runs", "num_gpus", server_default="0")
    op.drop_column("task_settings", "target_accuracy")
    op.drop_column("heartbeat_runs", "resolved_config")
    op.alter_column("registry_models", "task_objective", new_column_name="objective",
                    existing_type=sa.Text())
    op.alter_column("registry_models", "dataset_path", new_column_name="dataset",
                    existing_type=sa.Text())
    op.alter_column("registry_models", "training_method", new_column_name="method",
                    existing_type=sa.String(64))
    op.alter_column("task_settings", "training_method", new_column_name="method_hint")
    op.alter_column("tasks", "training_method", new_column_name="method_hint")
    op.drop_index("ix_work_products_role", table_name="work_products")
    op.alter_column("work_products", "role", new_column_name="kind",
                    existing_type=sa.String(48), type_=sa.String(32))

    op.drop_index("ix_execution_events_ticket_id", table_name="execution_events")
    op.drop_index("ix_execution_events_heartbeat_id", table_name="execution_events")
    op.drop_column("execution_events", "event_type")
    op.rename_table("execution_events", "pipeline_phases")
    op.create_index("ix_pipeline_phases_heartbeat_id", "pipeline_phases", ["heartbeat_id"])

    op.drop_table("heartbeat_results")
    op.drop_table("ticket_notices")
    op.drop_table("ticket_messages")
    op.create_table(
        "comments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("ticket_id", sa.String(64),
                  sa.ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("author", sa.String(64), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    op.drop_index("ix_tickets_lane", table_name="tickets")
    op.add_column("tickets", sa.Column("parent_id", sa.String(64), nullable=True))
    op.add_column("tickets", sa.Column("stage", sa.String(64), nullable=False,
                                        server_default=""))
    op.add_column("tickets", sa.Column("mode", sa.String(32), nullable=False,
                                        server_default="pipeline"))
    op.add_column("tickets", sa.Column("held_out", sa.Boolean(), nullable=False,
                                        server_default=sa.text("false")))
    op.add_column("tickets", sa.Column("depends_on", JSONB, nullable=False,
                                        server_default=sa.text("'[]'::jsonb")))
    op.add_column("tickets", sa.Column("refs", JSONB, nullable=False,
                                        server_default=sa.text("'{}'::jsonb")))
    op.drop_column("tickets", "supervisor_wake_deferred")
    op.drop_column("tickets", "inputs")
    op.drop_column("tickets", "customization")
    op.drop_column("tickets", "iteration")
    op.drop_column("tickets", "lane")
    op.drop_column("tickets", "input_format")

    op.drop_column("runs", "gpu_provider")
    op.alter_column("runs", "customizations", new_column_name="directives")
    op.alter_column("runs", "mode", new_column_name="run_mode")
