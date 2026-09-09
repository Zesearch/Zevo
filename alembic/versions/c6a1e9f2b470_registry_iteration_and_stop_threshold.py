"""record registry iteration and canonical stop threshold

Revision ID: c6a1e9f2b470
Revises: a0b1c2d3e4f5
Create Date: 2026-08-16
"""
from alembic import op
import sqlalchemy as sa


revision = "c6a1e9f2b470"
down_revision = "a0b1c2d3e4f5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("agents", "default_model", type_=sa.String(length=256))
    op.alter_column("heartbeat_runs", "model", type_=sa.String(length=256))
    # Execution events now have exactly one owner. Rows whose heartbeat was
    # already removed cannot satisfy that contract and must not survive as
    # unattached progress records.
    op.execute("""
        DELETE FROM execution_events AS event
        WHERE event.heartbeat_id = ''
           OR NOT EXISTS (
               SELECT 1 FROM heartbeat_runs AS heartbeat
               WHERE heartbeat.id = event.heartbeat_id
           )
    """)
    op.create_foreign_key(
        "fk_execution_events_heartbeat_id", "execution_events", "heartbeat_runs",
        ["heartbeat_id"], ["id"], ondelete="CASCADE",
    )
    op.alter_column("execution_events", "heartbeat_id", server_default=None)
    op.add_column(
        "registry_models",
        sa.Column("iteration", sa.Integer(), nullable=False, server_default="0"),
    )
    op.execute("""
        UPDATE registry_models AS rm
        SET iteration = retained.iteration
        FROM (
            SELECT DISTINCT ON (t.run_id)
                   t.run_id, t.iteration
            FROM tickets AS t
            JOIN heartbeat_results AS hr ON hr.ticket_id = t.id
            WHERE t.agent_id = 'registry'
              AND hr.output ->> 'retained' = 'true'
            ORDER BY t.run_id, hr.created_at DESC
        ) AS retained
        WHERE rm.run_id = retained.run_id
    """)
    op.alter_column("registry_models", "iteration", server_default=None)
    op.alter_column("runs", "target_accuracy", new_column_name="stop_threshold")
    op.alter_column(
        "task_settings", "target_accuracy", new_column_name="stop_threshold",
    )
    op.alter_column("runs", "stop_threshold", nullable=True, server_default=None)
    op.alter_column("task_settings", "stop_threshold", nullable=True, server_default=None)
    op.execute("UPDATE runs SET stop_threshold = NULL WHERE stop_threshold = 0")
    op.execute("UPDATE task_settings SET stop_threshold = NULL WHERE stop_threshold = 0")
    op.execute("""
        UPDATE runs
        SET history = COALESCE(
            (SELECT jsonb_agg(entry - 'notes')
             FROM jsonb_array_elements(history::jsonb) AS entry),
            '[]'::jsonb
        )::json
        WHERE json_typeof(history) = 'array'
    """)


def downgrade() -> None:
    op.execute("UPDATE runs SET stop_threshold = 0 WHERE stop_threshold IS NULL")
    op.execute("UPDATE task_settings SET stop_threshold = 0 WHERE stop_threshold IS NULL")
    op.alter_column("runs", "stop_threshold", nullable=False, server_default="0.0")
    op.alter_column("task_settings", "stop_threshold", nullable=False, server_default="0.0")
    op.alter_column("runs", "stop_threshold", new_column_name="target_accuracy")
    op.alter_column(
        "task_settings", "stop_threshold", new_column_name="target_accuracy",
    )
    op.alter_column("heartbeat_runs", "model", type_=sa.String(length=64))
    op.drop_constraint("fk_execution_events_heartbeat_id", "execution_events", type_="foreignkey")
    op.alter_column("execution_events", "heartbeat_id", server_default="")
    op.alter_column("agents", "default_model", type_=sa.String(length=64))
    op.drop_column("registry_models", "iteration")
