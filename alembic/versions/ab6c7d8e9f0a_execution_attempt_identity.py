"""give execution events a process-attempt identity

Revision ID: ab6c7d8e9f0a
Revises: c38d9e0f1a2b
"""

import sqlalchemy as sa
from alembic import op


revision = "ab6c7d8e9f0a"
down_revision = "c38d9e0f1a2b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "execution_events",
        sa.Column("attempt_id", sa.String(length=36), nullable=True),
    )
    # Rows created before attempt-aware telemetry represent one undifferentiated
    # execution per activation. Heartbeat UUID is a stable, non-empty identity
    # for that legacy group and keeps the API invariant simple.
    op.execute(sa.text("""
        UPDATE execution_events
        SET attempt_id = heartbeat_id
        WHERE attempt_id = ''
           OR attempt_id IS NULL
    """))
    # No permanent default: every writer must state which execution produced
    # the event. Silently storing an empty id would recreate the same curve
    # collision this migration removes.
    op.alter_column("execution_events", "attempt_id", nullable=False)
    op.drop_index("ux_execution_progress_step", table_name="execution_events")
    op.create_index(
        "ux_execution_progress_step",
        "execution_events",
        ["heartbeat_id", "attempt_id", "phase", "current_step"],
        unique=True,
        postgresql_where=sa.text("event_type = 'progress'"),
        sqlite_where=sa.text("event_type = 'progress'"),
    )


def downgrade() -> None:
    op.drop_index("ux_execution_progress_step", table_name="execution_events")
    # Multiple attempts legitimately reuse steps. Collapse them before restoring
    # the old uniqueness rule so downgrade remains executable.
    op.execute(sa.text("""
        DELETE FROM execution_events
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY heartbeat_id, phase, current_step
                           ORDER BY ts DESC, id DESC
                       ) AS duplicate_number
                FROM execution_events
                WHERE event_type = 'progress'
            ) ranked
            WHERE duplicate_number > 1
        )
    """))
    op.create_index(
        "ux_execution_progress_step",
        "execution_events",
        ["heartbeat_id", "phase", "current_step"],
        unique=True,
        postgresql_where=sa.text("event_type = 'progress'"),
        sqlite_where=sa.text("event_type = 'progress'"),
    )
    op.drop_column("execution_events", "attempt_id")
