"""make training progress idempotent per heartbeat/phase/step

Revision ID: b27c8d9e0f1a
Revises: 001a2b3c4d5e
"""

import sqlalchemy as sa
from alembic import op


revision = "b27c8d9e0f1a"
down_revision = "001a2b3c4d5e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Keep the newest pre-migration copy of each logical reading. The runtime
    # now merges numeric extras before insert, so future duplicates cannot form.
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


def downgrade() -> None:
    op.drop_index("ux_execution_progress_step", table_name="execution_events")
