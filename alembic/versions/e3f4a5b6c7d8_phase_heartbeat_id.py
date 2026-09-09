"""attribute each pipeline_phases row to the activation that emitted it

`pipeline_phases` was keyed only by `ticket_id`. That is enough for a stage
agent, which runs once per ticket, but the orchestrator is woken once per child
ticket — a four-iteration run wakes it 26 times on ONE ticket, so 26 separate
runs of the agent wrote their markers into the same undifferentiated pile.

Readers were left inferring which wake produced which row from `ts` alone,
which is guesswork at the boundaries: a marker emitted moments before the next
wake starts, or flushed after it, lands on the wrong activation. Recording the
heartbeat makes the attribution a fact instead.

Existing rows keep `''` — their wake is unknowable after the fact, and readers
fall back to the timestamp for those. The same applies to rows created by the
progress endpoint, which is called from the remote box with no heartbeat in
scope.

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-08-07

"""
import sqlalchemy as sa
from alembic import op


revision = "e3f4a5b6c7d8"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "pipeline_phases",
        sa.Column("heartbeat_id", sa.String(length=36), nullable=False, server_default=""),
    )
    # Read path: "every phase this heartbeat emitted", which is what the run
    # detail asks for each time you select an activation.
    op.create_index(
        "ix_pipeline_phases_heartbeat_id", "pipeline_phases", ["heartbeat_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_pipeline_phases_heartbeat_id", table_name="pipeline_phases")
    op.drop_column("pipeline_phases", "heartbeat_id")
