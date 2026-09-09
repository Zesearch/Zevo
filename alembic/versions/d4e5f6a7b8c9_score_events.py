"""score_events

One row per eval data point. Orchestrator inserts via
POST /runs/{id}/scores after each eval; backend exposes
GET /runs/{id}/scores for the UI sparkline.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-06-02 08:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "score_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source", sa.String(32), nullable=False, server_default="trained"),
        sa.Column("accuracy", sa.Float(), nullable=False, server_default="-1.0"),
        sa.Column("metric_name", sa.String(32), nullable=False, server_default="accuracy"),
        sa.Column("extras", sa.dialects.postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_score_events_run_id", "score_events", ["run_id"])
    op.create_index("ix_score_events_ts", "score_events", ["ts"])


def downgrade() -> None:
    op.drop_index("ix_score_events_ts", table_name="score_events")
    op.drop_index("ix_score_events_run_id", table_name="score_events")
    op.drop_table("score_events")
