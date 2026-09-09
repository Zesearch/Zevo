"""ticket_stage_mode

Rename tickets.task_type -> stage (the ticket's kind: synthesize / select /
infra / train / ... / orchestrate) and add tickets.mode (the invocation axis:
'pipeline' = an orchestrated stage, 'individual' = a direct user call). This
splits the two ideas that `task_type` used to conflate — in particular the old
`task_type='freeform'` value really meant "individual mode", so we backfill those
rows to mode='individual'.

Revision ID: d1e2f3a4b5c6
Revises: b8c9d0e1f2a3
Create Date: 2026-06-12 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "b8c9d0e1f2a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tickets",
        sa.Column("mode", sa.String(length=32), nullable=False, server_default="pipeline"),
    )
    op.alter_column("tickets", "task_type", new_column_name="stage")
    # Legacy freeform tickets were really individual-mode calls.
    op.execute("UPDATE tickets SET mode = 'individual' WHERE stage = 'freeform'")


def downgrade() -> None:
    op.execute("UPDATE tickets SET stage = 'freeform' WHERE mode = 'individual'")
    op.alter_column("tickets", "stage", new_column_name="task_type")
    op.drop_column("tickets", "mode")
