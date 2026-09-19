"""set the default Run queue wait limit to 48 hours

Revision ID: 9b7c6d5e4f3a
Revises: e2a3b4c5d6e7
"""

import sqlalchemy as sa
from alembic import op


revision = "9b7c6d5e4f3a"
down_revision = "e2a3b4c5d6e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.alter_column(
            "max_queue_wait_hours",
            existing_type=sa.Float(),
            existing_nullable=False,
            server_default="48.0",
        )


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.alter_column(
            "max_queue_wait_hours",
            existing_type=sa.Float(),
            existing_nullable=False,
            server_default="24.0",
        )
