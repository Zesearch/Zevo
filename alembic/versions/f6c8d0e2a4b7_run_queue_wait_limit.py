"""add per-Run GPU queue wait limit

Revision ID: f6c8d0e2a4b7
Revises: e5a7c9d1f3b2
"""

import sqlalchemy as sa
from alembic import op


revision = "f6c8d0e2a4b7"
down_revision = "e5a7c9d1f3b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column(
            "max_queue_wait_hours", sa.Float(), nullable=False,
            server_default="24.0",
        ))


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("max_queue_wait_hours")
