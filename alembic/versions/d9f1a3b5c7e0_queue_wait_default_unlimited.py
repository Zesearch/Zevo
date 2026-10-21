"""Default new Runs to unlimited queue wait; preserve existing explicit caps.

Revision ID: d9f1a3b5c7e0
Revises: c8d0e2f4a6b8
"""
import sqlalchemy as sa
from alembic import op

revision = "d9f1a3b5c7e0"
down_revision = "c8d0e2f4a6b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.alter_column("max_queue_wait_hours", existing_type=sa.Float(),
                           existing_nullable=False, server_default="0.0")


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.alter_column("max_queue_wait_hours", existing_type=sa.Float(),
                           existing_nullable=False, server_default="48.0")
