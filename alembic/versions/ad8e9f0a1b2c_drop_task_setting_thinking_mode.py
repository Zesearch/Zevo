"""drop obsolete user-selectable thinking mode

Revision ID: ad8e9f0a1b2c
Revises: ac7d8e9f0a1b
"""

import sqlalchemy as sa
from alembic import op


revision = "ad8e9f0a1b2c"
down_revision = "ac7d8e9f0a1b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("task_settings") as batch:
        batch.drop_column("thinking_mode")


def downgrade() -> None:
    with op.batch_alter_table("task_settings") as batch:
        batch.add_column(sa.Column(
            "thinking_mode",
            sa.String(length=16),
            nullable=False,
            server_default="",
        ))
