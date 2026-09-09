"""add a Run-only wall-clock limit

Revision ID: ae9f0a1b2c3d
Revises: ad8e9f0a1b2c
"""

import sqlalchemy as sa
from alembic import op


revision = "ae9f0a1b2c3d"
down_revision = "ad8e9f0a1b2c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column(
            "max_runtime_hours",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ))


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("max_runtime_hours")
