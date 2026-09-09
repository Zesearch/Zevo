"""widen provider transcript event names

Revision ID: c38d9e0f1a2b
Revises: b27c8d9e0f1a
"""

import sqlalchemy as sa
from alembic import op


revision = "c38d9e0f1a2b"
down_revision = "b27c8d9e0f1a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "transcript_events", "type",
        existing_type=sa.String(32), type_=sa.String(64), nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "transcript_events", "type",
        existing_type=sa.String(64), type_=sa.String(32), nullable=False,
    )
