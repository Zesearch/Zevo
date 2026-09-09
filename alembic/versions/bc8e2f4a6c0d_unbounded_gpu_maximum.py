"""make a blank Run GPU maximum unbounded

Revision ID: bc8e2f4a6c0d
Revises: bb7d1e3f5a9c
Create Date: 2026-09-01

``runs.num_gpus`` now stores a maximum rather than an exact allocation size.
Zero is the persisted sentinel for no user-imposed limit. Existing rows retain
their historical positive values as explicit limits; only the default changes.
"""

from alembic import op


revision = "bc8e2f4a6c0d"
down_revision = "bb7d1e3f5a9c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("runs", "num_gpus", server_default="0")


def downgrade() -> None:
    op.alter_column("runs", "num_gpus", server_default="1")
