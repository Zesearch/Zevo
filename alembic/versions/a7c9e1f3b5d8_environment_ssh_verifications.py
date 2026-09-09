"""persist verification status for env-backed SSH connections

Revision ID: a7c9e1f3b5d8
Revises: f6c8d0e2a4b7
"""

import sqlalchemy as sa
from alembic import op


revision = "a7c9e1f3b5d8"
down_revision = "f6c8d0e2a4b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "environment_ssh_verifications",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="unverified"),
        sa.Column("last_error", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("environment_ssh_verifications")
