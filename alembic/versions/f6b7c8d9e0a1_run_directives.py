"""Directed Run: run_mode + directives on runs

Adds runs.run_mode ('auto'|'directed') and runs.directives (JSONB per-agent
directive blocks executed verbatim in a Directed Run).

Revision ID: f6b7c8d9e0a1
Revises: f5a6b7c8d9e0
"""
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f6b7c8d9e0a1"
down_revision: Union[str, None] = "f5a6b7c8d9e0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("run_mode", sa.String(16), nullable=False, server_default="auto"),
    )
    op.add_column(
        "runs",
        sa.Column(
            "directives",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("runs", "directives")
    op.drop_column("runs", "run_mode")
