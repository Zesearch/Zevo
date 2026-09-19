"""Persist Run-level user instructions and coordinator decisions.

Revision ID: e1f2a3b4c5d6
Revises: e1a2b3c4d5f6
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "e1f2a3b4c5d6"
down_revision = "e1a2b3c4d5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_instructions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_ticket_id", sa.String(64), sa.ForeignKey("tickets.id", ondelete="SET NULL"), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("agent_response", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('queued', 'delivered', 'scheduled', 'applied', 'needs_input', 'declined')",
            name="ck_run_instructions_status",
        ),
    )
    op.create_index("ix_run_instructions_run_id", "run_instructions", ["run_id"])
    op.create_index("ix_run_instructions_source_ticket_id", "run_instructions", ["source_ticket_id"])
    op.create_index("ix_run_instructions_status", "run_instructions", ["status"])


def downgrade() -> None:
    op.drop_index("ix_run_instructions_status", table_name="run_instructions")
    op.drop_index("ix_run_instructions_source_ticket_id", table_name="run_instructions")
    op.drop_index("ix_run_instructions_run_id", table_name="run_instructions")
    op.drop_table("run_instructions")
