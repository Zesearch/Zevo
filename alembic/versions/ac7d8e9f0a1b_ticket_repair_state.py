"""add bounded Ticket repair state

Revision ID: ac7d8e9f0a1b
Revises: ab6c7d8e9f0a
"""

import sqlalchemy as sa
from alembic import op


revision = "ac7d8e9f0a1b"
down_revision = "ab6c7d8e9f0a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("tickets") as batch:
        batch.drop_constraint("ck_tickets_status", type_="check")
        batch.create_check_constraint(
            "ck_tickets_status",
            "status IN ('queued', 'running', 'repairing', 'awaiting_input', "
            "'succeeded', 'degraded', 'failed', 'skipped', 'cancelled')",
        )
        batch.add_column(sa.Column(
            "repair_attempts", sa.Integer(), nullable=False, server_default="0",
        ))
        batch.add_column(sa.Column(
            "repair_route", sa.String(length=16), nullable=False, server_default="",
        ))


def downgrade() -> None:
    op.execute("UPDATE tickets SET status = 'failed' WHERE status = 'repairing'")
    with op.batch_alter_table("tickets") as batch:
        batch.drop_column("repair_route")
        batch.drop_column("repair_attempts")
        batch.drop_constraint("ck_tickets_status", type_="check")
        batch.create_check_constraint(
            "ck_tickets_status",
            "status IN ('queued', 'running', 'awaiting_input', 'succeeded', "
            "'degraded', 'failed', 'skipped', 'cancelled')",
        )
