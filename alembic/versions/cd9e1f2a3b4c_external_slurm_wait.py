"""add external scheduler wait state and per-connection container image

Revision ID: cd9e1f2a3b4c
Revises: bc8e2f4a6c0d
"""

import sqlalchemy as sa
from alembic import op


revision = "cd9e1f2a3b4c"
down_revision = "bc8e2f4a6c0d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("tickets") as batch:
        batch.drop_constraint("ck_tickets_status", type_="check")
        batch.create_check_constraint(
            "ck_tickets_status",
            "status IN ('queued', 'running', 'repairing', 'awaiting_input', "
            "'waiting_external', 'succeeded', 'degraded', 'failed', 'skipped', "
            "'cancelled')",
        )
    with op.batch_alter_table("ssh_hosts") as batch:
        batch.add_column(sa.Column(
            "container_image", sa.Text(), nullable=False, server_default="",
        ))


def downgrade() -> None:
    op.execute(
        "UPDATE tickets SET status = 'failed' WHERE status = 'waiting_external'"
    )
    with op.batch_alter_table("ssh_hosts") as batch:
        batch.drop_column("container_image")
    with op.batch_alter_table("tickets") as batch:
        batch.drop_constraint("ck_tickets_status", type_="check")
        batch.create_check_constraint(
            "ck_tickets_status",
            "status IN ('queued', 'running', 'repairing', 'awaiting_input', "
            "'succeeded', 'degraded', 'failed', 'skipped', 'cancelled')",
        )
