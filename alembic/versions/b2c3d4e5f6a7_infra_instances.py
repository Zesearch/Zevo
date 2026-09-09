"""infra_instances

Creates the infra_instances table so the infrastructure agent can record
each GPU it provisions / connects to. Used for leak detection and
cumulative cost tracking.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-01 23:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "infra_instances",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("external_id", sa.String(128), nullable=False, server_default=""),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="provisioning"),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("ticket_id", sa.String(64), sa.ForeignKey("tickets.id", ondelete="SET NULL"), nullable=True),
        sa.Column("gpu_name", sa.String(64), nullable=False, server_default=""),
        sa.Column("gpu_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("vram_gb", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("dph", sa.Float(), nullable=False, server_default="0"),
        sa.Column("ssh_host", sa.String(128), nullable=False, server_default=""),
        sa.Column("ssh_port", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ssh_user", sa.String(64), nullable=False, server_default=""),
        sa.Column("meta", sa.dialects.postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("release_reason", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("ix_infra_instances_run_id", "infra_instances", ["run_id"])
    op.create_index("ix_infra_instances_ticket_id", "infra_instances", ["ticket_id"])
    op.create_index("ix_infra_instances_created_at", "infra_instances", ["created_at"])
    op.create_index("ix_infra_instances_released_at", "infra_instances", ["released_at"])
    op.create_index(
        "ix_infra_instances_provider_external_id",
        "infra_instances",
        ["provider", "external_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_infra_instances_provider_external_id", table_name="infra_instances")
    op.drop_index("ix_infra_instances_released_at", table_name="infra_instances")
    op.drop_index("ix_infra_instances_created_at", table_name="infra_instances")
    op.drop_index("ix_infra_instances_ticket_id", table_name="infra_instances")
    op.drop_index("ix_infra_instances_run_id", table_name="infra_instances")
    op.drop_table("infra_instances")
