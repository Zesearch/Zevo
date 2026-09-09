"""runs.ssh_host_id -> real FK to ssh_hosts with ON DELETE SET NULL

Before this, runs.ssh_host_id was a plain NOT NULL String(36) (default '') with
no foreign key, so deleting an SshHost left every run that used it pointing at a
non-existent id forever. Make it a real FK: deleting the profile now sets the
column back to NULL instead of dangling.

ON DELETE SET NULL requires the column to be nullable and forbids the '' default
(an empty string could never satisfy the FK), so we: make it nullable + drop the
'' server_default, normalize legacy values ('' and any dangling id -> NULL), then
add the FK. An index backs the SET NULL fan-out.

Revision ID: bc9f3a2e7d14
Revises: bb7d1e3f5a9c
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "bc9f3a2e7d14"
down_revision = "bb7d1e3f5a9c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1) nullable + drop the '' default so NULL becomes the "no profile" value.
    with op.batch_alter_table("runs") as batch:
        batch.alter_column(
            "ssh_host_id",
            existing_type=sa.String(length=36),
            nullable=True,
            server_default=None,
        )
    # 2) normalize legacy rows so the FK can be added without violation.
    op.execute(
        "UPDATE runs SET ssh_host_id = NULL "
        "WHERE ssh_host_id = '' "
        "OR ssh_host_id NOT IN (SELECT id FROM ssh_hosts)"
    )
    # 3) index + real FK with ON DELETE SET NULL.
    with op.batch_alter_table("runs") as batch:
        batch.create_index("ix_runs_ssh_host_id", ["ssh_host_id"])
        batch.create_foreign_key(
            "fk_runs_ssh_host_id",
            "ssh_hosts",
            ["ssh_host_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_constraint("fk_runs_ssh_host_id", type_="foreignkey")
        batch.drop_index("ix_runs_ssh_host_id")
    # Restore the NOT NULL '' contract: no NULLs may remain first.
    op.execute("UPDATE runs SET ssh_host_id = '' WHERE ssh_host_id IS NULL")
    with op.batch_alter_table("runs") as batch:
        batch.alter_column(
            "ssh_host_id",
            existing_type=sa.String(length=36),
            nullable=False,
            server_default="",
        )
