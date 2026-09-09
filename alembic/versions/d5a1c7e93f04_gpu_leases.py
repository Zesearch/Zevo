"""gpu_leases — arbitrate a shared `instance` allocation between runs.

`instance` is the one provider where several runs share hardware nobody
acquires: the user starts a Slurm allocation by hand, every run attaches to it
with `srun --overlap`, and `--overlap` is precisely the flag that tells Slurm
NOT to schedule. Before this table each run read the allocation's full GPU
count and launched an N-way job on it, so two concurrent runs put 2N processes
on N cards.

One row per (allocation, card, holder). The partial unique index is the actual
guarantee — read-then-write cannot be made safe in application code, because
two requests can both observe the same card free.

Revision ID: d5a1c7e93f04
Revises: b3f7e91c4d20
Create Date: 2026-08-13
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "d5a1c7e93f04"
down_revision = "b3f7e91c4d20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gpu_leases",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("jobid", sa.String(length=64), nullable=False),
        sa.Column("node", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("gpu_index", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("ticket_id", sa.String(length=64), nullable=True),
        sa.Column("gpu_name", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("vram_gb", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "acquired_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("release_reason", sa.Text(), nullable=False, server_default=""),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["ticket_id"], ["tickets.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_gpu_leases_jobid", "gpu_leases", ["jobid"])
    op.create_index("ix_gpu_leases_run_id", "gpu_leases", ["run_id"])
    op.create_index("ix_gpu_leases_acquired_at", "gpu_leases", ["acquired_at"])
    op.create_index("ix_gpu_leases_released_at", "gpu_leases", ["released_at"])
    # Partial, so a released row stays as history instead of blocking the card
    # forever. This is what makes a concurrent double-grant an IntegrityError
    # the API can retry rather than two runs quietly sharing card 3.
    op.create_index(
        "ux_gpu_lease_live_card",
        "gpu_leases",
        ["jobid", "gpu_index"],
        unique=True,
        postgresql_where=sa.text("released_at IS NULL"),
        sqlite_where=sa.text("released_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ux_gpu_lease_live_card", table_name="gpu_leases")
    op.drop_index("ix_gpu_leases_released_at", table_name="gpu_leases")
    op.drop_index("ix_gpu_leases_acquired_at", table_name="gpu_leases")
    op.drop_index("ix_gpu_leases_run_id", table_name="gpu_leases")
    op.drop_index("ix_gpu_leases_jobid", table_name="gpu_leases")
    op.drop_table("gpu_leases")
