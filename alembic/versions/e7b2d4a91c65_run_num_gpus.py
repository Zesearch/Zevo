"""runs.num_gpus — how many GPUs the launch form asked for.

Beside `runs.framework`, and for the same reason: a choice the user makes once
in the launch dialog, that the runner stamps onto ticket inputs rather than
routing through the orchestrator.

It matters most under `provider=instance`, where it is the LEASE size. That
allocation is the user's and is shared between runs, so a run that took all of
it would land on cards another run is already using.

Revision ID: e7b2d4a91c65
Revises: d5a1c7e93f04
Create Date: 2026-08-13
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "e7b2d4a91c65"
down_revision = "d5a1c7e93f04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1 for existing rows: that is what they actually ran with, since the
    # orchestrator's ticket template hardcoded num_gpus=1 until now.
    op.add_column(
        "runs",
        sa.Column("num_gpus", sa.Integer(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("runs", "num_gpus")
