"""runs.num_gpus: 0 now means "the user did not pick".

The column arrived defaulting to 1, which conflated "they asked for one card"
with "they said nothing". Those want opposite handling: a number the user typed
must survive whatever the orchestrator plans, while silence is exactly the case
the orchestrator should fill in, the same way it sizes RAM, cores and time.

Only the DEFAULT moves. Existing rows keep the 1 they were backfilled with,
because that is what they actually ran on: the orchestrator's ticket template
hardcoded num_gpus=1 for their whole lifetime.

Revision ID: a9d3f61e08b4
Revises: e7b2d4a91c65
Create Date: 2026-08-13
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a9d3f61e08b4"
down_revision = "e7b2d4a91c65"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("runs", "num_gpus", server_default="0")


def downgrade() -> None:
    op.alter_column("runs", "num_gpus", server_default="1")
