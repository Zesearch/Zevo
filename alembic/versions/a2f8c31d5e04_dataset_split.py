"""a setting's training data can name a hub split too

`validation_set` learned to carry a split and a config; `dataset` did not, so
the same hub repo could be pointed at precisely on one side of a setting and
only by name on the other. The runner filled the gap by looking the id up in the
catalogue, which works right up until the id was typed rather than registered —
and then the acquire step silently took `train`.

Revision ID: a2f8c31d5e04
Revises: f4a9d2e01b76
Create Date: 2026-08-11
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "a2f8c31d5e04"
down_revision = "f4a9d2e01b76"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("task_settings", sa.Column(
        "dataset_split", sa.String(length=64), nullable=False, server_default=""))
    op.add_column("task_settings", sa.Column(
        "dataset_config", sa.String(length=64), nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("task_settings", "dataset_config")
    op.drop_column("task_settings", "dataset_split")
