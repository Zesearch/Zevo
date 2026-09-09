"""a setting can name its own scorer for its own validation set

One `evaluation_script` scored both sets, which holds exactly as long as they
have the same shape. They need not: a validation set cut from the training data
carries the training file's columns, and one taken from a different hub split
carries that repo's. A scorer handed columns it does not read does not fail
loudly — it returns a number, and the loop steers by it.

Empty still means "the task's script", so nothing changes for a setting that
does not say otherwise.

Revision ID: c9d1f7b34e08
Revises: b7e4c02a9d15
Create Date: 2026-08-11
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "c9d1f7b34e08"
down_revision = "b7e4c02a9d15"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("task_settings", sa.Column(
        "validation_eval_script", sa.Text(), nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("task_settings", "validation_eval_script")
