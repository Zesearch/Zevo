"""drop tasks.answer_columns — three other fields already do its job

It named the ground-truth columns, for two purposes that no longer need it:

  * inference dropped them before prompting — but a task now supplies
    `test_set_public`, a questions-only file those columns are absent from, so
    the drop was a documented no-op;
  * evaluation read one of them as its gold column — but only on the fallback
    path taken when a task supplies no `evaluation_script`, and the scoring
    script is part of what a task IS.

`sample_submission` already fixes the shape of a prediction, so nothing is left
for it to say. The fallback scorer now reads the gold column off the test set's
header and records which one it picked.

Revision ID: f2b7d4e6a913
Revises: e7a2c5d9f318
Create Date: 2026-08-09
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "f2b7d4e6a913"
down_revision = "e7a2c5d9f318"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("tasks", "answer_columns")


def downgrade() -> None:
    """The column comes back empty: what it held was a per-task list nothing in
    the system reads any more, so there is nowhere to recover it from."""
    op.add_column(
        "tasks",
        sa.Column("answer_columns", sa.JSON(), nullable=False, server_default="[]"),
    )
