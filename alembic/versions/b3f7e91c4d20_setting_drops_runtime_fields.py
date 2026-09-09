"""a setting is how the problem is attacked, not where it is run

`gpu_provider` and `framework` answer a different question from the rest of a
setting. Which base model, which method, which data: those are a line of
attack, and two of them are worth comparing on a leaderboard. Which GPU you
rented today and which server generates the tokens are facts about a machine —
the same experiment run on a cluster and on a rented box is the same
experiment.

Keeping them was not only untidy, it broke launches. A setting recorded before
`cloud` had a value of its own stored the provider as "" meaning "whatever this
deployment defaults to". Picking that setting wrote the blank into a form whose
dropdown has no blank option, so the browser displayed the first one —
`instance` — while the state held "". The run went out with no provider, the
server resolved it to `cloud` because a Vast.ai key was present, and a run the
user had every reason to believe was attaching to their own allocation started
renting a GPU instead.

Both are per-run choices now, defaulted in the launch dialog and sent
explicitly, so nothing has to be resolved from an absence.

Revision ID: b3f7e91c4d20
Revises: f1a63d0c8b27
Create Date: 2026-08-12
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "b3f7e91c4d20"
down_revision = "f1a63d0c8b27"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("task_settings", "gpu_provider")
    op.drop_column("task_settings", "framework")


def downgrade() -> None:
    # The values are not recoverable — they were a per-run choice — so the
    # columns come back with the defaults a fresh row would have had.
    op.add_column("task_settings", sa.Column(
        "gpu_provider", sa.String(16), nullable=False, server_default=""))
    op.add_column("task_settings", sa.Column(
        "framework", sa.String(16), nullable=False, server_default="vllm"))
