"""hold the test set back: optimize on a validation split instead

The loop scored itself on the user's test set and handed that number to the
orchestrator, which then chose the next base model, the next method and when to
stop from it. Twenty iterations of that is twenty decisions fitted to the set
the run is judged by, and the reported accuracy stops predicting anything.

So the set the run reports on and the set it tunes on come apart here:

  * `tasks.validation_set` / `validation_set_public` — what every iteration is
    scored on, and the only score an agent ever sees. Empty means the harness
    carves one at run creation (see zevo.engine.method.validation_split).
  * `score_events.split` — 'validation' or 'test'. Existing rows become 'test':
    back then the loop really was tuning on the test set, and calling those
    numbers validation would rewrite what happened.
  * `runs.test_accuracy` / `best_test_accuracy` — the held-out numbers, written
    by the harness alone. `runs.best_accuracy` keeps its column but now means
    the best VALIDATION score, so the existing values (test scores) are copied
    across to `best_test_accuracy` before it takes on its new meaning.
  * `tickets.held_out` — the harness-spawned test infer/eval pair. These never
    wake the orchestrator and are filtered out of agent-facing reads.

Revision ID: c4f2a8e17b93
Revises: a4c8e1b5d720
Create Date: 2026-08-10
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "c4f2a8e17b93"
down_revision = "a4c8e1b5d720"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("validation_set", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "tasks",
        sa.Column("validation_set_public", sa.Text(), nullable=False, server_default=""),
    )

    # Left empty rather than pointed at a `validation.csv` sibling: no shipped
    # bundle has one, and a task naming a file that is not there fails at run
    # creation instead of falling through to the carve. Run creation adopts the
    # sibling when it does exist (zevo.engine.method.validation_split.resolve).
    conn = op.get_bind()

    op.add_column(
        "score_events",
        sa.Column("split", sa.String(length=16), nullable=False, server_default="test"),
    )
    op.add_column(
        "runs",
        sa.Column("test_accuracy", sa.Float(), nullable=False, server_default="-1.0"),
    )
    op.add_column(
        "runs",
        sa.Column("best_test_accuracy", sa.Float(), nullable=False, server_default="-1.0"),
    )
    op.add_column(
        "runs",
        sa.Column(
            "holdout",
            postgresql.JSONB().with_variant(sa.JSON(), "sqlite"),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    # Every score a finished run recorded was a test score. Move them across so
    # the leaderboard and the run list keep reading the same numbers they did
    # yesterday, then leave best_accuracy alone: it is a validation column from
    # here on, and for these runs the two happen to coincide.
    conn.execute(sa.text(
        "UPDATE runs SET best_test_accuracy = best_accuracy, test_accuracy = headline_accuracy"
    ))

    op.add_column(
        "tickets",
        sa.Column("held_out", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    op.drop_column("tickets", "held_out")
    op.drop_column("runs", "holdout")
    op.drop_column("runs", "best_test_accuracy")
    op.drop_column("runs", "test_accuracy")
    op.drop_column("score_events", "split")
    op.drop_column("tasks", "validation_set_public")
    op.drop_column("tasks", "validation_set")
