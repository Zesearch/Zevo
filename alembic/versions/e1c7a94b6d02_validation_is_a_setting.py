"""the validation set is part of the setting, not part of the task

A task is the PROBLEM: the objective, and the files that decide whether it was
solved — the held-out test set, its answer columns, the scorer, the submission
shape. Those must not drift, or the task stops being one task.

Which rows you tune against on the way there is not that. It is a choice about
how to attack the problem, the same kind of choice as which training data to
use — and it is usually cut out of that very training data, so the two belong
in the same place. Two runs that differ only in their validation split are two
attempts at one problem, which is exactly what a setting is for.

So the columns move from `tasks` to `task_settings`. Nothing is carried over:
no task row had a validation set, because the field existed for one migration
before this one and every shipped task left it empty (each run carves its own).

Revision ID: e1c7a94b6d02
Revises: d8b3c05f2a41
Create Date: 2026-08-10
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "e1c7a94b6d02"
down_revision = "d8b3c05f2a41"
branch_labels = None
depends_on = None


_JSON = postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def upgrade() -> None:
    op.add_column("task_settings", sa.Column(
        "validation_set", sa.Text(), nullable=False, server_default=""))
    op.add_column("task_settings", sa.Column(
        "validation_answer_columns", _JSON, nullable=False, server_default=sa.text("'[]'")))

    # Carry across anything a task did name, so an edit made between the two
    # migrations is not silently dropped. Settings are per-task, so every
    # setting of that task inherits it — which is the right reading: they were
    # all launched against the task as it stood.
    conn = op.get_bind()
    conn.execute(sa.text(
        "UPDATE task_settings s SET validation_set = t.validation_set, "
        "validation_answer_columns = t.validation_answer_columns "
        "FROM tasks t WHERE t.name = s.task_name AND t.validation_set <> ''"
    ))

    op.drop_column("tasks", "validation_answer_columns")
    op.drop_column("tasks", "validation_set")


def downgrade() -> None:
    op.add_column("tasks", sa.Column(
        "validation_set", sa.Text(), nullable=False, server_default=""))
    op.add_column("tasks", sa.Column(
        "validation_answer_columns", _JSON, nullable=False, server_default=sa.text("'[]'")))
    op.drop_column("task_settings", "validation_answer_columns")
    op.drop_column("task_settings", "validation_set")
