"""answer COLUMNS -> answer FIELDS: a scoring set is not always a table

The name was written when every scoring set was a CSV, so "which columns hold
the answers" was the whole question. It is not: a set can be JSON, where the
ground truth is a key rather than a column, and a multi-turn conversation is one
record whose answers live under a nested key. "Column" told the reader to go
looking for something the file does not have.

The values are unchanged — the same list of names, now called what they are.
Renamed rather than added-and-copied so there is one field, not two that can
disagree; `runs.holdout` carries the same two names inside its JSON and is
rewritten here for the same reason.

Revision ID: f1a63d0c8b27
Revises: e5b02c17d9a3
Create Date: 2026-08-11
"""
from __future__ import annotations

from alembic import op


revision = "f1a63d0c8b27"
down_revision = "e5b02c17d9a3"
branch_labels = None
depends_on = None


# (table, old, new)
_RENAMES = (
    ("tasks", "test_answer_columns", "test_answer_fields"),
    ("task_settings", "validation_answer_columns", "validation_answer_fields"),
)
# The same two names as keys inside runs.holdout.
_JSON_KEYS = (
    ("test_answer_columns", "test_answer_fields"),
    ("validation_answer_columns", "validation_answer_fields"),
)


def _rekey(old: str, new: str, forward: bool = True) -> str:
    """Move one key of runs.holdout, leaving every other key alone."""
    src, dst = (old, new) if forward else (new, old)
    return (
        f"UPDATE runs SET holdout = (holdout - '{src}') "
        f"|| jsonb_build_object('{dst}', holdout->'{src}') "
        f"WHERE holdout ? '{src}'"
    )


def upgrade() -> None:
    for table, old, new in _RENAMES:
        op.alter_column(table, old, new_column_name=new)
    for old, new in _JSON_KEYS:
        op.execute(_rekey(old, new))


def downgrade() -> None:
    for table, old, new in _RENAMES:
        op.alter_column(table, new, new_column_name=old)
    for old, new in _JSON_KEYS:
        op.execute(_rekey(old, new, forward=False))
