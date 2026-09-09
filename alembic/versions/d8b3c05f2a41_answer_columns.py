"""declare the answer columns instead of shipping a stripped duplicate

A task used to name four scoring files: the test set, a hand-made copy of it
with the answers removed, and the same pair again once validation arrived. The
copies are derivable — they are the original minus its ground-truth columns —
and a derived file kept by hand is a file that goes stale: edit `test.csv` and
`test_public.csv` silently describes a different set of rows, which nothing
detects and which shows up as an inference run misaligned with its own scorer.

So a task declares the COLUMNS now, and the data agent produces the
questions-only copies from them.

The backfill reads both files where they are still on disk and takes the
difference of their headers — that difference IS the answer columns, stated by
whoever made the pair. Where a file is missing or unreadable the columns are
left empty, which reads as "this task has no ground truth to hide"; a task in
that state has to be told its columns before its next run, and the alternative
(guessing from column names) would eventually guess wrong and leak an answer
into a prompt.

Revision ID: d8b3c05f2a41
Revises: c4f2a8e17b93
Create Date: 2026-08-10
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "d8b3c05f2a41"
down_revision = "c4f2a8e17b93"
branch_labels = None
depends_on = None


_JSON = postgresql.JSONB().with_variant(sa.JSON(), "sqlite")


def _header(path: str) -> list[str]:
    """The column names of a CSV, or [] if it cannot be read.

    A migration must not fail because a task points at a file someone deleted.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return []
        with p.open(newline="", encoding="utf-8") as fh:
            return list(next(csv.reader(fh), []))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def upgrade() -> None:
    op.add_column("tasks", sa.Column(
        "test_answer_columns", _JSON, nullable=False, server_default=sa.text("'[]'")))
    op.add_column("tasks", sa.Column(
        "validation_answer_columns", _JSON, nullable=False, server_default=sa.text("'[]'")))

    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT name, test_set, test_set_public FROM tasks"
    )).all()
    for r in rows:
        full = _header(r.test_set or "")
        public = _header(r.test_set_public or "")
        if not full or not public:
            continue
        answers = [c for c in full if c not in public]
        if not answers:
            continue
        conn.execute(
            sa.text("UPDATE tasks SET test_answer_columns = :cols WHERE name = :n"),
            {"cols": json.dumps(answers), "n": r.name},
        )

    op.drop_column("tasks", "validation_set_public")
    op.drop_column("tasks", "test_set_public")


def downgrade() -> None:
    op.add_column("tasks", sa.Column(
        "test_set_public", sa.Text(), nullable=False, server_default=""))
    op.add_column("tasks", sa.Column(
        "validation_set_public", sa.Text(), nullable=False, server_default=""))
    # The sibling convention is what the column held for every shipped task, so
    # rebuilding it from the test set's path restores the same values.
    conn = op.get_bind()
    conn.execute(sa.text(
        "UPDATE tasks SET test_set_public = "
        "regexp_replace(test_set, 'test\\.csv$', 'test_public.csv') "
        "WHERE test_set LIKE '%/test.csv'"
    ))
    op.drop_column("tasks", "validation_answer_columns")
    op.drop_column("tasks", "test_answer_columns")
