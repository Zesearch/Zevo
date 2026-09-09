"""Give a task an explicit questions-only test set.

Inference found it by filename convention — a `test_public.csv` sibling of the
test set — which meant the split existed but could not be seen, chosen, or
named anywhere in the UI. It is a field now; the convention stays as the
fallback so tasks and runs created before this keep working.

Backfills the shipped tasks whose sibling actually exists.

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "d2e3f4a5b6c7"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("test_set_public", sa.Text(), nullable=False,
                                     server_default=""))
    # Everything shipped names test.csv; its questions-only twin sits beside it.
    op.execute(
        "UPDATE tasks SET test_set_public = "
        "regexp_replace(test_set, 'test\\.csv$', 'test_public.csv') "
        "WHERE test_set LIKE '%/test.csv'"
    )


def downgrade() -> None:
    op.drop_column("tasks", "test_set_public")
