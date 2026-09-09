"""a setting can name its own submission template for its own validation set

The fourth and last of the files that define a scoring set. The other three
(the set, its answer columns, its scorer) already come in a validation flavour;
the template said which columns inference must emit and only ever came from the
test set. Predictions shaped for the test set's template are predictions the
validation scorer may not be able to read at all, which surfaces as a score
rather than as an error.

Empty means "the task's template", and failing that one the data agent writes
from the data itself, so nothing changes for a setting that does not say
otherwise.

Revision ID: e5b02c17d9a3
Revises: c9d1f7b34e08
Create Date: 2026-08-11
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "e5b02c17d9a3"
down_revision = "c9d1f7b34e08"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("task_settings", sa.Column(
        "validation_sample_submission", sa.Text(), nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("task_settings", "validation_sample_submission")
