"""a setting's validation set can be a hub split

`validation_set` could already hold a HuggingFace id, but an id alone does not
name a set: a repo publishes `train`, `validation` and `test`, often across
several configs. Guessing would be the worst option available — defaulting to
`train` puts training rows in the validation set, and nothing downstream can
tell that happened.

So the slice is declared alongside the id, and run creation fetches exactly it
to a local file before any agent starts.

Revision ID: f4a9d2e01b76
Revises: e1c7a94b6d02
Create Date: 2026-08-10
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "f4a9d2e01b76"
down_revision = "e1c7a94b6d02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("task_settings", sa.Column(
        "validation_split", sa.String(length=64), nullable=False, server_default=""))
    op.add_column("task_settings", sa.Column(
        "validation_config", sa.String(length=64), nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("task_settings", "validation_config")
    op.drop_column("task_settings", "validation_split")
