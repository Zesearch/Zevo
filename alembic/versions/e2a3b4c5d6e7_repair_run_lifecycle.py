"""Repair databases stamped past the Run lifecycle column.

Revision ID: e2a3b4c5d6e7
Revises: e1f2a3b4c5d6

Some development databases were stamped at ``e1f2a3b4c5d6`` without the
``e1a2b3c4d5f6`` column being present.  ORM reads then fail before any Run can
be scheduled.  Keep this repair idempotent so correctly migrated databases are
unchanged.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "e2a3b4c5d6e7"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("runs")}
    if "lifecycle" not in columns:
        op.add_column(
            "runs",
            sa.Column(
                "lifecycle",
                sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
                nullable=False,
                server_default="{}",
            ),
        )


def downgrade() -> None:
    # e1a2b3c4d5f6 already owns this column. Downgrading only this corrective
    # revision must restore that expected schema, which means leaving it in place.
    pass
