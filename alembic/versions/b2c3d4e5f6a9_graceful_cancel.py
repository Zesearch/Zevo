"""Graceful cancel: rescue the champion checkpoint before the GPU goes away.

Three columns on `runs`:

* `cancel_requested_at` -- set by POST /runs/{id}/cancel when the operator
  asks to keep the weights. The Run stays `running` (its tickets already
  cancelled) so no reconciler pass destroys the box, until the daemon has
  copied the checkpoint off it;
* `cancel_policy` -- the CancelWeightsPolicy the operator chose;
* `cancel_outcome` -- what the rescue did: model path, Hub URL, or error.

Revision ID: b2c3d4e5f6a9
Revises: f1a2b3c4d5e7
Create Date: 2026-09-07
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "b2c3d4e5f6a9"
down_revision = "f1a2b3c4d5e7"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("runs", sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("runs", sa.Column("cancel_policy", _JSON, nullable=False, server_default="{}"))
    op.add_column("runs", sa.Column("cancel_outcome", _JSON, nullable=False, server_default="{}"))


def downgrade() -> None:
    op.drop_column("runs", "cancel_outcome")
    op.drop_column("runs", "cancel_policy")
    op.drop_column("runs", "cancel_requested_at")
