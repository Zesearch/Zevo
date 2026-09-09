"""reconcile model-vs-DB drift found by `alembic check`

Three columns/constraints had drifted between the ORM models and the migrated
schema (models widened/tightened without a migration):

  - score_events.metric_name: model is String(64), DB was VARCHAR(32). Widen it
    (keeps the NOT NULL and the 'accuracy' server default).
  - ssh_hosts.gpu_info: model is NOT NULL (default {}), DB allowed NULL. Backfill
    any NULLs to '{}' then set NOT NULL.

The third drift -- the ck_registry_models_metric_direction check the model had
dropped -- is realigned by re-declaring the constraint on the model (it already
exists in the DB), so there is no DDL for it here.

After this migration `alembic check` reports no drift.

Revision ID: ce7a1b4f92d0
Revises: bc9f3a2e7d14
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "ce7a1b4f92d0"
down_revision = "bc9f3a2e7d14"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("score_events") as batch:
        batch.alter_column(
            "metric_name",
            existing_type=sa.String(length=32),
            type_=sa.String(length=64),
            existing_nullable=False,
            existing_server_default=sa.text("'accuracy'::character varying"),
        )
    # Satisfy the new NOT NULL for any legacy rows before enforcing it.
    op.execute("UPDATE ssh_hosts SET gpu_info = '{}' WHERE gpu_info IS NULL")
    with op.batch_alter_table("ssh_hosts") as batch:
        batch.alter_column(
            "gpu_info",
            existing_type=JSONB(astext_type=sa.Text()),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("ssh_hosts") as batch:
        batch.alter_column(
            "gpu_info",
            existing_type=JSONB(astext_type=sa.Text()),
            nullable=True,
        )
    with op.batch_alter_table("score_events") as batch:
        batch.alter_column(
            "metric_name",
            existing_type=sa.String(length=64),
            type_=sa.String(length=32),
            existing_nullable=False,
            existing_server_default=sa.text("'accuracy'::character varying"),
        )
