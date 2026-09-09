"""drop objectives + webhook_endpoints (features removed)

Removes the auto-improve (Objectives) and webhooks features entirely. The
ORM models, routers, daemon tick, and frontend pages are gone; this drops
their now-orphaned tables. Downgrade recreates the tables (empty) so the
migration chain stays reversible.

Revision ID: f3a4b5c6d7e8
Revises: e2f3a4b5c6d7
Create Date: 2026-06-13
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f3a4b5c6d7e8"
down_revision = "e2f3a4b5c6d7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("objectives")
    op.drop_table("webhook_endpoints")


def downgrade() -> None:
    op.create_table(
        "objectives",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("user_request", sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"), nullable=False),
        sa.Column("schedule_minutes", sa.Integer(), server_default="60", nullable=False),
        sa.Column("iteration_budget", sa.Integer(), server_default="3", nullable=False),
        sa.Column("target_accuracy", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("enabled", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_id", sa.String(length=36), nullable=False),
        sa.Column("best_accuracy_ever", sa.Float(), server_default="-1.0", nullable=False),
        sa.Column("best_registry_tag", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "webhook_endpoints",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("event_types", sa.String(256), nullable=False, server_default="*"),
        sa.Column("secret", sa.String(128), nullable=False, server_default=""),
        sa.Column("active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("label", sa.String(128), nullable=False, server_default=""),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status_code", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
