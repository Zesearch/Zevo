"""add runs.framework (generation backend: hf | vllm)

The whole run picks one generation backend — inference (predict.py) and any
training method that generates during training (GRPO rollouts). Stored on the
run so the harness runner can stamp it onto the inference/train ticket inputs
without the orchestrator having to forward it. Existing rows backfill to 'hf'.

Revision ID: f4a5b6c7d8e9
Revises: f3a4b5c6d7e8
Create Date: 2026-07-09
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "f4a5b6c7d8e9"
down_revision = "f3a4b5c6d7e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("framework", sa.String(length=16), nullable=False, server_default="hf"),
    )


def downgrade() -> None:
    op.drop_column("runs", "framework")
