"""add task pins and Run governance state

Revision ID: fc2d3e4f5a6b
Revises: fb1c2d3e4f5a
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "fc2d3e4f5a6b"
down_revision = "fb1c2d3e4f5a"
branch_labels = None
depends_on = None


_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    for name, kind in (
        ("prompt_framing", sa.String(length=256)),
        ("thinking_mode", sa.String(length=16)),
        ("system_prompt", sa.Text()),
    ):
        op.add_column(
            "task_settings",
            sa.Column(name, kind, nullable=False, server_default=""),
        )
    for name in ("loss_objective_config", "inference_config", "decoding_config"):
        op.add_column(
            "task_settings",
            sa.Column(name, _JSON, nullable=False, server_default="{}"),
        )
    for name in ("decision_pins", "model_lineages", "iteration_intents"):
        op.add_column(
            "runs", sa.Column(name, _JSON, nullable=False, server_default="{}"),
        )
    op.add_column(
        "runs",
        sa.Column(
            "active_rules", _JSON, nullable=False,
            server_default='{"version": 0, "rules": {}}',
        ),
    )


def downgrade() -> None:
    for name in (
        "active_rules", "iteration_intents", "model_lineages", "decision_pins",
    ):
        op.drop_column("runs", name)
    for name in (
        "decoding_config", "inference_config", "loss_objective_config",
        "system_prompt", "thinking_mode", "prompt_framing",
    ):
        op.drop_column("task_settings", name)
