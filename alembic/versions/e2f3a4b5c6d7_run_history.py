"""run_history

Adds runs.history — a JSON list the orchestrator appends to after each
Zevo model-improvement iteration: one entry per completed loop recording what it
chose and how it scored, e.g.
  {"iteration": 1, "method": "lora_sft", "base_model": "Qwen/Qwen2.5-7B-Instruct",
   "accuracy": 0.7, "notes": "..."}.
The runner surfaces this list in the orchestrate ticket payload at the start of
the NEXT iteration so the orchestrator can pick a different method/model.

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-06-13 06:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("history", sa.JSON(), nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("runs", "history")
