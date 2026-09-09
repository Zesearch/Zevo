"""make the per-agent sandbox contract explicit

Revision ID: c4d5e6f7a8b9
Revises: b3f2c8d91e04
Create Date: 2026-08-15
"""
from typing import Sequence, Union

from alembic import op


revision: str = "c4d5e6f7a8b9"
down_revision: Union[str, None] = "b3f2c8d91e04"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE agents SET sandbox = 'none' WHERE sandbox <> 'none'")
    op.execute("ALTER TABLE agents ALTER COLUMN sandbox SET DEFAULT 'none'")
    op.create_check_constraint(
        "ck_agents_sandbox_capability",
        "agents",
        "sandbox = 'none' OR "
        "(sandbox = 'openshell' AND id = 'orchestrator' "
        "AND default_driver = 'claude_cli')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_agents_sandbox_capability", "agents", type_="check")
    op.execute("ALTER TABLE agents ALTER COLUMN sandbox SET DEFAULT ''")
