"""per-agent sandbox (terminal) choice

Adds agents.sandbox: '' = inherit the global ZEVO_AGENT_SANDBOX default,
'none' = regular in-container terminal, 'openshell' = OpenShell sandbox.
Set from the Agent Configuration page's terminal toggle.

Revision ID: f5a6b7c8d9e0
Revises: f4a5b6c7d8e9
"""
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "f5a6b7c8d9e0"
down_revision: Union[str, None] = "f4a5b6c7d8e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent: backend + scheduler both run `alembic upgrade head` on startup,
    # so a cold `docker compose up` races them on this DDL. IF NOT EXISTS makes the
    # loser a no-op instead of crashing with DuplicateColumnError.
    op.execute(
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS "
        "sandbox VARCHAR(16) NOT NULL DEFAULT ''"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS sandbox")
