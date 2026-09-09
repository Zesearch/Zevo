"""add Run-scoped Agent memory

Revision ID: fe4f5a6b7c8d
Revises: fc2d3e4f5a6b
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "fe4f5a6b7c8d"
down_revision = "fc2d3e4f5a6b"
branch_labels = None
depends_on = None


_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "agent_memory_entries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("agent_id", sa.String(length=64), nullable=False),
        sa.Column(
            "lane", sa.String(length=24), nullable=False,
            server_default="optimization",
        ),
        sa.Column("iteration", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("details", _JSON, nullable=False, server_default="{}"),
        sa.Column(
            "visibility", sa.String(length=24), nullable=False,
            server_default="agent_local",
        ),
        sa.Column("applies_to", _JSON, nullable=False, server_default="{}"),
        sa.Column(
            "status", sa.String(length=16), nullable=False,
            server_default="active",
        ),
        sa.Column("source_ticket_id", sa.String(length=64), nullable=False),
        sa.Column("supersedes_id", sa.String(length=36), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "lane IN ('optimization', 'held_out_test')",
            name="ck_agent_memory_lane",
        ),
        sa.CheckConstraint(
            "kind IN ('verified_fact', 'pitfall', 'runtime_finding', "
            "'experiment_finding', 'artifact_reference', 'recommendation')",
            name="ck_agent_memory_kind",
        ),
        sa.CheckConstraint(
            "visibility IN ('agent_local', 'shared_candidate')",
            name="ck_agent_memory_visibility",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'accepted', 'dismissed')",
            name="ck_agent_memory_status",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_ticket_id"], ["tickets.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_id"], ["agent_memory_entries.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_agent_memory_entries_run_id", "agent_memory_entries", ["run_id"]
    )
    op.create_index(
        "ix_agent_memory_entries_agent_id", "agent_memory_entries", ["agent_id"]
    )
    op.create_index(
        "ix_agent_memory_entries_kind", "agent_memory_entries", ["kind"]
    )
    op.create_index(
        "ix_agent_memory_entries_key", "agent_memory_entries", ["key"]
    )
    op.create_index(
        "ix_agent_memory_entries_status", "agent_memory_entries", ["status"]
    )
    op.create_index(
        "ix_agent_memory_entries_source_ticket_id",
        "agent_memory_entries", ["source_ticket_id"],
    )
    op.create_index(
        "ix_agent_memory_scope",
        "agent_memory_entries", ["run_id", "agent_id", "lane", "status"],
    )


def downgrade() -> None:
    op.drop_table("agent_memory_entries")
