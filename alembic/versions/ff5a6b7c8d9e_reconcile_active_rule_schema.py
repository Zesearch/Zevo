"""reconcile the replaced Run-governance schema

Revision ID: ff5a6b7c8d9e
Revises: fe4f5a6b7c8d

The governance revisions were intentionally squashed before release.  A local
database may nevertheless already be stamped at their revision ids after
running the superseded definitions.  Bring that database to the schema now
defined by the models without retaining the removed contract workflow.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "ff5a6b7c8d9e"
down_revision = "fe4f5a6b7c8d"
branch_labels = None
depends_on = None


_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    run_columns = {column["name"] for column in inspector.get_columns("runs")}
    if "iteration_intents" not in run_columns:
        op.add_column(
            "runs",
            sa.Column(
                "iteration_intents", _JSON, nullable=False, server_default="{}"
            ),
        )
    if "active_rules" not in run_columns:
        op.add_column(
            "runs",
            sa.Column(
                "active_rules",
                _JSON,
                nullable=False,
                server_default='{"version": 0, "rules": {}}',
            ),
        )

    # The old proposal/review/freeze state is replaced, not aliased.
    for column_name in (
        "iteration_contracts",
        "inference_contract",
        "prompt_contract",
    ):
        if column_name in run_columns:
            op.drop_column("runs", column_name)

    if inspector.has_table("agent_memory_entries"):
        checks = {
            check["name"]: check.get("sqltext", "")
            for check in inspector.get_check_constraints("agent_memory_entries")
        }
        status_check = checks.get("ck_agent_memory_status", "")
        if "promoted" in status_check:
            op.drop_constraint(
                "ck_agent_memory_status",
                "agent_memory_entries",
                type_="check",
            )
            op.execute(
                sa.text(
                    "UPDATE agent_memory_entries "
                    "SET status = 'accepted' WHERE status = 'promoted'"
                )
            )
            op.create_check_constraint(
                "ck_agent_memory_status",
                "agent_memory_entries",
                "status IN ('active', 'superseded', 'accepted', 'dismissed')",
            )


def downgrade() -> None:
    # This repair only reconciles databases that ran superseded definitions of
    # already-squashed revisions.  The preceding revision in the current tree
    # defines the same schema, so there is nothing to reverse.
    pass
