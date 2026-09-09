"""persist the semantic operation of every heartbeat activation

Revision ID: a16b7c8d9e0f
Revises: ff5a6b7c8d9e
"""

import json

import sqlalchemy as sa
from alembic import op


revision = "a16b7c8d9e0f"
down_revision = "ff5a6b7c8d9e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "heartbeat_runs",
        sa.Column("operation", sa.String(length=64), nullable=False, server_default=""),
    )

    # Preserve truthful labels for activations already visible in the current
    # Run history. Successful results are authoritative. A validation failure
    # may have no HeartbeatResult at all; in that case the Ticket still carries
    # the operation that activation attempted.
    bind = op.get_bind()
    heartbeat_runs = sa.table(
        "heartbeat_runs",
        sa.column("id", sa.String),
        sa.column("ticket_id", sa.String),
        sa.column("operation", sa.String),
    )
    heartbeat_results = sa.table(
        "heartbeat_results",
        sa.column("heartbeat_id", sa.String),
        sa.column("output", sa.JSON),
    )
    tickets = sa.table(
        "tickets",
        sa.column("id", sa.String),
        sa.column("payload", sa.JSON),
    )

    def as_object(value: object) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
                return decoded if isinstance(decoded, dict) else {}
            except (TypeError, ValueError):
                return {}
        return {}

    result_operations = {
        str(heartbeat_id): str(as_object(output).get("operation") or "")
        for heartbeat_id, output in bind.execute(sa.select(
            heartbeat_results.c.heartbeat_id,
            heartbeat_results.c.output,
        ))
    }
    ticket_operations = {
        str(ticket_id): str(as_object(payload).get("operation") or "")
        for ticket_id, payload in bind.execute(sa.select(
            tickets.c.id,
            tickets.c.payload,
        ))
    }
    for heartbeat_id, ticket_id in bind.execute(sa.select(
        heartbeat_runs.c.id,
        heartbeat_runs.c.ticket_id,
    )):
        operation = (
            result_operations.get(str(heartbeat_id), "")
            or ticket_operations.get(str(ticket_id), "")
        )
        if operation:
            bind.execute(
                heartbeat_runs.update()
                .where(heartbeat_runs.c.id == heartbeat_id)
                .values(operation=operation)
            )


def downgrade() -> None:
    op.drop_column("heartbeat_runs", "operation")
