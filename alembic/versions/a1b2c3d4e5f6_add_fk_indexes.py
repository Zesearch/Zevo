"""add_fk_indexes

Adds indexes on the foreign-key columns that the scheduler / runner /
reconciler query on every tick. Without these, hot-path queries like
`select(Ticket).where(run_id=...)` or
`select(HeartbeatRun).where(ticket_id=...)` table-scan, which gets
noticeable past ~50 tickets per run.

Revision ID: a1b2c3d4e5f6
Revises: fbf81d644eca
Create Date: 2026-05-31 23:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'fbf81d644eca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Each tuple: (index_name, table, columns).
# Names follow Alembic's default convention so future autogenerate runs
# don't try to re-create them.
INDEXES = [
    ("ix_tickets_run_id",                    "tickets",                ["run_id"]),
    ("ix_tickets_agent_id",                  "tickets",                ["agent_id"]),
    ("ix_tickets_status",                    "tickets",                ["status"]),
    ("ix_heartbeat_runs_ticket_id",          "heartbeat_runs",         ["ticket_id"]),
    ("ix_heartbeat_runs_agent_id",           "heartbeat_runs",         ["agent_id"]),
    ("ix_heartbeat_runs_started_at",         "heartbeat_runs",         ["started_at"]),
    ("ix_work_products_ticket_id",           "work_products",          ["ticket_id"]),
    ("ix_comments_ticket_id",                "comments",               ["ticket_id"]),
    ("ix_pipeline_phases_ticket_id",         "pipeline_phases",        ["ticket_id"]),
    ("ix_agent_wakeup_requests_ticket_id",   "agent_wakeup_requests",  ["ticket_id"]),
    ("ix_agent_wakeup_requests_agent_id",    "agent_wakeup_requests",  ["agent_id"]),
    ("ix_agent_wakeup_requests_status",      "agent_wakeup_requests",  ["status"]),
]


def upgrade() -> None:
    for name, table, cols in INDEXES:
        op.create_index(name, table, cols, if_not_exists=True)


def downgrade() -> None:
    for name, table, _ in reversed(INDEXES):
        op.drop_index(name, table_name=table, if_exists=True)
