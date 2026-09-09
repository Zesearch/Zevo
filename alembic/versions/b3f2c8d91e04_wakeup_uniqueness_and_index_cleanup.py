"""wakeup coalescing becomes a constraint; drop the stale-named duplicate index

Two fixes in one housekeeping revision:

1. `queue_wakeup` coalesced duplicates with SELECT-then-INSERT and nothing
   backed it — two concurrent enqueues for the same ticket could both land
   as 'queued', and with the advisory lock now keyed per wakeup, both would
   run. A partial unique index makes one-'queued'-row-per-(agent, ticket)
   a database guarantee. Pre-existing duplicates are demoted to 'coalesced'
   first (keeping the newest), or the index could not build.

2. `ticket_contract_v2` renamed `pipeline_phases` → `execution_events` and
   created `ix_execution_events_ticket_id`, but Postgres keeps index names
   across table renames — so `ix_pipeline_phases_ticket_id` (from
   add_fk_indexes) survived as a duplicate index on the same column under a
   misleading name. Drop it.

Revision ID: b3f2c8d91e04
Revises: f7a79e2b6a30
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op


revision: str = "b3f2c8d91e04"
down_revision: Union[str, None] = "f7a79e2b6a30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Keep the newest queued row per (agent_id, ticket_id); demote the rest.
    op.execute("""
        UPDATE agent_wakeup_requests w
        SET status = 'coalesced',
            reason = 'coalesced by migration b3f2c8d91e04 (duplicate queued row)'
        WHERE w.status = 'queued'
          AND w.ticket_id IS NOT NULL
          AND EXISTS (
            SELECT 1 FROM agent_wakeup_requests newer
            WHERE newer.agent_id = w.agent_id
              AND newer.ticket_id = w.ticket_id
              AND newer.status = 'queued'
              AND newer.created_at > w.created_at
          )
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_wakeup_queued_per_ticket
        ON agent_wakeup_requests (agent_id, ticket_id)
        WHERE status = 'queued' AND ticket_id IS NOT NULL
    """)
    op.execute("DROP INDEX IF EXISTS ix_pipeline_phases_ticket_id")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_wakeup_queued_per_ticket")
    # The stale-named duplicate is not resurrected: execution_events keeps
    # its correctly-named index either way.
