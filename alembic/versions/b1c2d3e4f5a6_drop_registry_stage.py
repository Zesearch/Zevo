"""drop registry_models.stage

The model-lifecycle feature this column served — experimental -> staging ->
production, with accuracy gates, a unique production slot and a rollback —
lost its UI when the Registry page was rebuilt, and nothing replaced it: no
button, no CLI command, no other caller. The audit log records one promotion
and one rollback, on the same day, just before the buttons were removed. Every
row in the table sits at 'experimental'.

The endpoints are gone; this drops the column they wrote. Kept out of the
down-revision path is the data itself — a column whose only value was the
default carries nothing to preserve, so `downgrade` restores the column with
that default rather than pretending it can restore history.

Revision ID: b1c2d3e4f5a6
Revises: a9d3f61e08b4
Create Date: 2026-08-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, None] = 'a9d3f61e08b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('registry_models') as batch:
        batch.drop_column('stage')


def downgrade() -> None:
    with op.batch_alter_table('registry_models') as batch:
        batch.add_column(sa.Column('stage', sa.String(length=32),
                                   server_default='experimental', nullable=False))
