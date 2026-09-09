"""one name per thing: instance_id everywhere, providers are the trio

The infrastructure allocation id had three names — `allocation_id` in stored
ticket payloads, `external_id` on the infra_instances row, `instance_id` in the
typed contracts and device_info — and the provider column accepted legacy
aliases (`slurm`, `vastai`, `lambda`, ``''``). The code now uses `instance_id`
and exactly `cluster | cloud | instance`; this migration rewrites the stored
data to match, so no runtime alias handling survives.

Revision ID: f7a79e2b6a30
Revises: f8b9c0d1e2f3
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op


revision: str = "f7a79e2b6a30"
down_revision: Union[str, None] = "f8b9c0d1e2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_PROVIDER_CASE = """
    CASE {col}
        WHEN 'slurm' THEN 'cluster'
        WHEN 'vastai' THEN 'cloud'
        WHEN 'lambda' THEN 'cloud'
        WHEN '' THEN 'cloud'
        ELSE {col}
    END
"""


def upgrade() -> None:
    # infra_instances: external_id -> instance_id, and only canonical providers.
    op.alter_column("infra_instances", "external_id", new_column_name="instance_id")
    op.drop_index(
        "ix_infra_instances_provider_external_id", table_name="infra_instances"
    )
    op.create_index(
        "ix_infra_instances_provider_instance_id",
        "infra_instances",
        ["provider", "instance_id"],
    )
    op.execute(
        "UPDATE infra_instances SET provider = "
        + _PROVIDER_CASE.format(col="provider")
        + " WHERE provider IN ('slurm', 'vastai', 'lambda', '')"
    )

    # Stored infrastructure payloads: allocation_id -> instance_id.
    op.execute("""
        UPDATE tickets
        SET payload = (payload - 'allocation_id')
                      || jsonb_build_object('instance_id', payload->'allocation_id')
        WHERE agent_id = 'infrastructure' AND payload ? 'allocation_id'
    """)

    # device_info work-product meta: legacy provider aliases -> the trio.
    op.execute("""
        UPDATE work_products
        SET meta = meta || jsonb_build_object('provider',
            CASE meta->>'provider'
                WHEN 'slurm' THEN 'cluster'
                WHEN 'vastai' THEN 'cloud'
                WHEN 'lambda' THEN 'cloud'
                ELSE meta->>'provider'
            END)
        WHERE meta ? 'provider'
          AND meta->>'provider' IN ('slurm', 'vastai', 'lambda')
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE tickets
        SET payload = (payload - 'instance_id')
                      || jsonb_build_object('allocation_id', payload->'instance_id')
        WHERE agent_id = 'infrastructure' AND payload ? 'instance_id'
    """)
    op.drop_index(
        "ix_infra_instances_provider_instance_id", table_name="infra_instances"
    )
    op.alter_column("infra_instances", "instance_id", new_column_name="external_id")
    op.create_index(
        "ix_infra_instances_provider_external_id",
        "infra_instances",
        ["provider", "external_id"],
    )
    # Provider aliases are not restored: the canonical trio is valid before
    # and after this revision.
