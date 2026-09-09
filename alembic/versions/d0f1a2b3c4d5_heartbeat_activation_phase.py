"""label the purpose of each heartbeat activation

Revision ID: d0f1a2b3c4d5
Revises: cd9e1f2a3b4c
"""

import json

import sqlalchemy as sa
from alembic import op


revision = "d0f1a2b3c4d5"
down_revision = "cd9e1f2a3b4c"
branch_labels = None
depends_on = None


def _object(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def upgrade() -> None:
    op.add_column(
        "heartbeat_runs",
        sa.Column(
            "activation_phase", sa.String(length=32),
            nullable=False, server_default="",
        ),
    )

    # Recover submit/collect labels for existing finite Slurm activations from
    # their durable typed Results. Ordinary local/cloud activations stay blank.
    bind = op.get_bind()
    heartbeats = sa.table(
        "heartbeat_runs",
        sa.column("id", sa.String),
        sa.column("activation_phase", sa.String),
    )
    results = sa.table(
        "heartbeat_results",
        sa.column("heartbeat_id", sa.String),
        sa.column("output", sa.JSON),
    )
    for heartbeat_id, raw_output in bind.execute(sa.select(
        results.c.heartbeat_id, results.c.output,
    )):
        output = _object(raw_output)
        has_slurm_job = bool(
            output.get("slurm_script_path") or output.get("job_id")
        )
        if not has_slurm_job:
            continue
        phase = "submit" if output.get("status") == "deferred" else "collect"
        bind.execute(
            heartbeats.update()
            .where(heartbeats.c.id == heartbeat_id)
            .values(activation_phase=phase)
        )


def downgrade() -> None:
    op.drop_column("heartbeat_runs", "activation_phase")
