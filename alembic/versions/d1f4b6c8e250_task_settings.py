"""task_settings: a setting is a row, not just something a run once did

Settings were read back from run history. That answered "what has been tried",
but it could not answer "what do I want to try": a setting nobody had run yet
had nowhere to live, and one that turned out to be a dead end could not be
taken off the list.

So they are rows now. Every past run's setting is backfilled here, deduplicated
per task, and create_run keeps adding them as runs happen — deleting one never
touches a run, which keeps its own copy of the request it was launched with.

Revision ID: d1f4b6c8e250
Revises: c9e1a4b7d2f3
Create Date: 2026-08-09
"""
from __future__ import annotations

import uuid

import sqlalchemy as sa
from alembic import op


revision = "d1f4b6c8e250"
down_revision = "c9e1a4b7d2f3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "task_settings",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("task_name", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("dataset", sa.Text(), nullable=False, server_default=""),
        sa.Column("base_model", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("method_hint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("data_query", sa.Text(), nullable=False, server_default=""),
        sa.Column("gpu_provider", sa.String(length=16), nullable=False, server_default=""),
        sa.Column("framework", sa.String(length=16), nullable=False, server_default="vllm"),
        sa.Column("iteration_budget", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_cost_usd", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_task_settings_task_name", "task_settings", ["task_name"])

    # Backfill: one row per distinct setting a task has been run with. The
    # request each run was launched with lives in its supervisor ticket, which
    # is the only record of what it was actually handed.
    conn = op.get_bind()
    rows = conn.execute(sa.text("""
        SELECT r.task_name, r.iteration_budget, r.max_cost_usd, r.started_at, t.payload
          FROM runs r
          JOIN tickets t ON t.id = r.supervisor_ticket_id
         WHERE r.task_name <> ''
         ORDER BY r.started_at
    """))
    seen: set[tuple] = set()
    for r in rows:
        payload = r.payload if isinstance(r.payload, dict) else {}
        req = payload.get("user_request") or {}
        if not isinstance(req, dict):
            continue
        setting = {
            "task_name": r.task_name,
            "dataset": str(req.get("dataset") or ""),
            "base_model": str(req.get("base_model") or ""),
            "method_hint": str(req.get("method_hint") or ""),
            "gpu_provider": str(req.get("gpu_provider") or ""),
            "framework": str(req.get("framework") or "") or "vllm",
        }
        key = tuple(setting.values())
        if key in seen:
            continue
        seen.add(key)
        conn.execute(
            sa.text("""
                INSERT INTO task_settings
                  (id, task_name, dataset, base_model, method_hint, data_query,
                   gpu_provider, framework, iteration_budget, max_cost_usd, created_at)
                VALUES
                  (:id, :task_name, :dataset, :base_model, :method_hint, :data_query,
                   :gpu_provider, :framework, :iteration_budget, :max_cost_usd, :created_at)
            """),
            {
                **setting,
                "id": str(uuid.uuid4()),
                "data_query": str(req.get("data_query") or ""),
                "iteration_budget": int(r.iteration_budget or 0),
                "max_cost_usd": float(r.max_cost_usd or 0.0),
                "created_at": r.started_at,
            },
        )


def downgrade() -> None:
    op.drop_index("ix_task_settings_task_name", table_name="task_settings")
    op.drop_table("task_settings")
