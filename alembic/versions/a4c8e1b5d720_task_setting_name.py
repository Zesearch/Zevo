"""name each saved setting `s1`, `s2`, … per task

A setting had nothing to call it, so a run started from one could only be
described by reciting its five decisions. The names are handed out in creation
order and are not reused: the list is sorted by how often each has been run, so
a name derived from position would move whenever one overtook another.

Revision ID: a4c8e1b5d720
Revises: f2b7d4e6a913
Create Date: 2026-08-09
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "a4c8e1b5d720"
down_revision = "f2b7d4e6a913"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "task_settings",
        sa.Column("name", sa.String(length=32), nullable=False, server_default=""),
    )
    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT id, task_name FROM task_settings ORDER BY task_name, created_at, id"
    )).all()
    seen: dict[str, int] = {}
    for r in rows:
        seen[r.task_name] = seen.get(r.task_name, 0) + 1
        conn.execute(
            sa.text("UPDATE task_settings SET name = :n WHERE id = :id"),
            {"n": f"s{seen[r.task_name]}", "id": r.id},
        )


def downgrade() -> None:
    op.drop_column("task_settings", "name")
