"""Remove the bundled Task catalogue.

Zevo now starts with an empty Tasks page. This migration removes only the six
known bundled Tasks, identified by both their canonical name and their shipped
Test-set path. User-created Tasks, historical Runs, and reusable Files remain
untouched.

Revision ID: af0a1b2c3d4e
Revises: ae9f0a1b2c3d
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa


revision = "af0a1b2c3d4e"
down_revision = "ae9f0a1b2c3d"
branch_labels = None
depends_on = None


_BUNDLED_TASKS = {
    "capybara": "%/ifeval/test.csv",
    "code": "%/mbpp/test.csv",
    "math": "%/gsm8k/test.csv",
    "med": "%/medqa-usmle/test.csv",
    "science": "%/arc-challenge/test.csv",
    "tiny": "%/medqa-tiny/test.csv",
}


def upgrade() -> None:
    conn = op.get_bind()
    for name, test_path in _BUNDLED_TASKS.items():
        bundled = conn.execute(
            sa.text(
                "SELECT 1 FROM tasks "
                "WHERE name = :name AND test_set LIKE :test_path"
            ),
            {"name": name, "test_path": test_path},
        ).first()
        if bundled is None:
            continue
        conn.execute(
            sa.text("DELETE FROM task_settings WHERE task_name = :name"),
            {"name": name},
        )
        conn.execute(
            sa.text(
                "DELETE FROM tasks "
                "WHERE name = :name AND test_set LIKE :test_path"
            ),
            {"name": name, "test_path": test_path},
        )


def downgrade() -> None:
    # Deleted catalogue data is intentionally not recreated. Users can add any
    # desired Task from the UI or CLI after downgrading.
    pass
