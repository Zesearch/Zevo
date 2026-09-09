"""reshape the task catalogue: names state the level, one shape per level

Two changes to the shipped catalogue:

1. `b-med` / `c1-med` / `d2-med` / `f1-med` encoded the level in a letter you had
   to look up. The level is now the prefix: l1-med, l2-med, l3-med, l4-med.

2. The budget-capped twins (c2 / d3 / f2) and the off-ladder `d1` shape (model +
   method given, data withheld) are dropped. What is left is four levels × four
   domains, each level exactly one bargain over (training data, training model,
   training method) — the same catalogue, minus 16 near-duplicates. Anything
   dropped can be re-added from the UI, budget cap and all.

`ifeval-capybara` becomes `l1-capybara`: its training set IS specified, just as
a HuggingFace id (`trl-lib/Capybara`) rather than a local path, so all three
decisions are the user's.

Only rows the product shipped are touched. Tasks the user added are left alone,
and so are runs whose task has no counterpart here.

Revision ID: c3d4e5f6a8b9
Revises: b2c3d4e5f6a8
Create Date: 2026-07-31

"""
import sqlalchemy as sa
from alembic import op


revision = "c3d4e5f6a8b9"
down_revision = "b2c3d4e5f6a8"
branch_labels = None
depends_on = None

_DOMAINS = ["med", "science", "math", "code"]

# Old prefix -> new prefix, for the shapes that survive.
_KEPT = {"b": "l1", "c1": "l2", "d2": "l3", "f1": "l4"}
# Prefixes with no counterpart in the new catalogue: the $100 twins (c2/d3/f2)
# and d1, whose shape (data withheld, method given) is off the ladder.
_DROPPED = ["c2", "d1", "d3", "f2"]

RENAMES: dict[str, str] = {
    f"{old}-{d}": f"{new}-{d}" for old, new in _KEPT.items() for d in _DOMAINS
}
RENAMES["b-tiny"] = "l1-tiny"
RENAMES["ifeval-capybara"] = "l1-capybara"

DROPS: list[str] = [f"{p}-{d}" for p in _DROPPED for d in _DOMAINS]

# The hub id that makes l1-capybara's training data "given".
_CAPYBARA_DATASET = "trl-lib/Capybara"


def _rename(conn, mapping: dict[str, str]) -> None:
    """Rename task rows and the runs that point at them.

    Skips anything already renamed, missing, or whose target name is taken — the
    table is user-editable, so this must not fail on a database someone has
    already pruned or added to.
    """
    have = {r[0] for r in conn.execute(sa.text("SELECT name FROM tasks"))}
    for old, new in mapping.items():
        if old not in have or new in have:
            continue
        conn.execute(
            sa.text("UPDATE tasks SET name = :new WHERE name = :old"),
            {"new": new, "old": old},
        )
        # Past runs reference the task by name. Carry them along, or the
        # leaderboard would show `b-med` and `l1-med` as two unrelated tasks.
        conn.execute(
            sa.text("UPDATE runs SET task_name = :new WHERE task_name = :old"),
            {"new": new, "old": old},
        )
        have.discard(old)
        have.add(new)


def upgrade() -> None:
    conn = op.get_bind()
    _rename(conn, RENAMES)
    # Deleting a task never touches run history: runs keep task_name as plain
    # text, so a dropped variant's past results stay on the leaderboard.
    conn.execute(
        sa.text("DELETE FROM tasks WHERE name = ANY(:names)"), {"names": DROPS}
    )
    conn.execute(
        sa.text("UPDATE tasks SET dataset = :ds WHERE name = 'l1-capybara' AND dataset = ''"),
        {"ds": _CAPYBARA_DATASET},
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("UPDATE tasks SET dataset = '' WHERE name = 'l1-capybara' AND dataset = :ds"),
        {"ds": _CAPYBARA_DATASET},
    )
    _rename(conn, {v: k for k, v in RENAMES.items()})
    # The dropped variants are NOT restored: their definitions no longer exist
    # anywhere in the code to rebuild them from. Downgrading gets the old names
    # back, not the old row count.
