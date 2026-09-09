"""one task per domain: the level is a setting, not a name

`l1-med`, `l2-med`, `l3-med` and `l4-med` were four task rows describing ONE
problem. They shared the objective, the test set, the eval script, the sample
submission and the budget; they differed only in how much of (training data,
base model, training method) was pinned down — which is the SETTING a run is
launched with, not a different task.

So they collapse to `med`, and likewise for the other domains. The surviving row
keeps the fullest defaults (the l1 shape) and its objective loses the sentences
that described the setting: those are re-appended per run from whatever setting
that run actually uses (zevo.api.routers.ui.tasks.compose_objective), so the
agents read the same prose they always did, matching the run in front of them.

Runs are carried across with the rename, exactly as the c3d4e5f6a8b9 rename did
— a run's own level is recoverable from the request it recorded, so nothing is
lost by pointing its `task_name` at the task it was always an attempt at.

Revision ID: c9e1a4b7d2f3
Revises: b8d3f1a2c4e5
Create Date: 2026-08-09
"""
from __future__ import annotations

import re

import sqlalchemy as sa
from alembic import op


revision = "c9e1a4b7d2f3"
down_revision = "b8d3f1a2c4e5"
branch_labels = None
depends_on = None


# The sentences the shipped catalogue used to state a setting. Removed verbatim
# rather than parsed: an objective the user wrote themselves contains none of
# them and so comes through untouched.
_CLAUSES = (
    "Train on the PROVIDED training set.",
    "NO training set is provided — acquire and curate the training data yourself.",
    "Choose the training method(s) yourself.",
    "Pick the base model yourself.",
)
_METHOD_RE = re.compile(r"Use the [^.]+ training method\.")


def strip_setting(objective: str, base_model: str) -> str:
    """The problem, with the three setting sentences taken back out."""
    s = objective or ""
    for c in _CLAUSES:
        s = s.replace(c, "")
    s = _METHOD_RE.sub("", s)
    # The opening sentence names the model too ("Fine-tune Qwen/Qwen3-4B to
    # answer…" / "Fine-tune a base model you choose to answer…").
    s = s.replace("Fine-tune a base model you choose to ", "Fine-tune a model to ")
    if (base_model or "").strip():
        s = s.replace(f"Fine-tune {base_model.strip()} to ", "Fine-tune a model to ")
    return re.sub(r"\s{2,}", " ", s).strip()


def restore_setting(objective: str, dataset: str, base_model: str, method_hint: str) -> str:
    """The inverse used by downgrade(): put the setting back into the prose."""
    parts = [
        (objective or "").strip(),
        _CLAUSES[0] if (dataset or "").strip() else _CLAUSES[1],
        f"Use the {method_hint.strip()} training method."
        if (method_hint or "").strip() else _CLAUSES[2],
    ]
    if not (base_model or "").strip():
        parts.append(_CLAUSES[3])
    return " ".join(p for p in parts if p).strip()


_FIELDS = "name, domain, objective, dataset, base_model, method_hint"


def _richest(rows: list) -> object:
    """The row that pins down the most — the l1 shape. Name breaks ties so the
    choice does not depend on the order the database hands rows back."""
    return sorted(
        rows,
        key=lambda r: (
            -sum(1 for v in (r.dataset, r.base_model, r.method_hint) if (v or "").strip()),
            r.name,
        ),
    )[0]


def upgrade() -> None:
    conn = op.get_bind()
    rows = list(conn.execute(sa.text(f"SELECT {_FIELDS} FROM tasks")))
    names = {r.name for r in rows}

    by_domain: dict[str, list] = {}
    for r in rows:
        if (r.domain or "").strip():
            by_domain.setdefault(r.domain.strip(), []).append(r)

    for domain, group in sorted(by_domain.items()):
        members = {r.name for r in group}
        # Somebody already has a task by that name and it is not one of these —
        # leave the whole domain alone rather than merging into a stranger.
        if domain in names and domain not in members:
            continue
        keep = _richest(group)
        objective = strip_setting(keep.objective, keep.base_model)

        # Drop the other shapes first, so the name is free to take.
        doomed = sorted(members - {keep.name})
        if doomed:
            conn.execute(sa.text("DELETE FROM tasks WHERE name = ANY(:names)"),
                         {"names": doomed})
        conn.execute(
            sa.text("UPDATE tasks SET name = :new, objective = :obj WHERE name = :old"),
            {"new": domain, "obj": objective, "old": keep.name},
        )
        # Every run of every shape was an attempt at this one task.
        conn.execute(
            sa.text("UPDATE runs SET task_name = :new WHERE task_name = ANY(:old)"),
            {"new": domain, "old": sorted(members)},
        )
        names -= members
        names.add(domain)


def downgrade() -> None:
    """Give the l1 names and the setting prose back.

    The l2/l3/l4 rows are NOT rebuilt: they were four views of one row, and the
    only one that survived the merge is the one this restores. Same bargain the
    c3d4e5f6a8b9 downgrade struck — the old names come back, not the old count.
    """
    conn = op.get_bind()
    rows = list(conn.execute(sa.text(f"SELECT {_FIELDS} FROM tasks")))
    names = {r.name for r in rows}
    for r in rows:
        if not (r.domain or "").strip() or r.name != r.domain.strip():
            continue
        old = f"l1-{r.name}"
        if old in names:
            continue
        objective = restore_setting(r.objective, r.dataset, r.base_model, r.method_hint)
        conn.execute(
            sa.text("UPDATE tasks SET name = :new, objective = :obj WHERE name = :old"),
            {"new": old, "obj": objective, "old": r.name},
        )
        conn.execute(
            sa.text("UPDATE runs SET task_name = :new WHERE task_name = :old"),
            {"new": old, "old": r.name},
        )
