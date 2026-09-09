"""remove experiment limits from shipped Task objectives

Revision ID: f8b9c0d1e2f3
Revises: f7a8b9c0d1e2
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f8b9c0d1e2f3"
down_revision: Union[str, None] = "f7a8b9c0d1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_OBJECTIVES = {
    "med": (
        "Fine-tune a model to answer USMLE-style medical multiple-choice questions. "
        "reason, then give the final answer as a boxed letter, e.g. \\boxed{C}. "
        "There is no budget cap. Iterate freely (no iteration cap) to maximize "
        "test accuracy; keep the best and STOP when accuracy plateaus.",
        "Answer USMLE-style medical multiple-choice questions. Reason, then give "
        "the final answer as a boxed letter, e.g. \\boxed{C}.",
    ),
    "science": (
        "Fine-tune a model to answer grade-school science multiple-choice questions "
        "(ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. "
        "Stay within a hard $100 budget. Iterate freely (no iteration cap) to "
        "maximize test accuracy; keep the best and STOP when accuracy plateaus.",
        "Answer grade-school science multiple-choice questions (ARC-Challenge). "
        "Reason, then answer as a boxed letter, e.g. \\boxed{B}.",
    ),
    "math": (
        "Fine-tune a model to answer grade-school math word problems (GSM8K). "
        "reason step by step and end with the numeric answer in \\boxed{}, e.g. "
        "\\boxed{42}. There is no budget cap. Iterate freely (no iteration cap) "
        "to maximize test accuracy; keep the best and STOP when accuracy plateaus.",
        "Answer grade-school math word problems (GSM8K). Reason step by step and "
        "end with the numeric answer in \\boxed{}, e.g. \\boxed{42}.",
    ),
    "code": (
        "Fine-tune a model to answer Python functions that pass the given unit tests "
        "(MBPP). put the function in a ```python code block. Stay within a hard "
        "$100 budget. Iterate freely (no iteration cap) to maximize test accuracy; "
        "keep the best and STOP when accuracy plateaus.",
        "Write Python functions that pass the given unit tests (MBPP). Put the "
        "function in a ```python code block.",
    ),
}


def _replace(source_index: int, target_index: int) -> None:
    tasks = sa.table(
        "tasks",
        sa.column("name", sa.String()),
        sa.column("task_objective", sa.Text()),
    )
    for name, pair in _OBJECTIVES.items():
        source, target = pair[source_index], pair[target_index]
        # Exact-match guards preserve any Task the user edited after seeding.
        op.execute(
            tasks.update()
            .where(tasks.c.name == name)
            .where(tasks.c.task_objective == source)
            .values(task_objective=target)
        )


def upgrade() -> None:
    _replace(0, 1)


def downgrade() -> None:
    _replace(1, 0)
