"""cap the science + code tasks at $100

The shipped catalogue had no budget cap anywhere, so the UI only ever showed
`budget ∞`. Half the domains now carry a cap at every level — `science` and
`code` at $100, `med` and `math` unlimited — so both shapes exist without
doubling the catalogue.

The objective text is rewritten alongside the number, because the objective is
what the agents actually read: leaving "There is no budget cap." in place while
the runner enforces $100 would brief them against the config.

Only rows still at the shipped default (`max_cost_usd = 0`) are touched, so a
budget you set yourself is never overwritten.

Revision ID: d4e5f6a8b9c1
Revises: c3d4e5f6a8b9
Create Date: 2026-07-31

"""
import sqlalchemy as sa
from alembic import op


revision = "d4e5f6a8b9c1"
down_revision = "c3d4e5f6a8b9"
branch_labels = None
depends_on = None


# --- frozen ---------------------------------------------------------------
# name -> (capped cost, objective at that cost, objective at no cap). These
# used to be recomputed from fixtures/user_requests.py on every run, so a
# migration's effect changed whenever that module did — and a downgrade could
# write back wording the database had never held. That module is gone; the
# exact strings this revision moves between live here.
_CAPPED: dict[str, tuple[float, str, str]] = {
    "l1-code": (100.0,
        'Fine-tune Qwen/Qwen3-4B to answer Python functions that pass the given unit tests (MBPP). put the function in a ```python code block. Train on the PROVIDED training set. Use the rft training method. Stay within a hard $100 budget. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.',
        'Fine-tune Qwen/Qwen3-4B to answer Python functions that pass the given unit tests (MBPP). put the function in a ```python code block. Train on the PROVIDED training set. Use the rft training method. There is no budget cap. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.'),
    "l1-science": (100.0,
        'Fine-tune Qwen/Qwen3-4B to answer grade-school science multiple-choice questions (ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. Train on the PROVIDED training set. Use the rft training method. Stay within a hard $100 budget. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.',
        'Fine-tune Qwen/Qwen3-4B to answer grade-school science multiple-choice questions (ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. Train on the PROVIDED training set. Use the rft training method. There is no budget cap. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.'),
    "l2-code": (100.0,
        'Fine-tune Qwen/Qwen3-4B to answer Python functions that pass the given unit tests (MBPP). put the function in a ```python code block. Train on the PROVIDED training set. Choose the training method(s) yourself. Stay within a hard $100 budget. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.',
        'Fine-tune Qwen/Qwen3-4B to answer Python functions that pass the given unit tests (MBPP). put the function in a ```python code block. Train on the PROVIDED training set. Choose the training method(s) yourself. There is no budget cap. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.'),
    "l2-science": (100.0,
        'Fine-tune Qwen/Qwen3-4B to answer grade-school science multiple-choice questions (ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. Train on the PROVIDED training set. Choose the training method(s) yourself. Stay within a hard $100 budget. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.',
        'Fine-tune Qwen/Qwen3-4B to answer grade-school science multiple-choice questions (ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. Train on the PROVIDED training set. Choose the training method(s) yourself. There is no budget cap. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.'),
    "l3-code": (100.0,
        'Fine-tune Qwen/Qwen3-4B to answer Python functions that pass the given unit tests (MBPP). put the function in a ```python code block. NO training set is provided — acquire and curate the training data yourself. Choose the training method(s) yourself. Stay within a hard $100 budget. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.',
        'Fine-tune Qwen/Qwen3-4B to answer Python functions that pass the given unit tests (MBPP). put the function in a ```python code block. NO training set is provided — acquire and curate the training data yourself. Choose the training method(s) yourself. There is no budget cap. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.'),
    "l3-science": (100.0,
        'Fine-tune Qwen/Qwen3-4B to answer grade-school science multiple-choice questions (ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. NO training set is provided — acquire and curate the training data yourself. Choose the training method(s) yourself. Stay within a hard $100 budget. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.',
        'Fine-tune Qwen/Qwen3-4B to answer grade-school science multiple-choice questions (ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. NO training set is provided — acquire and curate the training data yourself. Choose the training method(s) yourself. There is no budget cap. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.'),
    "l4-code": (100.0,
        'Fine-tune a base model you choose to answer Python functions that pass the given unit tests (MBPP). put the function in a ```python code block. NO training set is provided — acquire and curate the training data yourself. Choose the training method(s) yourself. Pick the base model yourself. Stay within a hard $100 budget. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.',
        'Fine-tune a base model you choose to answer Python functions that pass the given unit tests (MBPP). put the function in a ```python code block. NO training set is provided — acquire and curate the training data yourself. Choose the training method(s) yourself. Pick the base model yourself. There is no budget cap. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.'),
    "l4-science": (100.0,
        'Fine-tune a base model you choose to answer grade-school science multiple-choice questions (ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. NO training set is provided — acquire and curate the training data yourself. Choose the training method(s) yourself. Pick the base model yourself. Stay within a hard $100 budget. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.',
        'Fine-tune a base model you choose to answer grade-school science multiple-choice questions (ARC-Challenge). reason, then answer as a boxed letter, e.g. \\boxed{B}. NO training set is provided — acquire and curate the training data yourself. Choose the training method(s) yourself. Pick the base model yourself. There is no budget cap. Iterate freely (no iteration cap) to maximize test accuracy; keep the best and STOP when accuracy plateaus.'),
}


def _sync(*, capped: bool) -> None:
    """Push each capped task's cost + objective to the requested state.

    `capped=False` reproduces the pre-cap state, which is how the downgrade
    restores the old wording.
    """
    conn = op.get_bind()
    for name, (cost, capped_obj, uncapped_obj) in _CAPPED.items():
        conn.execute(
            sa.text(
                "UPDATE tasks SET max_cost_usd = :cost, objective = :obj "
                "WHERE name = :name AND max_cost_usd = :expect"
            ),
            {
                "cost": cost if capped else 0.0,
                "obj": capped_obj if capped else uncapped_obj,
                "name": name,
                # Only move a row that is still where the previous state left
                # it; anything else is the user's own budget.
                "expect": 0.0 if capped else cost,
            },
        )


def upgrade() -> None:
    _sync(capped=True)


def downgrade() -> None:
    _sync(capped=False)
