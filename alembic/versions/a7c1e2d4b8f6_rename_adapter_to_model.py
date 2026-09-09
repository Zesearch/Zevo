"""call the trained model a model

`adapter/` was the output directory's name back when LoRA was the only method
the train agent knew. It is now one method among eleven: `full_sft` writes every
weight, and calling that an adapter is simply wrong. The instructions had grown
three separate paragraphs explaining that "adapter" was only a directory name
and not a promise of a LoRA delta — a sign the name was doing damage.

So the contract is renamed end to end: the output directory is `model/`, the
registry field is `model_path`, the work-product kind is `model`, and the
registry's kept copy is `best_model/`.

Two things are deliberately NOT renamed:

  * `adapter_config.json` / `adapter_model.safetensors` — PEFT writes those
    filenames itself. They stay whatever peft calls them.
  * The word "adapter" where it genuinely means a LoRA delta. `lora_sft` really
    does produce an adapter; that sentence was never the problem.

Existing rows are rewritten rather than left behind: a run whose artifacts are
filed under a kind the code no longer looks for would silently drop out of the
run detail's artifact list.

Revision ID: a7c1e2d4b8f6
Revises: e3f4a5b6c7d8
Create Date: 2026-08-07

"""
import sqlalchemy as sa
from alembic import op


revision = "a7c1e2d4b8f6"
down_revision = "e3f4a5b6c7d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("registry_models", "adapter_path", new_column_name="model_path")
    # The stored VALUES point at the old directory too, so rewriting only the
    # column name would leave every kept model unreachable on disk. The
    # accompanying directory rename is done by the same change.
    op.execute(
        "UPDATE registry_models SET model_path = "
        "replace(model_path, '/best_adapter', '/best_model') "
        "WHERE model_path LIKE '%/best_adapter%'"
    )
    op.execute("UPDATE work_products SET kind = 'model' WHERE kind = 'adapter'")


def downgrade() -> None:
    op.execute("UPDATE work_products SET kind = 'adapter' WHERE kind = 'model'")
    op.execute(
        "UPDATE registry_models SET model_path = "
        "replace(model_path, '/best_model', '/best_adapter') "
        "WHERE model_path LIKE '%/best_model%'"
    )
    op.alter_column("registry_models", "model_path", new_column_name="adapter_path")
