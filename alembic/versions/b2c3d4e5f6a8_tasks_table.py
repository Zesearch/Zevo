"""tasks table, seeded with the shipped catalogue

The predefined tasks lived only in fixtures/user_requests.py, so the catalogue
could not be added to or pruned without editing code. This moves them into a
table, which makes every task ordinary data: the shipped ones are now
indistinguishable from ones added later, and both can be deleted.

The seed data below is a FROZEN copy of that catalogue. It used to be imported
from fixtures/user_requests.py at migration time, which meant this migration
seeded whatever the application code happened to say that day — edit the
fixtures and a fresh database silently got a different catalogue than the one
every existing database was built from. That module is gone; the rows live here
now, and this is the one place that decides what a FRESH database starts with.
An existing database is not affected by edits here, and never was: seeding is
one-time. To change a task on a running instance, edit the tasks table — the
Tasks page does exactly that.

Seeding is idempotent — rows whose name already exists are left alone, so
re-running on a database that already has tasks is safe.

Revision ID: b2c3d4e5f6a8
Revises: a1b2c3d4e5f7
Create Date: 2026-07-31

"""
import os
from pathlib import Path

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON


revision = "b2c3d4e5f6a8"
down_revision = "a1b2c3d4e5f7"
branch_labels = None
depends_on = None

JsonCol = JSON().with_variant(JSONB(), "postgresql")

# --- frozen catalogue ------------------------------------------------------
# Paths are resolved at migration time, not frozen: the same catalogue seeds a
# container (/app/workspace/datasets) and a host checkout, so only the bundle
# name is data here.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATASETS = Path(os.environ.get("DATASETS_DIR", str(_PROJECT_ROOT / "workspace" / "datasets")))

_BASE_MODEL = "Qwen/Qwen3-4B"
_LIMITED_BUDGET = 100.0
_CAPPED_DOMAINS = {"science", "code"}

# domain -> (bundle name, answer columns, what it asks, required answer format)
_DOMAINS: dict[str, tuple[str, list[str], str, str]] = {
    "med": ("medqa-usmle", ["answer", "answer_idx"],
            "USMLE-style medical multiple-choice questions",
            "reason, then give the final answer as a boxed letter, e.g. \\boxed{C}"),
    "science": ("arc-challenge", ["answer", "answer_idx"],
                "grade-school science multiple-choice questions (ARC-Challenge)",
                "reason, then answer as a boxed letter, e.g. \\boxed{B}"),
    "math": ("gsm8k", ["gold"],
             "grade-school math word problems (GSM8K)",
             "reason step by step and end with the numeric answer in \\boxed{}, e.g. \\boxed{42}"),
    "code": ("mbpp", ["test_list"],
             "Python functions that pass the given unit tests (MBPP)",
             "put the function in a ```python code block"),
}

# name template -> (dataset given, base model given, method)
_LADDER: dict[str, tuple[bool, bool, str]] = {
    "l1-{dom}": (True, True, "rft"),
    "l2-{dom}": (True, True, ""),
    "l3-{dom}": (False, True, ""),
    "l4-{dom}": (False, False, ""),
}


def _objective(domain: str, *, base_given: bool, dataset_given: bool, method: str,
               max_cost: float) -> str:
    """Compose an objective from what the task pins down vs leaves to Zevo."""
    _, _, what, fmt = _DOMAINS[domain]
    base = _BASE_MODEL if base_given else "a base model you choose"
    parts = [f"Fine-tune {base} to answer {what}. {fmt}."]
    parts.append(
        "Train on the PROVIDED training set." if dataset_given
        else "NO training set is provided — acquire and curate the training data yourself."
    )
    parts.append(
        f"Use the {method} training method." if method
        else "Choose the training method(s) yourself."
    )
    if not base_given:
        parts.append("Pick the base model yourself.")
    parts.append(
        f"Stay within a hard ${max_cost:g} budget." if max_cost > 0
        else "There is no budget cap."
    )
    parts.append(
        "Iterate freely (no iteration cap) to maximize test accuracy; keep the best "
        "and STOP when accuracy plateaus."
    )
    return " ".join(parts)


def _row(name: str, *, domain: str, bundle: str, objective: str, method: str,
         dataset_given: bool, base_model: str, max_cost: float,
         iteration_budget: int) -> dict:
    d = _DATASETS / bundle
    _, answer_columns, _, _ = _DOMAINS[domain]
    return {
        "name": name,
        "objective": objective,
        "method_hint": method,
        "dataset": str(d / "train.csv") if dataset_given else "",
        # No training set -> the data agent derives what to acquire from the
        # objective, so the query stays empty here.
        "data_query": "",
        "base_model": base_model,
        "test_set": str(d / "test.csv"),
        "answer_columns": list(answer_columns),
        "sample_submission": str(d / "sample_submission.csv"),
        "evaluation_script": str(d / "eval.py"),
        "constraints": [],
        "gpu_provider": "instance",
        "framework": "vllm",
        "iteration_budget": iteration_budget,
        "max_cost_usd": max_cost,
        "target_accuracy": 0.0,
        # 'l1-med' -> domain 'med'; names without a dash keep an empty domain.
        "domain": name.partition("-")[2],
    }


def _catalogue() -> list[dict]:
    """The 18 shipped tasks: the four-level ladder over four domains, plus the
    tiny smoke test and the instruction-following task."""
    rows: list[dict] = []
    for tmpl, (data_given, base_given, method) in _LADDER.items():
        for dom, (bundle, _, _, _) in _DOMAINS.items():
            cost = _LIMITED_BUDGET if dom in _CAPPED_DOMAINS else 0.0
            rows.append(_row(
                tmpl.format(dom=dom), domain=dom, bundle=bundle,
                objective=_objective(dom, base_given=base_given,
                                     dataset_given=data_given, method=method,
                                     max_cost=cost),
                method=method, dataset_given=data_given,
                base_model=_BASE_MODEL if base_given else "",
                max_cost=cost, iteration_budget=0,
            ))

    # Cheap end-to-end smoke on the 10-row tiny bundle (L1 shape).
    rows.append(_row(
        "l1-tiny", domain="med", bundle="medqa-tiny",
        objective=(
            f"SMOKE TEST. Fine-tune {_BASE_MODEL} on the tiny provided MedQA training "
            f"set with LoRA SFT, one quick iteration. Goal: complete the pipeline "
            f"end-to-end cheaply."
        ),
        method="lora_sft", dataset_given=True, base_model=_BASE_MODEL,
        max_cost=0.0, iteration_budget=1,
    ))

    # Instruction following (FREE-FORM, not MCQ): full-parameter SFT of a
    # Qwen3-0.6B base on Capybara, scored by the full 541-prompt IFEval. The
    # training set is named as a HuggingFace hub id rather than a path, so
    # nothing on disk resolves it and the data agent fetches it at run time.
    capybara = _row(
        "l1-capybara", domain="med", bundle="ifeval",
        # Model, method, training set, test set, scorer and budget are all
        # columns on the row, so the objective states the goal and nothing else.
        # Two things it therefore does NOT say, both of which the agents infer:
        # that the answer is free-form (no boxed letter — the sample submission's
        # single `response` column and Capybara's own shape imply it), and that
        # the whole training set is to be used (the data agent may otherwise
        # take a subset of a provided set — see playbook/agents/data/goal.md).
        objective="Turn the base model into an instruction-following model",
        method="full_sft", dataset_given=False, base_model="Qwen/Qwen3-0.6B-Base",
        max_cost=0.0, iteration_budget=0,
    )
    capybara["dataset"] = "trl-lib/Capybara"
    capybara["data_query"] = (
        "The Capybara multi-turn instruction-following dataset — `trl-lib/Capybara` "
        "on HuggingFace. Reformat to chat SFT for instruction following."
    )
    capybara["answer_columns"] = ["instruction_id_list", "kwargs"]
    rows.append(capybara)
    return rows


def upgrade() -> None:
    tasks = op.create_table(
        "tasks",
        sa.Column("name", sa.String(length=64), primary_key=True),
        sa.Column("objective", sa.Text(), nullable=False, server_default=""),
        sa.Column("method_hint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("dataset", sa.Text(), nullable=False, server_default=""),
        sa.Column("data_query", sa.Text(), nullable=False, server_default=""),
        sa.Column("base_model", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("test_set", sa.Text(), nullable=False, server_default=""),
        sa.Column("answer_columns", JsonCol, nullable=False, server_default="[]"),
        sa.Column("sample_submission", sa.Text(), nullable=False, server_default=""),
        sa.Column("evaluation_script", sa.Text(), nullable=False, server_default=""),
        sa.Column("constraints", JsonCol, nullable=False, server_default="[]"),
        sa.Column("gpu_provider", sa.String(length=16), nullable=False, server_default=""),
        sa.Column("framework", sa.String(length=16), nullable=False, server_default="vllm"),
        sa.Column("iteration_budget", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_cost_usd", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("target_accuracy", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("domain", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    conn = op.get_bind()
    existing = {r[0] for r in conn.execute(sa.text("SELECT name FROM tasks"))}
    rows = [r for r in _catalogue() if r["name"] not in existing]
    if rows:
        op.bulk_insert(tasks, rows)


def downgrade() -> None:
    op.drop_table("tasks")
