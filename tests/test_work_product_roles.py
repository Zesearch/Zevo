"""One file may carry two work-product roles.

Since validation isolation the Data result names the frozen validation
population as both `validation_source` and `validation_dataset`. The
registration loop de-duplicated by path alone, so the second role was dropped
and no Train Ticket could bind `validation_dataset`: run 77b4ad61 halted at
Train 1 with "the optimization Data Ticket publishes no such Work Product".
"""
from __future__ import annotations

from zevo.engine.run.runner import _TICKET_ARTIFACTS, _unique_work_products


def test_same_path_keeps_every_role_and_drops_true_repeats() -> None:
    items = [
        ("/w/dataset.jsonl", "training_dataset", {}),
        ("/w/_splits/validation.csv", "validation_source", {}),
        ("/w/_splits/validation.csv", "validation_dataset", {}),
        ("/w/dataset.jsonl", "training_dataset", {"dup": True}),
        ("", "script", {}),
    ]
    kept = _unique_work_products(items)
    assert [(p, r) for p, r, _ in kept] == [
        ("/w/dataset.jsonl", "training_dataset"),
        ("/w/_splits/validation.csv", "validation_source"),
        ("/w/_splits/validation.csv", "validation_dataset"),
    ]


def test_data_result_publishes_both_validation_roles() -> None:
    roles = [role for _field, role in _TICKET_ARTIFACTS["data"]]
    assert "validation_source" in roles and "validation_dataset" in roles
