from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECK = runpy.run_path(str(ROOT / "examples" / "olmo_if_validation_evaluator.py"))[
    "check_constraints"
]


def test_official_constraint_checker_has_no_default_pass() -> None:
    spec = {
        "instruction_id": ["keywords:existence", "punctuation:no_comma"],
        "kwargs": [{"keywords": ["zevo"]}, None],
    }
    assert CHECK(spec, "") == (False, 0)
    assert CHECK(spec, "unrelated text") == (False, 1)
    assert CHECK(spec, "zevo works") == (True, 2)
    with pytest.raises(ValueError, match="no official verifier"):
        CHECK({"instruction_id": ["unknown:constraint"], "kwargs": [None]}, "text")
    with pytest.raises(ValueError, match="no official verifier"):
        CHECK({"instruction_id": ["unknown:constraint"], "kwargs": [None]}, "")


def test_pinned_validation_asset_has_500_multi_constraint_rows() -> None:
    path = ROOT / "assets" / "olmo" / "if_validation" / "if_validation.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 500
    assert len({row["id"] for row in rows}) == 500
    assert all(len(row["constraint_spec"]["instruction_id"]) >= 2 for row in rows)
    assert all(CHECK(row["constraint_spec"], "") == (False, 0) for row in rows)
