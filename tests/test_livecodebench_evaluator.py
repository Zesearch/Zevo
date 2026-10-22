from __future__ import annotations

import csv
import importlib.util
import json
import tracemalloc
from pathlib import Path

import pytest


@pytest.fixture
def evaluator():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluators/livecodebench.py"
    spec = importlib.util.spec_from_file_location("lcb_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_loader_bounds_memory_and_keeps_all_rows(tmp_path, evaluator, suffix):
    path = tmp_path / ("scoring" + suffix)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "private_test_cases"])
        if suffix == ".csv":
            writer.writeheader()
        for i in range(96):
            row = {"id": str(i), "private_test_cases": "x" * 262144}
            if suffix == ".csv":
                writer.writerow(row)
            else:
                handle.write(json.dumps(row) + "\n")
    tracemalloc.start()
    try:
        count = 0
        for row in evaluator.load_rows(str(path)):
            assert row["id"] == str(count)
            assert len(row["private_test_cases"]) == 262144
            count += 1
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == 96
    assert peak < 12 * 1024 * 1024


def test_main_scores_each_row_before_requesting_the_next(tmp_path, evaluator, monkeypatch):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text('id,prediction\n1,"```python\nprint(1)\n```"\n')
    metrics = tmp_path / "metrics.json"
    checked = []

    def rows(_):
        for i in range(2):
            assert len(checked) == i, "scoring must not pre-load/decode the population"
            yield {"id": str(i + 1), "difficulty": "easy",
                   "public_test_cases": '[{"input":"","output":"1"}]',
                   "private_test_cases": "[]", "metadata": "{}"}

    def check(args):
        sample, code, timeout = args
        assert json.loads(sample["input_output"])["outputs"] == ["1"]
        assert timeout == 6
        checked.append(code)
        return bool(code)

    monkeypatch.setattr(evaluator, "load_rows", rows)
    monkeypatch.setattr(evaluator, "check", check)
    monkeypatch.setattr(evaluator.sys, "argv", ["evaluator.py", str(predictions), "unused.csv", str(metrics)])
    evaluator.main()
    result = json.loads(metrics.read_text())
    assert result["score"] == 0.5
    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["missing_predictions"] == 1
    assert result["by_difficulty"] == {"easy": 0.5}
