#!/usr/bin/env python3
"""Strict, verifiable instruction-following Validation scorer.

The constraint registry is the official IFBench implementation, pinned by
Zevo's package dependency. Unknown or broken constraints fail the Evaluation
ticket; they can never silently earn credit.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from ifbench import instructions_registry


def check_constraints(spec: object, response: str) -> tuple[bool, int]:
    if not isinstance(spec, dict):
        raise ValueError("constraint_spec must be a JSON object")
    ids = spec.get("instruction_id")
    kwargs = spec.get("kwargs")
    if not isinstance(ids, list) or not ids or not isinstance(kwargs, list):
        raise ValueError("constraint_spec requires instruction_id and kwargs lists")
    if len(ids) != len(kwargs):
        raise ValueError("constraint IDs and kwargs must have the same length")
    checkers = []
    for key, arguments in zip(ids, kwargs):
        checker_class = instructions_registry.INSTRUCTION_DICT.get(key)
        if checker_class is None:
            raise ValueError(f"no official verifier for {key!r}")
        if arguments is not None and not isinstance(arguments, dict):
            raise ValueError(f"constraint {key!r} kwargs must be an object or null")
        checker = checker_class(key)
        checker.build_description(**(arguments or {}))
        checkers.append(checker)
    if not response.strip():
        return False, 0
    passed = sum(bool(checker.check_following(response)) for checker in checkers)
    return passed == len(ids), passed


def score_files(predictions_path: Path, scoring_set_path: Path) -> dict:
    with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    with scoring_set_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or len(predictions) != len(rows):
        raise ValueError("prediction and scoring row counts differ or are empty")
    prompt_passes = 0
    constraint_passes = 0
    constraint_total = 0
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        if prediction.get("id") != row.get("id"):
            raise ValueError(f"prediction ID mismatch at row {index}")
        spec = row.get("constraint_spec")
        if isinstance(spec, str):
            spec = json.loads(spec)
        passed, count = check_constraints(spec, prediction.get("prediction", ""))
        prompt_passes += int(passed)
        constraint_passes += count
        constraint_total += len(spec["instruction_id"])
    return {
        "score": prompt_passes / len(rows),
        "instruction_following_accuracy": prompt_passes / len(rows),
        "constraint_accuracy": constraint_passes / constraint_total,
        "n_scored": len(rows),
        "n_constraints": constraint_total,
        "score_range": [0, 1],
        "scoring_rule": "all declared constraints must pass",
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: evaluator.py predictions.csv scoring_set.jsonl metrics.json")
    metrics = score_files(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
