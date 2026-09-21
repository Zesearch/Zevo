#!/usr/bin/env python3
"""Score MATH-Hard's boxed final answers without exposing its solutions to Inference."""

from __future__ import annotations

import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path


def rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def last_boxed(value: str) -> str | None:
    text = str(value or "")
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def normalized(value: str) -> str:
    text = value.lower().strip().replace("$", "")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\,", "").replace(r"\!", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text).strip(" .,:;!\"'")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    fraction = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while fraction.search(text):
        text = fraction.sub(r"(\1)/(\2)", text)
    atom = re.compile(r"\(([+-]?(?:\\[a-z]+|[a-z0-9.]+))\)")
    while atom.search(text):
        text = atom.sub(r"\1", text)
    return text


def equivalent(actual: str, expected: str) -> bool:
    actual, expected = normalized(actual), normalized(expected)
    if not actual or not expected:
        return False
    if actual == expected:
        return True
    rational = re.compile(r"[+-]?\d+(?:/\d+)?")
    if rational.fullmatch(actual) and rational.fullmatch(expected):
        try:
            return Fraction(actual) == Fraction(expected)
        except ZeroDivisionError:
            return False
    return False


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: math-hard-evaluator.py predictions.csv scoring.csv metrics.json")
    predictions = rows(sys.argv[1])
    scoring = rows(sys.argv[2])
    if len(predictions) != len(scoring):
        raise ValueError("prediction/scoring row counts differ")
    if not scoring or "solution" not in scoring[0]:
        raise ValueError("MATH-Hard scoring set requires a solution column")
    if not predictions or "prediction" not in predictions[0]:
        raise ValueError("submission requires a prediction column")

    correct = 0
    missing_gold = 0
    missing_prediction = 0
    for candidate, gold in zip(predictions, scoring):
        expected = last_boxed(gold["solution"])
        actual = last_boxed(candidate["prediction"])
        missing_gold += expected is None
        missing_prediction += actual is None
        if expected is not None and actual is not None:
            correct += equivalent(actual, expected)
    if missing_gold:
        raise ValueError(f"{missing_gold} MATH-Hard reference solutions have no boxed answer")

    score = correct / len(scoring)
    Path(sys.argv[3]).write_text(json.dumps({
        "score": score,
        "answer_accuracy": score,
        "n_correct": correct,
        "n_total": len(scoring),
        "n_missing_prediction_boxed": missing_prediction,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
