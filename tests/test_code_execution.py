from __future__ import annotations

import base64
import csv
import json
import pickle
import zlib
from pathlib import Path

import pytest

from zevo.code_benchmarks import (
    code_execution_adapter_for,
    externalize_code_answers,
    resolve_code_answer,
    store_code_answer_buffer,
)
from zevo.contracts.orchestrator import TaskTestSet
from zevo.engine.code_execution.scorer import extract_python_code, score_code_benchmark


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _score(
    tmp_path: Path, *, adapter: str, scoring: dict[str, str], prediction: str,
) -> dict[str, object]:
    scoring_path = tmp_path / "scoring.csv"
    predictions_path = tmp_path / "predictions.csv"
    _write(scoring_path, [scoring])
    _write(predictions_path, [{"id": scoring["id"], "prediction": prediction}])
    return score_code_benchmark(
        adapter=adapter,
        predictions=predictions_path,
        scoring_set=scoring_path,
        prediction_column="prediction",
    )


def test_registered_references_share_code_execution_metric() -> None:
    assert code_execution_adapter_for("evalplus/humanevalplus") == "humaneval_plus"
    assert code_execution_adapter_for("hf://datasets/evalplus/mbppplus") == "mbpp_plus"
    assert code_execution_adapter_for(
        "https://huggingface.co/datasets/sam-paech/livecodebench-code_generation_lite"
    ) == "livecodebench"
    assert code_execution_adapter_for("deepmind/code_contests") == "code_contests"
    with pytest.raises(ValueError, match="registered coding benchmark"):
        TaskTestSet(
            name="unknown",
            test_set="owner/unregistered-code-data",
            inference_query="Write code.",
            sample_submission="/tmp/sample.csv",
            metric="pass_at_1",
            answer_fields=["tests"],
        )


def test_extract_python_code_prefers_fenced_payload() -> None:
    assert extract_python_code("reason\n```python\nprint(1)\n```") == "print(1)"


def test_large_livecodebench_answers_are_content_addressed(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    row = externalize_code_answers(
        "livecodebench", {"question_id": "x", "private_test_cases": "encoded"},
    )
    token = row["private_test_cases"]
    assert isinstance(token, str) and token.startswith("zevo-code-answer:v1:")
    assert resolve_code_answer(token) == "encoded"
    assert len(list((tmp_path / "holdout" / "code-execution-answers").iterdir())) == 1


def test_large_answer_never_encodes_the_complete_source_string(
    tmp_path: Path, monkeypatch,
) -> None:
    class NoWholeValueEncode(str):
        def encode(self, *args, **kwargs):  # pragma: no cover - failure path
            raise AssertionError("the complete hidden answer was encoded at once")

    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    raw = NoWholeValueEncode("private" * 300_000)
    row = externalize_code_answers(
        "livecodebench", {"question_id": "large", "private_test_cases": raw},
    )
    assert resolve_code_answer(row["private_test_cases"]) == raw


def test_arrow_style_buffer_is_externalized_without_a_python_string(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    raw = ("π-private-case\n" * 200_000).encode("utf-8")
    token = store_code_answer_buffer(memoryview(raw))
    assert resolve_code_answer(token) == raw.decode("utf-8")


def test_code_contests_hidden_answers_are_content_addressed(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    hidden = {"input": ["2 3\n"], "output": ["5\n"]}
    row = externalize_code_answers(
        "code_contests", {"id": "cc/0", "private_tests": hidden},
    )
    token = row["private_tests"]
    assert isinstance(token, str) and token.startswith("zevo-code-answer:v1:")
    assert json.loads(str(resolve_code_answer(token))) == hidden


def test_humaneval_and_mbpp_use_same_worker(tmp_path: Path) -> None:
    humaneval = _score(
        tmp_path,
        adapter="humaneval_plus",
        scoring={
            "id": "he/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        },
        prediction="```python\ndef add(a, b):\n    return a + b\n```",
    )
    assert humaneval["pass_at_1"] == 1.0

    completion = _score(
        tmp_path,
        adapter="humaneval_plus",
        scoring={
            "id": "he/1",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        },
        prediction="    return a + b\n",
    )
    assert completion["pass_at_1"] == 1.0

    mbpp = _score(
        tmp_path,
        adapter="mbpp_plus",
        scoring={
            "id": "mbpp/0",
            "test": "assert multiply(4, 5) == 20",
        },
        prediction="def multiply(a, b):\n    return a * b",
    )
    assert mbpp["pass_at_1"] == 1.0


def test_livecodebench_functional_and_stdio_cases(tmp_path: Path) -> None:
    encoded_private = base64.b64encode(zlib.compress(pickle.dumps(json.dumps([
        {"input": "4\n5", "output": "9", "testtype": "functional"},
    ])))).decode()
    functional = _score(
        tmp_path,
        adapter="livecodebench",
        scoring={
            "id": "lcb/functional",
            "metadata": json.dumps({"func_name": "add"}),
            "public_test_cases": "[]",
            "private_test_cases": encoded_private,
        },
        prediction="class Solution:\n    def add(self, a, b):\n        return a + b",
    )
    assert functional["pass_at_1"] == 1.0

    stdio = _score(
        tmp_path,
        adapter="livecodebench",
        scoring={
            "id": "lcb/stdio",
            "metadata": "{}",
            "public_test_cases": json.dumps([
                {"input": "8 7\n", "output": "15\n", "testtype": "stdin"},
            ]),
            "private_test_cases": "[]",
        },
        prediction="a, b = map(int, input().split())\nprint(a + b)",
    )
    assert stdio["pass_at_1"] == 1.0


def test_code_contests_uses_shared_stdio_worker(tmp_path: Path) -> None:
    metrics = _score(
        tmp_path,
        adapter="code_contests",
        scoring={
            "id": "cc/stdio",
            "public_tests": json.dumps({
                "input": ["8 7\n"], "output": ["15\n"],
            }),
            "private_tests": json.dumps({
                "input": ["2 3\n"], "output": ["5\n"],
            }),
            "generated_tests": json.dumps({"input": [], "output": []}),
        },
        prediction="a, b = map(int, input().split())\nprint(a + b)",
    )
    assert metrics["pass_at_1"] == 1.0


def test_wrong_code_is_a_zero_not_an_evaluator_failure(tmp_path: Path) -> None:
    metrics = _score(
        tmp_path,
        adapter="mbpp_plus",
        scoring={"id": "mbpp/1", "test": "assert answer() == 42"},
        prediction="def answer():\n    return 0",
    )
    assert metrics["status"] == "succeeded"
    assert metrics["score"] == 0.0
    assert metrics["outcomes"] == {"wrong_answer": 1}


def test_generated_code_cannot_read_host_files(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_text("must-not-appear", encoding="utf-8")
    metrics = _score(
        tmp_path,
        adapter="mbpp_plus",
        scoring={"id": "mbpp/unsafe", "test": "assert answer() == 42"},
        prediction=(
            "def answer():\n"
            f"    raise RuntimeError(open({str(secret)!r}).read())"
        ),
    )
    assert metrics["score"] == 0.0
    assert metrics["outcomes"] == {"runtime_error": 1}
    assert "must-not-appear" not in json.dumps(metrics)
