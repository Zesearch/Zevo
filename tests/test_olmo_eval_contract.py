"""Regression tests for the local OLMo Task's authored scorer assets."""

from __future__ import annotations

import importlib.util
import csv
import json
import os
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest

from zevo.contracts.prompting import render_inference_query
from zevo.contracts.orchestrator import UserRequest
from zevo.engine.method import model_judge


ROOT = Path(__file__).resolve().parents[1]


def module_at(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


test_eval = module_at("data/files/OLMo-3.1-Evaluation-Data/evaluator.py", "olmo_test_eval")
math_eval = module_at(
    "data/files/OLMo-3.1-Validation-Data/math-hard-evaluator.py", "olmo_math_eval",
)
validation_eval = module_at(
    "data/files/OLMo-3.1-Validation-Data/evaluator.py", "olmo_validation_eval",
)
safety_eval = module_at(
    "data/files/OLMo-3.1-Validation-Data/safety-wildjailbreak-evaluator.py",
    "olmo_safety_eval",
)
task_update = module_at("ops/update_olmo_eval_contract.py", "olmo_task_update")


def test_math_answers_accept_equivalent_formats_without_accepting_wrong_values():
    assert test_eval.math_equal(r"Reasoning. \boxed{(3,\pi/2)}", r"\left(3,\frac{\pi}{2}\right)")
    assert test_eval.math_equal(r"\boxed{70}", "070")
    assert not test_eval.math_equal(r"\boxed{71}", "070")
    assert math_eval.equivalent(r"(3,\pi/2)", r"\left(3,\frac{\pi}{2}\right)")


def test_popqa_uses_exact_alias_not_substring():
    gold = {"obj": "politician", "possible_answers": json.dumps(["politician", "pol"])}
    assert test_eval.score_row("popqa", gold, "politician") == 1.0
    assert test_eval.score_row("popqa", gold, "police officer") == 0.0


def test_agieval_math_rows_are_not_misread_as_option_indices():
    math_gold = {"options": "[]", "answer": "3", "label": ""}
    choice_gold = {"options": '["(A) one", "(B) two"]', "answer": "", "label": "B"}
    assert test_eval.score_row("agieval", math_gold, r"\boxed{3}") == 1.0
    assert test_eval.score_row("agieval", math_gold, r"\boxed{D}") == 0.0
    assert test_eval.score_row("agieval", choice_gold, r"\boxed{B}") == 1.0


def test_instruction_following_uses_every_constraint_and_skips_bad_metadata():
    pytest.importorskip("ifbench")
    row = {
        "instruction_id_list": json.dumps(["detectable_format:json_format"]),
        "kwargs": json.dumps([{}]),
    }
    assert test_eval.strict_constraints(row, '{"ok": true}') == (True, 1, 1)
    assert test_eval.strict_constraints(row, "Hello") == (False, 0, 1)

    exact_letter = {
        "instruction_id_list": json.dumps(["keywords:letter_frequency"]),
        "kwargs": json.dumps([{"letter": "x", "let_frequency": 2}]),
    }
    assert test_eval.strict_constraints(exact_letter, "xx") == (True, 1, 1)
    assert test_eval.strict_constraints(exact_letter, "xxx") == (False, 0, 1)

    malformed = {
        "instruction_id_list": json.dumps(["length_constraints:nth_paragraph_first_word"]),
        "kwargs": json.dumps([{"num_paragraphs": 3, "keyword": "schedule"}]),
    }
    with pytest.raises(test_eval.UnscorableRow):
        test_eval.strict_constraints(malformed, "schedule")


def write_csv(path, fieldnames, values):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def test_model_judge_pairwise_is_position_balanced_and_cacheable(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(model_judge, "_request_verdict", lambda item, **kwargs: "prediction")
    items = [model_judge.JudgeItem("pairwise", f"Question {n}", "good", "bad") for n in range(8)]
    keys = [model_judge._cache_key(item, model_judge.DEFAULT_MODEL) for item in items]
    assert {model_judge._candidate_order(key) for key in keys} == {True, False}
    results, model = model_judge.judge_many(items, tmp_path / "metrics.json")
    assert results == ["prediction"] * 8
    assert model == model_judge.DEFAULT_MODEL
    assert "Question" not in (tmp_path / "judge-cache.jsonl").read_text()
    progress = json.loads((tmp_path / "judge-progress.json").read_text())
    assert progress == {
        "schema_version": 1, "model": model_judge.DEFAULT_MODEL,
        "completed": 8, "total": 8, "cached": 0,
    }
    assert "Question" not in (tmp_path / "judge-progress.json").read_text()
    monkeypatch.delenv("OPENAI_API_KEY")
    assert model_judge.judge_many(items, tmp_path / "metrics.json")[0] == results
    assert json.loads((tmp_path / "judge-progress.json").read_text())["cached"] == 8


def test_model_judge_progress_counts_cached_and_duplicate_rows(tmp_path, monkeypatch):
    cached_item = model_judge.JudgeItem("pairwise", "Private cached prompt", "candidate", "reference")
    new_item = model_judge.JudgeItem("pairwise", "Private new prompt", "candidate", "reference")
    key = model_judge._cache_key(cached_item, model_judge.DEFAULT_MODEL)
    (tmp_path / "judge-cache.jsonl").write_text(
        json.dumps({"key": key, "verdict": "prediction"}) + "\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ZEVO_EVALUATION_JUDGE_WORKERS", "1")
    snapshots = []

    def fake_request(item, **kwargs):
        snapshots.append(json.loads((tmp_path / "judge-progress.json").read_text()))
        return "reference"

    monkeypatch.setattr(model_judge, "_request_verdict", fake_request)
    verdicts, _ = model_judge.judge_many(
        [cached_item, new_item, new_item], tmp_path / "metrics.json",
    )
    assert verdicts == ["prediction", "reference", "reference"]
    assert [(s["completed"], s["total"], s["cached"]) for s in snapshots] == [(1, 3, 1)]
    assert json.loads((tmp_path / "judge-progress.json").read_text())["completed"] == 3
    assert "Private" not in (tmp_path / "judge-progress.json").read_text()


def test_model_judge_requires_api_key_and_structured_response(tmp_path, monkeypatch):
    item = model_judge.JudgeItem("pairwise", "Question", "candidate", "reference")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        model_judge.judge_many([item], tmp_path / "metrics.json")
    captured = {}

    class FakeResponse(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return FakeResponse(json.dumps({
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": '{"verdict":"tie"}'},
            ]}],
        }).encode())

    monkeypatch.setattr(model_judge, "urlopen", fake_urlopen)
    verdict = model_judge._request_verdict(item, key="01" * 32, api_key="test-key", model="gpt-5.6-luna")
    assert verdict == "tie"
    assert captured["model"] == "gpt-5.6-luna"
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["reasoning"] == {"effort": "none"}
    assert captured["store"] is False


def test_obsolete_judge_selection_does_not_override_fixed_model(tmp_path, monkeypatch):
    item = model_judge.JudgeItem("safety", "Benign request", "Helpful answer")
    monkeypatch.setenv("ZEVO_EVALUATION_JUDGE_PROVIDER", "vertex_ai")
    monkeypatch.setenv("ZEVO_EVALUATION_JUDGE_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GOOGLE_CLOUD_VERTEX_API_KEY", "vertex-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    calls = []
    def fake_request(item, **kwargs):
        calls.append(kwargs)
        return "helpful_compliance"
    monkeypatch.setattr(model_judge, "_request_verdict", fake_request)
    verdicts, model = model_judge.judge_many([item], tmp_path / "metrics.json")
    assert verdicts == ["helpful_compliance"]
    assert model == "gpt-5.6-luna"
    assert calls[0]["model"] == "gpt-5.6-luna"
    assert calls[0]["api_key"] == "openai-test-key"


def test_dolly_uses_pairwise_judge_and_context(tmp_path, monkeypatch):
    scoring, predictions, metrics = (tmp_path / name for name in ("scoring.csv", "predictions.csv", "metrics.json"))
    write_csv(scoring, ["instruction", "context", "response"], [
        {"instruction": "What year?", "context": "It began in 2000.", "response": "2000"},
        {"instruction": "What color?", "context": "", "response": "Blue"},
    ])
    write_csv(predictions, ["benchmark", "prediction"], [
        {"benchmark": "dolly", "prediction": "2000"},
        {"benchmark": "dolly", "prediction": "Blue"},
    ])
    received = []
    def fake_judge(items, path):
        received.extend(items)
        return ["prediction", "tie"], "gpt-5.6-luna"
    monkeypatch.setattr(validation_eval, "judge_many", fake_judge)
    monkeypatch.setattr(sys, "argv", ["evaluator.py", str(predictions), str(scoring), str(metrics)])
    validation_eval.main()
    result = json.loads(metrics.read_text())
    assert result["pairwise_win_rate_vs_human_reference_model_judge"] == 0.75
    assert "It began in 2000." in received[0].prompt


def test_frozen_scorer_protocol_runs_in_subprocess_from_cached_verdict(tmp_path):
    scoring, predictions, metrics = (tmp_path / name for name in ("scoring.csv", "predictions.csv", "metrics.json"))
    write_csv(scoring, ["instruction", "context", "response"], [
        {"instruction": "What year?", "context": "It began in 2000.", "response": "2000"},
    ])
    write_csv(predictions, ["benchmark", "prediction"], [
        {"benchmark": "dolly", "prediction": "The year was 2000."},
    ])
    item = model_judge.JudgeItem("pairwise", "What year?\n\nContext:\nIt began in 2000.",
                                "The year was 2000.", "2000")
    key = model_judge._cache_key(item, model_judge.DEFAULT_MODEL)
    (tmp_path / "judge-cache.jsonl").write_text(json.dumps({
        "key": key, "verdict": "prediction",
    }) + "\n")
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run([
        sys.executable, str(ROOT / "data/files/OLMo-3.1-Validation-Data/evaluator.py"),
        str(predictions), str(scoring), str(metrics),
    ], env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(metrics.read_text())["score"] == 1.0


def test_alpacaeval_uses_pairwise_judge_not_token_overlap(tmp_path, monkeypatch):
    scoring, predictions, metrics = (tmp_path / name for name in ("scoring.csv", "predictions.csv", "metrics.json"))
    write_csv(scoring, ["instruction", "output"], [
        {"instruction": "Say hello", "output": "Hello there"},
        {"instruction": "Say goodbye", "output": "Goodbye"},
    ])
    write_csv(predictions, ["benchmark", "prediction"], [
        {"benchmark": "alpacaeval", "prediction": "A warm greeting"},
        {"benchmark": "alpacaeval", "prediction": "Goodbye"},
    ])
    monkeypatch.setattr(test_eval, "judge_many", lambda items, path: (["prediction", "reference"], "gpt-5.6-luna"))
    monkeypatch.setattr(sys, "argv", ["evaluator.py", str(predictions), str(scoring), str(metrics)])
    test_eval.main()
    result = json.loads(metrics.read_text())
    assert result["pairwise_win_rate_vs_reference_model_judge"] == 0.5
    assert "not official" in result["judge_protocol"]


def test_safety_validation_balances_benign_and_harmful_labels(tmp_path, monkeypatch):
    scoring = tmp_path / "scoring.csv"
    predictions = tmp_path / "predictions.csv"
    metrics = tmp_path / "metrics.json"
    write_csv(scoring, ["prompt", "label"], [
        {"prompt": "benign", "label": "adversarial_benign"},
        {"prompt": "harmful", "label": "adversarial_harmful"},
        {"prompt": "harmful again", "label": "adversarial_harmful"},
    ])
    write_csv(predictions, ["benchmark", "prediction"], [
        {"benchmark": "safety_validation", "prediction": "I can help."},
        {"benchmark": "safety_validation", "prediction": "I cannot help with that."},
        {"benchmark": "safety_validation", "prediction": "Here are the steps."},
    ])
    monkeypatch.setattr(safety_eval, "judge_many", lambda items, path: (
        ["helpful_compliance", "safe_refusal_or_redirect", "unsafe_compliance"],
        "gpt-5.6-luna",
    ))
    monkeypatch.setattr(sys, "argv", ["evaluator.py", str(predictions), str(scoring), str(metrics)])
    safety_eval.main()
    result = json.loads(metrics.read_text())
    assert result["benign_helpfulness_accuracy"] == 1.0
    assert result["harmful_safe_response_accuracy"] == 0.5
    assert result["balanced_safe_response_accuracy_model_judge"] == 0.75


def test_wildguardtest_excludes_unlabeled_rows_and_balances_classes(tmp_path, monkeypatch):
    scoring, predictions, metrics = (tmp_path / name for name in ("scoring.csv", "predictions.csv", "metrics.json"))
    write_csv(scoring, ["prompt", "label"], [
        {"prompt": "benign", "label": "unharmful"},
        {"prompt": "harmful", "label": "harmful"},
        {"prompt": "unknown", "label": ""},
    ])
    write_csv(predictions, ["benchmark", "prediction"], [
        {"benchmark": "safety", "prediction": "Answer"},
        {"benchmark": "safety", "prediction": "Decline"},
        {"benchmark": "safety", "prediction": "Whatever"},
    ])
    monkeypatch.setattr(test_eval, "judge_many", lambda items, path: (
        ["helpful_compliance", "safe_refusal_or_redirect"], "gpt-5.6-luna",
    ))
    monkeypatch.setattr(sys, "argv", ["evaluator.py", str(predictions), str(scoring), str(metrics)])
    test_eval.main()
    result = json.loads(metrics.read_text())
    assert result["balanced_safe_response_accuracy_model_judge"] == 1.0
    assert result["n_scored"] == 2
    assert result["n_excluded"] == 1


def test_preflight_blocks_model_judge_without_openai_api_key(monkeypatch):
    from zevo.api.routers.ui.preflight import _check_eval

    request = UserRequest(
        task_objective="test", metric_type="custom",
        metric="pairwise_win_rate_vs_reference_model_judge",
        metric_direction="max",
        evaluation_script=str(ROOT / "data/files/OLMo-3.1-Evaluation-Data/evaluator.py"),
        training_method="", dataset="", base_model="",
        test_set="/data/test.csv", test_answer_fields=["output"],
        test_sample_submission="/data/submission.csv",
        validation_metric_type="builtin", validation_metric="accuracy",
        validation_metric_direction="max", constraints=[],
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    items = []
    _check_eval(request, items)
    assert any(item.code == "model_judge_api_key_missing" for item in items)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    items = []
    _check_eval(request, items)
    assert any(item.code == "model_judge_api_key" and "gpt-5.6-luna" in item.message for item in items)
    monkeypatch.setenv("ZEVO_EVALUATION_JUDGE_PROVIDER", "vertex_ai")
    monkeypatch.delenv("OPENAI_API_KEY")
    items = []
    _check_eval(request, items)
    assert any(item.code == "model_judge_api_key_missing" and "OPENAI_API_KEY" in item.message for item in items)


def test_prediction_rows_must_match_reference_identity():
    test_eval.validate_alignment(
        [{"id": "1"}], [{"id": "1", "benchmark": "math", "prediction": "x"}],
    )
    with pytest.raises(ValueError, match="id differs"):
        test_eval.validate_alignment(
            [{"id": "1"}], [{"id": "2", "benchmark": "math", "prediction": "x"}],
        )


def test_queries_only_direct_generation_and_render_expected_fields():
    queries = {
        **task_update.TEST_QUERIES,
        **task_update.VALIDATION_QUERIES,
    }
    assert len(task_update.TEST_QUERIES) == 17
    assert len(task_update.VALIDATION_QUERIES) == 7
    for query in queries.values():
        assert not re.search(r"(?i)\b(?:set benchmark|copy .+ unchanged|csv columns)\b", query)
        assert "prediction" not in query.lower()
    rendered = render_inference_query(
        task_update.query_text(task_update.TEST_QUERIES["Math · MATH-500"]),
        {"problem": "What is 1+1?"},
    )
    assert rendered.endswith("What is 1+1?")
    assert r"\boxed{answer}" in rendered
    assert "\\n" not in rendered
