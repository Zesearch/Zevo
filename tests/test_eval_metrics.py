"""Tests for zevo.engine.method.eval_metrics — pin the built-in scorers + the
alignment validator so they never silently regress.

The orchestrator decides whether to keep iterating based on score
deltas; bad metrics = wrong stop decisions = wasted budget. Each
scorer here is exercised against hand-computed expected values.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from zevo.engine.method.eval_metrics import (
    accuracy, exact_match, f1, token_f1, bleu, rouge_l, mc_loglikelihood,
    validate_alignment, score_default, SCORERS,
)


# ─────────────────────────────── accuracy ───────────────────────────────────


def test_score_perfect_match() -> None:
    out = accuracy(["A", "B", "C"], ["A", "B", "C"])
    assert out["accuracy"] == 1.0
    assert out["n_correct"] == 3
    assert out["n_total"] == 3


def test_score_normalization_handles_case_and_articles() -> None:
    out = accuracy(["The cat", "Dogs"], ["a cat", "dogs"])
    # "The cat" -> "cat", "a cat" -> "cat"; "Dogs" -> "dogs", "dogs" -> "dogs"
    assert out["accuracy"] == 1.0


def test_score_strict_requires_exact() -> None:
    out = accuracy(["A", "b"], ["a", "B"], strict=True)
    assert out["accuracy"] == 0.0


def test_score_empty_returns_negative_one() -> None:
    assert accuracy([], [])["accuracy"] == -1.0


def test_score_unequal_lengths_uses_min() -> None:
    out = accuracy(["a", "b", "c"], ["a", "b"])
    assert out["n_total"] == 2


# ─────────────────────────────── exact_match ────────────────────────────────


def test_exact_match_is_strict() -> None:
    out = exact_match(["A", "B"], ["A", "b"])
    assert out["accuracy"] == 0.5
    assert out["metric_name"] == "exact_match"


# ─────────────────────────────── f1 ─────────────────────────────────────────


def test_f1_micro_overlap() -> None:
    # _normalize strips leading "the " so:
    #   "the cat sat" -> ["cat","sat"]
    #   "the cat ran" -> ["cat","ran"]
    # Overlap: "cat" (1 tok). P=1/2, R=1/2, F1=0.5
    out = f1(["the cat sat"], ["the cat ran"])
    assert abs(out["f1"] - 0.5) < 1e-3
    assert abs(out["precision"] - 0.5) < 1e-3
    assert abs(out["recall"] - 0.5) < 1e-3


def test_f1_macro_averages_per_example() -> None:
    out = f1(
        ["same words", "totally different"],
        ["same words", "completely unrelated"],
        average="macro",
    )
    # ex1: F1=1.0 ; ex2: F1=0 ; macro avg = 0.5
    assert abs(out["f1"] - 0.5) < 1e-3


def test_token_f1_uses_squad_normalization() -> None:
    out = token_f1(["The red, fox!"], ["a red fox"])
    assert out["token_f1"] == 1.0
    assert out["n_total"] == 1


def test_token_f1_macro_averages_multi_turn_records() -> None:
    out = token_f1(
        ['{"1":"same", "2":"wrong"}', "perfect answer"],
        ['{"1":"same", "2":"different"}', "perfect answer"],
    )
    assert out["token_f1"] == 0.75
    assert out["unit_token_f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert out["n_total"] == 2
    assert out["n_units"] == 3


# ─────────────────────────────── bleu ───────────────────────────────────────


def test_bleu_perfect_is_near_one() -> None:
    out = bleu(["the quick brown fox"], ["the quick brown fox"])
    assert out["bleu"] > 0.95  # smoothing keeps it under exact 1.0


def test_bleu_no_overlap_is_lower_than_perfect() -> None:
    # With add-1 smoothing on small inputs BLEU never hits 0, but the
    # no-overlap case must score MUCH lower than the perfect-match case.
    no_overlap = bleu(["zebra elephant penguin"], ["the quick brown fox"])
    perfect = bleu(["the quick brown fox"], ["the quick brown fox"])
    assert no_overlap["bleu"] < perfect["bleu"] - 0.3, (
        f"no-overlap BLEU {no_overlap['bleu']} should be much lower than "
        f"perfect {perfect['bleu']}"
    )


# ─────────────────────────────── rouge_l ────────────────────────────────────


def test_rouge_l_perfect_is_one() -> None:
    out = rouge_l(["the quick brown fox"], ["the quick brown fox"])
    assert out["rouge_l"] == 1.0


def test_rouge_l_partial() -> None:
    # After _normalize strips "the ":
    #   "cat sat" vs "dog sat" → LCS = ["sat"] = 1 token
    #   P=1/2, R=1/2, F=0.5
    out = rouge_l(["the cat sat"], ["the dog sat"])
    assert abs(out["rouge_l"] - 0.5) < 1e-3


# ────────────────────────── alignment validation ─────────────────────────────


def _write_csv(tmp_path: Path, name: str, header: list[str], rows: list[dict]) -> Path:
    p = tmp_path / name
    import csv
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def test_alignment_clean_csv(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "preds.csv", ["id", "answer"], [
        {"id": "1", "answer": "A"},
        {"id": "2", "answer": "B"},
    ])
    gold = _write_csv(tmp_path, "gold.csv", ["id", "answer"], [
        {"id": "1", "answer": "A"},
        {"id": "2", "answer": "C"},
    ])
    report = validate_alignment(preds, gold)
    assert report.ready
    assert report.answer_col == "answer"
    assert report.n_preds == 2
    assert report.n_golds == 2


def test_alignment_rowcount_mismatch(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "preds.csv", ["answer"], [
        {"answer": "A"}, {"answer": "B"},
    ])
    gold = _write_csv(tmp_path, "gold.csv", ["answer"], [
        {"answer": "A"}, {"answer": "B"}, {"answer": "C"},
    ])
    report = validate_alignment(preds, gold)
    assert not report.ready
    assert any(i.code == "rowcount_mismatch" for i in report.issues)


def test_alignment_rejects_empty_files(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "preds.csv", ["answer"], [])
    gold = _write_csv(tmp_path, "gold.csv", ["answer"], [])
    report = validate_alignment(preds, gold)
    assert not report.ready
    assert any(i.code == "empty_rows" for i in report.issues)


def test_alignment_unknown_score_columns(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "preds.csv", ["foo"], [{"foo": "A"}])
    gold = _write_csv(tmp_path, "gold.csv", ["bar"], [{"bar": "A"}])
    report = validate_alignment(preds, gold)
    assert not report.ready
    assert any(i.code == "no_shared_answer_col" for i in report.issues)


def test_alignment_allows_different_prediction_and_gold_columns(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "preds.csv", ["id", "prediction_idx"], [
        {"id": "1", "prediction_idx": "A"},
        {"id": "2", "prediction_idx": "B"},
    ])
    gold = _write_csv(tmp_path, "gold.csv", ["id", "gold"], [
        {"id": "1", "gold": "A"},
        {"id": "2", "gold": "B"},
    ])
    report = validate_alignment(preds, gold)
    assert report.ready
    assert report.prediction_col == "prediction_idx"
    assert report.answer_col == "gold"


def test_alignment_rejects_key_order_mismatch(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "preds.csv", ["id", "prediction"], [
        {"id": "2", "prediction": "B"},
        {"id": "1", "prediction": "A"},
    ])
    gold = _write_csv(tmp_path, "gold.csv", ["id", "answer"], [
        {"id": "1", "answer": "A"},
        {"id": "2", "answer": "B"},
    ])
    report = validate_alignment(preds, gold)
    assert not report.ready
    assert any(i.code == "row_order_mismatch" for i in report.issues)


def test_alignment_identical_predictions_warns(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "preds.csv", ["answer"],
                       [{"answer": "A"}] * 12)
    gold = _write_csv(tmp_path, "gold.csv", ["answer"],
                      [{"answer": str(i)} for i in range(12)])
    report = validate_alignment(preds, gold)
    assert any(i.code == "all_predictions_identical" for i in report.issues)


def test_alignment_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    g = tmp_path / "g.jsonl"
    p.write_text('{"answer":"A"}\n{"answer":"B"}\n', encoding="utf-8")
    g.write_text('{"answer":"A"}\n{"answer":"B"}\n', encoding="utf-8")
    report = validate_alignment(p, g)
    assert report.ready
    assert report.answer_col == "answer"


def test_alignment_json_array(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    g = tmp_path / "g.json"
    p.write_text('[{"prediction":"A"},{"prediction":"B"}]', encoding="utf-8")
    g.write_text('[{"answer":"A"},{"answer":"B"}]', encoding="utf-8")
    report = validate_alignment(p, g)
    assert report.ready
    assert report.prediction_col == "prediction"
    assert report.answer_col == "answer"


# ────────────────── mc_loglikelihood (option scoring) ───────────────────────


def _ll(**opts: dict) -> str:
    """Serialize a per-option {logprob,num_tokens} prediction cell as JSON."""
    return json.dumps(opts)


def test_mc_loglikelihood_argmax_of_normalized_scores() -> None:
    # Row 1: B has the highest per-token LL (-1.0) -> pick B (gold B) => correct.
    # Row 2: A has the highest per-token LL -> pick A, gold is C => wrong.
    preds = [
        _ll(A={"logprob": -12.0, "num_tokens": 4},
            B={"logprob": -3.0, "num_tokens": 3},
            C={"logprob": -20.0, "num_tokens": 5}),
        _ll(A={"logprob": -2.0, "num_tokens": 2},
            B={"logprob": -9.0, "num_tokens": 3},
            C={"logprob": -8.0, "num_tokens": 4}),
    ]
    golds = ["B", "C"]
    out = mc_loglikelihood(preds, golds)
    assert out["accuracy_norm"] == 0.5
    assert out["accuracy"] == 0.5  # mirror for headline resolution
    assert out["n_correct"] == 1
    assert out["n_total"] == 2
    assert out["n_unparsed"] == 0
    assert out["metric_name"] == "mc_loglikelihood"


def test_mc_loglikelihood_length_normalization_changes_the_winner() -> None:
    # A has the higher SUMMED logprob (-6 > -7) but B wins per-token
    # (-7/1=-7 vs A -6/6=-1)... construct so normalization flips the pick.
    # Raw sum: A=-6 (better than B=-7). Per-token: A=-6/6=-1.0, B=-7/10=-0.7.
    # Length-normalized picks B.
    pred = _ll(A={"logprob": -6.0, "num_tokens": 6},
               B={"logprob": -7.0, "num_tokens": 10})
    assert mc_loglikelihood([pred], ["B"])["accuracy_norm"] == 1.0
    assert mc_loglikelihood([pred], ["A"])["accuracy_norm"] == 0.0


def test_mc_loglikelihood_accepts_plain_numeric_scores() -> None:
    # Already-normalized per-option scores (e.g. first-token logit): argmax B.
    pred = json.dumps({"A": -2.5, "B": -0.4, "C": -3.1})
    assert mc_loglikelihood([pred], ["B"])["accuracy_norm"] == 1.0


def test_mc_loglikelihood_gold_by_integer_index() -> None:
    # Array form is position-keyed; gold "1" selects the second option.
    pred = json.dumps([-2.0, -0.5, -3.0])
    out = mc_loglikelihood([pred], ["1"])
    assert out["accuracy_norm"] == 1.0


def test_mc_loglikelihood_ties_break_by_label_ascending() -> None:
    pred = json.dumps({"B": -1.0, "A": -1.0})
    # Both tied; deterministic pick is "A" (label ascending).
    assert mc_loglikelihood([pred], ["A"])["accuracy_norm"] == 1.0
    assert mc_loglikelihood([pred], ["B"])["accuracy_norm"] == 0.0


def test_mc_loglikelihood_unparsed_prediction_counts_as_wrong() -> None:
    out = mc_loglikelihood(["not json at all", json.dumps({"A": -0.1, "B": -2.0})],
                           ["A", "A"])
    assert out["n_unparsed"] == 1
    assert out["n_correct"] == 1
    assert out["accuracy_norm"] == 0.5


def test_mc_loglikelihood_rejects_non_finite_logprob() -> None:
    # NaN/Inf must not slip through as a real score.
    out = mc_loglikelihood(['{"A": "NaN", "B": -1.0}'], ["B"])
    assert out["n_unparsed"] == 1


def test_mc_loglikelihood_empty_is_negative_one() -> None:
    out = mc_loglikelihood([], [])
    assert out["accuracy_norm"] == -1.0


def test_mc_loglikelihood_is_registered_in_scorers() -> None:
    assert "mc_loglikelihood" in SCORERS
    assert SCORERS["mc_loglikelihood"] is mc_loglikelihood


def test_score_default_mc_loglikelihood_via_csv(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "p.csv", ["id", "option_scores"], [
        {"id": "1", "option_scores": json.dumps(
            {"A": {"logprob": -2.0, "num_tokens": 2},
             "B": {"logprob": -9.0, "num_tokens": 3}})},
        {"id": "2", "option_scores": json.dumps(
            {"A": {"logprob": -8.0, "num_tokens": 2},
             "B": {"logprob": -1.0, "num_tokens": 3}})},
    ])
    gold = _write_csv(tmp_path, "g.csv", ["id", "answer"], [
        {"id": "1", "answer": "A"},
        {"id": "2", "answer": "B"},
    ])
    out = score_default(
        preds, gold, prediction_col_hint="option_scores",
        gold_col_hint="answer", metric="mc_loglikelihood",
    )
    assert out["status"] == "succeeded"
    assert out["metric_used"] == "mc_loglikelihood"
    assert out["score"] == 1.0
    assert out["accuracy_norm"] == 1.0


def test_score_default_accepts_accuracy_norm_alias(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "p.csv", ["scores"], [
        {"scores": json.dumps({"A": -0.2, "B": -3.0})},
    ])
    gold = _write_csv(tmp_path, "g.csv", ["answer"], [{"answer": "A"}])
    out = score_default(
        preds, gold, prediction_col_hint="scores",
        gold_col_hint="answer", metric="accuracy_norm",
    )
    assert out["status"] == "succeeded"
    assert out["metric_used"] == "mc_loglikelihood"
    assert out["score"] == 1.0


# ─────────────────────────── score_default e2e ───────────────────────────────


def test_score_default_clean(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "p.csv", ["answer"], [
        {"answer": "yes"}, {"answer": "no"}, {"answer": "yes"},
    ])
    gold = _write_csv(tmp_path, "g.csv", ["answer"], [
        {"answer": "yes"}, {"answer": "no"}, {"answer": "no"},
    ])
    out = score_default(preds, gold)
    assert out["status"] == "succeeded"
    # 2/3 correct = 0.666...
    assert abs(out["score"] - 2/3) < 1e-3
    assert out["accuracy"] == out["score"]
    assert out["answer_col"] == "answer"
    assert out["metric_used"] == "accuracy"


def test_score_default_alignment_failure_propagates(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "p.csv", ["pred"], [{"pred": "A"}])
    gold = _write_csv(tmp_path, "g.csv", ["gt"], [{"gt": "A"}])
    out = score_default(preds, gold)
    assert out["status"] == "failed"
    assert out["score"] is None
    assert any(i["code"] == "no_shared_answer_col" for i in out["alignment_issues"])


def test_score_default_maps_different_columns_and_named_metric(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "p.csv", ["prediction"], [
        {"prediction": "the red fox"}, {"prediction": "blue whale"},
    ])
    gold = _write_csv(tmp_path, "g.csv", ["gold"], [
        {"gold": "red fox"}, {"gold": "blue bird"},
    ])
    out = score_default(
        preds, gold, prediction_col_hint="prediction",
        gold_col_hint="gold", metric="rouge_l",
    )
    assert out["status"] == "succeeded"
    assert out["metric_used"] == "rouge_l"
    assert out["prediction_col"] == "prediction"
    assert out["answer_col"] == "gold"
    assert out["score"] == out["rouge_l"]


def test_score_default_rejects_unknown_metric(tmp_path: Path) -> None:
    preds = _write_csv(tmp_path, "p.csv", ["answer"], [{"answer": "A"}])
    gold = _write_csv(tmp_path, "g.csv", ["answer"], [{"answer": "A"}])
    out = score_default(preds, gold, metric="pass_at_1")
    assert out["status"] == "failed"
    assert out["score"] is None
    assert out["alignment_issues"][0]["code"] == "unknown_metric"


def test_all_scorers_handle_empty_inputs() -> None:
    """Every scorer must defensively return -1.0 on empty inputs, not crash."""
    for name, fn in SCORERS.items():
        try:
            out = fn([], [])
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"scorer {name!r} crashed on empty inputs: {e}")
        assert isinstance(out, dict), f"{name} must return a dict"
