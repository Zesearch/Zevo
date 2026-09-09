"""Built-in eval scorers used by the deterministic Evaluation runner when the
run carries no Task-owned eval script.

Pure-Python + stdlib. We deliberately keep this small and dependency-
free so it runs identically inside the system runner and in the
backend's preflight check.

Available scorers:
  accuracy(preds, golds) -> {accuracy, n_correct, n_total}
  exact_match(preds, golds) -> {accuracy, n_correct, n_total}
  f1(preds, golds, average="micro") -> {f1, precision, recall}
  bleu(preds, golds) -> {bleu}                   # simple BLEU-4
  rouge_l(preds, golds) -> {rouge_l}            # F1 over LCS
  mc_loglikelihood(preds, golds) -> {accuracy_norm, ...}  # MC option scoring

Multiple-choice option scoring (opt-in, additive):
  mc_loglikelihood is the field-standard MMLU-style scorer. Instead of
  generating a letter and string-matching it, Inference emits per-option
  (length-normalized) log-likelihoods for each answer choice, and this scorer
  deterministically argmaxes them and compares to the gold option. It never
  calls a model — it only compares numbers Inference already produced — so it
  preserves the "Evaluation is LLM-free" invariant. The existing generate +
  string-match path (accuracy / exact_match) is untouched; this is a separate,
  selectable metric.

Alignment validation:
  validate_alignment(preds_path, gold_path, prediction/gold column hints)
      -> AlignmentReport

The runner reads the resolved dataset paths directly. A custom eval script can
also import and reuse these functions instead of re-implementing them.
"""
from __future__ import annotations

import csv
import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────────── scorers ────────────────────────────────────


def _normalize(s: Any) -> str:
    """Lowercase + collapse whitespace + strip punctuation that
    typically doesn't matter for QA-style matching."""
    if s is None:
        return ""
    t = str(s).strip().lower()
    # Strip leading articles + trailing punctuation that often varies
    t = re.sub(r"^(a |an |the )", "", t)
    t = re.sub(r"[.,!?\;:]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def accuracy(preds: list[Any], golds: list[Any], *, strict: bool = False) -> dict[str, Any]:
    """Fraction of preds exactly matching golds (case+whitespace-tolerant
    unless strict=True). Returns the metric + counts for the durable metrics
    artifact."""
    n = min(len(preds), len(golds))
    if n == 0:
        return {"accuracy": -1.0, "n_correct": 0, "n_total": 0,
                "note": "empty predictions or golds"}
    correct = 0
    for p, g in zip(preds[:n], golds[:n]):
        if strict:
            if p == g:
                correct += 1
        else:
            if _normalize(p) == _normalize(g):
                correct += 1
    return {
        "accuracy": round(correct / n, 6),
        "n_correct": correct,
        "n_total": n,
    }


def exact_match(preds: list[Any], golds: list[Any]) -> dict[str, Any]:
    """Strict exact-string match. Useful when answer format matters
    (e.g. boxed letter, multi-choice)."""
    return {**accuracy(preds, golds, strict=True), "metric_name": "exact_match"}


def f1(preds: list[Any], golds: list[Any], *, average: str = "micro") -> dict[str, Any]:
    """Token-overlap F1 (Squad-style). `average="micro"` aggregates
    per-token over the whole set; `"macro"` averages per-example."""
    if not preds or not golds:
        return {"f1": -1.0, "precision": -1.0, "recall": -1.0,
                "note": "empty inputs"}
    n = min(len(preds), len(golds))

    def _toks(s: Any) -> list[str]:
        return _normalize(s).split()

    if average == "macro":
        per: list[float] = []
        for p, g in zip(preds[:n], golds[:n]):
            tp = sum((Counter(_toks(p)) & Counter(_toks(g))).values())
            pp = max(1, len(_toks(p)))
            gg = max(1, len(_toks(g)))
            prec = tp / pp
            rec = tp / gg
            per.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return {"f1": round(sum(per) / len(per), 6), "n_total": n,
                "metric_name": "f1_macro"}
    # micro
    tp_total = pp_total = gg_total = 0
    for p, g in zip(preds[:n], golds[:n]):
        ctp = Counter(_toks(p))
        ctg = Counter(_toks(g))
        tp_total += sum((ctp & ctg).values())
        pp_total += sum(ctp.values())
        gg_total += sum(ctg.values())
    prec = tp_total / pp_total if pp_total else 0.0
    rec = tp_total / gg_total if gg_total else 0.0
    f1v = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "f1": round(f1v, 6), "precision": round(prec, 6), "recall": round(rec, 6),
        "n_total": n, "metric_name": "f1_micro",
    }


_TOKEN_F1_PUNCT = str.maketrans("", "", string.punctuation)
_TOKEN_F1_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _token_f1_tokens(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    return _TOKEN_F1_ARTICLES.sub(
        " ", text.lower().translate(_TOKEN_F1_PUNCT),
    ).split()


def _token_f1_pair(prediction: Any, reference: Any) -> float:
    pred = _token_f1_tokens(prediction)
    gold = _token_f1_tokens(reference)
    if not pred or not gold:
        return float(not pred and not gold)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _structured_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def token_f1(preds: list[Any], golds: list[Any]) -> dict[str, Any]:
    """SQuAD-normalized token overlap, macro-averaged per scoring record.

    A dict-valued answer (for example a multi-turn conversation) is scored by
    key and averaged once for that record, so long conversations do not carry
    more weight than short ones. Scalar answer columns use the same metric.
    """
    n = min(len(preds), len(golds))
    if n == 0:
        return {"token_f1": -1.0, "n_total": 0, "note": "empty inputs"}
    record_scores: list[float] = []
    unit_scores: list[float] = []
    for raw_pred, raw_gold in zip(preds[:n], golds[:n]):
        pred = _structured_value(raw_pred)
        gold = _structured_value(raw_gold)
        if isinstance(gold, dict):
            pred_map = pred if isinstance(pred, dict) else {"1": pred}
            scores = [
                _token_f1_pair(pred_map.get(str(key), pred_map.get(key, "")), value)
                for key, value in gold.items()
            ]
            unit_scores.extend(scores)
            record_scores.append(sum(scores) / len(scores) if scores else 0.0)
        else:
            score = _token_f1_pair(pred, gold)
            unit_scores.append(score)
            record_scores.append(score)
    headline = sum(record_scores) / len(record_scores)
    return {
        "token_f1": round(headline, 6),
        "unit_token_f1": round(sum(unit_scores) / len(unit_scores), 6),
        "n_total": n,
        "n_units": len(unit_scores),
        "metric_name": "token_f1",
    }


def bleu(preds: list[str], golds: list[str]) -> dict[str, Any]:
    """Corpus BLEU-4 with smoothing (no NLTK dep). Returns 0..1."""
    if not preds or not golds:
        return {"bleu": -1.0, "note": "empty inputs"}
    n = min(len(preds), len(golds))
    # geometric mean of n-gram precisions, with brevity penalty
    p_ns = []
    for ngram in (1, 2, 3, 4):
        match = total = 0
        for p, g in zip(preds[:n], golds[:n]):
            pt = _normalize(p).split()
            gt = _normalize(g).split()
            if len(pt) < ngram:
                continue
            pred_ngrams = Counter(tuple(pt[i:i + ngram]) for i in range(len(pt) - ngram + 1))
            gold_ngrams = Counter(tuple(gt[i:i + ngram]) for i in range(max(0, len(gt) - ngram + 1)))
            match += sum((pred_ngrams & gold_ngrams).values())
            total += sum(pred_ngrams.values())
        # Add-1 smoothing so a single 0 doesn't sink the geometric mean
        p_ns.append((match + 1) / (total + 1))
    # Brevity penalty
    pred_len = sum(len(_normalize(p).split()) for p in preds[:n])
    gold_len = sum(len(_normalize(g).split()) for g in golds[:n])
    if pred_len == 0:
        return {"bleu": 0.0, "n_total": n}
    import math
    bp = 1.0 if pred_len >= gold_len else math.exp(1 - gold_len / pred_len)
    geo = math.exp(sum(math.log(p) for p in p_ns) / len(p_ns))
    return {"bleu": round(bp * geo, 6), "n_total": n}


def rouge_l(preds: list[str], golds: list[str]) -> dict[str, Any]:
    """Corpus-level ROUGE-L: F1 of longest common subsequence between
    each (pred, gold) pair, then macro-averaged."""
    if not preds or not golds:
        return {"rouge_l": -1.0, "note": "empty inputs"}

    def _lcs(a: list[str], b: list[str]) -> int:
        if not a or not b:
            return 0
        prev = [0] * (len(b) + 1)
        for ai in a:
            cur = [0] * (len(b) + 1)
            for j, bj in enumerate(b, 1):
                if ai == bj:
                    cur[j] = prev[j - 1] + 1
                else:
                    cur[j] = max(prev[j], cur[j - 1])
            prev = cur
        return prev[-1]

    n = min(len(preds), len(golds))
    per: list[float] = []
    for p, g in zip(preds[:n], golds[:n]):
        pt = _normalize(p).split()
        gt = _normalize(g).split()
        lcs = _lcs(pt, gt)
        if not pt or not gt or lcs == 0:
            per.append(0.0)
            continue
        prec = lcs / len(pt)
        rec = lcs / len(gt)
        per.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return {"rouge_l": round(sum(per) / len(per), 6), "n_total": n}


# ─────────────── multiple-choice option scoring (log-likelihood) ─────────────


def _mc_option_scores(raw: Any) -> dict[str, float] | None:
    """Parse one prediction cell of per-option log-likelihoods into a
    ``{option_label: length_normalized_score}`` mapping, or ``None`` if the cell
    cannot be interpreted as option scores.

    Inference is expected to emit, for each row, the log-likelihood the model
    assigns to *each* answer option under the frozen prompt. Two shapes are
    accepted so the inference engine can emit whichever is cheaper:

      * ``{"A": {"logprob": -12.3, "num_tokens": 4}, "B": {...}, ...}`` —
        the raw summed completion log-probability plus its token count; this
        scorer length-normalizes as ``logprob / num_tokens`` (the MMLU
        ``acc_norm`` convention), which stops long options from being unfairly
        penalized.
      * ``{"A": -3.07, "B": -2.11, ...}`` — an already-normalized per-option
        score (e.g. per-token log-likelihood or a first-token logit); used as-is.

    A JSON array is accepted too and is keyed by position (``"0"``, ``"1"``…),
    which lines up with an integer / letter gold answer.
    """
    obj: Any = raw
    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
    if isinstance(obj, list):
        obj = {str(i): v for i, v in enumerate(obj)}
    if not isinstance(obj, dict) or not obj:
        return None
    scores: dict[str, float] = {}
    for label, value in obj.items():
        key = str(label)
        if isinstance(value, dict):
            if "logprob" not in value:
                return None
            logprob = value.get("logprob")
            num_tokens = value.get("num_tokens", value.get("length", 1))
            if isinstance(logprob, bool) or isinstance(num_tokens, bool):
                return None
            try:
                logprob_f = float(logprob)
                num_tokens_f = float(num_tokens)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(logprob_f):
                return None
            scores[key] = logprob_f / num_tokens_f if num_tokens_f > 0 else logprob_f
        elif isinstance(value, bool):
            return None
        elif isinstance(value, (int, float)):
            score = float(value)
            if not math.isfinite(score):
                return None
            scores[key] = score
        else:
            return None
    return scores or None


def _mc_gold_key(gold: Any, labels: list[str]) -> str | None:
    """Resolve the gold answer to one of ``labels``.

    Accepts the option label directly (case-insensitive, e.g. ``"B"``) or a
    zero-based integer index into the option order (e.g. ``"1"`` -> second
    option). Returns ``None`` when the gold cannot be mapped, which the scorer
    surfaces as an unparsed row rather than silently guessing.
    """
    text = "" if gold is None else str(gold).strip()
    if not text:
        return None
    for label in labels:
        if label.lower() == text.lower():
            return label
    try:
        index = int(text)
    except ValueError:
        return None
    if 0 <= index < len(labels):
        return labels[index]
    return None


def mc_loglikelihood(preds: list[Any], golds: list[Any]) -> dict[str, Any]:
    """Length-normalized multiple-choice accuracy over per-option log-likelihoods.

    For each row the predicted option is ``argmax`` of the per-option scores
    produced by :func:`_mc_option_scores`; the row is correct when that equals
    the gold option resolved by :func:`_mc_gold_key`. Ties break by option label
    ascending so the result is deterministic regardless of dict ordering.

    A row whose prediction cell or gold answer cannot be interpreted counts as
    incorrect (and is reported under ``n_unparsed``) — the same way generation
    scoring counts an unparseable model answer as wrong — so a broken inference
    emit can never inflate the score. Accuracy is over all ``n`` rows.
    """
    n = min(len(preds), len(golds))
    if n == 0:
        return {
            "accuracy_norm": -1.0, "accuracy": -1.0,
            "n_correct": 0, "n_total": 0, "n_unparsed": 0,
            "metric_name": "mc_loglikelihood",
            "note": "empty predictions or golds",
        }
    correct = 0
    unparsed = 0
    for raw_pred, raw_gold in zip(preds[:n], golds[:n]):
        scores = _mc_option_scores(raw_pred)
        if scores is None:
            unparsed += 1
            continue
        labels = list(scores.keys())
        gold_key = _mc_gold_key(raw_gold, labels)
        if gold_key is None:
            unparsed += 1
            continue
        predicted = min(labels, key=lambda k: (-scores[k], k))
        if predicted == gold_key:
            correct += 1
    value = round(correct / n, 6)
    return {
        # `accuracy_norm` is the conventional lm-eval-harness name; `accuracy`
        # mirrors it so the standard headline resolver records the score.
        "accuracy_norm": value,
        "accuracy": value,
        "n_correct": correct,
        "n_total": n,
        "n_unparsed": unparsed,
        "metric_name": "mc_loglikelihood",
    }


SCORERS: dict[str, Any] = {
    "accuracy": accuracy,
    "exact_match": exact_match,
    "f1": f1,
    "token_f1": token_f1,
    "bleu": bleu,
    "rouge_l": rouge_l,
    "mc_loglikelihood": mc_loglikelihood,
}


# ─────────────────────── alignment check (preflight) ─────────────────────────


@dataclass
class AlignmentIssue:
    code: str
    severity: str  # info | warn | error
    message: str


@dataclass
class AlignmentReport:
    n_preds: int
    n_golds: int
    answer_col: str            # ground-truth field
    prediction_col: str = ""   # prediction field; need not share the same name
    sample_preds: list[Any] = field(default_factory=list)
    sample_golds: list[Any] = field(default_factory=list)
    issues: list[AlignmentIssue] = field(default_factory=list)
    ready: bool = False        # safe to score?


def _load_table(path: Path) -> tuple[list[str], list[dict]]:
    """Return (columns, rows). Supports CSV, JSONL, and JSON objects/arrays."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL record on line {line_no}") from exc
                if not isinstance(obj, dict):
                    raise ValueError("every JSONL record must be an object")
                rows.append(obj)
        cols: list[str] = []
        for r in rows[:200]:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        return cols, rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            rows = [data]
        elif isinstance(data, list) and all(isinstance(row, dict) for row in data):
            rows = list(data)
        else:
            raise ValueError("JSON scoring data must be an object or array of objects")
        cols: list[str] = []
        for r in rows[:200]:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        return cols, rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = list(reader.fieldnames or [])
            return cols, list(reader)
    raise ValueError(f"unsupported file suffix {suffix!r}; want .csv|.jsonl|.json")


def validate_alignment(
    preds_path: Path,
    gold_path: Path,
    *,
    prediction_col_hint: str = "",
    gold_col_hint: str = "",
) -> AlignmentReport:
    """Check that predictions + ground truth line up and expose explicit
    score columns. Prediction and gold columns may have different names.
    The orchestrator should run this
    BEFORE dispatching a finetune ticket — if alignment is broken,
    training is wasted compute."""
    p_cols, p_rows = _load_table(preds_path)
    g_cols, g_rows = _load_table(gold_path)
    issues: list[AlignmentIssue] = []

    if not p_rows or not g_rows:
        issues.append(AlignmentIssue(
            code="empty_rows", severity="error",
            message=(
                f"predictions has {len(p_rows)} rows and gold has "
                f"{len(g_rows)}; both must be non-empty."
            ),
        ))

    if prediction_col_hint and prediction_col_hint not in p_cols:
        issues.append(AlignmentIssue(
            code="missing_prediction_col", severity="error",
            message=f"declared prediction column {prediction_col_hint!r} is absent.",
        ))
    if gold_col_hint and gold_col_hint not in g_cols:
        issues.append(AlignmentIssue(
            code="missing_gold_col", severity="error",
            message=f"declared gold column {gold_col_hint!r} is absent.",
        ))
    if issues:
        return AlignmentReport(
            n_preds=len(p_rows), n_golds=len(g_rows),
            answer_col=gold_col_hint if gold_col_hint in g_cols else "",
            prediction_col=(prediction_col_hint
                            if prediction_col_hint in p_cols else ""),
            issues=issues, ready=False,
        )

    # 1. Row count match.
    if len(p_rows) != len(g_rows):
        issues.append(AlignmentIssue(
            code="rowcount_mismatch", severity="error",
            message=(
                f"predictions has {len(p_rows)} rows, gold has {len(g_rows)}; "
                f"scoring will be misleading or crash."
            ),
        ))

    # 2. Resolve each side independently. Inference normally writes
    # `prediction_idx` while the full scoring set carries `gold`/`answer`, so
    # requiring one shared name rejects the system's own standard handoff.
    p_candidates = [c for c in (
        prediction_col_hint,
        "prediction_idx", "prediction", "answer", "label", "target", "y",
        "output", "pred", "response",
    ) if c]
    g_candidates = [c for c in (
        gold_col_hint,
        "answer", "label", "target", "y", "output", "gold",
        "ground_truth", "response", "completion",
    ) if c]
    prediction_col = next((c for c in p_candidates if c in p_cols), "")
    answer_col = next((c for c in g_candidates if c in g_cols), "")
    if not prediction_col or not answer_col:
        issues.append(AlignmentIssue(
            code="no_shared_answer_col", severity="error",
            message=(
                "could not resolve both score columns. "
                f"predictions cols: {p_cols}; gold cols: {g_cols}; "
                f"prediction candidates: {p_candidates}; "
                f"gold candidates: {g_candidates}. Pass explicit hints."
            ),
        ))
        return AlignmentReport(
            n_preds=len(p_rows), n_golds=len(g_rows),
            answer_col=answer_col, prediction_col=prediction_col,
            issues=issues, ready=False,
        )

    # 3. When stable keys exist on both sides, prove row order rather than
    # merely assuming equal counts imply equal examples.
    for key in ("id", "row_id", "example_id", "key"):
        if key not in p_cols or key not in g_cols:
            continue
        if any(str(p.get(key)) != str(g.get(key)) for p, g in zip(p_rows, g_rows)):
            issues.append(AlignmentIssue(
                code="row_order_mismatch", severity="error",
                message=f"shared key {key!r} does not align row-for-row.",
            ))
        break

    # 4. Missing answers in either side. Empty model outputs are a warning and
    # legitimate evidence; missing gold makes the requested metric undefined.
    def _cell(row: dict, col: str) -> str:
        value = row.get(col)
        return "" if value is None else str(value).strip()

    p_missing = sum(1 for r in p_rows if not _cell(r, prediction_col))
    g_missing = sum(1 for r in g_rows if not _cell(r, answer_col))
    if p_missing:
        issues.append(AlignmentIssue(
            code="pred_missing_answers", severity="warn",
            message=f"{p_missing} prediction(s) have empty {prediction_col!r}.",
        ))
    if g_missing:
        issues.append(AlignmentIssue(
            code="gold_missing_answers", severity="error",
            message=f"{g_missing} gold row(s) have empty {answer_col!r}.",
        ))

    # 5. Identical-prediction smell: model output is the same for every row.
    if p_rows:
        unique_preds = {_cell(r, prediction_col) for r in p_rows}
        if len(unique_preds) == 1 and len(p_rows) >= 10:
            issues.append(AlignmentIssue(
                code="all_predictions_identical", severity="warn",
                message=(
                    f"All {len(p_rows)} predictions are identical "
                    f"({list(unique_preds)[0]!r}); model may not have "
                    f"trained / inference may be broken."
                ),
            ))

    return AlignmentReport(
        n_preds=len(p_rows), n_golds=len(g_rows),
        answer_col=answer_col, prediction_col=prediction_col,
        sample_preds=[_cell(r, prediction_col) for r in p_rows[:5]],
        sample_golds=[_cell(r, answer_col) for r in g_rows[:5]],
        issues=issues,
        ready=not any(i.severity == "error" for i in issues),
    )


# ───────────────────────── default scorer entry ──────────────────────────────


def score_default(
    preds_path: Path,
    gold_path: Path,
    *,
    prediction_col_hint: str = "",
    gold_col_hint: str = "",
    metric: str = "accuracy",
    strict: bool = False,
    f1_average: str = "micro",
) -> dict[str, Any]:
    """Run alignment + the requested metric. Returns a metrics dict
    safe to write as metrics.json; includes a finite top-level `score`, which
    the runner records as the authoritative Evaluation score."""
    aliases = {
        "accuracy": "accuracy",
        "exact_match": "exact_match",
        "f1": "f1",
        "token_f1": "token_f1",
        "bleu": "bleu",
        "rouge_l": "rouge_l",
        "mc_loglikelihood": "mc_loglikelihood",
        # `accuracy_norm` is the familiar lm-eval-harness spelling of the same
        # length-normalized option-scoring metric.
        "accuracy_norm": "mc_loglikelihood",
    }
    requested = str(metric or "").strip().lower()
    canonical = aliases.get(requested)
    if canonical is None:
        return {
            "status": "failed",
            "score": None,
            "metric_used": requested,
            "alignment_issues": [{
                "code": "unknown_metric", "severity": "error",
                "message": f"unsupported fallback metric {metric!r}",
            }],
            "answer_col": gold_col_hint,
            "prediction_col": prediction_col_hint,
        }
    if f1_average not in ("micro", "macro"):
        return {
            "status": "failed", "score": None, "metric_used": canonical,
            "alignment_issues": [{
                "code": "invalid_f1_average", "severity": "error",
                "message": "f1_average must be 'micro' or 'macro'",
            }],
            "answer_col": gold_col_hint,
            "prediction_col": prediction_col_hint,
        }

    report = validate_alignment(
        preds_path, gold_path,
        prediction_col_hint=prediction_col_hint,
        gold_col_hint=gold_col_hint,
    )
    if not report.ready:
        return {
            "status": "failed",
            "score": None,
            "metric_used": canonical,
            "alignment_issues": [
                {"code": i.code, "severity": i.severity, "message": i.message}
                for i in report.issues
            ],
            "answer_col": report.answer_col,
            "prediction_col": report.prediction_col,
        }
    # Extract aligned vectors
    _, p_rows = _load_table(preds_path)
    _, g_rows = _load_table(gold_path)
    preds = [r.get(report.prediction_col) for r in p_rows]
    golds = [r.get(report.answer_col) for r in g_rows]
    if canonical == "accuracy":
        out = accuracy(preds, golds, strict=strict)
    elif canonical == "exact_match":
        out = exact_match(preds, golds)
    elif canonical == "f1":
        out = f1(preds, golds, average=f1_average)
    elif canonical == "token_f1":
        out = token_f1(preds, golds)
    elif canonical == "mc_loglikelihood":
        out = mc_loglikelihood(preds, golds)
    elif canonical == "bleu":
        out = bleu(preds, golds)
    else:
        out = rouge_l(preds, golds)
    # Standardize a single headline number so the runner can record the
    # Evaluation result without parsing per-metric keys.
    headline = (
        out.get("accuracy") if "accuracy" in out
        else out.get("f1") if "f1" in out
        else out.get("token_f1") if "token_f1" in out
        else out.get("bleu") if "bleu" in out
        else out.get("rouge_l") if "rouge_l" in out
        else None
    )
    return {
        "status": "succeeded",
        "score": float(headline) if headline is not None else None,
        "metric_used": canonical,
        "prediction_col": report.prediction_col,
        "answer_col": report.answer_col,
        **out,
        "alignment_issues": [
            {"code": i.code, "severity": i.severity, "message": i.message}
            for i in report.issues
        ],
    }
