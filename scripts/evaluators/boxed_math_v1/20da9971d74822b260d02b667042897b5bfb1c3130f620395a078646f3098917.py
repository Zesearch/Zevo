# Evaluation protocol revision: boxed-math-v1
#!/usr/bin/env python3
# =============================================================================
#  math_evaluator.py  --  Zevo custom Test/Validation evaluator (Xiang)
# =============================================================================
#
#  ZEVO'S CALLING CONTRACT  (fixed, three positional paths)
#  --------------------------------------------------------
#      python3 evaluator.py PREDICTIONS_CSV SCORING_SET METRICS_JSON
#
#    PREDICTIONS_CSV  the model's predictions, columns fixed by the sample
#                     submission -> here:  id,prediction
#    SCORING_SET      the answer-bearing rows (.jsonl / .json / .csv) with
#                     fields: id, question, answer
#    METRICS_JSON     the path this script MUST write one UTF-8 JSON object
#                     to, containing a finite numeric top-level "score",
#                     e.g. {"score": 0.83, "correct": 83, "total": 100}
#
#  A nonzero exit, a timeout, a missing file, invalid JSON, or a non-finite
#  score all make Zevo mark Evaluation as failed. So failures here are loud on
#  purpose: a silent 0.0 is worse than a crash, because a crash tells you the
#  contract broke while a 0.0 looks like a weak model.
#
#  WHY THIS IS A CUSTOM EVALUATOR AND NOT builtin `exact_match`
#  -----------------------------------------------------------
#  Zevo's built-in metrics are string-level. On math benchmarks that fails
#  three ways at once:
#    * AIME 2024 gold answers are zero-padded three-character strings
#      ("025"). A model emitting 25 would be scored wrong.
#    * MATH-500 gold answers are LaTeX. \frac{1}{2}, (1)/(2) and 0.5 are the
#      same answer and compare unequal as strings.
#    * College Math gold answers may be multi-valued -- $(45,2),(-10,-9)$ --
#      where solution order carries no meaning.
#
#  WHY ROUTING BY `id` PREFIX
#  --------------------------
#  A Zevo Task has ONE evaluation script, shared by every attached test set.
#  Whether Zevo calls this script once per test set or once over a merged
#  scoring set is not documented -- routing per row on the `id` prefix
#  (`gsm8k-00001`, `aime2024-00003`, `collegemath-01234`, `val-00007`) is
#  correct under either behavior.
#
#  KEEP IN SYNC
#  ------------
#  normalize_answer() / extract_boxed() are duplicated from
#  prepare_math_data.py. If you change one, change both, or gold answers
#  written to disk stop matching the canonicalization applied here.
#
#  LOCAL SELF-TEST BEFORE UPLOADING  (do this -- it is free, a failed Zevo
#  Evaluation is not)
#  ------------------------------------------------------------------------
#      python3 - <<'EOF'
#      import json, csv
#      rows = [json.loads(l) for l in
#              open('zevo_math_files/smoke/gsm8k_smoke.jsonl')]
#      with open('fake_preds.csv','w',newline='') as f:
#          w = csv.writer(f); w.writerow(['id','prediction'])
#          for i,r in enumerate(rows):
#              w.writerow([r['id'],
#                          f"reasoning...\\boxed{{{r['answer']}}}" if i%2==0
#                          else "\\boxed{999999}"])
#      EOF
#      python3 math_evaluator.py fake_preds.csv \
#              zevo_math_files/smoke/gsm8k_smoke.jsonl /tmp/m.json
#      cat /tmp/m.json      # score should be ~0.5
#
#  Also run --selftest to exercise the comparators with no files at all:
#      python3 math_evaluator.py --selftest
# =============================================================================

import csv
import json
import math
import os
import re
import sys
from fractions import Fraction

NUMERIC_TOL = 1e-6

ID_FIELDS = ("id", "ID", "Id", "index")
ANSWER_FIELDS = ("answer", "answers", "label", "target", "solution")
PREDICTION_FIELDS = ("prediction", "pred", "output", "answer")


# =============================================================================
#  SECTION 1 -- normalization  (mirror of prepare_math_data.py)
# =============================================================================

_BOXED = re.compile(r"\\boxed\s*{")


def extract_boxed(text):
    """Last boxed command, with balanced braces; incomplete/empty final box fails.

    One token pass, including nested LaTeX and escaped literal braces. Never
    fall back to an earlier intermediate result when the final box is broken.
    """
    if not text:
        return ''
    stack = []
    last_start = None
    answer = ''
    for match in re.finditer(r'\\boxed\s*\{|\\[{}\\]|[{}]', str(text)):
        token = match.group()
        if token.startswith('\\boxed'):
            last_start = match.end()
            answer = ''
            stack.append(last_start)
        elif token == '{':
            stack.append(None)
        elif token == '}' and stack:
            start = stack.pop()
            if start is not None and start == last_start:
                answer = str(text)[start:match.start()].strip()
    return answer


def normalize_answer(ans):
    """Presentation-noise removal only. See prepare_math_data.py for details."""
    if ans is None:
        return ""
    s = str(ans).strip()

    boxed = extract_boxed(s)
    if boxed:
        s = boxed

    s = s.replace("$", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\s*{([^}]*)}", r"\1", s)
    s = re.sub(r"\\mbox\s*{([^}]*)}", r"\1", s)
    s = re.sub(r"\\d?frac\s*{([^{}]+)}\s*{([^{}]+)}", r"(\1)/(\2)", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"\\+(percent|%)", "", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".")


def as_number(s):
    """Exact rational parsing avoids float rounding of large integer answers."""
    from fractions import Fraction
    if not s:
        return None
    s = re.sub(r'(\\times|\\cdot|\*|×)10\^\{?(-?\d+)\}?', r'e\2', s)
    if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', s):
        s = s.replace(',', '')
    s = re.sub(r'^10\^\{?(-?\d+)\}?$', r'1e\1', s)
    ratio = re.fullmatch(r'\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?', s)
    try:
        if ratio:
            return Fraction(ratio[1]) / Fraction(ratio[2])
        return Fraction(s)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def symbolically_equal(a, b):
    """
    sympy-backed LaTeX equivalence, used only as a last resort.

    Optional dependency on purpose: if sympy or its LaTeX parser is absent in
    the Zevo container this returns False rather than crashing, and scoring
    degrades to normalized string equality instead of failing the whole
    Evaluation stage.
    """
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
    except Exception:
        return False
    try:
        return bool(simplify(parse_latex(a) - parse_latex(b)) == 0)
    except Exception:
        return False


# =============================================================================
#  SECTION 2 -- per-dataset comparators
# =============================================================================

def _close(pn, gn):
    # An integer reference must match exactly, including a near-integer guess.
    if gn.denominator == 1:
        return pn == gn
    tolerance = globals().get('REL_TOL', globals().get('NUMERIC_TOL', 1e-6))
    scale = max(abs(pn), abs(gn)) if 'REL_TOL' in globals() else max(1, abs(gn))
    return abs(pn - gn) <= tolerance * scale


def check_integer(pred, gold):
    """
    GSM8K, AIME 2024, AIME 2025.

    All three have integer gold answers, but stored differently: GSM8K after
    the '####' marker, AIME 2024 zero-padded as "025", AIME 2025 as int64.
    Numeric comparison makes all three forms agree. Falls back to the general
    comparator so a stray non-integer gold row cannot silently score zero.
    """
    p, g = normalize_answer(pred), normalize_answer(gold)
    pn, gn = as_number(p), as_number(g)
    if pn is not None and gn is not None:
        if math.isfinite(pn) and math.isfinite(gn):
            return _close(pn, gn)
        return False
    return check_general(pred, gold)


def _split_multivalue(s):
    """
    Split a multi-solution answer into an order-independent set of
    numeric-canonical elements.

    Depth-aware: a comma or ';' at brace/paren depth 0 separates solutions; a
    comma inside (), [], {} is a coordinate and is kept. This covers every
    shape College Math actually uses (audited over all 2818 gold answers):
        (45,2),(-10,-9)   -> {(45,2), (-10,-9)}   ')('-delimited     (10 rows)
        x=1;x=-3          -> {x=1, x=-3}          ';'-delimited      (43 rows)
        35,36,37 / 14,16  -> {35,36,37} / {14,16} bare comma lists  (417 rows)
    A parenthesized tuple (45,2) stays a single element because its comma sits
    at depth > 0 -- so (45,2) and (2,45) still do NOT compare equal, while a
    bare list like 35,36,37 is compared order-independently.

    Each element is canonicalized numerically when possible (("n", value)),
    else kept as its normalized string (("s", text)), so 1/2 and 0.5 match
    inside a set exactly as they do for scalar answers.
    """
    s = normalize_answer(s)
    if not s:
        return frozenset()

    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            if depth > 0:
                depth -= 1
            cur.append(ch)
        elif (ch == "," or ch == ";") and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))

    out = set()
    for p in (x.strip() for x in parts):
        if not p:
            continue
        n = as_number(p)
        out.add(("n", round(n, 6)) if (n is not None and math.isfinite(n))
                else ("s", p))
    return frozenset(out)


def check_multivalue(pred, gold):
    """
    College Math.

    Gold answers here are $-wrapped and may list several solutions, e.g.
    $(45,2),(-10,-9)$, where the order is not meaningful. Compare as sets
    first, then fall through to the general comparator for the single-valued
    majority of rows.
    """
    gp, pp = _split_multivalue(gold), _split_multivalue(pred)
    if len(gp) > 1 or len(pp) > 1:
        if gp and gp == pp:
            return True
    return check_general(pred, gold)


def check_general(pred, gold):
    """
    MATH-500, the validation set, and anything unrecognized.

    Three tiers, cheapest first: normalized string equality, numeric equality
    with tolerance, then sympy LaTeX equivalence.
    """
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == "" or g == "":
        return False
    if p == g:
        return True
    pn, gn = as_number(p), as_number(g)
    if pn is not None and gn is not None:
        if math.isfinite(pn) and math.isfinite(gn):
            return _close(pn, gn)
        return False
    return symbolically_equal(p, g)


def route(row_id):
    """
    Pick a comparator from the `id` prefix written by prepare_math_data.py.

    Unknown prefixes fall through to check_general, so adding a test set
    without touching this file degrades to reasonable behavior rather than
    breaking.
    """
    rid = str(row_id)
    if rid.startswith("gsm8k") or rid.startswith("aime"):
        return check_integer
    if rid.startswith("collegemath"):
        return check_multivalue
    return check_general          # math500, val, anything else


def is_correct(row_id, pred, gold):
    answer = extract_boxed(pred)
    return bool(answer) and route(row_id)(answer, gold)


# =============================================================================
#  SECTION 3 -- file IO
# =============================================================================

def pick_field(row, candidates, what, path):
    for c in candidates:
        if c in row:
            return c
    raise SystemExit(f"{what}: none of {candidates} found in {path}; "
                     f"columns were {list(row)}")


def read_predictions(path):
    """Read id -> prediction from the CSV Zevo hands us."""
    if not os.path.exists(path):
        raise SystemExit(f"predictions file not found: {path}")
    # utf-8-sig: strips a BOM if one appears, which otherwise corrupts the
    # first column name and makes the id lookup fail on every row.
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"predictions file is empty: {path}")
    id_field = pick_field(rows[0], ID_FIELDS, "predictions id", path)
    pr_field = pick_field(rows[0], PREDICTION_FIELDS, "predictions value", path)
    out = {}
    for r in rows:
        rid = str(r[id_field]).strip()
        if rid:
            out[rid] = r[pr_field]
    return out


def read_scoring_set(path):
    """Read the answer-bearing rows. Accepts jsonl, json, csv, tsv."""
    if not os.path.exists(path):
        raise SystemExit(f"scoring set not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    rows = []
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("data", [])
    elif ext in (".csv", ".tsv"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(
                f, delimiter="\t" if ext == ".tsv" else ","))
    else:
        raise SystemExit(f"unsupported scoring set extension: {ext}")
    if not rows:
        raise SystemExit(f"scoring set is empty: {path}")
    return rows


# =============================================================================
#  SECTION 4 -- self-test
# =============================================================================

CASES = [
    # (row_id, prediction, gold, expected)
    ("gsm8k-00001", "reasoning... \\boxed{72}", "72", True),
    ("gsm8k-00002", "\\boxed{1,234}", "1234", True),
    ("gsm8k-00003", "\\boxed{71}", "72", False),
    ("aime2024-00001", "\\boxed{25}", "025", True),      # zero-padded gold
    ("aime2024-00002", "\\boxed{025}", "25", True),
    ("aime2025-00001", "\\boxed{70}", 70, True),         # int64 gold
    ("math500-00001", "\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", True),
    ("math500-00002", "\\boxed{0.5}", "\\frac{1}{2}", True),
    ("math500-00003", "\\boxed{\\sqrt{5}}", "\\sqrt{5}", True),
    ("math500-00004", "\\boxed{\\frac{1}{3}}", "\\frac{1}{2}", False),
    ("collegemath-00001", "\\boxed{(45,2),(-10,-9)}",
     "$(45,2),(-10,-9)$", True),
    ("collegemath-00002", "\\boxed{(-10,-9),(45,2)}",
     "$(45,2),(-10,-9)$", True),                         # order-independent
    ("collegemath-00003", "\\boxed{(45,2)}", "$(45,2),(-10,-9)$", False),
    ("collegemath-00004", "\\boxed{37,35,36}", "$35,36,37$", True),   # bare-comma list, order-independent
    ("collegemath-00005", "\\boxed{35,36}", "$35,36,37$", False),     # a value is missing
    ("collegemath-00006", "\\boxed{0.5,3}", "$1/2,3$", True),         # numeric-equivalent element in a set
    ("val-00001", "\\boxed{12}", "12", True),
    ("gsm8k-00004", "", "72", False),                    # empty prediction
]


def selftest():
    failures = 0
    for rid, pred, gold, expected in CASES:
        got = is_correct(rid, pred, gold)
        ok = "ok  " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"  {ok} {rid:22s} pred={pred!r:34s} gold={gold!r:22s} "
              f"-> {got} (expected {expected})")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


# =============================================================================
#  SECTION 5 -- main
# =============================================================================

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())

    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: evaluator.py PREDICTIONS_CSV SCORING_SET METRICS_JSON\n"
            "       evaluator.py --selftest")

    pred_path, gold_path, metrics_path = sys.argv[1:4]

    preds = read_predictions(pred_path)
    gold_rows = read_scoring_set(gold_path)

    id_field = pick_field(gold_rows[0], ID_FIELDS, "scoring set id", gold_path)
    ans_field = pick_field(gold_rows[0], ANSWER_FIELDS,
                           "scoring set answer", gold_path)

    total = correct = 0
    missing = []
    # Per-dataset breakdown. Zevo only reads `score`, but these extra keys land
    # in the metrics JSON and make a bad run diagnosable without re-running:
    # if one prefix is at 0.0 while the others are fine, the problem is that
    # dataset's answer format, not the model.
    by_set = {}

    for r in gold_rows:
        rid = str(r[id_field]).strip()
        prefix = rid.rsplit("-", 1)[0] if "-" in rid else rid
        bucket = by_set.setdefault(prefix, {"correct": 0, "total": 0})

        total += 1
        bucket["total"] += 1

        if rid not in preds:
            missing.append(rid)
            continue
        if is_correct(rid, preds[rid], r[ans_field]):
            correct += 1
            bucket["correct"] += 1

    score = (correct / total) if total else 0.0
    if not math.isfinite(score):
        raise SystemExit("computed score is not finite")

    result = {
        "score": round(score, 6),
        "correct": correct,
        "total": total,
        "missing_predictions": len(missing),
        "extra_predictions": max(0, len(preds) - (total - len(missing))),
        "by_dataset": {
            k: {"correct": v["correct"],
                "total": v["total"],
                "score": round(v["correct"] / v["total"], 6) if v["total"] else 0.0}
            for k, v in sorted(by_set.items())
        },
    }
    if missing:
        result["missing_examples"] = missing[:5]

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    # Echo to stdout too -- it shows up in the Zevo ticket's execution events,
    # which is the fastest place to see what happened.
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
