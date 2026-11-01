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
#                     submission -> here: id,complete_prediction,extracted_answer
#                     complete_prediction keeps the model's FULL response
#                     (reasoning + \boxed{}); this script re-extracts \boxed{}
#                     from it, so the reasoning stays inspectable for debugging.
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
import signal
import sys
from fractions import Fraction

REL_TOL = 1e-4   # SimpleRL-Zoo grader: isclose(reference, prediction, rel_tol=1e-4)


def _close(pn, gn):
    # An integer reference must match exactly, including a near-integer guess.
    if gn.denominator == 1:
        return pn == gn
    tolerance = globals().get('REL_TOL', globals().get('NUMERIC_TOL', 1e-6))
    scale = max(abs(pn), abs(gn)) if 'REL_TOL' in globals() else max(1, abs(gn))
    return abs(pn - gn) <= tolerance * scale

ID_FIELDS = ("id", "ID", "Id", "index")
ANSWER_FIELDS = ("answer", "answers", "label", "target", "solution")
# The prediction column carries the model's COMPLETE generation (reasoning +
# \boxed{...}); this evaluator does its own \boxed extraction, so reasoning is
# preserved in the saved prediction for debugging. `complete_prediction` is
# preferred; `extracted_answer` and `prediction` are accepted as fallbacks.
PREDICTION_FIELDS = ("complete_prediction", "prediction", "extracted_answer",
                     "pred", "output", "answer")


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

    # commas are never removed here: "1,000" (one number) and "153,370,371,407"
    # (four answers) look identical, so grouping is resolved in as_number()
    s = s.replace("\\$", "").replace("$", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\s*{([^}]*)}", r"\1", s)
    s = re.sub(r"\\mbox\s*{([^}]*)}", r"\1", s)
    s = re.sub(r"\\d?frac\s*{([^{}]+)}\s*{([^{}]+)}", r"(\1)/(\2)", s)
    s = re.sub(r"\\+(percent|%)", "", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".")


_GROUPED = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")
_SCI = re.compile(r"(\\times|\\cdot|\*|×)10\^\{?(-?\d+)\}?")


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


_KNOWN_FUNCS = ("sqrt", "pi", "oo", "sin", "cos", "tan", "log", "ln", "exp")


def _to_sympy_expr(s):
    """
    Convert a normalized answer into a sympy expression WITHOUT the LaTeX
    parser. parse_latex needs antlr4, which neither our local env nor the Zevo
    container has, so symbolically_equal() silently returns False and
    `x + 11` vs `11+x` scores wrong. Returns None for anything that isn't a
    plain algebraic expression (tuples, intervals, equations, words).
    """
    from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                            parse_expr, standard_transformations)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi").replace("\\infty", "oo")
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s)
    prev = None
    while prev != s:  # \frac{a}{b} with nested content, once \sqrt{} is gone
        prev = s
        s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = s.replace("{", "(").replace("}", ")")
    if not s or re.search(r"\\[A-Za-z]", s) or "=" in s or "," in s:
        return None
    # implicit multiplication would read "yes" as y*e*s, so bail on any word
    if re.search(r"[A-Za-z]{2,}", re.sub("|".join(_KNOWN_FUNCS), "", s)):
        return None
    return parse_expr(s, transformations=standard_transformations
                      + (implicit_multiplication_application, convert_xor), evaluate=True)


class _SympyTimeout(Exception):
    pass


def _raise_timeout(*_):
    raise _SympyTimeout()


def sympy_equal(a, b, timeout=2):
    """Algebraic equivalence via sympy's plain-expression parser, time-boxed."""
    try:
        from sympy import simplify
    except Exception:
        return False
    alarm = hasattr(signal, "SIGALRM")
    if alarm:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(timeout)
    try:
        ea, eb = _to_sympy_expr(a), _to_sympy_expr(b)
        return ea is not None and eb is not None and bool(simplify(ea - eb) == 0)
    except Exception:
        return False
    finally:
        if alarm:
            signal.alarm(0)


# =============================================================================
#  SECTION 2 -- per-dataset comparators
# =============================================================================

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
        return _close(pn, gn)
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
    with tolerance, then sympy equivalence (LaTeX parser, else plain parser).
    """
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == "" or g == "":
        return False
    if p == g:
        return True
    pn, gn = as_number(p), as_number(g)
    if pn is not None and gn is not None:
        return _close(pn, gn)
    return symbolically_equal(p, g) or sympy_equal(p, g)


_LEADING_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def check_minerva(pred, gold):
    """Accept a numeric answer with recognized units, never an expression prefix."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    gn = as_number(g)
    p = _SCI.sub(r'e\2', p)
    # Only known unit symbols, optionally with SI prefixes and integer powers.
    unit = r'(?:[fpnumcdhkMGT]?(?:mol|Pa|Hz|eV|Wb|Bq|Gy|Sv|kat|rad|sr|[msgAKNJWCVFHTSL])|kg|atm|bar|cal|kcal|L|liter|liters)'
    power = r'(?:\^(?:\{-?\d+\}|-?\d+)|[23])?'
    units = unit + power + r'(?:(?:[*/·]|\\cdot)?' + unit + power + r')*'
    match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)(' + units + r')', p)
    if gn is not None and match:
        return _close(as_number(match[1]), gn)
    return check_general(pred, gold)


def route(row_id):
    """
    Pick a comparator from the `id` prefix written by the data builders.

    Unknown prefixes fall through to check_general, so adding a test set
    without touching this file degrades to reasonable behavior rather than
    breaking.
    """
    rid = str(row_id)
    if rid.startswith(("gsm8k", "aime", "amc")):
        return check_integer
    if rid.startswith(("collegemath", "olympiad")):
        return check_multivalue   # tuples / several answers, order-free
    if rid.startswith("minerva"):
        return check_minerva
    return check_general          # math500, validation sets, anything else


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
    ("math500-00005", "\\boxed{x + 11}", "11+x", True),           # commutative: needs the plain sympy parser
    ("math500-00006", "\\boxed{\\frac{\\sqrt{2}}{2}}", "\\frac{1}{\\sqrt{2}}", True),
    ("math500-00007", "\\boxed{x+12}", "11+x", False),
    ("omnimath-00001", "\\boxed{yes}", "sey", False),             # words must not be read as y*e*s
    ("minerva-00001", "\\boxed{4.5 \\times 10^{33}}", "4.5e33", True),   # scientific notation
    ("minerva-00002", "\\boxed{41.8 \\text{ kJ}}", "41.8", True),        # trailing unit
    ("minerva-00003", "\\boxed{0.00601}", "0.006", False),               # outside rel 1e-4
    ("minerva-00004", "\\boxed{0.0060003}", "0.006", True),              # inside rel 1e-4
    ("amc23-00001", "\\boxed{27}", "27", True),
    ("olympiad-00001", "\\boxed{(2,7,13),(1,8,19),(4,5,7)}", "(1,8,19),(2,7,13),(4,5,7)", True),
    ("gsm8k-00005", "\\boxed{1000050}", "1000000", False),               # integers stay exact
    ("math500-00008", "\\boxed{11,111,111,100}", "11111111100", True),      # grouped thousands
    ("math500-00009", "\\boxed{(2,12)\\cup(12,102)}", "(2,12)\\cup(12,102)", True),  # ",102" is not a thousands group
    ("math500-00010", "\\boxed{\\$32,348}", "32348", True),               # escaped dollar
    ("olympiad-00002", "\\boxed{(3,5,101,107),(1,7,103,105)}", "(1,7,103,105),(3,5,101,107)", True),
    ("olympiad-00003", "\\boxed{(1,7103,105)}", "(1,7,103,105)", False),    # must not merge 7,103
    ("valomni-00001", "\\boxed{153,370,371,407}", "153,370,371,407", True),  # list that looks like thousands
    ("gsm8k-00006", "\\boxed{1,000}", "1000", True),
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
