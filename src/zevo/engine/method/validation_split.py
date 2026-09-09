"""Give every run a validation set while keeping a final Test population.

A run that scores itself on the user's test set and feeds that number to the
orchestrator is tuning on the set it is judged by: after a dozen iterations of
"that base model scored worse, try another", the reported score is a fitted
number, not a measured one. So the loop optimizes a VALIDATION set and the test
set is scored behind its back (zevo.engine.run.runner, the held-out lane).

When a task does not ship Validation, Zevo deterministically takes 20% of its
Test rows before the Run starts.  At least 200 Validation rows are required; a
smaller Test contract is rejected and the user must upload Validation.  The
remaining 80% stays in the private held-out lane.

  * **The scoring binding stays aligned.** The Test answer fields are split by
    the same row indices. The sample submission is a schema/example contract,
    not a second copy of the dataset: its columns and example formatting are
    reused for both Validation and remaining Test, regardless of its row count.
  * **It must be the same split every time.** The rows are ordered by a hash of
    their own content, not shuffled by a RNG, so the same input file always
    yields the same validation set — on a re-run, on another machine, in
    another container. Two runs of one task stay comparable, which is the whole
    point of a fixed evaluation set.

Training is untouched.  Rows promoted to Validation are removed from the final
Test population, so the two scoring lanes remain disjoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# How much of Test becomes Validation when none is supplied.
FRACTION = 0.20
MIN_VALIDATION_ROWS = 200


class SplitError(RuntimeError):
    """The supplied Test population cannot produce a valid Validation set."""


@dataclass
class CarveResult:
    """Where the run's evaluation paths ended up, and why.

    Only FULL sets, with their ground-truth columns intact. The engine creates
    the optimization questions-only copy after Data selection; private held-out
    preparation remains a separate harness operation.
    """

    validation_set: str
    # The answer fields of whatever the validation set turned out to be. A
    # carve inherits them from Test, because that is the file it splits.
    validation_answer_fields: list[str] = field(default_factory=list)
    # Set when the carve consumed rows from a file the run also uses: the
    # trimmed copy the run must point at instead of the original. Empty when
    # nothing had to be rewritten.
    dataset: str = ""
    test_set: str = ""
    source: str = ""          # 'user' | 'test'
    # Legacy setup signal retained in Run metadata; binding is engine-owned.
    needs_metric_binding: bool = False
    n_validation: int = 0
    n_remaining: int = 0
    notes: list[str] = field(default_factory=list)


# ─────────────────────────── reading + writing ───────────────────────────────


def _fmt(path: Path) -> str:
    """'json' | 'jsonl' | 'csv', from the name. What is written back matches
    what was read: a validation set carved out of a JSON file and handed back as
    a CSV has had its nested records flattened into quoted strings, and whatever
    reads it next has to know that happened."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    return "csv"


def _read_raw(path: Path) -> tuple[list[str], list[dict]]:
    """(fields, records) with values left as they are.

    A JSON record can hold an object — a multi-turn conversation keeps every
    turn under one key — and flattening that to a string is lossy the moment the
    file is written back out. The flattened view is `_read_rows`, for the callers
    that genuinely need strings.
    """
    fmt = _fmt(path)
    if fmt == "csv":
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return list(reader.fieldnames or []), [dict(r) for r in reader]

    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SplitError(f"{path}: expected a JSON array of records")
        records = data
    else:
        records = []
        # JSONL records are separated by physical LF bytes. str.splitlines()
        # additionally splits at U+2028/U+2029, which are legal inside a JSON
        # string and must remain part of the same record.
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise SplitError(f"{path}: expected one JSON object per line")
                records.append(obj)

    cols: list[str] = []
    for obj in records:
        if not isinstance(obj, dict):
            raise SplitError(f"{path}: expected records, got {type(obj).__name__}")
        for k in obj:
            if k not in cols:
                cols.append(k)
    return cols, records


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV, JSON or JSONL table as (columns, rows), values as strings.

    The column set of a JSON file is the union over its records, so a sparse
    file still round-trips.
    """
    cols, rows = _read_raw(path)
    if _fmt(path) == "csv":
        return cols, rows
    return cols, [{k: _flatten(v) for k, v in r.items()} for r in rows]


def _flatten(v: object) -> str:
    """JSONL cells can hold lists/dicts; a CSV cell cannot."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: _flatten(r.get(c, "")) for c in columns})


def _write_table(path: Path, columns: list[str], rows: list[dict], fmt: str) -> None:
    """Write in the format the rows came from, so nothing is flattened on the
    way out and Data can map the exact original values into the Task binding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    elif fmt == "jsonl":
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")
    else:
        _write_csv(path, columns, rows)


def _order_key(row: dict[str, str], index: int) -> tuple[str, int]:
    """A row's position in the shuffle, decided by the row itself.

    Hashing the content rather than seeding a RNG keeps the split stable across
    Python versions and platforms, and keeps it stable when the file is
    re-exported with its rows in a different order. The index is a SEPARATE
    tiebreaker for genuinely identical rows — folding it into the hash itself
    made every key depend on file position, which is exactly the reorder
    instability the content hash exists to prevent.
    """
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return (hashlib.sha256(payload.encode()).hexdigest(), index)


def _n_validation(total: int) -> int:
    """Exactly 20% of Test, subject to the minimum useful population."""
    if total < 2:
        raise SplitError(f"cannot carve a validation set from {total} Test row(s)")
    # Floor makes the minimum unambiguous: a 999-row Test has only 199 whole
    # rows in its 20% allocation and therefore cannot qualify by rounding up.
    selected = min(int(total * FRACTION), total - 1)
    if selected < MIN_VALIDATION_ROWS:
        raise SplitError(
            f"20% of the Test set is only {selected} row(s); derived Validation "
            f"requires at least {MIN_VALIDATION_ROWS}. Upload a separate "
            "Validation set instead (or provide a Test set with at least 1,000 rows)."
        )
    return selected


# ───────────────────────────── source choice ─────────────────────────────────


def pick_source(
    *, dataset: str, test_set: str,
) -> tuple[str, str]:
    """Return the Test file used to derive Validation."""
    if test_set and Path(test_set).exists():
        return "test", test_set
    raise SplitError(
        "Test data must be a readable local file before Zevo can derive Validation"
    )


# ─────────────────────────────── the carve ───────────────────────────────────


def carve(
    *,
    dataset: str,
    test_set: str,
    test_answer_fields: list[str],
    out_dir: str,
    remaining_out_dir: str | None = None,
    declared_columns: list[str] | None = None,
) -> CarveResult:
    """Split Validation out of Test without changing Training.

    Writes into `out_dir` and returns the paths the run should use from here
    on. The source file is rewritten (minus the validation rows) alongside, so
    the caller repoints the run at the trimmed copy: the originals are shared
    task inputs and are never touched.
    """
    kind, source = pick_source(dataset=dataset, test_set=test_set)
    src = Path(source)
    # Raw, not flattened: a JSON record whose value is an object survives the
    # round trip only if it is never turned into a string on the way in.
    fmt = _fmt(src)
    columns, rows = _read_raw(src)
    if not rows:
        raise SplitError(f"{src} has no rows to split")

    answer_cols = list(test_answer_fields or declared_columns or [])
    missing = [c for c in answer_cols if c not in columns]
    if missing:
        raise SplitError(
            "Test answer fields are absent from the Test data: "
            + ", ".join(map(repr, missing))
        )

    ordered = sorted(range(len(rows)), key=lambda i: _order_key(rows[i], i))
    n_val = _n_validation(len(rows))
    chosen = sorted(ordered[:n_val])
    val_idx = set(chosen)
    # Keep both populations in original order so prediction rows can be checked
    # against their respective scoring populations.
    val_rows = [rows[i] for i in range(len(rows)) if i in val_idx]
    rest_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]

    out = Path(out_dir)
    suffix = {"json": ".json", "jsonl": ".jsonl"}.get(fmt, ".csv")
    val_path = out / f"validation{suffix}"
    _write_table(val_path, columns, val_rows, fmt)

    result = CarveResult(
        validation_set=str(val_path),
        validation_answer_fields=answer_cols,
        source=kind,
        needs_metric_binding=False,
        n_validation=len(val_rows),
        n_remaining=len(rest_rows),
    )

    remaining = Path(remaining_out_dir or out_dir)
    trimmed = remaining / f"test_minus_validation{suffix}"
    _write_table(trimmed, columns, rest_rows, fmt)
    result.dataset = dataset
    result.test_set = str(trimmed)
    result.notes.append(
        f"carved {len(val_rows)} validation rows out of {src.name}; "
        f"final Test uses the remaining {len(rest_rows)} rows"
    )
    return result


def copy_sample_submission(
    *, source: str, validation_out: str, remaining_out: str,
) -> tuple[str, str]:
    """Copy one submission schema/example contract into both scoring lanes.

    A sample submission describes the columns, their order, and representative
    cell formatting. It is not required to contain one placeholder for every
    scoring row. Actual predictions are checked against the scoring population
    for row count and identity later, at the Inference/Evaluation boundaries.
    """
    src = Path(source)
    with src.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not columns or not rows:
        raise SplitError("Test sample submission must be a non-empty CSV")
    if len(columns) != len(set(columns)):
        raise SplitError("Test sample submission contains duplicate columns")
    val_path, rest_path = Path(validation_out), Path(remaining_out)
    _write_csv(val_path, columns, rows)
    _write_csv(rest_path, columns, rows)
    return str(val_path), str(rest_path)


def _assert_fields_exist(path: Path, fields: list[str]) -> None:
    """The declared answer fields have to be IN the validation set.

    A named Validation set must carry the explicitly declared Validation
    fields. Test fields never substitute for them because the two lanes may
    have unrelated schemas and metrics.

    Checked here because here is where it is cheap. The validation set is
    settled at run creation, before any agent starts, so this costs one file
    read and turns an hour-long failure into a refused request.
    """
    if not fields:
        return
    try:
        present, _ = _read_raw(path)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as e:
        raise SplitError(
            f"validation set {str(path)!r} could not be read to check its "
            f"answer fields: {e}"
        ) from e
    missing = [f for f in fields if f not in present]
    if missing:
        raise SplitError(
            f"validation set {str(path)!r} has no {', '.join(map(repr, missing))}. "
            f"It carries {', '.join(map(repr, present)) or 'nothing'}. "
            "Name the fields this file actually uses (a column of a CSV, a key "
            "of a JSON record). Test fields are not a Validation fallback."
        )


def resolve(
    *,
    validation_set: str,
    validation_answer_fields: list[str],
    dataset: str,
    test_set: str,
    test_answer_fields: list[str],
    out_dir: str,
) -> CarveResult:
    """Use explicit Validation or derive it from Test."""
    columns = list(validation_answer_fields or [])

    if validation_set:
        if not columns:
            raise SplitError(
                "a named validation set requires its own validation_answer_fields; "
                "Test answer fields are not a fallback"
            )
        if not Path(validation_set).exists():
            # Falling through to the carve here would be the worst outcome: the
            # run would proceed, tune on a set the user did not choose, and
            # never say so. A hub id is the likely mistake — the validation set
            # is settled before any agent runs, so nothing has downloaded
            # anything yet.
            raise SplitError(
                f"validation set {validation_set!r} is not a file. It has to be "
                "readable when the run is created — the split is settled before "
                "any agent starts, so a HuggingFace id cannot be used here. "
                "Fetch it into a dataset first, or leave the field empty to "
                "derive one from the Test set."
            )
        _assert_fields_exist(Path(validation_set), columns)
        return CarveResult(
            validation_set=validation_set,
            validation_answer_fields=columns,
            source="user",
        )

    return carve(
        dataset=dataset,
        test_set=test_set,
        test_answer_fields=list(test_answer_fields or []),
        declared_columns=list(validation_answer_fields or []),
        out_dir=out_dir,
    )
