"""Portable, answer-free baseline length calibration gate (also copied to GPUs)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

SAMPLE_SIZE = 64
# Compare integer counts, so exactly 5% passes without floating point drift.
MAX_TRUNCATION_PERCENT = 5
REPORT_NAME = "length_calibration.json"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_rows(questions: str | Path) -> tuple[list[str], list[dict], list[int]]:
    csv.field_size_limit(1 << 30)
    with open(questions, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not rows:
        raise ValueError("calibration requires nonempty prepared questions-only CSV")
    # Deterministic length-stratified selection, including both tails. Use only
    # supplied public inputs; never load reference answers or a scorer.
    ordered = sorted(range(len(rows)), key=lambda i: (
        sum(len(str(v)) for k, v in rows[i].items() if k != "id"), i,
    ))
    size = min(SAMPLE_SIZE, len(rows))
    indices = sorted(ordered[i * (len(rows) - 1) // (size - 1)]
                     for i in range(size)) if size > 1 else [0]
    return fields, [rows[i] for i in indices], indices


def _validate_records(records: list[dict], size: int) -> int:
    if not isinstance(records, list) or not records:
        raise ValueError("calibration has no generation records")
    rows = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("calibration records must be objects")
        row = record.get("row_index")
        if type(row) is not int or not 0 <= row < size:
            raise ValueError("calibration row_index is outside the selected sample")
        if record.get("request_index") != index:
            raise ValueError("calibration request indices must be dense and ordered")
        if record.get("finish_reason") not in {"stop", "length"}:
            raise ValueError("calibration requires backend stop/length finish reasons")
        tokens = record.get("generated_tokens")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("calibration requires nonnegative generated-token counts")
        rows.add(row)
    if rows != set(range(size)):
        raise ValueError("calibration must cover every selected sample row")
    # Count affected questions, including any truncated turn in multi-turn work.
    truncated = {r["row_index"] for r in records if r["finish_reason"] == "length"}
    return len(truncated)


def _accept(evidence: dict, size: int, indices: list[int], config_hash: str,
            questions_hash: str) -> str:
    count = _validate_records(evidence.get("records"), size)
    if count * 100 <= size * MAX_TRUNCATION_PERCENT:
        return "within_target"
    waiver = evidence.get("output_behavior_waiver")
    if not isinstance(waiver, dict):
        raise ValueError(
            f"length calibration failed: {count}/{size} questions truncated; "
            "maximum is 5%. Adjust the unpinned budget and repeat the same sample. "
            "After 2–3 adjustments, documented model output behavior may justify a waiver."
        )
    attempts = waiver.get("previous_attempts")
    if not isinstance(attempts, list) or not 2 <= len(attempts) <= 3:
        raise ValueError("output behavior waiver requires 2–3 previous attempts")
    hashes = {config_hash}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("previous calibration attempts must be objects")
        digest = attempt.get("config_sha256", "")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest) or digest in hashes):
            raise ValueError("waiver requires distinct adjusted configuration hashes")
        hashes.add(digest)
        if (attempt.get("questions_sha256") != questions_hash
                or attempt.get("sample_row_indices") != indices):
            raise ValueError("waiver attempts must use the same questions and sample")
        if _validate_records(attempt.get("records"), size) * 100 <= size * MAX_TRUNCATION_PERCENT:
            raise ValueError("waiver requires persistent truncation above target")
        if not isinstance(attempt.get("adjustment"), str) or not attempt["adjustment"].strip():
            raise ValueError("waiver must describe each budget/stop-control adjustment")
    for key in ("reason", "stop_controls_checked"):
        if not isinstance(waiver.get(key), str) or not waiver[key].strip():
            raise ValueError(f"output behavior waiver requires {key}")
    examples = waiver.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("waiver requires observed output examples")
    truncated_requests = {r["request_index"] for r in evidence["records"]
                          if r["finish_reason"] == "length"}
    for example in examples:
        if (not isinstance(example, dict)
                or example.get("request_index") not in truncated_requests
                or any(not isinstance(example.get(k), str) or not example[k].strip()
                       for k in ("output_excerpt", "observation"))):
            raise ValueError("waiver examples must describe actual truncated requests")
    return "accepted_output_behavior"


def check(config: str | Path, questions: str | Path, diagnostics: str | Path) -> Path:
    """Gate measured generation; write evidence after meeting the target or a documented waiver."""
    _, _, indices = sample_rows(questions)
    report = Path(config).with_name(REPORT_NAME)
    report.unlink(missing_ok=True)  # A failed recheck must invalidate prior evidence.
    evidence = json.loads(Path(diagnostics).read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("calibration diagnostics must be an object")
    records = evidence.get("records")
    config_hash, questions_hash = _sha256(config), _sha256(questions)
    acceptance = _accept(evidence, len(indices), indices, config_hash, questions_hash)
    report.write_text(json.dumps({
        "schema_version": 1,
        "config_sha256": _sha256(config),
        "questions_sha256": _sha256(questions),
        "sample_row_indices": indices,
        "records": records,
        "acceptance": acceptance,
        "truncated_questions": _validate_records(records, len(indices)),
        "sample_size": len(indices),
        "output_behavior_waiver": evidence.get("output_behavior_waiver"),
    }, indent=2) + "\n", encoding="utf-8")
    return report


def validate(config: str | Path, questions: str | Path) -> None:
    """Recompute acceptance; never trust an agent-authored passed/rate flag."""
    report = json.loads(Path(config).with_name(REPORT_NAME).read_text(encoding="utf-8"))
    _, _, indices = sample_rows(questions)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("unsupported length calibration report version")
    if report.get("config_sha256") != _sha256(config):
        raise ValueError("length calibration does not match the final configuration")
    if report.get("questions_sha256") != _sha256(questions):
        raise ValueError("length calibration does not match the assigned questions")
    if report.get("sample_row_indices") != indices:
        raise ValueError("length calibration must use the system-selected 64-row sample")
    _accept(report, len(indices), indices, _sha256(config), _sha256(questions))


def run_measured(suite: str | Path, command: list[str]) -> int:
    """Check all members before launching any measured generation process."""
    import yaml

    members = json.loads(Path(suite).read_text(encoding="utf-8")).get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("measured inference requires a nonempty suite manifest")
    for member in members:
        config_path, questions = member["config"], member["questions"]
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        scoring_mode = config.get("measurement", {}).get("inference_config", {}).get(
            "scoring_mode", "generate",
        )
        if scoring_mode != "option_loglikelihood":
            validate(config_path, questions)
    if not command:
        raise ValueError("measured inference command is empty")
    return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--questions", required=True)
    prepare.add_argument("--output", required=True)
    gate = sub.add_parser("check")
    gate.add_argument("--config", required=True)
    gate.add_argument("--questions", required=True)
    gate.add_argument("--diagnostics", required=True)
    run = sub.add_parser("run")
    run.add_argument("--suite", required=True)
    run.add_argument("workload", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        fields, rows, _ = sample_rows(args.questions)
        if Path(args.output).resolve() == Path(args.questions).resolve():
            raise ValueError("calibration output must not overwrite assigned questions")
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    elif args.command == "check":
        check(args.config, args.questions, args.diagnostics)
    else:
        command = args.workload[1:] if args.workload[:1] == ["--"] else args.workload
        return run_measured(args.suite, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
