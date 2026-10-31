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
DEFAULT_POLICY = {"sample_size": SAMPLE_SIZE, "max_truncation_percent": MAX_TRUNCATION_PERCENT,
                  "max_configurations": 4}


def calibration_policy(config: str | Path | None = None) -> dict:
    """Defaults for historical YAML; explicit choices freeze with new configurations."""
    import yaml
    raw = yaml.safe_load(Path(config).read_text()) if config is not None else {}
    raw = raw or {}
    values = raw.get("implementation_config", {}).get("length_calibration_policy", {})
    if not isinstance(values, dict) or set(values) - set(DEFAULT_POLICY):
        raise ValueError("invalid length_calibration_policy keys")
    policy = {**DEFAULT_POLICY, **values}
    for name in ("sample_size", "max_configurations"):
        if type(policy[name]) is not int or policy[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    target = policy["max_truncation_percent"]
    if type(target) not in (int, float) or not 0 <= target < 100:
        raise ValueError("max_truncation_percent must be finite and in [0, 100)")
    return policy



REPETITION_RULE = "word8-v1"
REPETITION_DOMINANCE = 0.5


def repetitive_output(text: str) -> bool:
    """Conservative post-generation signal; never changes or stops generation."""
    words = text.split()
    if len(words) < 200:
        return False
    grams = [tuple(words[i:i + 8]) for i in range(len(words) - 7)]
    tail = words[-512:]
    tail_grams = [tuple(tail[i:i + 8]) for i in range(len(tail) - 7)]
    return (len(set(grams)) / len(grams) <= .3 or
            (len(tail) >= 256 and len(set(tail_grams)) / len(tail_grams) <= .1))


def repetition_evidence(records: list[dict], outputs: dict[int, str]) -> dict:
    limited = {r["row_index"] for r in records if r["finish_reason"] == "length"}
    # A row-level final reply cannot establish repetition in an earlier turn.
    counts = {}
    for record in records:
        counts[record["row_index"]] = counts.get(record["row_index"], 0) + 1
    covered = {i for i in limited if i in outputs and counts[i] == 1}
    repeated = {i for i in covered if repetitive_output(outputs[i])}
    complete = covered == limited
    return {"rule": REPETITION_RULE, "truncated_questions": len(limited),
            "covered_questions": len(covered), "repetitive_questions": len(repeated),
            "repetitive_row_indices": sorted(repeated),
            "repetitive_fraction_of_truncated": len(repeated) / len(limited) if limited else 0,
            "dominated": complete and len(limited) >= 3 and
                len(repeated) >= REPETITION_DOMINANCE * len(limited)}


def prediction_outputs(path: str | Path, records: list[dict], column: str,
                       expected_ids: list[str] | None = None) -> dict[int, str]:
    csv.field_size_limit(1 << 30)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError("repetition analysis requires the complete raw output column")
        limited = {r["row_index"] for r in records if r["finish_reason"] == "length"}
        outputs, ids = {}, []
        for i, row in enumerate(reader):
            ids.append(row.get("id"))
            if i in limited:
                outputs[i] = row[column]
    if {r["row_index"] for r in records} != set(range(len(ids))):
        raise ValueError("predictions and diagnostics row coverage differ")
    if expected_ids is not None and ids != expected_ids:
        raise ValueError("calibration prediction IDs/order differ from the sample")
    return outputs


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_rows(questions: str | Path, sample_size: int = SAMPLE_SIZE) -> tuple[list[str], list[dict], list[int]]:
    path = Path(questions)
    csv.field_size_limit(1 << 30)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
            rows = json.load(handle) if path.suffix.lower() == ".json" else [json.loads(line) for line in handle if line.strip()]
            if not isinstance(rows, list):
                raise ValueError("calibration JSON must contain an array of rows")
            if any(not isinstance(row, dict) or not row for row in rows):
                raise ValueError("calibration JSONL rows must be nonempty objects")
            fields = list(dict.fromkeys(key for row in rows for key in row))
        else:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError("calibration CSV rows must match the header")
    if not fields or not rows:
        raise ValueError("calibration requires nonempty questions-only CSV, JSONL or JSON")
    # Deterministic length-stratified selection, including both tails. Use only
    # supplied public inputs; never load reference answers or a scorer.
    ordered = sorted(range(len(rows)), key=lambda i: (
        sum(len(str(v)) for k, v in rows[i].items() if k != "id"), i,
    ))
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("sample_size must be positive")
    size = min(sample_size, len(rows))
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
            questions_hash: str, policy: dict | None = None) -> str:
    target = (policy or DEFAULT_POLICY)["max_truncation_percent"]
    count = _validate_records(evidence.get("records"), size)
    if count * 100 <= size * target:
        return "within_target"
    outputs = evidence.get("repetition_outputs", {})
    if isinstance(outputs, dict):
        outputs = {int(k): v for k, v in outputs.items() if isinstance(v, str)}
        if repetition_evidence(evidence["records"], outputs)["dominated"]:
            return "accepted_repetition"
    waiver = evidence.get("output_behavior_waiver")
    if not isinstance(waiver, dict):
        raise ValueError(
            f"length calibration failed: {count}/{size} questions truncated; "
            f"maximum is {target}%. Adjust the unpinned budget within the calibration "
            "resource plan, or provide observed evidence and a residual-truncation rationale."
        )
    attempts = waiver.get("previous_attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("previous_attempts must be a list of observed attempts")
    if not attempts:
        for key in ("resource_constraints", "acceptance_rationale"):
            if not isinstance(waiver.get(key), str) or not waiver[key].strip():
                raise ValueError(f"a waiver without prior attempts requires {key}")
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
        if _validate_records(attempt.get("records"), size) * 100 <= size * target:
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


def record_configuration(config: str | Path, questions: str | Path, policy: dict) -> None:
    """Count distinct checked configurations per benchmark, including failed checks.

    Retrying identical bytes consumes no additional configuration. The ledger
    lives with the member YAML so separate helper invocations share the budget.
    """
    path = Path(config).with_name("length_calibration_trials_" + _sha256(questions) + ".json")
    trials = json.loads(path.read_text()) if path.exists() else {"policy": policy, "configurations": []}
    if trials["policy"] != policy:
        raise ValueError("calibration policy must remain fixed across candidates")
    digest = _sha256(config)
    if digest not in trials["configurations"]:
        if len(trials["configurations"]) >= policy["max_configurations"]:
            raise ValueError("calibration configurations exceed the declared resource budget")
        trials["configurations"].append(digest)
        path.write_text(json.dumps(trials, indent=2) + "\n")


def check(config: str | Path, questions: str | Path, diagnostics: str | Path,
          predictions: str | Path | None = None) -> Path:
    """Gate measured generation; write evidence after meeting the target or a documented waiver."""
    policy = calibration_policy(config)
    _, sample, indices = sample_rows(questions, policy["sample_size"])
    report = Path(config).with_name(REPORT_NAME)
    report.unlink(missing_ok=True)  # A failed recheck must invalidate prior evidence.
    evidence = json.loads(Path(diagnostics).read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("calibration diagnostics must be an object")
    records = evidence.get("records")
    _validate_records(records, len(indices))
    # Only the helper reads prediction bytes; do not accept caller-authored statistics.
    evidence.pop("repetition_outputs", None)
    if predictions is not None:
        import yaml
        cfg = yaml.safe_load(Path(config).read_text())
        column = cfg.get("implementation_config", {}).get("raw_output_column", "complete_prediction")
        outputs = prediction_outputs(predictions, records, column, [str(r["id"]) for r in sample])
        limited = {r["row_index"] for r in records if r["finish_reason"] == "length"}
        evidence["repetition_outputs"] = {str(i): outputs[i] for i in limited}
    record_configuration(config, questions, policy)
    config_hash, questions_hash = _sha256(config), _sha256(questions)
    acceptance = _accept(evidence, len(indices), indices, config_hash, questions_hash, policy)
    report.write_text(json.dumps({
        "schema_version": 1,
        "policy": policy,
        "config_sha256": _sha256(config),
        "questions_sha256": _sha256(questions),
        "sample_row_indices": indices,
        "records": records,
        "acceptance": acceptance,
        "truncated_questions": _validate_records(records, len(indices)),
        "sample_size": len(indices),
        "output_behavior_waiver": evidence.get("output_behavior_waiver"),
        "repetition_outputs": evidence.get("repetition_outputs", {}),
        "repetition": repetition_evidence(records, {
            int(k): v for k, v in evidence.get("repetition_outputs", {}).items()
        }),
    }, indent=2) + "\n", encoding="utf-8")
    return report


def validate(config: str | Path, questions: str | Path) -> None:
    """Recompute acceptance; never trust an agent-authored passed/rate flag."""
    report = json.loads(Path(config).with_name(REPORT_NAME).read_text(encoding="utf-8"))
    policy = calibration_policy(config)
    _, _, indices = sample_rows(questions, policy["sample_size"])
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("unsupported length calibration report version")
    if report.get("policy", DEFAULT_POLICY) != policy:
        raise ValueError("length calibration policy differs from the frozen configuration")
    if report.get("config_sha256") != _sha256(config):
        raise ValueError("length calibration does not match the final configuration")
    if report.get("questions_sha256") != _sha256(questions):
        raise ValueError("length calibration does not match the assigned questions")
    if report.get("sample_row_indices") != indices:
        raise ValueError("length calibration must use the policy-selected sample")
    _accept(report, len(indices), indices, _sha256(config), _sha256(questions), policy)


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
    prepare.add_argument("--config", help="Provisional config carrying the calibration policy")
    gate = sub.add_parser("check")
    gate.add_argument("--config", required=True)
    gate.add_argument("--questions", required=True)
    gate.add_argument("--diagnostics", required=True)
    gate.add_argument("--predictions")
    run = sub.add_parser("run")
    run.add_argument("--suite", required=True)
    run.add_argument("workload", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        fields, rows, _ = sample_rows(args.questions, calibration_policy(args.config)["sample_size"])
        if Path(args.output).resolve() == Path(args.questions).resolve():
            raise ValueError("calibration output must not overwrite assigned questions")
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    elif args.command == "check":
        check(args.config, args.questions, args.diagnostics, args.predictions)
    else:
        command = args.workload[1:] if args.workload[:1] == ["--"] else args.workload
        return run_measured(args.suite, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
