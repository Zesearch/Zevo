"""Versioned, content-addressed evidence required for cross-Run comparisons.

Paths and metric labels do not identify a benchmark. Missing historical
provenance is unknown, never permission to claim a numeric winner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def snapshot_test_contract(suite: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Freeze actual population, template, scorer, query and aggregation bytes.

    Conservative file hashing may reject equivalent encodings; it must never
    accept different populations merely because their paths/metric names match.
    The caller stores this once at settlement and verifies it at measurement.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        builtin = digest_json({
            name: file_digest(root / name) for name in (
                "engine/method/eval_metrics.py", "engine/method/model_judge.py",
                "engine/agent/drivers/evaluation_runner.py", "code_benchmarks.py",
            )
        })
        members = []
        for member in suite:
            kind = str(member.get("metric_type") or "builtin")
            scorer = builtin
            if kind == "custom":
                scorer = file_digest(str(member.get("evaluation_script") or ""))
                if scorer != member.get("evaluator_sha256"):
                    return None
            name = str(member.get("name") or "")
            query = str(member.get("inference_query") or "")
            answers = list(member.get("answer_fields") or [])
            if not name or not query or not answers or not member.get("metric"):
                return None
            members.append({
                "name": name,
                "population_sha256": file_digest(str(member["test_set"])),
                "submission_sha256": file_digest(str(member["sample_submission"])),
                "metric_type": kind, "metric": member["metric"],
                "metric_direction": member.get("metric_direction") or "max",
                "answer_fields": answers, "inference_query": query,
                "scorer_sha256": scorer,
                "code_execution_adapter": member.get("code_execution_adapter") or "",
            })
        if not members or len({item["name"] for item in members}) != len(members):
            return None
        return {"version": 1, "aggregation": "unweighted_mean",
                "members": sorted(members, key=lambda item: item["name"])}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def inference_identity(configuration: Any) -> dict[str, Any] | None:
    """Keep the engine-validated realized measurement, not suggested settings."""
    if not isinstance(configuration, dict) or any(
        key not in configuration for key in ("measurement", "prompt", "generation_backend")
    ):
        return None
    # Model weights are the thing being compared, not part of the benchmark.
    # Tokenizer/template, decoding and framework settings remain authoritative.
    return {key: value for key, value in configuration.items()
            if key not in {"base_model", "prompt_example", "suggestion_decisions"}}


def complete_identity(contract: Any, components: dict[str, dict]) -> dict | None:
    if not isinstance(contract, dict) or contract.get("version") != 1:
        return None
    names = {str(item["name"]) for item in contract["members"]}
    if set(components) != names or any(
        not item.get("inference_identity") or item.get("test_contract") != contract
        for item in components.values()
    ):
        return None
    payload = {"test_contract": contract, "inference": {
        name: components[name]["inference_identity"] for name in sorted(names)
    }}
    return {"version": 1, "sha256": digest_json(payload), "contract": payload}


def compatible_identities(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    try:
        return (a.get("version") == b.get("version") == 1
                and a["sha256"] == b["sha256"]
                and a["sha256"] == digest_json(a["contract"])
                and b["sha256"] == digest_json(b["contract"]))
    except (KeyError, TypeError, ValueError):
        return False
