"""Score artifact helpers for evaluation and run-level improvement state."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _read_metrics_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def materialize_metrics_artifact(metrics_path: str, work_dir: str) -> tuple[str, dict[str, Any]]:
    """Return `(metrics_path, metrics_dict)` for an EvaluationResult value.

    The contract requires an existing file path. ``work_dir`` remains in the
    signature because callers use one uniform artifact hook for all Agents.
    """
    del work_dir
    raw = str(metrics_path or "").strip()
    if not raw:
        return "", {}

    candidate = Path(raw)
    return (str(candidate), _read_metrics_file(candidate)) if candidate.exists() else (raw, {})


def extract_score(metrics: dict[str, Any], fallback: float | None = None) -> float | None:
    """Extract any finite task score, preferring the durable artifact."""
    for key in ("score",):
        raw = metrics.get(key)
        if isinstance(raw, bool):
            continue
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    if isinstance(fallback, bool):
        return None
    if fallback is None:
        return None
    try:
        value = float(fallback)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
