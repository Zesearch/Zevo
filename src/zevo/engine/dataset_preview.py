"""Bounded UI samples from verified remote training data, never training inputs."""
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from zevo.contracts.infrastructure import InfrastructureDeviceInfo
from zevo.engine.remote_training_data import (
    _write_json_atomic,
    load_remote_dataset_receipt,
)
from zevo.engine.run.remote_jobs import _ssh


# Executed on the assigned data host with only the Python standard library.
# Bound records, individual reads, and output before anything crosses SSH.
_REMOTE_READ = '''import json, sys
def clip(value, depth=0):
    if depth > 6:
        return "…"
    if isinstance(value, str):
        return value[:4000] + ("…" if len(value) > 4000 else "")
    if isinstance(value, list):
        return [clip(v, depth+1) for v in value[:30]]
    if isinstance(value, dict):
        return {str(k)[:200]: clip(v, depth+1) for k,v in list(value.items())[:30]}
    return value
rows = []
try:
    with open(sys.argv[1], encoding="utf-8") as source:
        scanned = 0
        while len(rows) < 100 and scanned < 8*1024*1024:
            line = source.readline(256*1024+1)
            if not line:
                break
            scanned += len(line)
            if len(line) > 256*1024:
                raise ValueError("record too large")
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("not a record")
            row = clip(row)
            rendered = json.dumps(row, ensure_ascii=False)
            if len(rendered) > 16000:
                row = {"preview": rendered[:16000] + "…"}
            rows.append(row)
    print(json.dumps({"rows": rows}, ensure_ascii=False))
except FileNotFoundError:
    print(json.dumps({"error": "missing"}))
except PermissionError:
    print(json.dumps({"error": "permission"}))
except (ValueError, UnicodeError, OSError, RecursionError):
    print(json.dumps({"error": "unreadable"}))
'''


async def remote_training_preview(
    *, info: InfrastructureDeviceInfo | None, dataset_path: str,
    receipt_path: str,
) -> tuple[list[dict[str, Any]], str]:
    """Read the persisted sample first; backfill old artifacts on first view.

    The engine receipt binds both cache identity and the exact post-filter file.
    Failure is a preview failure only and never invalidates a Data result.
    """
    try:
        receipt = load_remote_dataset_receipt(receipt_path)
        if receipt.dataset_path != dataset_path:
            return [], "Preview unavailable: the verified dataset reference does not match."
    except (OSError, ValueError):
        return [], "Preview unavailable: the verified dataset receipt is unavailable."
    identity = hashlib.sha256(
        f"v1\n{receipt.data_signature}\n{receipt.dataset_sha256}\n{dataset_path}".encode()
    ).hexdigest()
    cache = Path(receipt_path).parent / ".dataset-previews" / f"{identity}.json"
    try:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        rows = cached["rows"]
        if cached.get("identity") == identity and isinstance(rows, list) and len(rows) <= 100 \
                and all(isinstance(row, dict) for row in rows):
            return rows, ""
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if info is None:
        return [], "Preview unavailable: no connection to the dataset host is available."
    command = shlex.join(["python3", "-c", _REMOTE_READ, dataset_path])
    result = await _ssh(info, command, timeout_seconds=15, stdout_limit=2_000_000)
    if not result.get("ok"):
        return [], "Preview unavailable: the dataset host could not be reached or the read timed out."
    if result.get("stdout_truncated"):
        return [], "Preview unavailable: the sample exceeded the preview size limit."
    try:
        body = json.loads(result["stdout"])
        reason = body.get("error")
        if reason:
            messages = {
                "missing": "the dataset file is no longer present on the remote host",
                "permission": "the remote dataset file is not readable with the assigned connection",
                "unreadable": "the dataset contains an unreadable or oversized record",
            }
            return [], f"Preview unavailable: {messages.get(reason, 'the remote read failed')}."
        rows = body["rows"]
        if not isinstance(rows, list) or len(rows) > 100 or not all(isinstance(row, dict) for row in rows):
            raise ValueError("invalid sample")
    except (ValueError, KeyError, TypeError, AttributeError):
        return [], "Preview unavailable: the remote host returned an invalid sample."
    try:
        _write_json_atomic(cache, {"identity": identity, "rows": rows})
    except OSError:
        # The current response can still show the sample if the cache is unwritable.
        pass
    return rows, ""
