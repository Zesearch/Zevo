"""Shared helpers for reporting WorkProduct / artifact paths + existence.

One place so every endpoint (runs, tickets, registry) agrees on:
  - host_path():     map an in-container path to where it lives on the HOST
                     (host and container agree: ./data is /app/data).
  - artifact_stat(): (exists, size_bytes) that handles FILES *and* DIRECTORIES
                     — LoRA adapters are saved as a directory (`adapter/`), so a
                     plain is_file() check wrongly reported them 'missing'.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def host_paths_in_text(s: str) -> str:
    """Rewrite in-container path SUBSTRINGS (/app/data/runs/..., /app/...) to HOST
    paths inside free text — e.g. an agent message 'Wrote train.py to
    /app/data/runs/<run>/train-007/train.py'. Display-only; the stored text keeps
    the container path. Prefix-gated so only real container paths are touched."""
    if not s:
        return s
    root = (os.environ.get("ZEVO_WORK_DIR", "/app/data/runs") or "/app/data/runs").rstrip("/")
    pat = re.compile(r'(?:' + re.escape(root) + r'|/app)/[^\s`"\';,)\]]+')
    return pat.sub(lambda m: host_path(m.group(0)), s)


def host_path(container_path: str) -> str:
    """Map an in-container path to the HOST path the user sees on their machine.

    The repo root is `/app`, and every mount now keeps its name across the
    boundary — `./data/runs` is `/app/data/runs`, `./data/files` is
    `/app/data/files` — so this is one rule: drop the `/app/` prefix.

    `ZEVO_WORK_DIR` is still honoured for a deployment that puts run artifacts
    somewhere else entirely; there the name genuinely can differ.
    Anything outside `/app` is returned unchanged.
    """
    if not container_path:
        return container_path
    root = (os.environ.get("ZEVO_WORK_DIR", "/app/data/runs") or "/app/data/runs").rstrip("/")
    if root != "/app/data/runs" and container_path.startswith(root + "/"):
        return "data/runs/" + container_path[len(root) + 1:]
    if container_path.startswith("/app/"):
        return container_path[len("/app/"):]
    return container_path


def host_paths_in(obj):
    """Recursively rewrite in-container paths to HOST paths inside a JSON-ish
    value (dict/list/str), for DISPLAY only. Used on ticket payloads returned to
    the UI so a user_request never shows /app/data/runs/... or /app/data/files/...
    — the STORED payload keeps container paths (agents read those to open/scp
    files); only the API response is rewritten. Prefix-gated, so non-path strings
    are untouched."""
    if isinstance(obj, str):
        if obj.startswith("/app/") or obj.startswith(
            (os.environ.get("ZEVO_WORK_DIR", "/app/data/runs") or "/app/data/runs").rstrip("/") + "/"
        ):
            return host_path(obj)
        return obj
    if isinstance(obj, dict):
        return {k: host_paths_in(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [host_paths_in(v) for v in obj]
    return obj


def artifact_stat(p: Path | None) -> tuple[bool, int]:
    """(exists, size_bytes) for an artifact path, handling files AND dirs. For a
    directory (e.g. a trained model), size is the recursive sum of its files."""
    if not p or not p.exists():
        return False, 0
    try:
        if p.is_dir():
            return True, sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return True, p.stat().st_size
    except OSError:
        return True, 0


def host_path_abs(container_path: str) -> str:
    """The HOST path, absolute, when the deployment told us where the repo is.

    `host_path` can only produce a repo-RELATIVE string: the container knows
    `/app`, not what the user cloned it into. That is enough to show where a
    file sits, and not enough to build a command that copies it somewhere else.
    `ZEVO_HOST_REPO` is set from `${PWD}` by compose, so it is the host's own
    view of this directory. Falls back to the relative path when it is unset —
    a shorter answer, never a wrong one.
    """
    rel = host_path(container_path)
    root = (os.environ.get("ZEVO_HOST_REPO", "") or "").rstrip("/")
    if not root or rel.startswith("/"):
        return rel
    return f"{root}/{rel}"
