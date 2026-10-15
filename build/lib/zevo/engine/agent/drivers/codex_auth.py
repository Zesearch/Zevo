"""Interpret the Codex CLI credential cache without exposing its secrets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


CodexAuthMode = Literal["chatgpt", "api_key", "invalid", "missing"]


def codex_auth_mode(path: str | Path) -> CodexAuthMode:
    """Return the billing identity represented by a Codex ``auth.json``."""
    auth_path = Path(path)
    try:
        is_file = auth_path.is_file()
    except OSError:
        is_file = False
    if not is_file:
        return "missing"
    try:
        raw = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid"
    mode = raw.get("auth_mode") if isinstance(raw, dict) else None
    if mode == "chatgpt" and isinstance(raw.get("tokens"), dict):
        return "chatgpt"
    if mode == "apikey" and bool(raw.get("OPENAI_API_KEY")):
        return "api_key"
    return "invalid"
