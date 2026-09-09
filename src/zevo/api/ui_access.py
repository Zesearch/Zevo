"""Lightweight trust boundary between the dashboard proxy and Agent callers."""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from fastapi import Request, WebSocket


UI_ACCESS_HEADER = "x-zevo-ui-access"


@lru_cache(maxsize=1)
def _ui_access_token() -> str:
    token_file = Path(
        os.environ.get("ZEVO_UI_TOKEN_FILE", "/run/zevo-ui-auth/token")
    )
    try:
        return token_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _trusted(headers) -> bool:
    expected = _ui_access_token()
    supplied = (headers.get(UI_ACCESS_HEADER) or "").strip()
    return bool(expected and supplied and secrets.compare_digest(expected, supplied))


def is_trusted_ui_request(request: Request) -> bool:
    return _trusted(request.headers)


def is_trusted_ui_websocket(websocket: WebSocket) -> bool:
    return _trusted(websocket.headers)
