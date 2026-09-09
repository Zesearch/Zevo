"""Origin/Referer CSRF guard for state-changing requests.

The backend binds to loopback and has no login, but a web page the user visits
in the same browser can still POST to http://127.0.0.1:<port>/api/... — a
classic CSRF. Browsers attach an ``Origin`` header (and a ``Referer``) to such
cross-site state-changing requests; a non-browser client (the ``zevo`` CLI, the
in-cluster agents, ``curl``) sends neither. So we can block the browser CSRF
vector without touching any programmatic caller and without any token:

  - Safe methods (GET/HEAD/OPTIONS/TRACE) always pass — including the CORS
    preflight, which reaches the CORS middleware untouched.
  - A state-changing request with NO Origin/Referer is a non-browser caller
    (CLI, agent, curl) → allowed.
  - A state-changing request whose Origin is same-origin (matches the Host it
    arrived on) or is in the CORS allow-list → allowed (the dashboard).
  - Anything else is a cross-site browser request → 403.

Set ``ZEVO_DISABLE_CSRF_GUARD=1`` to turn this off entirely (escape hatch).
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from zevo.api.config import settings


_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


def _request_origin(request: Request) -> str | None:
    """The browser-declared origin (scheme://host[:port]) or None if absent.

    Prefer the ``Origin`` header; fall back to deriving one from ``Referer``.
    Non-browser clients send neither and get None (→ allowed by the caller).
    """
    origin = (request.headers.get("origin") or "").strip()
    if origin and origin.lower() != "null":
        return origin.rstrip("/")
    referer = (request.headers.get("referer") or "").strip()
    if referer:
        parts = urlsplit(referer)
        if parts.scheme and parts.netloc:
            return f"{parts.scheme}://{parts.netloc}"
    return None


def _origin_allowed(origin: str, request: Request) -> bool:
    allowed = {o.rstrip("/") for o in settings.cors_origins}
    if "*" in allowed:
        return True
    if origin in allowed:
        return True
    # Same-origin: the browser is on the very host the request arrived on
    # (the dashboard talking to its own /api). Compare the Origin's netloc to
    # the request Host, which is what a same-origin fetch carries.
    host = (request.headers.get("host") or "").strip()
    if host and urlsplit(origin).netloc == host:
        return True
    return False


class CsrfOriginMiddleware(BaseHTTPMiddleware):
    """Reject state-changing requests carrying a cross-site browser Origin."""

    async def dispatch(self, request: Request, call_next):
        if (
            request.method in _SAFE_METHODS
            or os.environ.get("ZEVO_DISABLE_CSRF_GUARD", "").strip() in ("1", "true", "True")
        ):
            return await call_next(request)
        origin = _request_origin(request)
        if origin is not None and not _origin_allowed(origin, request):
            return JSONResponse(
                status_code=403,
                content={"detail": "cross-site request blocked (CSRF)"},
            )
        return await call_next(request)
