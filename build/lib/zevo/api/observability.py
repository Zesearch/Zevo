"""Structured logging, request correlation, and optional error tracking.

- `configure_logging()` routes the root logger through a JSON formatter to
  stdout (CloudWatch/most log shippers parse JSON line-per-record).
- `RequestContextMiddleware` assigns/propagates an `X-Request-ID` per request,
  exposes it to every log record via a contextvar, and emits one structured
  access line per request (method, path, status, duration).
- `init_error_tracking()` initializes Sentry iff `SENTRY_DSN` is set and the
  SDK is installed — a no-op otherwise, so dev/local needs nothing.

No required new dependency: the JSON formatter is stdlib; Sentry is optional.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Correlation id for the in-flight request; empty string outside a request.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

_REQUEST_ID_HEADER = "X-Request-ID"


class _RequestIdFilter(logging.Filter):
    """Attach the current request id to every record so the formatter can emit
    it (records created outside a request get an empty id)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")
        return True


class _JsonFormatter(logging.Formatter):
    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "request_id", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "") or "",
        }
        # Merge structured extras passed via logger.info(..., extra={...}).
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                payload.setdefault(k, v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Idempotently route the root logger through the JSON formatter to stdout."""
    level = os.environ.get("ZEVO_LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(level)
    # Replace existing handlers so uvicorn's default plain handlers don't
    # double-log; keep exactly one JSON stream handler.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)
    # Align uvicorn's loggers with the root config (JSON, no duplicate handlers).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Per-request correlation id + a structured access log line."""

    _log = logging.getLogger("zevo.access")

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[_REQUEST_ID_HEADER] = rid
            return response
        finally:
            dur_ms = round((time.perf_counter() - start) * 1000, 1)
            # Don't spam the access log with health/readiness probes.
            if request.url.path not in ("/health", "/ready"):
                self._log.info(
                    "request",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "duration_ms": dur_ms,
                        "client": request.client.host if request.client else "",
                    },
                )
            request_id_var.reset(token)


def init_error_tracking() -> None:
    """Initialize Sentry iff configured + installed. Safe no-op otherwise."""
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return
    try:
        import sentry_sdk
    except Exception:
        logging.getLogger("zevo").warning("SENTRY_DSN set but sentry_sdk not installed")
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("ZEVO_ENV", "development"),
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.0") or 0.0),
    )
    logging.getLogger("zevo").info("error tracking initialized")
