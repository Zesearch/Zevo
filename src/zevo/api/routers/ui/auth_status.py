"""GET /api/auth-status — surface which provider credentials the
backend container can see, so the UI's per-driver auth badge stops
being a heuristic.

We only report PRESENCE of a credential, never its value and never a
redacted preview -- even a first-6/last-4 slice leaks entropy and confirms
a key's shape, so no part of a secret leaves the backend.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter()


class CredStatus(BaseModel):
    name: str          # env var or file name
    present: bool      # True if found
    preview: str = ""  # always "" — retained for response-shape stability
    source: str = ""   # "env" | "file" | ""


class DriverAuthStatus(BaseModel):
    driver: str            # claude_cli | codex_cli | bedrock | openrouter
    mode: Literal["subscription", "api_key", "either", "none"]
    has_subscription: bool = False
    has_api_key: bool = False
    ready: bool = False    # at least one valid auth path is set
    effective: str = "none"  # credential the driver will actually select
    creds: list[CredStatus] = Field(default_factory=list)


class AuthStatusResponse(BaseModel):
    drivers: list[DriverAuthStatus]


def _env_cred(env_name: str) -> CredStatus:
    v = os.environ.get(env_name) or ""
    # Presence only -- never a preview of the value (see module docstring).
    return CredStatus(name=env_name, present=bool(v), preview="", source="env" if v else "")


def _file_cred(path: str, display: str) -> CredStatus:
    p = Path(path)
    try:
        exists = p.is_file() and p.stat().st_size > 0
    except OSError:
        # GitHub-hosted runners and hardened deployments may make /root itself
        # unreadable. An inaccessible credential is simply unavailable to
        # Zevo; the status endpoint must still report env-based fallbacks.
        exists = False
    return CredStatus(name=display, present=exists, preview="", source="file" if exists else "")


@router.get("/auth-status", response_model=AuthStatusResponse)
async def get_auth_status() -> AuthStatusResponse:
    drivers: list[DriverAuthStatus] = []

    # ─── claude_cli ─────────────────────────────────────────────────────────
    cl_oauth = _env_cred("CLAUDE_CODE_OAUTH_TOKEN")
    cl_api_key = _env_cred("ANTHROPIC_API_KEY")
    cl_auth_token = _env_cred("ANTHROPIC_AUTH_TOKEN")
    cl_creds_file = _file_cred("/root/.claude/.credentials.json", "~/.claude/.credentials.json")
    has_subscription = cl_oauth.present or cl_creds_file.present
    has_api_key = cl_api_key.present or cl_auth_token.present
    drivers.append(DriverAuthStatus(
        driver="claude_cli",
        mode="either",
        has_subscription=has_subscription,
        has_api_key=has_api_key,
        ready=has_subscription or has_api_key,
        effective=(
            "oauth_token" if cl_oauth.present
            else "oauth_session" if cl_creds_file.present
            else "auth_token" if cl_auth_token.present
            else "api_key" if cl_api_key.present
            else "none"
        ),
        creds=[cl_oauth, cl_api_key, cl_auth_token, cl_creds_file],
    ))

    # ─── bedrock ────────────────────────────────────────────────────────────
    bearer = _env_cred("AWS_BEARER_TOKEN_BEDROCK")
    aws_region = _env_cred("AWS_REGION")
    drivers.append(DriverAuthStatus(
        driver="bedrock",
        mode="api_key",
        has_api_key=bearer.present,
        ready=bearer.present,
        effective="bearer_token" if bearer.present else "none",
        creds=[bearer, aws_region],
    ))

    # ─── codex_cli ──────────────────────────────────────────────────────────
    # Personal ChatGPT-plan auth is the complete Codex-managed login cache.
    # A browser-session JWT copied out of that file is not a Codex access token.
    from zevo.engine.agent.drivers.codex_auth import codex_auth_mode

    cx_key = _env_cred("OPENAI_API_KEY")
    cx_path = "/root/.codex/auth.json"
    cx_mode = codex_auth_mode(cx_path)
    cx_session = _file_cred(cx_path, "~/.codex/auth.json")
    cx_session.present = cx_mode in ("chatgpt", "api_key")
    if cx_mode == "invalid":
        cx_session.source = "invalid_file"
    drivers.append(DriverAuthStatus(
        driver="codex_cli",
        mode="either",
        has_subscription=cx_mode == "chatgpt",
        has_api_key=cx_mode == "api_key" or cx_key.present,
        # An invalid cache blocks the CLI; do not silently change billing path.
        ready=cx_session.present or (cx_mode == "missing" and cx_key.present),
        effective=(
            "chatgpt_plan" if cx_mode == "chatgpt"
            else "api_key" if cx_mode == "api_key" or (cx_mode == "missing" and cx_key.present)
            else "invalid" if cx_mode == "invalid"
            else "none"
        ),
        creds=[cx_key, cx_session],
    ))

    # ─── openrouter ─────────────────────────────────────────────────────────
    or_key = _env_cred("OPENROUTER_API_KEY")
    drivers.append(DriverAuthStatus(
        driver="openrouter",
        mode="api_key",
        has_api_key=or_key.present,
        ready=or_key.present,
        effective="api_key" if or_key.present else "none",
        creds=[or_key],
    ))

    return AuthStatusResponse(drivers=drivers)
