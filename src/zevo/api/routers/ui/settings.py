"""GET/POST /api/settings — the Settings page: provider credentials written
into the host .env file, so the UI can replace hand-editing.

Named for the page a person opens, which is called Settings and shows more than
keys (driver readiness, which cloud backend is selected). The values it writes
are still secrets, and the DTOs below say so.

NB the module is imported as `settings_router` in main.py: `zevo.api.config`
already exports a `settings` object, and one file importing two different
`settings` is a trap worth one alias.

Design constraints:
- Allowlist of writable keys (typo'd / arbitrary env names are rejected
  with 400 — we never want the UI to be a vector for setting
  PYTHONPATH, LD_PRELOAD, etc.).
- GET returns presence flags + full values for non-secret config only
  (same shape as the /auth-status row); real secrets carry no preview at
  all, never a redaction and never the raw value.
- A successful POST writes to /app/.env (which is the host's repo-root
  .env via the compose bind mount) but does NOT update the running
  container's os.environ; the user must restart compose for new values
  to take effect. The response includes a `next_step` hint.
- File ops are atomic: write to a sibling tmp file + os.replace (with an
  in-place fallback for the one case where replace cannot work: a single
  bind-mounted .env, whose inode is pinned by the mount).
"""
from __future__ import annotations

import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from zevo.api.database import get_db
from zevo.contracts._base import StrictBody
from zevo.db import EnvironmentSshVerification, SshHost
from zevo.providers import resolve_ssh_key


router = APIRouter()


# Keys the UI is allowed to set / clear. Anything else POST-ed is rejected.
ALLOWED_KEYS = {
    # claude_cli driver
    "CLAUDE_CODE_OAUTH_TOKEN",  # Claude CLI OAuth (Max/Pro)
    "ANTHROPIC_API_KEY",        # Anthropic Console, usage billed
    "ANTHROPIC_AUTH_TOKEN",     # explicit Claude bearer-token override
    # codex_cli driver -- ChatGPT-plan auth comes from the mounted auth.json;
    # this key is the optional usage-billed fallback.
    "OPENAI_API_KEY",
    # bedrock driver
    "AWS_BEARER_TOKEN_BEDROCK", # Bedrock API key (bearer token) -- the only Bedrock auth we use
    "AWS_REGION",
    # openrouter driver -- one key fronts every model it offers
    "OPENROUTER_API_KEY",
    # GPU — cloud mode has two backends (pick with ZEVO_CLOUD_BACKEND)
    "VASTAI_API_KEY",
    "LAMBDA_API_KEY",       # Lambda Cloud (lambda.ai) API key
    "LAMBDA_SSH_KEY_NAME",  # optional: name of a pre-registered Lambda account SSH key
    "ZEVO_CLOUD_BACKEND",    # "vastai" | "lambda" — which cloud backend to rent on
    # Hugging Face Hub access for gated/private model and dataset downloads.
    "HF_TOKEN",
    # Optional public training telemetry.
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
}

# Keys that are configuration, not credentials: a region name and a choice
# between two literals. Redacting these only hides the current setting from the
# page that exists to show it, so they come back in the clear.
PLAIN_KEYS = {"AWS_REGION", "ZEVO_CLOUD_BACKEND", "WANDB_ENTITY", "WANDB_PROJECT"}

# Lenient per-key format check so the UI catches obvious paste errors
# before we write garbage to .env. Empty string is always allowed
# (= clear the key).
KEY_FORMATS: dict[str, re.Pattern] = {
    "CLAUDE_CODE_OAUTH_TOKEN":   re.compile(r"^sk-ant-oat\d+-[A-Za-z0-9\-_]{20,}$"),
    "ANTHROPIC_API_KEY":         re.compile(r"^sk-ant-[A-Za-z0-9\-_]{20,}$"),
    "ANTHROPIC_AUTH_TOKEN":      re.compile(r"^\S{20,}$"),
    # Bedrock API key is a base64 blob (decodes to "BedrockAPIKey-...:secret").
    "AWS_BEARER_TOKEN_BEDROCK":  re.compile(r"^[A-Za-z0-9+/=]{40,}$"),
    "OPENAI_API_KEY":            re.compile(r"^sk-[A-Za-z0-9\-_]{20,}$"),
    "OPENROUTER_API_KEY":        re.compile(r"^sk-or-v1-[a-f0-9]{32,}$"),
    "AWS_REGION":                re.compile(r"^[a-z]{2}-[a-z]+-\d$"),
    "VASTAI_API_KEY":            re.compile(r"^[a-f0-9]{32,128}$"),
    # Lambda Cloud keys look like 'secret_<label>_<hex>'; stay lenient.
    "LAMBDA_API_KEY":            re.compile(r"^secret_[A-Za-z0-9._-]{16,}$"),
    "LAMBDA_SSH_KEY_NAME":       re.compile(r"^[A-Za-z0-9 ._-]{1,64}$"),
    "ZEVO_CLOUD_BACKEND":         re.compile(r"^(vastai|lambda)$"),
    "HF_TOKEN":                    re.compile(r"^hf_[A-Za-z0-9]{20,}$"),
    "WANDB_API_KEY":              re.compile(r"^[A-Za-z0-9_-]{20,}$"),
    "WANDB_ENTITY":               re.compile(r"^[A-Za-z0-9_.-]{1,128}$"),
    "WANDB_PROJECT":              re.compile(r"^[A-Za-z0-9_.-]{1,128}$"),
}


ENV_PATH = Path(os.environ.get("ZEVO_ENV_FILE", "/app/.env"))


class SecretEntry(BaseModel):
    name: str
    present: bool
    preview: str = ""


class EnvironmentSshStatus(BaseModel):
    """Safe readiness summary for an env-backed SSH connection.

    Instance vs. managed Cluster remains an internal execution detail.  The
    Settings page presents both exactly like any other SSH-reachable machine.
    Connection facts stay server-side; the browser receives only an id, a
    human label, and whether all required values are present.
    """

    id: str
    label: str
    configured: bool
    status: str
    last_error: str = ""
    last_verified_at: str | None = None


class SecretsResponse(BaseModel):
    env_path: str
    entries: list[SecretEntry]
    ssh_connections: list[EnvironmentSshStatus]


class SetSecretsBody(StrictBody):
    # Empty string in the value = clear the key.
    values: dict[str, str]


class SetSecretsResponse(BaseModel):
    updated: list[str]
    cleared: list[str]
    next_step: str = (
        "Run `docker compose up -d --force-recreate backend scheduler` "
        "to load the new values into the running containers."
    )


def _read_env(path: Path) -> dict[str, str]:
    """Parse a .env file into {KEY: VALUE}. Comments + blank lines
    skipped. Quoted values supported (single OR double)."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if v.startswith(("'", '"')) and v.endswith(("'", '"')) and len(v) >= 2:
            v = v[1:-1]
        out[k] = v
    return out


def _environment_ssh_config(connection_id: str) -> dict[str, str | int | bool]:
    if connection_id not in {"instance", "cluster"}:
        raise KeyError(connection_id)
    prefix = "ZEVO_INSTANCE" if connection_id == "instance" else "ZEVO_CLUSTER"
    configured_name = os.environ.get(f"{prefix}_SSH_NAME", "").strip()
    host = os.environ.get(f"{prefix}_SSH_HOST", "").strip()
    username = os.environ.get(f"{prefix}_SSH_USER", "").strip()
    remote_dir = os.environ.get(f"{prefix}_REMOTE_DIR", "").strip()
    env_setup = os.environ.get(f"{prefix}_ENV_SETUP", "").strip()
    port_text = os.environ.get(f"{prefix}_SSH_PORT", "22").strip() or "22"
    try:
        port = int(port_text)
    except ValueError:
        port = 0
    password_path = os.environ.get(f"{prefix}_SSH_PASSWORD_FILE", "").strip()
    key_path = (
        "" if password_path else resolve_ssh_key(f"{prefix}_SSH_KEY")
    )
    label = configured_name or connection_id.title()
    configured = bool(
        host and username and remote_dir and env_setup and 1 <= port <= 65535
        and (bool(key_path) != bool(password_path))
    )
    key_marker = ""
    try:
        key_stat = Path(key_path or password_path).stat()
        key_marker = f"{key_stat.st_size}:{key_stat.st_mtime_ns}"
    except OSError:
        pass
    fingerprint = sha256("\0".join((
        host, str(port), username, remote_dir, env_setup,
        key_path, password_path, key_marker,
    )).encode("utf-8")).hexdigest()
    return {
        "id": connection_id,
        "prefix": prefix,
        "host": host,
        "port": port,
        "username": username,
        "remote_dir": remote_dir,
        "env_setup": env_setup,
        "key_path": key_path,
        "password_path": password_path,
        "label": label,
        "configured": configured,
        "fingerprint": fingerprint,
    }


def _environment_ssh_statuses(
    checks: Mapping[str, EnvironmentSshVerification] | None = None,
) -> list[EnvironmentSshStatus]:
    """Report active env-backed SSH connections without execution categories."""
    rows: list[EnvironmentSshStatus] = []
    checks = checks or {}
    for connection_id in ("instance", "cluster"):
        config = _environment_ssh_config(connection_id)
        check = checks.get(connection_id)
        current = bool(
            check is not None
            and check.fingerprint == config["fingerprint"]
        )
        status = check.status if current else "unverified"
        rows.append(EnvironmentSshStatus(
            id=connection_id,
            label=str(config["label"]),
            configured=bool(config["configured"]),
            status=status,
            last_error=check.last_error if current and check else "",
            last_verified_at=(
                check.last_verified_at.isoformat()
                if current and check and check.last_verified_at else None
            ),
        ))
    return rows


def _write_env_atomic(
    path: Path,
    kv: dict[str, str],
    *,
    comment_header: str | None = None,
    remove_keys: Iterable[str] = (),
    remove_comment_prefixes: Iterable[str] = (),
) -> None:
    """Update or remove keys while preserving unrelated dotenv content.

    Existing key order and comments are retained, new keys are appended, and
    values containing whitespace or ``#`` are quoted.
    """
    removed = set(remove_keys)
    removed_comment_prefixes = tuple(remove_comment_prefixes)
    existing_lines: list[str] = []
    existing_keys: set[str] = set()
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.rstrip()
            if line.lstrip().startswith(removed_comment_prefixes):
                continue
            if not line or line.lstrip().startswith("#") or "=" not in line:
                existing_lines.append(line)
                continue
            k = line.split("=", 1)[0].strip()
            if k in removed:
                continue
            existing_keys.add(k)
            if k in kv:
                v = kv[k]
                if v == "":
                    # Cleared — keep the key commented out for visibility
                    existing_lines.append(f"# {k}=")
                else:
                    existing_lines.append(f"{k}={_quote_if_needed(v)}")
            else:
                existing_lines.append(line)

    # Append brand-new keys (not present in the existing file)
    new_keys = [k for k in kv if k not in existing_keys and kv[k] != ""]
    if new_keys:
        if comment_header:
            existing_lines.append("")
            existing_lines.append(f"# {comment_header}")
        for k in new_keys:
            existing_lines.append(f"{k}={_quote_if_needed(kv[k])}")

    # Atomic write: serialize into a sibling temp file, fsync, then
    # os.replace onto the target so a crash mid-write can never leave a
    # half-written .env. One caveat: a SINGLE bind-mounted file (Docker
    # mounts .env this way) cannot be replaced — the kernel returns EBUSY
    # because the mount is pinned to the inode — so on that OSError we fall
    # back to truncate-and-write in place, with the bytes already fully
    # serialized in memory.
    new_bytes = ("\n".join(existing_lines).rstrip() + "\n").encode("utf-8")
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(path.parent), prefix=path.name + ".", suffix=".tmp",
        delete=False,
    )
    try:
        with tmp:
            tmp.write(new_bytes)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp.name, 0o600)
        os.replace(tmp.name, path)
    except OSError:
        # Bind-mounted target (EBUSY) or replace not permitted: write in place.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        with open(path, "wb") as fh:
            fh.write(new_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(path, 0o600)
        except OSError:
            # Best-effort chmod; on bind-mounted files some hosts deny this.
            pass


def _quote_if_needed(v: str) -> str:
    if any(ch in v for ch in (" ", "#", "\t", "\n")):
        # Use double quotes; escape any existing "
        return '"' + v.replace('"', r'\"') + '"'
    return v


@router.get("/settings", response_model=SecretsResponse)
async def get_secrets(db: AsyncSession = Depends(get_db)) -> SecretsResponse:
    env = _read_env(ENV_PATH)
    entries = []
    for k in sorted(ALLOWED_KEYS):
        v = env.get(k, "")
        entries.append(SecretEntry(
            name=k, present=bool(v),
            # Non-secret config (region, cloud-backend selection, W&B names) is
            # shown in full; real secrets return NO preview at all -- not even a
            # first-6/last-4 redaction, which still leaks entropy and confirms a
            # key's shape. The UI renders `preview || "(redacted)"`, so an empty
            # string shows a clean "(redacted)" marker via presence alone.
            preview=v if k in PLAIN_KEYS else "",
        ))
    check_rows = (await db.execute(select(EnvironmentSshVerification))).scalars().all()
    managed_hosts = (await db.execute(select(SshHost))).scalars().all()
    managed_identities = {
        (row.category, row.host.strip().casefold(), row.port, row.username.strip())
        for row in managed_hosts
    }
    env_connections = []
    for row in _environment_ssh_statuses({row.id: row for row in check_rows}):
        if not row.configured:
            continue
        config = _environment_ssh_config(row.id)
        identity = (
            row.id,
            str(config["host"]).strip().casefold(),
            int(config["port"]),
            str(config["username"]).strip(),
        )
        # A verified UI profile is also mirrored into .env. Return only one
        # status entry for that physical connection.
        if identity not in managed_identities:
            env_connections.append(row)
    return SecretsResponse(
        env_path=str(ENV_PATH),
        entries=entries,
        ssh_connections=env_connections,
    )


@router.post("/settings", response_model=SetSecretsResponse)
async def set_secrets(body: SetSecretsBody) -> SetSecretsResponse:
    if not body.values:
        raise HTTPException(400, "values is empty")

    unknown = [k for k in body.values if k not in ALLOWED_KEYS]
    if unknown:
        raise HTTPException(
            400,
            f"unknown key(s): {unknown}. Only {sorted(ALLOWED_KEYS)} can be set via UI.",
        )

    # Validate non-empty values against their per-key format check.
    bad_format: list[str] = []
    for k, v in body.values.items():
        if v == "":
            continue
        pat = KEY_FORMATS.get(k)
        if pat and not pat.match(v):
            bad_format.append(k)
    if bad_format:
        raise HTTPException(
            400,
            f"value(s) failed format check for: {bad_format}. "
            f"Check for paste errors / trailing whitespace.",
        )

    # Apply.
    env = _read_env(ENV_PATH)
    updated: list[str] = []
    cleared: list[str] = []
    for k, v in body.values.items():
        if v == "":
            if env.get(k):
                cleared.append(k)
            env[k] = ""
        else:
            if env.get(k) != v:
                updated.append(k)
            env[k] = v

    _write_env_atomic(
        ENV_PATH, env,
        comment_header="Added by /api/settings (the Settings page)",
    )

    return SetSecretsResponse(updated=updated, cleared=cleared)
