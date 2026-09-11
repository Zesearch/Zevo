"""CRUD and local verification for Cluster/Instance SSH connections.

The UI accepts a host private-key path or a write-only password. Host private
keys remain in the host's read-only ``~/.ssh`` mount; password credentials
are stored as 0600 files under the shared Zevo data root. Credential contents
never enter response bodies or logs. A successful
verification proves all three facts the runtime needs: SSH authentication,
creation of the configured ``.../zevo`` remote root, and environment
activation. It then synchronizes that connection to the matching ``.env``
prefix so CLI and UI launches share one configuration.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

import frontmatter
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.api.database import get_db
from zevo.api.hardware.ssh_verify import verify_ssh_workspace
from zevo.api.routers.ui.settings import (
    ENV_PATH,
    EnvironmentSshStatus,
    _environment_ssh_config,
    _environment_ssh_statuses,
    _read_env,
    _write_env_atomic,
)
from zevo.db import EnvironmentSshVerification, SshHost
from zevo.db import get_session_factory
from zevo.engine.ssh_auth import authentication_method

router = APIRouter()


_PROJECT_ROOT = Path(__file__).resolve().parents[5]
INFRA_SKILLS_ROOT = Path(os.environ.get(
    "ZEVO_INFRA_SKILLS_ROOT",
    str(_PROJECT_ROOT / "playbook" / "skills" / "infrastructure"),
))
_SKILL_START = "<!-- zevo:operator-instructions:start -->"
_SKILL_END = "<!-- zevo:operator-instructions:end -->"


class SshHostDTO(BaseModel):
    """Public connection data; credential contents never leave the backend."""

    id: str
    name: str
    category: Literal["cluster", "instance"]
    host: str
    port: int
    username: str
    private_key_path: str
    # True when the key was uploaded through the UI and lives in Zevo's own
    # credential store rather than the host's ~/.ssh mount.
    private_key_uploaded: bool = False
    authentication: Literal["private_key", "password", "none"]
    remote_dir: str
    env_setup: str
    container_image: str
    skill_path: str
    status: str
    gpu_info: Any
    last_error: str
    created_at: str
    last_verified_at: str | None


def _to_dto(h: SshHost) -> SshHostDTO:
    try:
        auth = authentication_method(
            key_path=h.key_path or "",
            password_path=h.password_path or "",
        )
    except ValueError:
        auth = "none"
    return SshHostDTO(
        id=h.id,
        name=h.label,
        category=h.category,
        host=h.host,
        port=h.port,
        username=h.username,
        private_key_path=_host_private_key_path(h.key_path or ""),
        private_key_uploaded=_is_managed_credential(h.key_path or ""),
        authentication=auth,
        remote_dir=h.remote_dir,
        env_setup=h.env_setup,
        container_image=h.container_image,
        skill_path=(
            _infra_skill_display_path(h)
            if _infra_skill_path(h).is_file() else ""
        ),
        status=h.status,
        gpu_info=h.gpu_info,
        last_error=h.last_error,
        created_at=h.created_at.isoformat() if h.created_at else "",
        last_verified_at=h.last_verified_at.isoformat() if h.last_verified_at else None,
    )


async def _load(db: AsyncSession, host_id: str) -> SshHost:
    """Load an `SshHost` by id; 404 if it does not exist."""
    h = (await db.execute(
        select(SshHost).where(SshHost.id == host_id)
    )).scalar_one_or_none()
    if h is None:
        raise HTTPException(404, f"ssh host {host_id} not found")
    return h


class CreateSshHostBody(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    category: Literal["cluster", "instance"]
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(22, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=64)
    private_key_path: str = ""
    # Contents of the private key file (PEM / OpenSSH). Stored by Zevo under
    # its credential root, so it works when the browser is not on the host.
    private_key: str = Field("", max_length=20_000)
    password: str = ""
    # Parent directory selected by the user. Zevo creates/uses its `zevo`
    # child, unless the supplied path already ends in `zevo`.
    remote_parent_dir: str = Field(min_length=1)
    env_setup: str = Field(min_length=1)
    container_image: str = ""
    skill_filename: str = ""
    skill_markdown: str | None = Field(None, max_length=50_000)

    @model_validator(mode="after")
    def require_one_credential(self) -> "CreateSshHostBody":
        given = sum(bool(v.strip()) for v in (self.private_key_path, self.private_key, self.password))
        if given != 1:
            raise ValueError("provide exactly one of an uploaded private key, private_key_path or password")
        if any(ch in self.password for ch in ("\n", "\r", "\0")):
            raise ValueError("password cannot contain line breaks or NUL")
        if self.skill_markdown is not None:
            if self.skill_filename != "SKILL.md":
                raise ValueError("infrastructure skill file must be named SKILL.md")
            if "\0" in self.skill_markdown:
                raise ValueError("skill_markdown cannot contain NUL")
        return self


SSH_CREDENTIAL_ROOT = Path(os.environ.get(
    "ZEVO_SSH_CREDENTIAL_ROOT",
    os.environ.get("ZEVO_SSH_KEY_ROOT", "/app/data/ssh-connections"),
))
HOST_SSH_ROOT = Path(
    os.environ.get("ZEVO_HOST_HOME", str(Path.home()))
).expanduser() / ".ssh"
CONTAINER_SSH_ROOT = Path(os.environ.get("ZEVO_CONTAINER_SSH_ROOT", "/root/.ssh"))


def _remote_zevo_dir(parent: str) -> str:
    value = parent.strip().rstrip("/")
    if (
        not value.startswith("/")
        or value.startswith("~")
        or any(ch in value for ch in ("\n", "\r", "\0"))
    ):
        raise HTTPException(400, "Remote directory must be an absolute path")
    return value if Path(value).name == "zevo" else f"{value}/zevo"


def _connection_credential_dir(name: str, host_id: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-._").lower()
    slug = slug or "connection"
    return SSH_CREDENTIAL_ROOT / f"{slug}-{host_id[:8]}"


def _connection_password_path(name: str, host_id: str) -> Path:
    return _connection_credential_dir(name, host_id) / "password"


def _connection_key_path(name: str, host_id: str) -> Path:
    return _connection_credential_dir(name, host_id) / "id_key"


def _is_managed_credential(path: str) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(SSH_CREDENTIAL_ROOT.resolve())
        return True
    except (OSError, ValueError):
        return False


def _write_private_key(path: Path, contents: str) -> None:
    """Store an uploaded private key with the permissions ssh insists on."""
    text = contents.replace("\r\n", "\n").strip()
    if "PRIVATE KEY" not in text or "\0" in text:
        raise HTTPException(400, "Upload the private key file (OpenSSH or PEM), not the .pub public key")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _infra_skill_slug(h: SshHost) -> str:
    """Path-safe connection display name used as its private Skill folder."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", h.label.strip()).strip("-._")
    if not safe:
        safe = f"ssh-{sha256(h.id.encode('utf-8')).hexdigest()[:16]}"
    return safe


def _infra_skill_path(h: SshHost) -> Path:
    return INFRA_SKILLS_ROOT / _infra_skill_slug(h) / "SKILL.md"


def _legacy_infra_skill_path(h: SshHost) -> Path:
    """Former connection-id path, used only to migrate an installed Skill."""
    safe_id = re.sub(r"[^a-z0-9-]+", "-", h.id.strip().lower()).strip("-")
    safe_id = safe_id or sha256(h.id.encode("utf-8")).hexdigest()[:16]
    return INFRA_SKILLS_ROOT / f"ssh-{safe_id}" / "SKILL.md"


def _infra_skill_display_path(h: SshHost) -> str:
    return f"playbook/skills/infrastructure/{_infra_skill_slug(h)}/SKILL.md"


def _read_infra_skill_path(path: Path) -> str:
    """Read only the operator-authored section of a managed site Skill."""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    if _SKILL_START not in text or _SKILL_END not in text:
        # Preserve the body of a hand-authored/uploaded Skill if startup sync
        # encounters it before Zevo has added connection-owned routing.
        try:
            return frontmatter.loads(text).content.strip()
        except Exception:
            return text.strip()
    return text.split(_SKILL_START, 1)[1].split(_SKILL_END, 1)[0].strip()


def _read_infra_skill_instructions(h: SshHost) -> str:
    return _read_infra_skill_path(_infra_skill_path(h))


def _uploaded_skill_body(markdown: str) -> str:
    """Validate an uploaded Agent Skill and return its instruction body.

    Zevo owns the connection-specific routing frontmatter so a renamed host or
    connection cannot leave stale triggers behind. The uploaded file supplies
    the operator-authored body.
    """
    if _SKILL_START in markdown or _SKILL_END in markdown:
        raise HTTPException(400, "SKILL.md cannot contain Zevo managed markers")
    try:
        document = frontmatter.loads(markdown)
    except Exception as exc:
        raise HTTPException(400, f"Invalid SKILL.md frontmatter: {exc}") from None
    metadata = dict(document.metadata)
    if not str(metadata.get("name") or "").strip():
        raise HTTPException(400, "SKILL.md frontmatter must include name")
    if not str(metadata.get("description") or "").strip():
        raise HTTPException(400, "SKILL.md frontmatter must include description")
    body = document.content.strip()
    if not body:
        raise HTTPException(400, "SKILL.md must include an instruction body")
    return body


def _write_infra_skill(
    h: SshHost,
    skill_markdown: str | None = None,
    *,
    previous_path: Path | None = None,
) -> None:
    """Materialize the connection's private Infrastructure ``SKILL.md``.

    Connection and credential values stay in the typed runtime contract. The
    Skill contributes site policy: scheduler usage for a cluster or direct GPU
    process rules for an instance. Its stable id avoids orphaning a local Skill
    when the display name changes.
    """
    # No file in an update means preserve the current instructions while still
    # refreshing connection-owned routing metadata after a rename. A
    # connection without an uploaded Skill does not receive a synthetic one.
    path = _infra_skill_path(h)
    source_path = previous_path or (
        path if path.is_file() else _legacy_infra_skill_path(h)
    )
    if skill_markdown is None:
        if not source_path.is_file():
            return
        operator = _read_infra_skill_path(source_path)
    else:
        if not skill_markdown.strip():
            _remove_infra_skill_path(source_path)
            if source_path != path:
                _remove_infra_skill_path(path)
            return
        operator = _uploaded_skill_body(skill_markdown)
    slug = _infra_skill_slug(h)
    label = " ".join(h.label.split()) or "SSH GPU connection"
    host = "".join(h.host.split())
    description = (
        f"Operate Zevo on the {label} {h.category} connection. "
        f"Use only when ssh_host_id is {h.id} or the SSH host is {host}."
    )
    if h.category == "cluster":
        execution = """- Treat this resource as a Slurm cluster, not a fixed instance.
- Write a finite `train.sbatch` or `predict.sbatch`, submit that file, record the job id, and monitor queued, running, and terminal scheduler states with bounded backoff.
- Use the typed container image, environment command, remote directory, GPU maximum, and live partition/account/QOS evidence. Never create an empty holder allocation."""
    else:
        execution = """- Treat this resource as a fixed GPU instance and execute the stage directly over SSH; do not invoke Slurm.
- Select only an idle GPU slice within the Run maximum, record the remote process, checkpoint resumable work, and stop only the exact Zevo-owned process."""
    text = f"""---
name: {slug}
description: {json.dumps(description)}
---

# {label}

This Skill applies only to the typed `{h.category}` connection identified by
`ssh_host_id={h.id}` or SSH host `{host}`. Connection values and credentials
come from the Run's typed Infrastructure input; never copy secrets into this
file or replace those values here.

## Use the GPU resource

{execution}
- Validate live resource and environment state before execution. Live SSH,
  scheduler, and device output outrank this file.
- Keep caches and resumable checkpoints in the typed remote workspace; clean
  only Ticket-local temporary files after the stage reaches a terminal state.
- Report concrete resource, job/process, output, and terminal evidence.

## Site-specific operator instructions

{_SKILL_START}
{operator}
{_SKILL_END}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".SKILL.md.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    if source_path != path:
        _remove_infra_skill_path(source_path)


def _remove_infra_skill_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _remove_infra_skill(h: SshHost) -> None:
    _remove_infra_skill_path(_infra_skill_path(h))
    _remove_infra_skill_path(_legacy_infra_skill_path(h))


def _host_relative_key_path(value: str) -> Path:
    raw = value.strip()
    if raw.startswith("~/"):
        source = HOST_SSH_ROOT.parent / raw[2:]
    else:
        source = Path(raw)
    if not source.is_absolute():
        raise HTTPException(400, "Private key path must be absolute or start with ~/.ssh/")
    for root in (HOST_SSH_ROOT, CONTAINER_SSH_ROOT):
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        if ".." in relative.parts or not relative.parts:
            break
        return relative
    raise HTTPException(400, "Private key path must be inside the host ~/.ssh directory")


def _resolve_host_private_key_path(value: str) -> str:
    """Map a host ``~/.ssh`` path to the existing read-only container mount."""
    relative = _host_relative_key_path(value)
    raw = value.strip()
    native = HOST_SSH_ROOT / relative
    mapped = CONTAINER_SSH_ROOT / relative
    candidate = native if native.is_file() else mapped
    try:
        resolved = candidate.resolve(strict=True)
        allowed_root = (HOST_SSH_ROOT if candidate == native else CONTAINER_SSH_ROOT).resolve()
        resolved.relative_to(allowed_root)
    except (OSError, ValueError):
        raise HTTPException(400, f"Private key file is not available to Zevo: {raw}") from None
    if resolved.suffix == ".pub":
        raise HTTPException(400, "Select the private key file, not the .pub public key")
    if not os.access(resolved, os.R_OK):
        raise HTTPException(400, f"Private key file is not readable by Zevo: {raw}")
    return str(resolved)


def _host_private_key_path(key_path: str) -> str:
    """Return the operator-facing host path rather than the container mount."""
    if not key_path:
        return ""
    try:
        relative = Path(key_path).relative_to(CONTAINER_SSH_ROOT)
    except ValueError:
        return key_path
    return str(HOST_SSH_ROOT / relative)


def _remove_managed_credential(path: str) -> None:
    """Remove only Zevo-owned password/legacy-key files, never host SSH keys."""
    if not path:
        return
    credential_path = Path(path)
    try:
        credential_path.resolve().relative_to(SSH_CREDENTIAL_ROOT.resolve())
        credential_path.unlink(missing_ok=True)
        credential_path.parent.rmdir()
    except (OSError, ValueError):
        pass


def _write_password(path: Path, password: str) -> None:
    if not password or any(ch in password for ch in ("\n", "\r", "\0")):
        raise HTTPException(400, "SSH password is empty or contains a line break")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(password + "\n")
        handle.flush()
        os.fsync(handle.fileno())


SSH_PROFILE_ROOT = "ZEVO_SSH_CONNECTION_"
SSH_PROFILE_FIELDS = (
    "ID",
    "NAME",
    "CATEGORY",
    "SSH_HOST",
    "SSH_PORT",
    "SSH_USER",
    "SSH_KEY",
    "SSH_PASSWORD_FILE",
    "REMOTE_DIR",
    "ENV_SETUP",
    "CONTAINER_IMAGE",
)
LEGACY_CONNECTION_FIELDS = (
    "SSH_NAME",
    "SSH_HOST",
    "SSH_PORT",
    "SSH_USER",
    "SSH_KEY",
    "SSH_PASSWORD_FILE",
    "REMOTE_DIR",
    "ENV_SETUP",
    "CONTAINER_IMAGE",
)


def _connection_env_token(connection_id: str) -> str:
    """Return a short stable env-safe token for one database connection."""
    if connection_id.startswith("env-"):
        direct_token = re.sub(r"[^A-Z0-9_]", "_", connection_id[4:].upper())
        if direct_token:
            return direct_token[:24]
    return sha256(connection_id.encode("utf-8")).hexdigest()[:16].upper()


def _connection_env_prefix(connection_id: str) -> str:
    return f"{SSH_PROFILE_ROOT}{_connection_env_token(connection_id)}"


def _connection_env_values(h: SshHost) -> dict[str, str]:
    """Serialize one UI-managed connection without sharing a category slot."""
    prefix = _connection_env_prefix(h.id)
    return {
        f"{prefix}_ID": h.id,
        f"{prefix}_NAME": h.label,
        f"{prefix}_CATEGORY": h.category,
        f"{prefix}_SSH_HOST": h.host,
        f"{prefix}_SSH_PORT": str(h.port),
        f"{prefix}_SSH_USER": h.username,
        f"{prefix}_SSH_KEY": h.key_path or "",
        f"{prefix}_SSH_PASSWORD_FILE": h.password_path or "",
        f"{prefix}_REMOTE_DIR": h.remote_dir,
        f"{prefix}_ENV_SETUP": h.env_setup,
        f"{prefix}_CONTAINER_IMAGE": h.container_image or "",
    }


def _clear_connection_env_if_current(values: dict[str, str]) -> None:
    """Remove only the independently keyed profile for this connection."""
    id_key = next(key for key in values if key.endswith("_ID"))
    current = _read_env(ENV_PATH)
    if current.get(id_key, "").strip() != values[id_key].strip():
        return
    _write_env_atomic(ENV_PATH, {}, remove_keys=values)
    for key in values:
        os.environ.pop(key, None)


def _sync_connection_to_env(h: SshHost) -> None:
    """Persist one UI connection just like any other Settings value."""
    values = _connection_env_values(h)
    _write_env_atomic(
        ENV_PATH,
        values,
        comment_header=f"SSH connection: {h.label} ({h.category})",
    )
    # Keep the backend Settings/readiness response current. Other containers
    # receive these values after the restart already called out in the UI.
    os.environ.update(values)


def _sync_connection_blocks_to_env(rows: list[SshHost]) -> None:
    """Rewrite all managed profiles as readable, separated dotenv blocks."""
    current = _read_env(ENV_PATH)
    old_profile_keys = {
        key for key in current if key.startswith(SSH_PROFILE_ROOT)
    }
    _write_env_atomic(
        ENV_PATH,
        {},
        remove_keys=old_profile_keys,
        remove_comment_prefixes=("# SSH connection:",),
    )
    for key in old_profile_keys:
        os.environ.pop(key, None)
    for row in rows:
        _sync_connection_to_env(row)


def _profile_rows_from_env(values: dict[str, str]) -> list[dict[str, Any]]:
    """Parse every independent SSH profile from a dotenv mapping."""
    tokens: set[str] = set()
    pattern = re.compile(
        rf"^{re.escape(SSH_PROFILE_ROOT)}([A-Z0-9_]{{1,24}})_(?:{'|'.join(SSH_PROFILE_FIELDS)})$"
    )
    for key in values:
        if match := pattern.fullmatch(key):
            tokens.add(match.group(1))

    rows: list[dict[str, Any]] = []
    for token in sorted(tokens):
        prefix = f"{SSH_PROFILE_ROOT}{token}"
        category = values.get(f"{prefix}_CATEGORY", "").strip().lower()
        port_text = values.get(f"{prefix}_SSH_PORT", "22").strip() or "22"
        try:
            port = int(port_text)
        except ValueError:
            continue
        key_path = values.get(f"{prefix}_SSH_KEY", "").strip()
        password_path = values.get(f"{prefix}_SSH_PASSWORD_FILE", "").strip()
        row = {
            "token": token,
            "id": values.get(f"{prefix}_ID", "").strip(),
            "label": values.get(f"{prefix}_NAME", "").strip(),
            "category": category,
            "host": values.get(f"{prefix}_SSH_HOST", "").strip(),
            "port": port,
            "username": values.get(f"{prefix}_SSH_USER", "").strip(),
            "key_path": key_path,
            "password_path": password_path,
            "remote_dir": values.get(f"{prefix}_REMOTE_DIR", "").strip(),
            "env_setup": values.get(f"{prefix}_ENV_SETUP", "").strip(),
            "container_image": values.get(f"{prefix}_CONTAINER_IMAGE", "").strip(),
        }
        if (
            category in {"cluster", "instance"}
            and row["host"]
            and row["username"]
            and row["remote_dir"]
            and row["env_setup"]
            and 1 <= port <= 65535
            and (bool(key_path) != bool(password_path))
        ):
            rows.append(row)
    return rows


def _legacy_profile_from_env(
    values: dict[str, str], category: Literal["cluster", "instance"]
) -> dict[str, Any] | None:
    """Read one old category slot so startup can migrate it losslessly."""
    prefix = f"ZEVO_{category.upper()}"
    host = values.get(f"{prefix}_SSH_HOST", "").strip()
    username = values.get(f"{prefix}_SSH_USER", "").strip()
    remote_dir = values.get(f"{prefix}_REMOTE_DIR", "").strip()
    env_setup = values.get(f"{prefix}_ENV_SETUP", "").strip()
    container_image = values.get(f"{prefix}_CONTAINER_IMAGE", "").strip()
    key_path = values.get(f"{prefix}_SSH_KEY", "").strip()
    password_path = values.get(f"{prefix}_SSH_PASSWORD_FILE", "").strip()
    port_text = values.get(f"{prefix}_SSH_PORT", "22").strip() or "22"
    try:
        port = int(port_text)
    except ValueError:
        return None
    if not (
        host and username and remote_dir and env_setup
        and 1 <= port <= 65535
        and (bool(key_path) != bool(password_path))
    ):
        return None
    return {
        "token": category.upper(),
        "id": "",
        "label": values.get(f"{prefix}_SSH_NAME", "").strip()
        or category.title(),
        "category": category,
        "host": host,
        "port": port,
        "username": username,
        "key_path": key_path,
        "password_path": password_path,
        "remote_dir": remote_dir,
        "env_setup": env_setup,
        "container_image": container_image,
    }


async def sync_ssh_connections_with_env() -> int:
    """Two-way startup sync for all SSH profiles.

    Independent ``ZEVO_SSH_CONNECTION_<token>_*`` blocks are imported into
    ``ssh_hosts``. Existing UI rows are then serialized back to their own
    blocks. The two legacy category slots are imported once and removed only
    after a database row exists, so same-category connections never overwrite
    each other again.
    """
    values = _read_env(ENV_PATH)
    profiles = _profile_rows_from_env(values)
    legacy_profiles = [
        row for category in ("instance", "cluster")
        if (row := _legacy_profile_from_env(values, category)) is not None
    ]
    Session = get_session_factory()
    async with Session() as db:
        for profile in [*profiles, *legacy_profiles]:
            row = None
            profile_id = str(profile["id"] or "").strip()
            if profile_id and len(profile_id) <= 36:
                row = await db.get(SshHost, profile_id)
            if row is None:
                row = (await db.execute(
                    select(SshHost).where(
                        SshHost.host == profile["host"],
                        SshHost.port == profile["port"],
                        SshHost.username == profile["username"],
                    ).limit(1)
                )).scalars().first()
            if row is None:
                stable_id = (
                    profile_id
                    if profile_id and len(profile_id) <= 36
                    else f"env-{str(profile['token']).lower().replace('_', '-')}"
                )
                if len(stable_id) > 36:
                    stable_id = str(uuid5(NAMESPACE_URL, f"zevo:ssh:{profile['token']}"))
                row = SshHost(id=stable_id, host=profile["host"], username=profile["username"])
                db.add(row)
                changed = True
            else:
                changed = any(
                    getattr(row, field) != profile[field]
                    for field in (
                        "label", "category", "host", "port", "username",
                        "key_path", "password_path", "remote_dir", "env_setup",
                        "container_image",
                    )
                )
            for field in (
                "label", "category", "host", "port", "username",
                "key_path", "password_path", "remote_dir", "env_setup",
                "container_image",
            ):
                setattr(row, field, profile[field])
            if changed:
                row.status = "unverified"
                row.last_error = "Connection changed in .env; verify it in Settings."
                row.last_verified_at = None
        await db.commit()

        rows = (await db.execute(select(SshHost).order_by(SshHost.created_at))).scalars().all()
        synchronized = len(rows)
        for row in rows:
            _write_infra_skill(row)
        _sync_connection_blocks_to_env(list(rows))

    migrated_categories = {str(row["category"]) for row in legacy_profiles}
    if migrated_categories:
        remove_keys = {
            f"ZEVO_{category.upper()}_{field}"
            for category in migrated_categories
            for field in LEGACY_CONNECTION_FIELDS
        }
        _write_env_atomic(
            ENV_PATH,
            {},
            remove_keys=remove_keys,
            remove_comment_prefixes=("# SSH connection saved from Settings:",),
        )
        for key in remove_keys:
            os.environ.pop(key, None)
    return synchronized


async def _verify_and_persist(
    h: SshHost,
    db: AsyncSession,
    *,
    previous_env_values: dict[str, str] | None = None,
    skill_markdown: str | None = None,
    previous_skill_path: Path | None = None,
) -> SshHostDTO:
    result = await verify_ssh_workspace(
        host=h.host,
        port=h.port,
        username=h.username,
        key_path=h.key_path or "",
        password_path=h.password_path or "",
        remote_dir=h.remote_dir,
        env_setup=h.env_setup,
    )
    h.status = "verified" if result["ok"] else "failed"
    h.gpu_info = {}
    h.last_error = str(result["error"])
    h.last_verified_at = datetime.now(timezone.utc)
    # Configuration and readiness are separate facts. Persist every saved
    # profile to its own env block even when verification fails, so UI and
    # direct .env configuration remain the same source set.
    if previous_env_values:
        old_id_key = next(key for key in previous_env_values if key.endswith("_ID"))
        new_id_key = next(
            key for key in _connection_env_values(h) if key.endswith("_ID")
        )
        if old_id_key != new_id_key:
            _clear_connection_env_if_current(previous_env_values)
    _write_infra_skill(h, skill_markdown, previous_path=previous_skill_path)
    await db.flush()
    rows = (await db.execute(
        select(SshHost).order_by(SshHost.created_at)
    )).scalars().all()
    _sync_connection_blocks_to_env(list(rows))
    await db.commit()
    await db.refresh(h)
    return _to_dto(h)


@router.post("/hardware/ssh", response_model=SshHostDTO)
async def create_ssh_host(
    body: CreateSshHostBody,
    db: AsyncSession = Depends(get_db),
) -> SshHostDTO:
    name = body.name.strip()
    existing = (await db.execute(
        select(SshHost).where(SshHost.label == name)
    )).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(409, f"SSH connection {name!r} already exists")

    h = SshHost(
        label=name,
        category=body.category,
        host=body.host.strip(),
        port=body.port,
        username=body.username.strip(),
        remote_dir=_remote_zevo_dir(body.remote_parent_dir),
        env_setup=body.env_setup.strip(),
        container_image=body.container_image.strip(),
        status="unverified",
    )
    db.add(h)
    await db.flush()
    if body.private_key.strip():
        h.key_path = str(_connection_key_path(name, h.id))
    elif body.private_key_path.strip():
        h.key_path = _resolve_host_private_key_path(body.private_key_path)
    else:
        h.key_path = ""
    h.password_path = (
        str(_connection_password_path(name, h.id)) if body.password else ""
    )
    try:
        if body.private_key.strip():
            _write_private_key(Path(h.key_path), body.private_key)
        if h.password_path:
            _write_password(Path(h.password_path), body.password)
    except Exception:
        await db.rollback()
        raise
    return await _verify_and_persist(
        h, db, skill_markdown=body.skill_markdown,
    )


@router.get("/hardware/ssh", response_model=list[SshHostDTO])
async def list_ssh_hosts(
    db: AsyncSession = Depends(get_db),
) -> list[SshHostDTO]:
    rows = (await db.execute(
        select(SshHost).order_by(desc(SshHost.created_at))
    )).scalars().all()
    return [_to_dto(r) for r in rows]


@router.post(
    "/hardware/ssh/environment/{connection_id}/verify",
    response_model=EnvironmentSshStatus,
)
async def verify_environment_ssh(
    connection_id: str,
    db: AsyncSession = Depends(get_db),
) -> EnvironmentSshStatus:
    try:
        config = _environment_ssh_config(connection_id)
    except KeyError:
        raise HTTPException(404, "SSH connection not found") from None
    if not config["configured"]:
        raise HTTPException(400, "SSH connection is not fully configured")

    result = await verify_ssh_workspace(
        host=str(config["host"]),
        port=int(config["port"]),
        username=str(config["username"]),
        key_path=str(config["key_path"]),
        password_path=str(config["password_path"]),
        remote_dir=str(config["remote_dir"]),
        env_setup=str(config["env_setup"]),
    )
    check = await db.get(EnvironmentSshVerification, connection_id)
    if check is None:
        check = EnvironmentSshVerification(id=connection_id)
        db.add(check)
    check.fingerprint = str(config["fingerprint"])
    check.status = "verified" if result["ok"] else "failed"
    check.last_error = str(result["error"])
    check.last_verified_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(check)
    statuses = _environment_ssh_statuses({connection_id: check})
    return next(row for row in statuses if row.id == connection_id)


class ReverifyBody(BaseModel):
    private_key_path: str = ""  # optional: switch to/replace key authentication
    private_key: str = Field("", max_length=20_000)  # optional: uploaded key contents
    password: str = ""     # optional: switch to/replace password authentication

    @model_validator(mode="after")
    def at_most_one_credential(self) -> "ReverifyBody":
        if sum(bool(v.strip()) for v in (self.private_key_path, self.private_key, self.password)) > 1:
            raise ValueError("provide only one of an uploaded private key, private_key_path or password")
        if any(ch in self.password for ch in ("\n", "\r", "\0")):
            raise ValueError("password cannot contain line breaks or NUL")
        return self


class UpdateSshHostBody(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    category: Literal["cluster", "instance"]
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(22, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=64)
    private_key_path: str = ""
    private_key: str = Field("", max_length=20_000)
    password: str = ""
    remote_parent_dir: str = Field(min_length=1)
    env_setup: str = Field(min_length=1)
    # ``None`` means an older UI did not send this newly added field. Preserve
    # the configured Cluster image in that case instead of silently erasing it.
    container_image: str | None = None
    # None preserves the installed Skill. An uploaded file replaces its
    # operator-authored instruction body.
    skill_filename: str = ""
    skill_markdown: str | None = Field(None, max_length=50_000)

    @model_validator(mode="after")
    def validate_optional_replacement_credential(self) -> "UpdateSshHostBody":
        if sum(bool(v.strip()) for v in (self.private_key_path, self.private_key, self.password)) > 1:
            raise ValueError("provide only one of an uploaded private key, private_key_path or password")
        if any(ch in self.password for ch in ("\n", "\r", "\0")):
            raise ValueError("password cannot contain line breaks or NUL")
        if self.skill_markdown is not None:
            if self.skill_filename != "SKILL.md":
                raise ValueError("infrastructure skill file must be named SKILL.md")
            if "\0" in self.skill_markdown:
                raise ValueError("skill_markdown cannot contain NUL")
        return self


@router.put("/hardware/ssh/{host_id}", response_model=SshHostDTO)
async def update_ssh_host(
    host_id: str,
    body: UpdateSshHostBody,
    db: AsyncSession = Depends(get_db),
) -> SshHostDTO:
    h = await _load(db, host_id)
    previous_env_values = _connection_env_values(h)
    previous_skill_path = _infra_skill_path(h)
    name = body.name.strip()
    duplicate = (await db.execute(
        select(SshHost).where(SshHost.label == name, SshHost.id != host_id)
    )).scalar_one_or_none()
    if duplicate is not None:
        raise HTTPException(409, f"SSH connection {name!r} already exists")

    new_key_path = body.private_key_path.strip()
    if body.private_key.strip():
        key_path = _connection_key_path(name, h.id)
        _write_private_key(key_path, body.private_key)
        _remove_managed_credential(h.password_path or "")
        if h.key_path and Path(h.key_path) != key_path:
            _remove_managed_credential(h.key_path)
        h.key_path = str(key_path)
        h.password_path = ""
    elif new_key_path:
        resolved_key_path = _resolve_host_private_key_path(new_key_path)
        _remove_managed_credential(h.password_path or "")
        _remove_managed_credential(h.key_path or "")
        h.key_path = resolved_key_path
        h.password_path = ""
    elif body.password:
        password_path = h.password_path or str(_connection_password_path(name, h.id))
        _write_password(Path(password_path), body.password)
        _remove_managed_credential(h.key_path or "")
        h.key_path = ""
        h.password_path = password_path

    h.label = name
    h.category = body.category
    h.host = body.host.strip()
    h.port = body.port
    h.username = body.username.strip()
    h.remote_dir = _remote_zevo_dir(body.remote_parent_dir)
    h.env_setup = body.env_setup.strip()
    if body.category == "instance":
        h.container_image = ""
    elif body.container_image is not None:
        h.container_image = body.container_image.strip()
    return await _verify_and_persist(
        h,
        db,
        previous_env_values=previous_env_values,
        skill_markdown=body.skill_markdown,
        previous_skill_path=previous_skill_path,
    )


@router.post("/hardware/ssh/{host_id}/verify", response_model=SshHostDTO)
async def reverify_ssh_host(
    host_id: str,
    body: ReverifyBody | None = None,
    db: AsyncSession = Depends(get_db),
) -> SshHostDTO:
    h = await _load(db, host_id)

    new_key_path = ((body.private_key_path if body else "") or "").strip()
    new_key = ((body.private_key if body else "") or "").strip()
    new_password = (body.password if body else "") or ""
    if new_key:
        key_path = _connection_key_path(h.label, h.id)
        _write_private_key(key_path, new_key)
        if h.password_path:
            _remove_managed_credential(h.password_path)
            h.password_path = ""
        if h.key_path and Path(h.key_path) != key_path:
            _remove_managed_credential(h.key_path)
        h.key_path = str(key_path)
    elif new_key_path:
        resolved_key_path = _resolve_host_private_key_path(new_key_path)
        if h.password_path:
            _remove_managed_credential(h.password_path)
            h.password_path = ""
        h.key_path = resolved_key_path
    elif new_password:
        if not h.password_path:
            h.password_path = str(_connection_password_path(h.label, h.id))
        _write_password(Path(h.password_path), new_password)
        if h.key_path:
            _remove_managed_credential(h.key_path)
            h.key_path = ""
    credential_path = h.key_path or h.password_path
    if not credential_path or not Path(credential_path).is_file():
        h.status = "failed"
        h.last_error = "No SSH private key or password file — provide one to verify."
        h.last_verified_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(h)
        return _to_dto(h)
    return await _verify_and_persist(h, db)


@router.delete("/hardware/ssh/{host_id}")
async def delete_ssh_host(
    host_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    h = await _load(db, host_id)
    credential_paths = [value for value in (h.key_path, h.password_path) if value]
    skill_paths = (_infra_skill_path(h), _legacy_infra_skill_path(h))
    await db.delete(h)
    await db.flush()
    remaining = (await db.execute(
        select(SshHost).order_by(SshHost.created_at)
    )).scalars().all()
    _sync_connection_blocks_to_env(list(remaining))
    await db.commit()
    for credential_path in credential_paths:
        _remove_managed_credential(credential_path)
    for skill_path in skill_paths:
        _remove_infra_skill_path(skill_path)
    return {"status": "deleted", "id": host_id}
