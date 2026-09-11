"""Resolve the installation-wide default compute target.

The setting names one concrete option from the Launch Run picker instead of
only naming a provider category.  That distinction matters when an
installation has several verified Cluster or Instance connections.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.db import SshHost


DEFAULT_COMPUTE_KEY = "ZEVO_DEFAULT_COMPUTE"
DEFAULT_COMPUTE_PATTERN = re.compile(
    r"^(?:cloud:(?:vastai|lambda)|environment:(?:cluster|instance)|"
    r"connection:[A-Za-z0-9._-]{1,128})$"
)


class DefaultComputeError(ValueError):
    """The configured default is missing, malformed, or no longer usable."""


ComputeProvider = Literal["cluster", "cloud", "instance"]


@dataclass(frozen=True)
class ComputeTarget:
    value: str
    provider: ComputeProvider
    cloud_backend: str = ""
    ssh_host_id: str = ""


def _env_path() -> Path:
    return Path(os.environ.get("ZEVO_ENV_FILE", "/app/.env"))


def _read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (
            value.startswith(("'", '"'))
            and value.endswith(("'", '"'))
            and len(value) >= 2
        ):
            value = value[1:-1]
        values[key] = value
    return values


def runtime_setting(name: str) -> str:
    """Read the live Settings file, falling back to process env when absent.

    The Settings page writes the bind-mounted ``.env`` before containers are
    recreated.  Reading the file makes a default-compute choice effective for
    the next Run immediately while keeping existing environment-only installs
    working.
    """
    path = _env_path()
    if path.exists():
        return _read_env(path).get(name, "").strip()
    return os.environ.get(name, "").strip()


def parse_compute_target(value: str) -> ComputeTarget:
    normalized = (value or "").strip()
    if not DEFAULT_COMPUTE_PATTERN.fullmatch(normalized):
        raise DefaultComputeError(
            "Default compute must name an available Cloud backend or SSH connection."
        )
    kind, identifier = normalized.split(":", 1)
    if kind == "cloud":
        return ComputeTarget(normalized, "cloud", cloud_backend=identifier)
    if kind == "environment":
        return ComputeTarget(normalized, cast(ComputeProvider, identifier))
    return ComputeTarget(normalized, "instance", ssh_host_id=identifier)


async def resolve_default_compute(db: AsyncSession) -> ComputeTarget:
    raw = runtime_setting(DEFAULT_COMPUTE_KEY)
    if not raw:
        raise DefaultComputeError(
            "GPU backend is required. Choose one for this Run or set a default in Settings."
        )
    target = parse_compute_target(raw)
    if target.provider == "cloud":
        credential_key = (
            "VASTAI_API_KEY"
            if target.cloud_backend == "vastai"
            else "LAMBDA_API_KEY"
        )
        legacy_lambda = (
            target.cloud_backend == "lambda"
            and bool(runtime_setting("LAMBDA_CLOUD_API_KEY"))
        )
        if not runtime_setting(credential_key) and not legacy_lambda:
            raise DefaultComputeError(
                f"Default compute {target.cloud_backend} is no longer available. "
                "Add its API key or choose another default in Settings."
            )
        return target
    if target.value.startswith("environment:"):
        prefix = target.provider.upper()
        required = (
            runtime_setting(f"ZEVO_{prefix}_SSH_HOST"),
            runtime_setting(f"ZEVO_{prefix}_SSH_USER"),
            runtime_setting(f"ZEVO_{prefix}_REMOTE_DIR"),
            runtime_setting(f"ZEVO_{prefix}_ENV_SETUP"),
        )
        if not all(required):
            raise DefaultComputeError(
                f"Default {target.provider} connection is no longer configured. "
                "Choose another default in Settings."
            )
        return target

    host = (await db.execute(
        select(SshHost).where(SshHost.id == target.ssh_host_id)
    )).scalar_one_or_none()
    if host is None:
        raise DefaultComputeError(
            "The default SSH connection no longer exists. "
            "Choose another default in Settings."
        )
    if host.status != "verified":
        raise DefaultComputeError(
            f"Default SSH connection {host.label or host.host!r} is not verified."
        )
    return ComputeTarget(
        value=target.value,
        provider=cast(ComputeProvider, host.category),
        ssh_host_id=host.id,
    )
