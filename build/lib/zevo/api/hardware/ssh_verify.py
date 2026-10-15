"""Verification for Cluster/Instance SSH connections.

Connection keys are managed as 0600 files under Zevo's shared data directory.
A connection becomes selectable only after SSH authentication, creation of its
remote Zevo workspace, and environment activation all succeed.
"""
from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any

from zevo.engine.ssh_auth import (
    authentication_method,
    credential_file_exists,
    ssh_base_args,
)

# Conservative safe-character sets for values that get embedded in an argv
# token handed to `ssh`. Neither pattern allows a leading `-`, which is what
# stops a value from being interpreted as an ssh option (defense-in-depth on
# top of the `--` end-of-options marker used below).
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_HOST_RE = re.compile(r"^[A-Za-z0-9.:_-]+$")


def _validate_inputs(host: str, port: int, username: str) -> str | None:
    """Return an error message if inputs are unsafe/invalid, else None."""
    if not username or username.startswith("-") or not _USERNAME_RE.match(username):
        return "invalid host/username/port"
    if not host or host.startswith("-") or not _HOST_RE.match(host):
        return "invalid host/username/port"
    if not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535):
        return "invalid host/username/port"
    return None


async def verify_ssh_workspace(
    host: str,
    port: int,
    username: str,
    key_path: str,
    remote_dir: str,
    env_setup: str,
    password_path: str = "",
    timeout: float = 45.0,
) -> dict[str, Any]:
    """Verify SSH, create the Zevo root, and prove activation succeeds."""
    validation_error = _validate_inputs(host, port, username)
    if validation_error is not None:
        return {"ok": False, "error": validation_error}
    try:
        method = authentication_method(
            key_path=key_path,
            password_path=password_path,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not credential_file_exists(
        key_path=key_path,
        password_path=password_path,
    ):
        return {"ok": False, "error": f"SSH {method} credential file was not found"}
    if (
        not remote_dir.startswith("/")
        or remote_dir.startswith("~")
        or any(ch in remote_dir for ch in ("\n", "\r", "\0"))
    ):
        return {"ok": False, "error": "remote directory must be an absolute path"}
    activation = env_setup.strip()
    if not activation or any(ch in activation for ch in ("\n", "\r", "\0")):
        return {"ok": False, "error": "environment activation command is required"}

    marker = "zevo-ssh-workspace-ok"
    remote_cmd = (
        "set -e; "
        f"mkdir -p -- {shlex.quote(remote_dir)}; "
        f"{activation}; "
        f"test -d {shlex.quote(remote_dir)}; "
        f"printf {shlex.quote(marker)}"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_base_args(
                key_path=key_path,
                password_path=password_path,
                port=port,
                strict_host_key_checking="accept-new",
                connect_timeout=10,
            ),
            "--",
            f"{username}@{host}",
            remote_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return {"ok": False, "error": f"failed to start ssh: {exc}"}

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, Exception):
            pass
        return {"ok": False, "error": "timeout"}

    if proc.returncode == 0 and stdout.decode("utf-8", errors="replace").strip().endswith(marker):
        return {"ok": True, "error": ""}
    return {"ok": False, "error": _friendly_error(stderr)}
def _friendly_error(stderr: bytes) -> str:
    text = (stderr or b"").decode("utf-8", errors="replace").strip()
    if text:
        # ssh sometimes emits multiple lines (warnings + the real error) — the
        # last non-empty line is almost always the actionable one.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text
    return "ssh connection failed"
