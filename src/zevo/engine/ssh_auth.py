"""Build non-interactive SSH/SCP commands without exposing passwords.

Key authentication passes only the key path. Password authentication uses
``sshpass -f`` so the password stays in a mode-0600 file and never appears in
argv, process listings, Ticket payloads, or logs.
"""
from __future__ import annotations

from pathlib import Path


def authentication_method(*, key_path: str = "", password_path: str = "") -> str:
    """Return the configured method, requiring exactly one credential path."""
    has_key = bool(key_path.strip())
    has_password = bool(password_path.strip())
    if has_key == has_password:
        raise ValueError("exactly one of key_path or password_path is required")
    return "private_key" if has_key else "password"


def validate_ssh_target(*, host: str, user: str, port: int) -> None:
    """Reject host/user/port that could inject ssh options or break argv.

    ``ssh`` (and ``sshpass``/``scp``) treat any argument beginning with ``-``
    as an option, so an attacker-supplied host like ``-oProxyCommand=<cmd>``
    or user like ``-oProxyCommand=...`` is remote-code-execution even under
    ``create_subprocess_exec`` (no shell needed). Callers must ALSO place a
    ``--`` end-of-options marker before the ``user@host`` target; this guard
    is the belt to that suspenders and rejects the obviously-hostile inputs
    up front with a clear message.
    """
    for label, value in (("host", host), ("user", user)):
        if not value or not value.strip():
            raise ValueError(f"{label} must not be empty")
        if value != value.strip():
            raise ValueError(f"{label} must not have leading/trailing whitespace")
        if value.lstrip().startswith("-"):
            raise ValueError(f"{label} must not start with '-' (option injection)")
        if any(c.isspace() or ord(c) < 0x20 for c in value):
            raise ValueError(f"{label} must not contain whitespace or control characters")
    if "@" in host:
        raise ValueError("host must not contain '@'")
    if not (1 <= int(port) <= 65535):
        raise ValueError("port must be between 1 and 65535")


def credential_file_exists(*, key_path: str = "", password_path: str = "") -> bool:
    try:
        method = authentication_method(
            key_path=key_path,
            password_path=password_path,
        )
    except ValueError:
        return False
    path = key_path if method == "private_key" else password_path
    return Path(path).is_file()


def ssh_base_args(
    *,
    key_path: str = "",
    password_path: str = "",
    port: int,
    strict_host_key_checking: str = "no",
    connect_timeout: int = 15,
) -> list[str]:
    method = authentication_method(
        key_path=key_path,
        password_path=password_path,
    )
    prefix = ["sshpass", "-f", password_path] if method == "password" else []
    args = [*prefix, "ssh"]
    if method == "private_key":
        args += [
            "-i", key_path,
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
        ]
    else:
        args += [
            "-o", "BatchMode=no",
            "-o", "PreferredAuthentications=password,keyboard-interactive",
            "-o", "PubkeyAuthentication=no",
            "-o", "NumberOfPasswordPrompts=1",
        ]
    return [
        *args,
        "-p", str(port),
        "-o", f"StrictHostKeyChecking={strict_host_key_checking}",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", f"ConnectTimeout={connect_timeout}",
    ]


def scp_base_args(
    *,
    key_path: str = "",
    password_path: str = "",
    port: int,
    recursive: bool = False,
) -> list[str]:
    method = authentication_method(
        key_path=key_path,
        password_path=password_path,
    )
    prefix = ["sshpass", "-f", password_path] if method == "password" else []
    args = [*prefix, "scp"]
    if method == "private_key":
        args += [
            "-i", key_path,
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
        ]
    else:
        args += [
            "-o", "BatchMode=no",
            "-o", "PreferredAuthentications=password,keyboard-interactive",
            "-o", "PubkeyAuthentication=no",
            "-o", "NumberOfPasswordPrompts=1",
        ]
    args += [
        "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
    ]
    if recursive:
        args.append("-r")
    return args
