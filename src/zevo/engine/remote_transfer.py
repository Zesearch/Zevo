"""Safe SSH/SCP file transfer driven by an exact ``device_info.json`` route.

The CLI deliberately owns SSH/SCP option construction.  Specialists provide
only absolute local/remote paths and therefore cannot accidentally reuse
SSH's lowercase ``-p`` for SCP, whose port option is uppercase ``-P``.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Sequence

from zevo.contracts.infrastructure import DeviceSshRoute, InfrastructureDeviceInfo
from zevo.engine.ssh_auth import (
    scp_base_args as _authenticated_scp_base_args,
    ssh_base_args as _authenticated_ssh_base_args,
)


def load_ssh_route(device_info_path: str | Path) -> DeviceSshRoute:
    path = Path(device_info_path)
    if not path.is_absolute():
        raise ValueError(f"device_info must be an absolute path: {path}")
    try:
        info = InfrastructureDeviceInfo.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load valid device_info.json {path}: {exc}") from exc
    return info.ssh


def ssh_base_args(route: DeviceSshRoute) -> list[str]:
    return _authenticated_ssh_base_args(
        key_path=route.key_path,
        password_path=route.password_path,
        port=route.port,
    )


def scp_base_args(route: DeviceSshRoute, *, recursive: bool = False) -> list[str]:
    return _authenticated_scp_base_args(
        key_path=route.key_path,
        password_path=route.password_path,
        port=route.port,
        recursive=recursive,
    )


def _remote_path(value: str, *, label: str) -> str:
    if not value or not PurePosixPath(value).is_absolute():
        raise ValueError(f"{label} must be an absolute remote path: {value!r}")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError(f"{label} contains a forbidden control character")
    return value


def _target(route: DeviceSshRoute) -> str:
    return f"{route.user}@{route.host}"


def build_upload_commands(
    route: DeviceSshRoute,
    *,
    sources: Sequence[str],
    remote_dir: str,
    recursive: bool = False,
) -> list[list[str]]:
    destination = _remote_path(remote_dir, label="remote_dir").rstrip("/") or "/"
    if not sources:
        raise ValueError("upload requires at least one --source")
    local_sources: list[str] = []
    for source_value in sources:
        source = Path(source_value)
        if not source.is_absolute():
            raise ValueError(f"upload source must be absolute: {source}")
        if not source.exists():
            raise ValueError(f"upload source does not exist: {source}")
        if source.is_dir() and not recursive:
            raise ValueError(f"directory upload requires --recursive: {source}")
        local_sources.append(str(source))
    mkdir = ssh_base_args(route) + [
        _target(route), f"mkdir -p -- {shlex.quote(destination)}",
    ]
    remote_spec = f"{_target(route)}:{shlex.quote(destination + '/')}"
    copy = scp_base_args(route, recursive=recursive) + [
        *local_sources, remote_spec,
    ]
    return [mkdir, copy]


def build_download_command(
    route: DeviceSshRoute,
    *,
    remote_paths: Sequence[str],
    local_dir: str,
    recursive: bool = False,
) -> list[str]:
    if not remote_paths:
        raise ValueError("download requires at least one --remote-path")
    destination = Path(local_dir)
    if not destination.is_absolute():
        raise ValueError(f"local_dir must be an absolute path: {destination}")
    sources = [
        f"{_target(route)}:{shlex.quote(_remote_path(path, label='remote_path'))}"
        for path in remote_paths
    ]
    return scp_base_args(route, recursive=recursive) + [*sources, str(destination)]


def _run(command: Sequence[str]) -> None:
    completed = subprocess.run(list(command), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"remote transfer command exited {completed.returncode}: {command[0]}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer Zevo artifacts using the route in device_info.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    upload = subparsers.add_parser("upload")
    upload.add_argument("--device-info", required=True)
    upload.add_argument("--remote-dir", required=True)
    upload.add_argument("--source", action="append", required=True)
    upload.add_argument("--recursive", action="store_true")
    upload.add_argument("--dry-run", action="store_true")

    download = subparsers.add_parser("download")
    download.add_argument("--device-info", required=True)
    download.add_argument("--local-dir", required=True)
    download.add_argument("--remote-path", action="append", required=True)
    download.add_argument("--recursive", action="store_true")
    download.add_argument("--dry-run", action="store_true")
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        route = load_ssh_route(args.device_info)
        if args.command == "upload":
            commands = build_upload_commands(
                route,
                sources=args.source,
                remote_dir=args.remote_dir,
                recursive=args.recursive,
            )
        else:
            Path(args.local_dir).mkdir(parents=True, exist_ok=True)
            commands = [build_download_command(
                route,
                remote_paths=args.remote_path,
                local_dir=args.local_dir,
                recursive=args.recursive,
            )]
        if args.dry_run:
            print(json.dumps(commands, ensure_ascii=False))
            return 0
        for command in commands:
            _run(command)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"remote-transfer failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(_main())
