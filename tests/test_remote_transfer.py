from __future__ import annotations

from pathlib import Path

import pytest

from zevo.contracts.infrastructure import DeviceSshRoute
from zevo.engine.remote_transfer import (
    build_download_command,
    build_upload_commands,
)


@pytest.fixture
def route() -> DeviceSshRoute:
    return DeviceSshRoute(
        host="login.example.edu",
        port=22,
        user="researcher",
        key_path="/keys/id_ed25519",
    )


def test_upload_owns_distinct_ssh_and_scp_port_options(
    tmp_path: Path, route: DeviceSshRoute,
) -> None:
    source = tmp_path / "predict.py"
    source.write_text("print('ok')\n", encoding="utf-8")

    mkdir, copy = build_upload_commands(
        route,
        sources=[str(source)],
        remote_dir="/orange/project/run/ticket",
    )

    assert mkdir[0] == "ssh"
    assert mkdir[mkdir.index("-p") + 1] == "22"
    assert "-P" not in mkdir
    assert copy[0] == "scp"
    assert copy[copy.index("-P") + 1] == "22"
    assert "-p" not in copy
    assert copy[-2] == str(source)
    assert copy[-1].startswith("researcher@login.example.edu:")


def test_download_rejects_relative_remote_paths(
    tmp_path: Path, route: DeviceSshRoute,
) -> None:
    with pytest.raises(ValueError, match="absolute remote path"):
        build_download_command(
            route,
            remote_paths=["model/checkpoint"],
            local_dir=str(tmp_path),
            recursive=True,
        )


def test_download_uses_scp_uppercase_port(
    tmp_path: Path, route: DeviceSshRoute,
) -> None:
    command = build_download_command(
        route,
        remote_paths=["/orange/project/run/model"],
        local_dir=str(tmp_path),
        recursive=True,
    )

    assert command[0] == "scp"
    assert command[command.index("-P") + 1] == "22"
    assert "-p" not in command
    assert "-r" in command


def test_password_route_uses_sshpass_file_without_exposing_password(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    route = DeviceSshRoute(
        host="gpu.example.org",
        port=2222,
        user="researcher",
        password_path="/credentials/password",
    )
    mkdir, copy = build_upload_commands(
        route,
        sources=[str(source)],
        remote_dir="/srv/zevo/run",
    )
    assert mkdir[:4] == ["sshpass", "-f", "/credentials/password", "ssh"]
    assert copy[:4] == ["sshpass", "-f", "/credentials/password", "scp"]
    assert "-i" not in mkdir and "-i" not in copy
