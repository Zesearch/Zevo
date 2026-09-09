"""validate_ssh_target must reject option-injection vectors in host/user/port.

A host or user beginning with `-` (e.g. `-oProxyCommand=<cmd>`) is remote code
execution against ssh/sshpass/scp even under create_subprocess_exec (no shell),
because ssh parses any leading-dash argument as an option. The guard rejects
those, and callers additionally place a `--` end-of-options marker.
"""
from __future__ import annotations

import pytest

from zevo.engine.ssh_auth import validate_ssh_target


def test_accepts_ordinary_target() -> None:
    validate_ssh_target(host="10.0.0.5", user="root", port=22)
    validate_ssh_target(host="gpu-box.example.com", user="ubuntu", port=2222)


@pytest.mark.parametrize("host", [
    "-oProxyCommand=touch /tmp/pwned",
    "-J attacker",
    "",
    "   ",
    "has space",
    "has\ttab",
    "bad\nhost",
    "user@sneaky",
])
def test_rejects_hostile_host(host: str) -> None:
    with pytest.raises(ValueError):
        validate_ssh_target(host=host, user="root", port=22)


@pytest.mark.parametrize("user", [
    "-oProxyCommand=touch /tmp/pwned",
    "",
    " root",
    "root ",
    "ro ot",
])
def test_rejects_hostile_user(user: str) -> None:
    with pytest.raises(ValueError):
        validate_ssh_target(host="10.0.0.5", user=user, port=22)


@pytest.mark.parametrize("port", [0, -1, 65536, 999999])
def test_rejects_bad_port(port: int) -> None:
    with pytest.raises(ValueError):
        validate_ssh_target(host="10.0.0.5", user="root", port=port)
