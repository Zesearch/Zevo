"""One SSH-readiness probe for every provider that hands back a fresh host."""
from __future__ import annotations

import asyncio


async def wait_for_ssh(
    *,
    host: str,
    port: int,
    user: str,
    key_path: str,
    timeout: int = 420,
    poll_interval: int = 10,
) -> bool:
    """Poll until the box ACCEPTS an SSH connection.

    A provider's "running"/"active" state precedes sshd accepting connections,
    so callers must also wait on real SSH. Waits in PYTHON (asyncio.sleep + a
    subprocess ssh probe), never a shell `sleep` loop, which some agent
    sandboxes BLOCK. Returns True once reachable, else False at the deadline.
    Raises ValueError on an empty host: probing nothing can only time out, and
    the caller should re-read the instance instead of burning its deadline.
    """
    if not str(host or "").strip():
        raise ValueError(
            "wait_for_ssh requires a host; the instance has no IP yet "
            "(call wait_for_ready / get_ssh_details again)"
        )
    elapsed = 0
    while elapsed < timeout:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh", "-i", key_path, "-p", str(port),
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                f"{user}@{host}", "echo __ssh_ok__",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            if proc.returncode == 0 and b"__ssh_ok__" in (out or b""):
                return True
        except Exception:
            pass  # ssh not up yet / transient — keep polling
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    return False
