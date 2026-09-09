"""GPU providers, plus the one rule for resolving outbound SSH keys."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_ssh_key(env_var: str = "") -> str:
    """The private key for outbound SSH: ``$<env_var>`` when set, else the
    first key that actually exists under /root/.ssh (id_ed25519, then
    id_rsa). Every SSH-ing component — cluster, instance, cloud providers —
    resolves its key through here, so an ed25519-only or rsa-only host
    works the same way everywhere. Pass no env var to get just the probe.
    """
    value = os.environ.get(env_var, "").strip() if env_var else ""
    if value:
        return value
    for cand in ("/root/.ssh/id_ed25519", "/root/.ssh/id_rsa"):
        try:
            if Path(cand).is_file():
                return cand
        except OSError:
            # Treat an unreadable mount like an absent key. Verification will
            # surface a useful SSH error if no explicit key/password is set.
            continue
    return "/root/.ssh/id_ed25519"
