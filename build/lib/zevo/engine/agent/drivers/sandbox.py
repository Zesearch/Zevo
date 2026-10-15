"""OpenShell lifecycle used by the ``claude_cli`` orchestrator.

Today the only isolation an agent CLI gets is the Docker container it runs in,
launched with `--dangerously-skip-permissions` — so an agent can touch the whole
container plus every host path we bind-mount, and reach any network. That's the
`none` mode below and stays the default.

`openshell` mode runs the orchestrator inside an **NVIDIA OpenShell**
sandbox: per-agent kernel-level isolation (Landlock), a default-deny network
opened by a declarative YAML policy, and a full allow/deny audit trail. OpenShell
runs Claude Code unmodified.

    openshell sandbox create --policy <policy.yaml> -- claude -p … -

The claude_cli driver creates a keepalive sandbox, uploads the Ticket workspace,
executes Claude with proxied stdio, downloads outputs, and deletes the sandbox.

Deployment (chosen): a **host-side gateway + Docker compute driver**. The gateway
runs on the host; the containerized scheduler points at it via
`$ZEVO_OPENSHELL_GATEWAY`. OpenShell then spawns each sandbox as a sibling Docker
container and injects credentials as env vars (never onto the sandbox FS), so our
**Max OAuth** (`CLAUDE_CODE_OAUTH_TOKEN`) is forwarded rather than an API key.

Config (all optional; the Agent row is the only place that selects the mode):

  ZEVO_OPENSHELL_BIN          path to the openshell CLI   (default: "openshell")
  ZEVO_OPENSHELL_IMAGE        BYOC sandbox image (--from)  (default: zevo-openshell-sandbox:latest)
  ZEVO_OPENSHELL_PROVIDER     provider carrying OAuth      (default: claude-code)
  ZEVO_OPENSHELL_POLICY       policy YAML path            (default: /app/ops/openshell/zevo.yaml)
  ZEVO_OPENSHELL_GATEWAY      gateway endpoint URL (host)  → OPENSHELL_GATEWAY_ENDPOINT
  ZEVO_OPENSHELL_GATEWAY_INSECURE  skip TLS verify (self-signed local gw)  (default: "1")
  ZEVO_OPENSHELL_CREATE_ARGS  extra `sandbox create` args, shlex-split. Escape hatch
                             for anything the defaults don't cover.

VERIFIED end-to-end against OpenShell 0.0.86 (Docker driver, host gateway):
  1. `sandbox create --no-tty --` proxies stdin+stdout — a piped prompt reaches
     claude and stream-json flows back (apiKeySource=none, i.e. Max OAuth).
  2. The Max OAuth token is injected via the `claude-code` provider created with
     `--credential CLAUDE_CODE_OAUTH_TOKEN` (see ops/openshell/README.md).
  3. Network egress needs a `binaries` allowlist per endpoint (ops/openshell/zevo.yaml).

WORKSPACE I/O: the agent's per-ticket workspace lives on
the host, not in the sandbox FS, so the driver drives a lifecycle around the run:
`open_sandbox()` creates a keepalive sandbox and UPLOADS the workspace into it;
the agent runs via `sandbox exec` (stdio proxied) with its cwd set to the uploaded
workspace; `close_sandbox()` DOWNLOADS the workspace back (collecting artifacts)
and deletes the sandbox. `sandbox_exec_argv()` builds the exec command.

KNOWN LIMITATION: only the ticket's OWN workspace dir round-trips. Agents that
read/write ABSOLUTE host paths outside it (cross-ticket inputs, /app/data, the
run root) or SSH to GPU boxes need those paths inside the sandbox + opened in the
policy — a per-run concern beyond this per-ticket mechanism.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Remote root inside the sandbox we upload the ticket workspace under.
_SANDBOX_ROOT = "/sandbox"
# create-keepalive processes, keyed by sandbox name, so close_sandbox can reap them.
_KEEPALIVE: dict[str, "asyncio.subprocess.Process"] = {}

_DEFAULT_POLICY = "/app/ops/openshell/zevo.yaml"
# Our BYOC sandbox image (built from ops/openshell/sandbox/) — has claude +
# iproute2 + the high-uid `sandbox` user OpenShell requires. Avoids the gated
# ghcr.io/nvidia/openshell-community images.
_DEFAULT_IMAGE = "zevo-openshell-sandbox:latest"
# The OpenShell provider carrying our Max OAuth token (built-in `claude-code`
# type, created with --credential CLAUDE_CODE_OAUTH_TOKEN). Satisfies the agent
# binding AND injects the subscription token (apiKeySource=none in-sandbox).
_DEFAULT_PROVIDER = "claude-code"


def resolve_mode(explicit: str = "") -> str:
    """Resolve the canonical per-agent choice; there is no global override."""
    e = (explicit or "").strip().lower()
    return e if e in ("none", "openshell") else "none"


def configure_sandbox_env(env: dict[str, str], mode: str = "") -> dict[str, str]:
    """Mutate + return the child env so the OpenShell CLI can reach the host
    gateway. No-op unless the explicit per-agent mode is `openshell`.

    The local gateway listens on an HTTPS endpoint with a self-signed cert
    (e.g. https://host.docker.internal:17670 from inside a container), so we set
    the endpoint var the CLI reads (OPENSHELL_GATEWAY_ENDPOINT) plus
    OPENSHELL_GATEWAY_INSECURE to skip TLS verification. Both are overridable."""
    if resolve_mode(mode) != "openshell":
        return env
    gateway = os.environ.get("ZEVO_OPENSHELL_GATEWAY", "").strip()
    if gateway:
        env["OPENSHELL_GATEWAY_ENDPOINT"] = gateway
        # Local gateways use a self-signed cert; default to skipping verification
        # unless the operator explicitly turns it off.
        if os.environ.get("ZEVO_OPENSHELL_GATEWAY_INSECURE", "1").strip() not in ("0", "false", ""):
            env["OPENSHELL_GATEWAY_INSECURE"] = "1"
    return env


# ───────────────────────── workspace-I/O lifecycle ──────────────────────────
#
# The driver calls, in order:
#   name = await open_sandbox(agent_id, workspace_dir)      # create + upload
#   cmd  = sandbox_exec_argv(name, base_cmd, workspace_dir) # -> `sandbox exec …`
#   <run cmd as a subprocess; stdio proxies as proven>
#   await close_sandbox(name, workspace_dir)                # download + delete
# open_sandbox returns None in `none` mode, so the driver falls back to running
# the command directly (today's behaviour) with zero changes.


def _bin() -> str:
    return os.environ.get("ZEVO_OPENSHELL_BIN", "openshell").strip() or "openshell"


def _sandbox_name(agent_id: str, workspace_dir: str) -> str:
    leaf = Path(workspace_dir).name
    return f"zevo-{agent_id}-{leaf}"[:63]


def remote_workdir(workspace_dir: str) -> str:
    """Where the uploaded workspace lands in the sandbox. `upload <ws> /sandbox`
    nests under the source basename, so it's `/sandbox/<leaf>`."""
    return f"{_SANDBOX_ROOT}/{Path(workspace_dir).name}"


def _cli_env() -> dict[str, str]:
    env = {**os.environ}
    return configure_sandbox_env(env, mode="openshell")


async def _run_cli(args: list[str], *, timeout: float = 180.0) -> tuple[int, str, str]:
    """Run an `openshell …` CLI call, return (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        _bin(), *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=_cli_env(),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "", f"openshell {args[0] if args else ''} timed out after {timeout}s"
    return proc.returncode or 0, (out or b"").decode(errors="replace"), (err or b"").decode(errors="replace")


def sandbox_exec_argv(name: str, argv: list[str], workspace_dir: str) -> list[str]:
    """`openshell sandbox exec` form that runs `argv` inside the live sandbox with
    cwd = the uploaded workspace (so Claude Code discovers <ws>/.claude/skills)."""
    return [
        _bin(), "sandbox", "exec", "-n", name, "--no-tty",
        "--workdir", remote_workdir(workspace_dir), "--", *argv,
    ]


async def open_sandbox(agent_id: str, workspace_dir: str, mode: str = "") -> str | None:
    """Create a keepalive sandbox and upload the ticket workspace into it.
    Returns the sandbox name, or None unless the effective mode is `openshell`
    (`mode` is the canonical per-agent setting). Raises
    (after cleaning up) if the sandbox never reaches Ready or the upload fails —
    so an opted-in agent fails loudly rather than silently running unsandboxed."""
    if resolve_mode(mode) != "openshell":
        return None

    name = _sandbox_name(agent_id, workspace_dir)
    image = os.environ.get("ZEVO_OPENSHELL_IMAGE", _DEFAULT_IMAGE).strip()
    provider = os.environ.get("ZEVO_OPENSHELL_PROVIDER", _DEFAULT_PROVIDER).strip()
    policy = os.environ.get("ZEVO_OPENSHELL_POLICY", _DEFAULT_POLICY).strip()
    ttl = os.environ.get("ZEVO_OPENSHELL_TTL", "3600").strip() or "3600"

    create = [_bin(), "sandbox", "create", "--name", name, "--no-tty"]
    if image:
        create += ["--from", image]
    if provider:
        create += ["--provider", provider]
    if policy and Path(policy).is_file():
        create += ["--policy", policy]
    extra = os.environ.get("ZEVO_OPENSHELL_CREATE_ARGS", "").strip()
    if extra:
        create += shlex.split(extra)
    # Keepalive: the sandbox stays up running `sleep <ttl>` while we upload / exec
    # / download; `sandbox delete` (in close_sandbox) tears it down early.
    create += ["--", "sleep", ttl]

    # Launch detached — don't await; readiness is polled via `sandbox get`.
    proc = await asyncio.create_subprocess_exec(
        _bin(), *create[1:],
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        env=_cli_env(), start_new_session=True,
    )
    _KEEPALIVE[name] = proc

    # Poll until Ready (Docker "Up" precedes OpenShell "Ready" — use the phase).
    ready = False
    for _ in range(48):  # ~240s
        rc, out, _ = await _run_cli(["sandbox", "get", name], timeout=30)
        if rc == 0 and "Phase:" in out and "Ready" in out.split("Phase:", 1)[1][:40]:
            ready = True
            break
        await asyncio.sleep(5)
    if not ready:
        await close_sandbox(name, workspace_dir, download=False)
        raise RuntimeError(f"OpenShell sandbox {name!r} never reached Ready")

    rc, _, err = await _run_cli(
        ["sandbox", "upload", name, workspace_dir, _SANDBOX_ROOT], timeout=300)
    if rc != 0:
        await close_sandbox(name, workspace_dir, download=False)
        raise RuntimeError(f"OpenShell upload of {workspace_dir} failed: {err.strip()}")
    return name


async def close_sandbox(name: str, workspace_dir: str, *, download: bool = True) -> None:
    """Download the workspace back (collecting artifacts the agent wrote) then
    delete the sandbox. Best-effort: logs but never raises — cleanup must not mask
    the run's real outcome. Always reaps the keepalive process."""
    if download:
        tmp = tempfile.mkdtemp(prefix="zevo-openshell-dl-")
        try:
            rc, _, err = await _run_cli(
                ["sandbox", "download", name, remote_workdir(workspace_dir), tmp], timeout=300)
            if rc == 0:
                # `download <remote_dir> <tmp>` places the remote dir's CONTENTS
                # directly under <tmp> (unlike upload, which nests under the source
                # basename) — merge those contents back into the host workspace.
                if any(Path(tmp).iterdir()):
                    shutil.copytree(tmp, workspace_dir, dirs_exist_ok=True)
            else:
                log.warning("[sandbox] download from %s failed: %s", name, err.strip())
        except Exception as e:
            log.warning("[sandbox] download from %s errored: %s", name, e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    rc, _, err = await _run_cli(["sandbox", "delete", name], timeout=60)
    if rc != 0:
        log.warning("[sandbox] delete %s failed: %s", name, err.strip())

    proc = _KEEPALIVE.pop(name, None)
    if proc and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
