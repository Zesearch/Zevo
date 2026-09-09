"""Lambda Cloud (lambda.ai) GPU provider client.

A second `cloud`-mode backend alongside Vast.ai. The public surface deliberately
MIRRORS `zevo.providers.vastai.VastAIProvider` so the infrastructure agent's cloud block can
swap one for the other with the same calls:

  async with LambdaCloudProvider(api_key=key) as p:  # __aenter__/__aexit__ are no-ops
      offers = await p.search_gpus(min_gpu_ram_gb=24, gpu_type="a10", max_dph=1.0)
      inst   = await p.create_instance(offer_id=offers[0]["id"],
                                       region=offers[0]["region"])
      info   = await p.wait_for_ready(inst["instance_id"], timeout=600)
      ssh    = await p.get_ssh_details(inst["instance_id"])       # user='ubuntu', port 22
      ok     = await p.wait_for_ssh(host=ssh["ssh_host"], port=22, user="ubuntu", key_path=...)
      done   = await p.destroy_instance(inst["instance_id"])

Differences from Vast.ai that callers must know (documented on each method):
  - Lambda has no per-offer marketplace: you pick an `instance_type_name` (e.g.
    `gpu_1x_a10`) that has capacity in a `region_name`. `search_gpus` returns one
    offer per (type, region-with-capacity) pair so a caller can retry regions.
  - `create_instance` needs BOTH the type name (offer_id) and a region, plus a
    PRE-REGISTERED account SSH key. We register the mounted public key once (see
    `resolve_ssh_key_name`) so downstream `ssh -i <priv> ubuntu@<ip>` works.
  - Login user is `ubuntu` on port 22 (Vast is `root` on a random high port).
  - Ready status is `"active"` (Vast is `"running"`). No docker image / disk arg —
    a Lambda box boots the Lambda Stack (CUDA + PyTorch) on bare Ubuntu.

`load_api_key()` checks `$LAMBDA_API_KEY` (also `$LAMBDA_CLOUD_API_KEY`) then
`~/.lambda/api_key`, raising FileNotFoundError with a clear hint if neither works.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

LAMBDA_API_BASE = "https://cloud.lambda.ai/api/v1"
_DEFAULT_API_KEY_PATH = Path.home() / ".lambda" / "api_key"

# The private key mounted into the scheduler/backend container. Its .pub is what
# we register with Lambda so launched boxes accept `ssh -i <this> ubuntu@<ip>`.
# Resolved via $LAMBDA_SSH_KEY_PATH, else the shared id_ed25519 → id_rsa probe
# (zevo.providers.resolve_ssh_key) — same rule as every other SSH key here.
from zevo.providers import resolve_ssh_key as _resolve_ssh_key


def _default_ssh_key_path() -> str:
    return _resolve_ssh_key("LAMBDA_SSH_KEY_PATH")

# Lambda's login user + port are fixed (unlike Vast's per-instance ssh_port).
LAMBDA_SSH_USER = "ubuntu"
LAMBDA_SSH_PORT = 22


def _sanitize_key(k: str) -> str:
    """Drop non-ASCII / whitespace chars from a key so a stray pasted character
    can't make httpx raise UnicodeEncodeError when building the Authorization
    header (same guard as the Vast.ai client)."""
    return "".join(c for c in (k or "") if c.isascii() and not c.isspace())


def load_api_key(explicit: Optional[str] = None) -> str:
    """Resolve a Lambda Cloud API key from (in order): the explicit arg,
    $LAMBDA_API_KEY / $LAMBDA_CLOUD_API_KEY, then ~/.lambda/api_key.
    Raises FileNotFoundError with a clear hint if none of those work.
    """
    if explicit:
        return _sanitize_key(explicit)
    for var in ("LAMBDA_API_KEY", "LAMBDA_CLOUD_API_KEY"):
        env = os.environ.get(var, "").strip()
        if env:
            return _sanitize_key(env)
    if _DEFAULT_API_KEY_PATH.exists():
        return _sanitize_key(_DEFAULT_API_KEY_PATH.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "No Lambda Cloud API key found. Set $LAMBDA_API_KEY or write the key to "
        f"{_DEFAULT_API_KEY_PATH}. Generate one at "
        "https://cloud.lambda.ai/api-keys."
    )


class LambdaCloudProvider:
    """Search, launch, and manage GPU instances on Lambda Cloud (lambda.ai)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = load_api_key(api_key)
        self._headers = {"Authorization": f"Bearer {self.api_key}"}

    # Context-manager parity with VastAIProvider (both open a client per call).
    async def __aenter__(self) -> "LambdaCloudProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """Authenticated request. Lambda wraps every payload in `{"data": ...}`
        and errors in `{"error": {"code","message","suggestion"}}`; we surface a
        readable message from the latter before raise_for_status swallows it."""
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.request(
                method, f"{LAMBDA_API_BASE}{path}", headers=self._headers, **kwargs,
            )
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    err = resp.json().get("error", {})
                    detail = f"{err.get('code','')}: {err.get('message','')} " \
                             f"{err.get('suggestion','')}".strip() or detail
                except Exception:
                    pass
                raise httpx.HTTPStatusError(
                    f"Lambda API {method} {path} -> {resp.status_code}: {detail}",
                    request=resp.request, response=resp,
                )
            return resp.json() if resp.content else {}

    # -----------------------------------------------------------------
    # Marketplace (instance types with regional capacity)
    # -----------------------------------------------------------------

    async def search_gpus(
        self,
        min_gpu_ram_gb: float = 16.0,
        gpu_type: Optional[str] = None,
        num_gpus: int = 1,
        max_dph: Optional[float] = None,
        order: str = "dph_total",  # accepted for VastAIProvider parity; always price-asc
        limit: int = 20,
        # Accepted for VastAIProvider parity and deliberately unused: Lambda
        # sells fixed instance TYPES whose host RAM and cores come with the
        # GPU, so there is nothing to filter on. The caller passes the same
        # arguments to either backend; here they are a no-op, not a promise.
        min_cpu_ram_gb: float = 0.0,
        min_cpu_cores: int = 0,
    ) -> list[dict]:
        """Return bootable Lambda offers, cheapest-first.

        Unlike Vast.ai, Lambda exposes fixed instance TYPES (e.g. `gpu_1x_a10`)
        that are only launchable in regions currently reporting capacity. We emit
        one offer per (type, region-with-capacity) pair — dropping types with no
        capacity anywhere — so the caller can fall through to another region if a
        launch races capacity away. Each offer carries `id` (instance_type_name)
        and `region` (region_name), both required by `create_instance`.
        """
        data = (await self._request("GET", "/instance-types")).get("data", {}) or {}
        offers: list[dict] = []
        for type_name, entry in data.items():
            it = entry.get("instance_type", {}) or {}
            regions = entry.get("regions_with_capacity_available", []) or []
            if not regions:
                continue  # no capacity anywhere -> not launchable right now
            specs = it.get("specs", {}) or {}
            n_gpus = int(specs.get("gpus", 0) or 0)
            if num_gpus and n_gpus != num_gpus:
                continue
            vram_gb = _parse_vram_gb(it.get("gpu_description") or it.get("description") or "")
            if min_gpu_ram_gb and vram_gb and vram_gb < min_gpu_ram_gb:
                continue
            dph = round((it.get("price_cents_per_hour", 0) or 0) / 100.0, 4)
            if max_dph and dph > max_dph:
                continue
            if gpu_type and gpu_type.lower().replace("_", "") not in \
                    (type_name + " " + str(it.get("gpu_description", ""))).lower().replace("_", ""):
                continue
            for reg in regions:
                offers.append(_normalize_offer(type_name, it, specs, vram_gb, dph, reg))
        offers.sort(key=lambda o: (o["dph_total"], o["region"]))
        return offers[:limit]

    # -----------------------------------------------------------------
    # SSH key registration (Lambda launches need a NAMED account key)
    # -----------------------------------------------------------------

    async def list_ssh_keys(self) -> list[dict]:
        return (await self._request("GET", "/ssh-keys")).get("data", []) or []

    async def resolve_ssh_key_name(self, key_path: str = "") -> str:
        """Return the name of an account SSH key whose public half matches our
        mounted key, registering it if absent.

        Order: `$LAMBDA_SSH_KEY_NAME` (trust the user pre-registered it) →ELSE→
        read `<key_path>.pub`, look for an already-registered key with the same
        public_key, →ELSE→ POST it under a stable `zevo-<fingerprint>` name. This
        is what makes `ssh -i <key_path> ubuntu@<ip>` work after launch, since
        Lambda only installs keys named in the launch request."""
        pinned = os.environ.get("LAMBDA_SSH_KEY_NAME", "").strip()
        if pinned:
            return pinned
        key_path = key_path or _default_ssh_key_path()
        pub_path = Path(f"{key_path}.pub")
        if not pub_path.exists():
            raise FileNotFoundError(
                f"Lambda launch needs an SSH public key at {pub_path} to register "
                "with the account (or set $LAMBDA_SSH_KEY_NAME to a key you already "
                "registered at https://cloud.lambda.ai/ssh-keys)."
            )
        pub = pub_path.read_text(encoding="utf-8").strip()
        # Compare on the key body (type + base64), ignoring any trailing comment.
        want = " ".join(pub.split()[:2])
        for k in await self.list_ssh_keys():
            existing = " ".join(str(k.get("public_key", "")).split()[:2])
            if existing and existing == want:
                return str(k["name"])
        name = "zevo-" + re.sub(r"[^a-zA-Z0-9]", "", want.split()[-1])[-16:]
        await self._request("POST", "/ssh-keys", json={"name": name, "public_key": pub})
        logger.info("Registered SSH key %r with Lambda Cloud", name)
        return name

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def create_instance(
        self,
        offer_id: str,
        region: str,
        ssh_key_name: Optional[str] = None,
        ssh_key_path: str = "",
        name: str = "zevo",
        # Accepted for VastAIProvider signature parity; Lambda ignores these
        # (bare-Ubuntu box, fixed root disk — no docker image / disk arg).
        docker_image: Optional[str] = None,
        disk_gb: int = 0,
        env: Optional[dict[str, str]] = None,
        onstart_cmd: Optional[str] = None,
    ) -> dict:
        """Launch one instance of `offer_id` (an instance_type_name) in `region`.
        Returns {instance_id, success}.

        SAFETY CAP mirrors the Vast client: refuses to launch when the account
        already holds >= ZEVO_LAMBDA_MAX_INSTANCES (default 1) live instances, so a
        runaway infra agent / rerun loop can't stack boxes and burn money.
        """
        max_active = int(os.environ.get("ZEVO_LAMBDA_MAX_INSTANCES", "1"))
        try:
            existing = await self.list_instances()
        except Exception as e:
            logger.warning("Lambda safety check could not list instances: %s", e)
            existing = []
        active = [
            i for i in existing
            if str(i.get("status", "")).lower() not in ("terminated", "terminating")
        ]
        if len(active) >= max_active:
            ids = [i.get("id") for i in active]
            raise RuntimeError(
                f"Lambda safety cap hit: {len(active)} active instance(s) already "
                f"running {ids}; limit ZEVO_LAMBDA_MAX_INSTANCES={max_active}. "
                "Refusing to launch another -- terminate existing instances first."
            )

        key_name = ssh_key_name or await self.resolve_ssh_key_name(ssh_key_path)
        payload: dict[str, Any] = {
            "region_name": region,
            "instance_type_name": offer_id,
            "ssh_key_names": [key_name],
            "name": name,
        }
        data = (await self._request(
            "POST", "/instance-operations/launch", json=payload)).get("data", {}) or {}
        ids = data.get("instance_ids") or []
        if not ids:
            raise RuntimeError(f"No instance id in Lambda launch response: {data}")
        return {"instance_id": str(ids[0]), "success": True}

    async def get_instance(self, instance_id: str) -> dict:
        """Fetch one instance, normalized to the Vast-shaped dict downstream reads."""
        inst = (await self._request("GET", f"/instances/{instance_id}")).get("data", {}) or {}
        it = inst.get("instance_type", {}) or {}
        specs = it.get("specs", {}) or {}
        region = inst.get("region", {}) or {}
        return {
            "id": str(inst.get("id", instance_id)),
            "status": inst.get("status", "unknown"),
            "ssh_host": inst.get("ip", "") or "",
            "ssh_port": LAMBDA_SSH_PORT,
            "ssh_user": LAMBDA_SSH_USER,
            "gpu_name": it.get("gpu_description", it.get("description", "")),
            "num_gpus": int(specs.get("gpus", 0) or 0),
            "dph_total": round((it.get("price_cents_per_hour", 0) or 0) / 100.0, 4),
            "region": region.get("name", ""),
        }

    async def list_instances(self) -> list[dict]:
        """List all instances currently on this account (raw Lambda dicts)."""
        return (await self._request("GET", "/instances")).get("data", []) or []

    async def destroy_instance(self, instance_id: str) -> bool:
        """Terminate an instance (stops billing). Returns True on success."""
        try:
            await self._request(
                "POST", "/instance-operations/terminate",
                json={"instance_ids": [instance_id]},
            )
            logger.info("Terminated Lambda instance %s", instance_id)
            return True
        except Exception as e:
            logger.error("Failed to terminate Lambda instance %s: %s", instance_id, e)
            return False

    async def get_ssh_details(self, instance_id: str) -> dict:
        """{ssh_host, ssh_port, ssh_user, instance_id} — user is always 'ubuntu'."""
        info = await self.get_instance(instance_id)
        return {
            "ssh_host": info.get("ssh_host", ""),
            "ssh_port": LAMBDA_SSH_PORT,
            "ssh_user": LAMBDA_SSH_USER,
            "instance_id": instance_id,
        }

    async def wait_for_ready(
        self,
        instance_id: str,
        timeout: int = 600,
        poll_interval: int = 15,
    ) -> dict:
        """Poll until status == 'active'. Lambda boots take longer than Vast, so
        the default timeout is higher. 'active' is API-ready, NOT SSH-ready — the
        caller must additionally `wait_for_ssh` afterwards."""
        elapsed = 0
        info: dict = {}
        while elapsed < timeout:
            info = await self.get_instance(instance_id)
            status = str(info.get("status", "")).lower()
            if status == "active":
                return info
            if status in ("terminated", "terminating", "unhealthy"):
                raise RuntimeError(
                    f"Lambda instance {instance_id} entered terminal state: {status}")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(
            f"Lambda instance {instance_id} not active after {timeout}s "
            f"(last status: {info.get('status')})"
        )

    async def wait_for_ssh(
        self,
        *,
        host: str,
        port: int = LAMBDA_SSH_PORT,
        user: str = LAMBDA_SSH_USER,
        key_path: str = "",
        timeout: int = 420,
        poll_interval: int = 10,
    ) -> bool:
        """Poll until the box ACCEPTS an SSH connection.

        `status=active` precedes sshd accepting connections, so callers must also
        wait on real SSH. Waits in PYTHON (asyncio.sleep + a subprocess ssh probe)
        — never a shell `sleep` loop, which some agent sandboxes BLOCK, turning a
        slow boot into a spurious failure. Returns True once reachable, else False.
        """
        key_path = key_path or _default_ssh_key_path()
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


def _parse_vram_gb(text: str) -> float:
    """Best-effort GB-of-VRAM from a Lambda gpu/instance description string, e.g.
    '1x A10 (24 GB PCIe)' -> 24.0, 'H100 (80 GB SXM5)' -> 80.0. 0.0 if not found
    (callers treat 0 as 'unknown' and skip the VRAM filter rather than exclude)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*GB", text or "", re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def _normalize_offer(
    type_name: str, it: dict, specs: dict, vram_gb: float, dph: float, region: dict,
) -> dict:
    """One (instance_type, region) offer, shaped like a Vast.ai normalized offer so
    the infra agent's cloud block treats both providers uniformly."""
    return {
        "id": type_name,                       # -> create_instance(offer_id=...)
        "region": region.get("name", ""),      # -> create_instance(region=...)
        "region_desc": region.get("description", ""),
        "gpu_name": it.get("gpu_description", type_name),
        "num_gpus": int(specs.get("gpus", 1) or 1),
        "gpu_ram_gb": vram_gb,
        "cpu_cores": int(specs.get("vcpus", 0) or 0),
        "ram_gb": float(specs.get("memory_gib", 0) or 0),
        "disk_gb": float(specs.get("storage_gib", 0) or 0),
        "dph_total": dph,
        "reliability": 1.0,                    # first-party hardware; no dud tier
        "verified": True,
        "description": it.get("description", ""),
    }
