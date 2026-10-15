"""Vast.ai cloud GPU provider client.

Ported from the previous TuneLLM Labs project. Public surface is unchanged:

  async with VastAIProvider(api_key=key) as p:
      offers = await p.search_gpus(min_gpu_ram_gb=16, gpu_type="RTX_3090", max_dph=0.5)
      inst   = await p.create_instance(offer_id=..., docker_image=..., disk_gb=50)
      info   = await p.wait_for_ready(instance_id=inst["instance_id"], timeout=300)
      ssh    = await p.get_ssh_details(instance_id=inst["instance_id"])
      ok     = await p.destroy_instance(instance_id=inst["instance_id"])

`load_api_key()` is a convenience: checks `$VASTAI_API_KEY` first, then
falls back to `~/.config/vastai/vast_api_key` (the location the official
`vastai` CLI writes to). Raises FileNotFoundError with a clear hint if
neither is present.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx

from zevo.providers import resolve_ssh_key

logger = logging.getLogger(__name__)

VASTAI_API_BASE = "https://console.vast.ai/api/v0"
# Vast.ai retired /api/v0/instances/ (410 "deprecated_endpoint"); the rest of
# the v0 surface we use still answers, so only this collection moves.
VASTAI_INSTANCES_URL = "https://console.vast.ai/api/v1/instances/"
_DEFAULT_API_KEY_PATH = Path.home() / ".config" / "vastai" / "vast_api_key"


def _sanitize_key(k: str) -> str:
    """Drop any non-ASCII / whitespace chars from a key. A stray non-ASCII
    character (e.g. a pasted full-width '。' \\u3002) makes httpx raise
    UnicodeEncodeError when building the Authorization header and breaks every
    Vast.ai call, so we scrub it once at the source rather than relying on each
    caller to clean up."""
    return "".join(c for c in (k or "") if c.isascii() and not c.isspace())


def load_api_key(explicit: Optional[str] = None) -> str:
    """Resolve a Vast.ai API key from (in order): the explicit arg,
    $VASTAI_API_KEY env var, ~/.config/vastai/vast_api_key file.
    Raises FileNotFoundError with a clear hint if none of those work.
    """
    if explicit:
        return _sanitize_key(explicit)
    env = os.environ.get("VASTAI_API_KEY", "").strip()
    if env:
        return _sanitize_key(env)
    if _DEFAULT_API_KEY_PATH.exists():
        return _sanitize_key(_DEFAULT_API_KEY_PATH.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No Vast.ai API key found. Set $VASTAI_API_KEY or write the key to "
        f"{_DEFAULT_API_KEY_PATH} (the location the `vastai` CLI uses). "
        f"Get the key at https://cloud.vast.ai/account/."
    )


class VastAIProvider:
    """Interact with the Vast.ai API to search, rent, and manage GPU instances."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = load_api_key(api_key)
        self._headers = {"Authorization": f"Bearer {self.api_key}"}

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict | list:
        """Authenticated request. Opens a fresh client per call -- cheap for the
        infrequent rate of marketplace ops; avoids long-lived state in tools."""
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.request(
                method,
                f"{VASTAI_API_BASE}{path}",
                headers=self._headers,
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()

    # -----------------------------------------------------------------
    # Marketplace
    # -----------------------------------------------------------------

    async def search_gpus(
        self,
        min_gpu_ram_gb: float = 16.0,
        gpu_type: Optional[str] = None,
        num_gpus: int = 1,
        max_dph: Optional[float] = None,
        order: str = "dph_total",
        limit: int = 20,
        min_cpu_ram_gb: float = 0.0,
        min_cpu_cores: int = 0,
    ) -> list[dict]:
        """Search Vast.ai for available GPU offers. Sorted by price ascending.

        `min_cpu_ram_gb` / `min_cpu_cores` are the HOST side, and they default
        to unfiltered because that is what this did before they existed. That
        default is worth knowing about: results are ordered by price ascending,
        so with no host filter the cheapest box clearing the GPU bar wins, and
        on this marketplace that is regularly 4 cores and 16GB behind a 24GB
        GPU. Fine for a small LoRA; not fine for a job that stages a 14B model
        through host memory or tokenizes a large corpus.
        """
        # Vast.ai's marketplace query field `gpu_ram` is in MEGABYTES
        # (callers think in GB; without this conversion the filter is a
        # no-op because every offer's gpu_ram value is in the thousands).
        def _build_query(verified: bool) -> dict[str, Any]:
            q: dict[str, Any] = {
                "gpu_ram": {"gte": int(min_gpu_ram_gb * 1024)},
                "num_gpus": {"eq": num_gpus},
                "rentable": {"eq": True},
                "rented": {"eq": False},
                "type": "on-demand",
                "order": [[order, "asc"]],
                # Pull a wide slice so verified hosts (usually pricier than the
                # very cheapest junk) actually appear before we trim.
                "limit": max(limit, 64),
            }
            if verified:
                q["verified"] = {"eq": True}
            if gpu_type:
                q["gpu_name"] = {"eq": gpu_type}
            # Vast reports host RAM in MEGABYTES under `cpu_ram`, the same
            # units trap as `gpu_ram` above. `cpu_cores_effective` is the
            # share of cores this offer actually gets, which is the number
            # that matters on a shared host — `cpu_cores` is the machine's.
            if min_cpu_ram_gb:
                q["cpu_ram"] = {"gte": int(min_cpu_ram_gb * 1024)}
            if min_cpu_cores:
                q["cpu_cores_effective"] = {"gte": int(min_cpu_cores)}
            if max_dph:
                q["dph_total"] = {"lte": max_dph}
            return q

        async def _fetch(verified: bool) -> list[dict]:
            data = await self._request(
                "GET", "/bundles/", params={"q": _json.dumps(_build_query(verified))},
            )
            offers = data.get("offers", []) if isinstance(data, dict) else data
            return [_normalize_offer(o) for o in (offers or [])]

        try:
            # Ultra-cheap Vast.ai offers (e.g. $0.02/hr) frequently never boot to
            # SSH-ready: they sit on UNVERIFIED hosts (a high reliability2 score
            # is NOT enough — the broken $0.02 V100 reports reliability 0.99).
            # The raw API sort is price-ascending, so a caller taking the
            # cheapest lands on exactly those duds. Ask Vast.ai for VERIFIED
            # hosts first; among those return cheapest-first to still minimise
            # cost. Only if zero verified hosts match do we fall back to the
            # unverified pool (ranked by reliability) so we never return empty.
            verified = await _fetch(True)
            if verified:
                verified.sort(key=lambda o: o["dph_total"])  # cheapest verified first
                return verified[:limit]
            pool = await _fetch(False)
            pool.sort(key=lambda o: (-o["reliability"], o["dph_total"]))
            return pool[:limit]
        except Exception as e:
            logger.error(f"Failed to search Vast.ai GPUs: {e}")
            raise

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def create_instance(
        self,
        offer_id: int,
        docker_image: str,
        disk_gb: int = 50,
        env: Optional[dict[str, str]] = None,
        onstart_cmd: Optional[str] = None,
    ) -> dict:
        """Rent a GPU. Returns {instance_id, success}.

        SAFETY CAP: refuses to rent when the account already holds
        >= ZEVO_VASTAI_MAX_INSTANCES (default 1) active instances. Without
        this, a runaway infra agent or ticket re-run loop rents box after
        box and burns money (observed: 3 V100s rented across one failed
        run). This is the last line of defense regardless of how the
        caller misbehaves. Raise ZEVO_VASTAI_MAX_INSTANCES if you truly
        need concurrent rentals.
        """
        max_active = int(os.environ.get("ZEVO_VASTAI_MAX_INSTANCES", "1"))
        try:
            existing = await self.list_instances()
        except Exception as e:
            logger.warning(f"Vast.ai safety check could not list instances: {e}")
            existing = []
        active = [
            i for i in existing
            if str(i.get("actual_status", "")).lower() not in ("exited", "destroyed")
        ]
        if len(active) >= max_active:
            ids = [i.get("id") for i in active]
            raise RuntimeError(
                f"Vast.ai safety cap hit: {len(active)} active instance(s) "
                f"already rented {ids}; limit ZEVO_VASTAI_MAX_INSTANCES="
                f"{max_active}. Refusing to rent another -- destroy existing "
                f"instances first."
            )

        payload: dict[str, Any] = {
            "client_id": "me",
            "image": docker_image,
            "disk": disk_gb,
            "runtype": "ssh",
        }
        if env:
            payload["env"] = env
        if onstart_cmd:
            payload["onstart"] = onstart_cmd

        try:
            data = await self._request("PUT", f"/asks/{offer_id}/", json=payload)
            instance_id = data.get("new_contract")
            if not instance_id:
                raise RuntimeError(f"No instance ID in response: {data}")
            return {"instance_id": str(instance_id), "success": bool(data.get("success", True))}
        except Exception as e:
            logger.error(f"Failed to create Vast.ai instance: {e}")
            raise

    async def get_instance(self, instance_id: str) -> dict:
        """Fetch one instance's details."""
        try:
            data = await self._request("GET", f"/instances/{instance_id}/")
            instances = data.get("instances", [data]) if isinstance(data, dict) else data
            if not instances:
                raise RuntimeError(f"Instance {instance_id} not found")
            inst = instances[0] if isinstance(instances, list) else instances
            return {
                "id": str(inst.get("id")),
                "status": inst.get("actual_status", inst.get("status_msg", "unknown")),
                "ssh_host": inst.get("ssh_host", ""),
                "ssh_port": inst.get("ssh_port", 0),
                "gpu_name": inst.get("gpu_name", ""),
                "num_gpus": inst.get("num_gpus", 0),
                "dph_total": float(inst.get("dph_total") or 0.0),
                "image_uuid": inst.get("image_uuid", ""),
            }
        except Exception as e:
            logger.error(f"Failed to get Vast.ai instance {instance_id}: {e}")
            raise

    async def list_instances(self) -> list[dict]:
        """List all instances currently on this account (raw Vast.ai dicts)."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                VASTAI_INSTANCES_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            data = r.json()
        instances = data.get("instances", []) if isinstance(data, dict) else []
        return instances if isinstance(instances, list) else []

    async def destroy_instance(self, instance_id: str) -> bool:
        """Destroy (terminate) a rented instance. Returns True on success."""
        try:
            await self._request("DELETE", f"/instances/{instance_id}/")
            logger.info(f"Destroyed Vast.ai instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy Vast.ai instance {instance_id}: {e}")
            return False

    async def get_ssh_details(self, instance_id: str) -> dict:
        """{ssh_host, ssh_port, instance_id}."""
        info = await self.get_instance(instance_id)
        return {
            "ssh_host": info.get("ssh_host", ""),
            "ssh_port": int(info.get("ssh_port") or 0),
            "instance_id": instance_id,
        }

    async def wait_for_ready(
        self,
        instance_id: str,
        timeout: int = 300,
        poll_interval: int = 10,
    ) -> dict:
        """Poll until status == 'running'. API-ready is NOT the same as SSH-ready;
        callers should additionally poll SSH after this returns."""
        elapsed = 0
        info: dict = {}
        while elapsed < timeout:
            info = await self.get_instance(instance_id)
            status = info.get("status", "")
            if status == "running":
                return info
            if status in ("exited", "error"):
                raise RuntimeError(f"Instance {instance_id} entered terminal state: {status}")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(
            f"Instance {instance_id} not ready after {timeout}s (last status: {info.get('status')})"
        )

    async def wait_for_ssh(
        self,
        *,
        host: str,
        port: int,
        user: str = "root",
        key_path: str = "",
        timeout: int = 420,
        poll_interval: int = 10,
    ) -> bool:
        """Poll until the box ACCEPTS an SSH connection.

        A Vast.ai instance reports status='running' (see wait_for_ready) a while
        before sshd is actually accepting connections, so callers must also wait
        on real SSH. Crucially this waits in PYTHON (asyncio.sleep + a subprocess
        ssh probe) rather than a shell `sleep`-and-retry loop, which some agent
        sandboxes BLOCK — that block is what turns a slow boot into a spurious
        provisioning failure. Returns True once reachable, False on timeout.
        """
        key_path = key_path or resolve_ssh_key()
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


def _normalize_offer(o: dict) -> dict:
    """Pull a stable subset of fields out of a Vast.ai bundles/offers row."""
    gpu_ram = o.get("gpu_ram", 0)
    return {
        "id": o.get("id"),
        "gpu_name": o.get("gpu_name", "Unknown"),
        "num_gpus": o.get("num_gpus", 1),
        # Vast.ai returns gpu_ram in MB for most rows; if it looks like MB, convert to GB.
        "gpu_ram_gb": round(gpu_ram / 1024, 1) if gpu_ram > 100 else float(gpu_ram),
        "cpu_cores": o.get("cpu_cores_effective", 0),
        "ram_gb": round(o.get("cpu_ram", 0) / 1024, 1),
        "disk_gb": round(o.get("disk_space", 0), 0),
        "dph_total": round(o.get("dph_total", 0), 4),
        "reliability": round(o.get("reliability2", 0), 3),
        "inet_down": round(o.get("inet_down", 0), 1),
        "inet_up": round(o.get("inet_up", 0), 1),
        "cuda_max_good": o.get("cuda_max_good"),
        "machine_id": o.get("machine_id", o.get("id")),
        "verified": o.get("verification", "") == "verified",
        "geolocation": o.get("geolocation", "") or "",
        "dlperf": round(o.get("dlperf", 0) or 0, 1),
    }
