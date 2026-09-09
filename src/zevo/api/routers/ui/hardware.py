"""GET /hardware, POST /hardware/detect-local, and the cloud-GPU routes.

The cloud routes are parameterised by backend — `vastai` or `lambda` — because
the harness supports renting from either, and a page that can only search one
of them cannot answer "where should this run train?". Both providers already
return offers in the same normalized shape, so the only per-backend work here
is picking the class and mapping its offer fields.

Renting costs real money and is rate-limited at the provider's end; the UI
surfaces a confirm step before POST /rent.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from zevo.contracts._base import StrictBody

from zevo.providers.lambda_labs.provider import LambdaCloudProvider
from zevo.providers.vastai import VastAIProvider, load_api_key


router = APIRouter()


# ---------- local GPU probe ------------------------------------------------

class LocalGPUInfo(BaseModel):
    has_gpu: bool
    gpu_count: int
    gpu_name: str
    vram_gb: float
    driver_version: str
    cuda_version: str
    source: str
    error: str = ""


@router.post("/hardware/detect-local", response_model=LocalGPUInfo)
async def detect_local() -> LocalGPUInfo:
    """Run nvidia-smi if available; report what we find."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
            first = lines[0].split(",")
            name = first[0].strip()
            vram_mb = float(first[1].strip())
            driver = first[2].strip()
            # CUDA version from a second nvidia-smi call
            cuda = ""
            r2 = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            try:
                rr = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                for ln in rr.stdout.splitlines():
                    if "CUDA Version" in ln:
                        cuda = ln.split("CUDA Version:")[-1].strip().split()[0]
                        break
            except Exception:
                pass
            return LocalGPUInfo(
                has_gpu=True, gpu_count=len(lines), gpu_name=name,
                vram_gb=round(vram_mb / 1024, 1),
                driver_version=driver, cuda_version=cuda,
                source="nvidia-smi",
            )
    except FileNotFoundError:
        pass
    except Exception as e:
        return LocalGPUInfo(
            has_gpu=False, gpu_count=0, gpu_name="", vram_gb=0.0,
            driver_version="", cuda_version="",
            source="error", error=f"{type(e).__name__}: {e}",
        )

    return LocalGPUInfo(
        has_gpu=False, gpu_count=0, gpu_name="", vram_gb=0.0,
        driver_version="", cuda_version="", source="absent",
        error="nvidia-smi not found (no NVIDIA driver installed)",
    )


# ---------- cloud GPUs: vast.ai or lambda.ai -------------------------------

# Both providers, by the id the rest of the system uses for them
# (ZEVO_CLOUD_BACKEND, gpu_provider blocks, the Settings page).
BACKENDS: dict[str, dict[str, Any]] = {
    "vastai": {"label": "Vast.ai", "cls": VastAIProvider},
    "lambda": {"label": "Lambda.ai", "cls": LambdaCloudProvider},
}


def _provider(backend: str) -> Any:
    spec = BACKENDS.get(backend)
    if spec is None:
        raise HTTPException(404, f"unknown cloud backend {backend!r}; expected one of {sorted(BACKENDS)}")
    try:
        return spec["cls"]()
    except Exception as e:
        raise HTTPException(503, f"{spec['label']} not configured: {e}")


@router.get("/hardware/cloud/backends")
async def cloud_backends() -> dict[str, Any]:
    """Which clouds this build supports, and whether each one has a key.

    `configured` drives the UI: an unconfigured backend is still listed, so the
    page can say "set a key" rather than pretend the provider does not exist.
    """
    out = []
    for bid, spec in BACKENDS.items():
        try:
            spec["cls"]()
            configured = True
        except Exception:
            configured = False
        out.append({"id": bid, "label": spec["label"], "configured": configured})
    return {"backends": out}


class CloudOffer(BaseModel):
    """One rentable machine, in the shape both providers normalize to.

    `id` is a string because Vast.ai keys offers by integer id while Lambda
    keys them by instance-type name. `region` and `dlperf` are only meaningful
    for one provider each; the other leaves them empty.
    """

    id: str
    gpu_name: str
    num_gpus: int
    gpu_ram_gb: float
    dph_total: float
    cpu_cores: int
    ram_gb: float
    disk_space_gb: float
    reliability: float
    location: str = ""
    dlperf: float = 0.0
    region: str = ""


class SearchRequest(StrictBody):
    min_vram_gb: float = 16.0
    gpu_type: str = ""
    num_gpus: int = 1
    max_dph: float = 0.0  # 0 = no cap
    limit: int = Query(10, ge=1, le=100)


def _offer_from(o: dict) -> CloudOffer:
    """Map a provider offer onto CloudOffer.

    One mapping serves both clouds: `search_gpus` normalizes before returning,
    so a Vast.ai offer and a Lambda offer arrive with the same keys. The router
    used to read Vast.ai's RAW field names here (`gpu_ram`, `cpu_ram`,
    `reliability2`), which the normalizer had already renamed — every row
    rendered 0 GB / 0 cores / 0% reliability.
    """
    return CloudOffer(
        id=str(o.get("id", "")),
        gpu_name=str(o.get("gpu_name", "")),
        num_gpus=int(o.get("num_gpus", 1) or 1),
        gpu_ram_gb=float(o.get("gpu_ram_gb", 0.0) or 0.0),
        dph_total=float(o.get("dph_total", 0.0) or 0.0),
        cpu_cores=int(o.get("cpu_cores", 0) or 0),
        ram_gb=float(o.get("ram_gb", 0.0) or 0.0),
        disk_space_gb=float(o.get("disk_gb", 0.0) or 0.0),
        reliability=float(o.get("reliability", 0.0) or 0.0),
        # Vast.ai reports a geolocation string; Lambda a named region.
        location=str(o.get("geolocation", "") or o.get("region_desc", "") or o.get("region", "")),
        dlperf=float(o.get("dlperf", 0.0) or 0.0),
        region=str(o.get("region", "") or ""),
    )


@router.post("/hardware/cloud/{backend}/search", response_model=list[CloudOffer])
async def cloud_search(backend: str, req: SearchRequest) -> list[CloudOffer]:
    provider = _provider(backend)
    label = BACKENDS[backend]["label"]
    try:
        offers = await provider.search_gpus(
            min_gpu_ram_gb=req.min_vram_gb,
            gpu_type=req.gpu_type or None,
            num_gpus=req.num_gpus,
            max_dph=req.max_dph if req.max_dph > 0 else None,
            limit=req.limit,
        )
    except Exception as e:
        raise HTTPException(502, f"{label} search failed: {e}")
    return [_offer_from(o) for o in offers]


class RentRequest(StrictBody):
    offer_id: str
    docker_image: str = "nvidia/cuda:12.1.1-devel-ubuntu22.04"
    disk_gb: int = 50
    region: str = ""     # Lambda needs one; Vast.ai ignores it


class RentResponse(BaseModel):
    instance_id: str
    success: bool
    detail: dict[str, Any]


@router.post("/hardware/cloud/{backend}/rent", response_model=RentResponse)
async def cloud_rent(backend: str, req: RentRequest) -> RentResponse:
    provider = _provider(backend)
    label = BACKENDS[backend]["label"]
    try:
        if backend == "vastai":
            result = await provider.create_instance(
                offer_id=int(req.offer_id),
                docker_image=req.docker_image,
                disk_gb=req.disk_gb,
            )
        else:
            result = await provider.create_instance(
                offer_id=req.offer_id,
                region=req.region,
            )
    except Exception as e:
        raise HTTPException(502, f"{label} rent failed: {e}")
    return RentResponse(
        instance_id=str(result.get("instance_id", "")),
        success=bool(result.get("success", False)),
        detail=result if isinstance(result, dict) else {},
    )


@router.delete("/hardware/cloud/{backend}/instances/{instance_id}")
async def cloud_destroy(backend: str, instance_id: str) -> dict[str, Any]:
    provider = _provider(backend)
    ok = await provider.destroy_instance(instance_id)
    return {"instance_id": instance_id, "destroyed": ok}


@router.get("/hardware/cloud/{backend}/instances")
async def cloud_instances(backend: str) -> dict[str, Any]:
    """Current rentals for whichever key this backend is configured with."""
    provider = _provider(backend)
    label = BACKENDS[backend]["label"]
    try:
        raw = await provider.list_instances()
    except Exception as e:
        raise HTTPException(502, f"{label} list failed: {e}")

    out = []
    for i in raw or []:
        if backend == "vastai":
            out.append({
                "id": str(i.get("id", "")),
                "gpu_name": i.get("gpu_name", ""),
                "actual_status": i.get("actual_status", ""),
                "dph_total": float(i.get("dph_total", 0.0) or 0.0),
                "ssh_host": i.get("ssh_host", ""),
                "ssh_port": int(i.get("ssh_port", 0) or 0),
            })
        else:
            it = i.get("instance_type", {}) or {}
            out.append({
                "id": str(i.get("id", "")),
                "gpu_name": it.get("gpu_description", it.get("name", "")),
                "actual_status": i.get("status", ""),
                "dph_total": round(float(it.get("price_cents_per_hour", 0) or 0) / 100.0, 4),
                "ssh_host": i.get("ip", ""),
                "ssh_port": 22,
            })
    return {"instances": out}
