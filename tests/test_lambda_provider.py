"""Unit tests for zevo.providers.lambda_labs.LambdaCloudProvider.

Lambda is the second `cloud`-mode backend. Its API differs from Vast.ai in ways
that would silently break provisioning if a regression crept in:
  - offers come from /instance-types keyed by name, with per-region capacity —
    we must drop types with NO capacity and emit one offer per available region;
  - launch needs (region_name, instance_type_name, ssh_key_names[one]) and a
    PRE-REGISTERED account key;
  - the ready state is 'active' (not Vast's 'running'); login user is 'ubuntu'.

We mock the provider's single HTTP chokepoint `_request` so no network is hit,
and pin the request shapes + response normalisation the infra agent relies on.
"""
from __future__ import annotations

import pytest

from zevo.providers.lambda_labs.provider import (
    LambdaCloudProvider, _parse_vram_gb, _normalize_offer,
    LAMBDA_SSH_USER, LAMBDA_SSH_PORT,
)


# instance-types payload with a mix a real account sees: a type with capacity,
# a same-type-different-region duplicate, a NO-capacity type, and a 8x box.
_INSTANCE_TYPES = {
    "data": {
        "gpu_1x_a10": {
            "instance_type": {
                "name": "gpu_1x_a10",
                "description": "1x A10 (24 GB PCIe)",
                "gpu_description": "A10 (24 GB PCIe)",
                "price_cents_per_hour": 75,
                "specs": {"vcpus": 30, "memory_gib": 200, "storage_gib": 1400, "gpus": 1},
            },
            "regions_with_capacity_available": [
                {"name": "us-west-1", "description": "California, USA"},
                {"name": "us-east-1", "description": "Virginia, USA"},
            ],
        },
        "gpu_1x_h100_pcie": {
            "instance_type": {
                "name": "gpu_1x_h100_pcie",
                "description": "1x H100 (80 GB PCIe)",
                "gpu_description": "H100 (80 GB PCIe)",
                "price_cents_per_hour": 249,
                "specs": {"vcpus": 26, "memory_gib": 200, "storage_gib": 1024, "gpus": 1},
            },
            "regions_with_capacity_available": [],  # sold out -> must be dropped
        },
        "gpu_8x_a100_80gb_sxm4": {
            "instance_type": {
                "name": "gpu_8x_a100_80gb_sxm4",
                "description": "8x A100 (80 GB SXM4)",
                "gpu_description": "A100 (80 GB SXM4)",
                "price_cents_per_hour": 1436,
                "specs": {"vcpus": 240, "memory_gib": 1800, "storage_gib": 20000, "gpus": 8},
            },
            "regions_with_capacity_available": [{"name": "us-south-1", "description": "Texas"}],
        },
    }
}


def _provider_with(monkeypatch, responses: dict):
    """Build a provider whose `_request` returns canned payloads by (method, path).
    `responses` maps 'METHOD /path' (path startswith match) -> dict, or a callable
    (method, path, kwargs) -> dict for asserting request bodies."""
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_test_" + "a" * 20)
    p = LambdaCloudProvider()
    calls: list[tuple[str, str, dict]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        for key, val in responses.items():
            m, pfx = key.split(" ", 1)
            if m == method and path.startswith(pfx):
                return val(method, path, kwargs) if callable(val) else val
        raise AssertionError(f"unexpected request {method} {path}")

    p._request = fake_request  # type: ignore[assignment]
    return p, calls


# ─────────────────────────── helpers ────────────────────────────────────────

def test_parse_vram_gb_reads_various_descriptions() -> None:
    assert _parse_vram_gb("A10 (24 GB PCIe)") == 24.0
    assert _parse_vram_gb("H100 (80 GB SXM5)") == 80.0
    assert _parse_vram_gb("1x A6000 (48GB)") == 48.0
    assert _parse_vram_gb("no size here") == 0.0


def test_normalize_offer_shapes_like_vast() -> None:
    it = _INSTANCE_TYPES["data"]["gpu_1x_a10"]["instance_type"]
    off = _normalize_offer("gpu_1x_a10", it, it["specs"], 24.0, 0.75,
                           {"name": "us-west-1", "description": "California"})
    assert off["id"] == "gpu_1x_a10"        # -> create_instance(offer_id=...)
    assert off["region"] == "us-west-1"     # -> create_instance(region=...)
    assert off["num_gpus"] == 1
    assert off["gpu_ram_gb"] == 24.0
    assert off["dph_total"] == 0.75
    assert off["verified"] is True


# ─────────────────────────── search_gpus ────────────────────────────────────

@pytest.mark.asyncio
async def test_search_drops_no_capacity_and_expands_regions(monkeypatch) -> None:
    p, _ = _provider_with(monkeypatch, {"GET /instance-types": _INSTANCE_TYPES})
    offers = await p.search_gpus(min_gpu_ram_gb=16, num_gpus=1)
    # h100 has no capacity -> gone. a10 has 2 regions -> 2 offers. 8x excluded by num_gpus=1.
    ids = [(o["id"], o["region"]) for o in offers]
    assert ("gpu_1x_a10", "us-west-1") in ids
    assert ("gpu_1x_a10", "us-east-1") in ids
    assert all(o["id"] != "gpu_1x_h100_pcie" for o in offers)
    assert all(o["id"] != "gpu_8x_a100_80gb_sxm4" for o in offers)


@pytest.mark.asyncio
async def test_search_num_gpus_and_vram_filters(monkeypatch) -> None:
    p, _ = _provider_with(monkeypatch, {"GET /instance-types": _INSTANCE_TYPES})
    # num_gpus=8 -> only the 8x box (has capacity in us-south-1)
    eight = await p.search_gpus(min_gpu_ram_gb=40, num_gpus=8)
    assert [o["id"] for o in eight] == ["gpu_8x_a100_80gb_sxm4"]
    # min_gpu_ram_gb=48 with num_gpus=1 -> a10 (24GB) filtered out entirely
    big1 = await p.search_gpus(min_gpu_ram_gb=48, num_gpus=1)
    assert big1 == []


@pytest.mark.asyncio
async def test_search_sorts_cheapest_first(monkeypatch) -> None:
    p, _ = _provider_with(monkeypatch, {"GET /instance-types": _INSTANCE_TYPES})
    offers = await p.search_gpus(min_gpu_ram_gb=0, num_gpus=0)  # 0 = no count filter
    prices = [o["dph_total"] for o in offers]
    assert prices == sorted(prices)


# ─────────────────────────── create_instance ────────────────────────────────

@pytest.mark.asyncio
async def test_create_instance_posts_launch_body_and_returns_id(monkeypatch) -> None:
    captured = {}

    def launch(method, path, kwargs):
        captured.update(kwargs.get("json", {}))
        return {"data": {"instance_ids": ["i-abc123"]}}

    p, _ = _provider_with(monkeypatch, {
        "GET /instances": {"data": []},                 # safety-cap list
        "GET /ssh-keys": {"data": [{"name": "mykey", "public_key": "ssh-ed25519 AAAAC3 body"}]},
        "POST /instance-operations/launch": launch,
    })
    # pin the ssh key name so we don't touch the filesystem for a .pub
    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "mykey")

    res = await p.create_instance(offer_id="gpu_1x_a10", region="us-west-1")
    assert res == {"instance_id": "i-abc123", "success": True}
    assert captured["region_name"] == "us-west-1"
    assert captured["instance_type_name"] == "gpu_1x_a10"
    assert captured["ssh_key_names"] == ["mykey"]        # exactly one, pre-registered


@pytest.mark.asyncio
async def test_create_instance_safety_cap_refuses_second_box(monkeypatch) -> None:
    p, _ = _provider_with(monkeypatch, {
        "GET /instances": {"data": [{"id": "i-live", "status": "active"}]},
    })
    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "mykey")
    monkeypatch.setenv("ZEVO_LAMBDA_MAX_INSTANCES", "1")
    with pytest.raises(RuntimeError, match="safety cap"):
        await p.create_instance(offer_id="gpu_1x_a10", region="us-west-1")


# ─────────────────────────── lifecycle ──────────────────────────────────────

@pytest.mark.asyncio
async def test_get_instance_normalises_ip_and_ubuntu_user(monkeypatch) -> None:
    inst = {"data": {
        "id": "i-abc", "status": "active", "ip": "203.0.113.7",
        "instance_type": {"gpu_description": "A10 (24 GB PCIe)",
                          "price_cents_per_hour": 75, "specs": {"gpus": 1}},
        "region": {"name": "us-west-1"},
    }}
    p, _ = _provider_with(monkeypatch, {"GET /instances/i-abc": inst})
    info = await p.get_instance("i-abc")
    assert info["ssh_host"] == "203.0.113.7"
    assert info["ssh_port"] == LAMBDA_SSH_PORT == 22
    assert info["ssh_user"] == LAMBDA_SSH_USER == "ubuntu"
    assert info["dph_total"] == 0.75
    assert info["num_gpus"] == 1


@pytest.mark.asyncio
async def test_wait_for_ready_returns_on_active(monkeypatch) -> None:
    inst = {"data": {"id": "i-abc", "status": "active",
                     "instance_type": {"specs": {"gpus": 1}}, "region": {}}}
    p, _ = _provider_with(monkeypatch, {"GET /instances/i-abc": inst})
    info = await p.wait_for_ready("i-abc", timeout=5, poll_interval=1)
    assert str(info["status"]).lower() == "active"


@pytest.mark.asyncio
async def test_wait_for_ready_raises_on_terminated(monkeypatch) -> None:
    inst = {"data": {"id": "i-abc", "status": "terminated",
                     "instance_type": {"specs": {"gpus": 1}}, "region": {}}}
    p, _ = _provider_with(monkeypatch, {"GET /instances/i-abc": inst})
    with pytest.raises(RuntimeError, match="terminal state"):
        await p.wait_for_ready("i-abc", timeout=5, poll_interval=1)


@pytest.mark.asyncio
async def test_destroy_posts_terminate_with_id_list(monkeypatch) -> None:
    captured = {}

    def terminate(method, path, kwargs):
        captured.update(kwargs.get("json", {}))
        return {"data": {"terminated_instances": [{"id": "i-abc"}]}}

    p, _ = _provider_with(monkeypatch, {"POST /instance-operations/terminate": terminate})
    ok = await p.destroy_instance("i-abc")
    assert ok is True
    assert captured == {"instance_ids": ["i-abc"]}


@pytest.mark.asyncio
async def test_resolve_ssh_key_name_reuses_existing_by_public_key(monkeypatch, tmp_path) -> None:
    key = tmp_path / "id_ed25519"
    key.write_text("PRIVATE")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAAC3NzaC1 body user@host\n")
    p, _ = _provider_with(monkeypatch, {
        "GET /ssh-keys": {"data": [
            {"name": "already-there", "public_key": "ssh-ed25519 AAAAC3NzaC1 different-comment"},
        ]},
    })
    monkeypatch.delenv("LAMBDA_SSH_KEY_NAME", raising=False)
    name = await p.resolve_ssh_key_name(str(key))
    # matched on type+body (ignoring comment) -> reused, NOT re-registered
    assert name == "already-there"
