"""The shipped cloud acquisition sequence: one implementation of create ->
bookkeeping -> ready -> SSH -> probe -> health gate -> cleanup, so agents stop
rewriting it per activation. Three rewrites on 2026-10-01 each destroyed a
healthy rental (unit slip, mis-spelled nvidia-smi section, ssh["host"] for
ssh_host)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HELPER = Path(__file__).resolve().parents[1] / "playbook" / "runners" / "cloud_acquire.py"
spec = importlib.util.spec_from_file_location("zevo_cloud_acquire", HELPER)
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)  # type: ignore[union-attr]


class FakeProvider:
    def __init__(self, *, ssh_ok=True, details=None, fail_create=False):
        self.ssh_ok = ssh_ok
        self.details = details or {"ssh_host": "203.0.113.5", "ssh_port": 22, "ssh_user": "ubuntu"}
        self.fail_create = fail_create
        self.destroyed: list[str] = []
        self.instances = []

    async def create_instance(self, *a, **k):
        if self.fail_create:
            raise RuntimeError("capacity")
        self.instances.append({"id": "i-1", "status": "active"})
        return {"instance_id": "i-1", "success": True}

    async def wait_for_ready(self, instance_id, timeout=0):
        return {"status": "active", "ssh_host": self.details["ssh_host"]}

    async def get_ssh_details(self, instance_id):
        return dict(self.details, instance_id=instance_id)

    async def wait_for_ssh(self, **k):
        return self.ssh_ok

    async def destroy_instance(self, instance_id):
        self.destroyed.append(instance_id)
        self.instances = [dict(i, status="terminating") for i in self.instances]
        return True

    async def list_instances(self):
        return self.instances


def _args(tmp_path, **over):
    base = dict(backend="lambda", offer_id="gpu_1x_a10", region="us-east-1", docker_image="", disk_gb=0,
                ssh_key_path="/tmp/k", name="zevo-test", run_id="r1", ticket_id="infra-r1-001", dph=1.29,
                state=str(tmp_path / "acquire_state.json"), ready_timeout=5, ssh_timeout=5)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def api(monkeypatch):
    calls: list[tuple[str, str, dict]] = []

    def fake_api(method, path, body=None):
        calls.append((method, path, body or {}))
        return {"id": "row-1"}

    monkeypatch.setattr(acquire, "_api", fake_api)
    return calls


def test_ssh_route_reads_the_clients_keys_not_host_or_ip() -> None:
    route = acquire.ssh_route({"ssh_host": "203.0.113.5", "ssh_port": 22, "ssh_user": "ubuntu"}, "lambda")
    assert route == {"host": "203.0.113.5", "port": 22, "user": "ubuntu"}
    with pytest.raises(RuntimeError, match="incomplete"):
        acquire.ssh_route({"host": "203.0.113.5", "ip": "203.0.113.5"}, "lambda")
    vast = acquire.ssh_route({"ssh_host": "ssh4.vast.ai", "ssh_port": 31234}, "vastai")
    assert vast["user"] == "root" and vast["port"] == 31234


def test_probe_parser_reads_nvidia_smi_csv() -> None:
    devices = acquire.parse_probe("0, NVIDIA A10, 23028\n1, NVIDIA A10, 23028\n")
    assert devices == [{"index": 0, "name": "NVIDIA A10", "vram_mb": 23028},
                       {"index": 1, "name": "NVIDIA A10", "vram_mb": 23028}]


@pytest.mark.asyncio
async def test_healthy_path_registers_then_marks_the_row_ready(tmp_path, monkeypatch, api) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(acquire, "make_provider", lambda backend: provider)
    monkeypatch.setattr(acquire, "probe", lambda route, key: {
        "devices": [{"index": 0, "name": "NVIDIA A10", "vram_mb": 23028}],
        "driver_version": "570.148.08", "cuda_version": "12.8"})
    monkeypatch.setattr(acquire, "health", lambda route, key, idx: {"ok": True, "defects": [], "evidence": [], "exit": 0})
    rc = await acquire.run(_args(tmp_path))
    assert rc == 0
    state = json.loads((tmp_path / "acquire_state.json").read_text())
    assert state["phase"] == "ready" and state["row_id"] == "row-1"
    assert state["ssh"] == {"host": "203.0.113.5", "port": 22, "user": "ubuntu"}
    methods = [(m, p) for m, p, _ in api]
    assert methods[0] == ("POST", "/api/infra/instances")
    assert api[0][2]["status"] == "provisioning" and api[0][2]["meta"]["auto_release"] is True
    assert methods[-1] == ("PATCH", "/api/infra/instances/row-1") and api[-1][2]["status"] == "ready"
    assert api[-1][2]["vram_gb"] == 22 and provider.destroyed == []


@pytest.mark.asyncio
async def test_defective_gpu_is_destroyed_released_and_exits_2(tmp_path, monkeypatch, api) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(acquire, "make_provider", lambda backend: provider)
    monkeypatch.setattr(acquire, "probe", lambda route, key: {
        "devices": [{"index": 0, "name": "NVIDIA A10", "vram_mb": 23028}], "driver_version": "570", "cuda_version": "12.8"})
    monkeypatch.setattr(acquire, "health", lambda route, key, idx: {
        "ok": False, "defects": ["idx0: Remapping Failure Occurred : Yes"], "evidence": [], "exit": 1})
    rc = await acquire.run(_args(tmp_path))
    assert rc == 2
    state = json.loads((tmp_path / "acquire_state.json").read_text())
    assert state["error"].startswith("GPU hardware defect:")
    assert provider.destroyed == ["i-1"] and state["destroyed"] is True and state["row_released"] is True
    assert api[-1][2]["status"] == "released" and "GPU hardware defect" in api[-1][2]["release_reason"]


@pytest.mark.asyncio
async def test_ssh_timeout_cleans_up_and_exits_1(tmp_path, monkeypatch, api) -> None:
    provider = FakeProvider(ssh_ok=False)
    monkeypatch.setattr(acquire, "make_provider", lambda backend: provider)
    rc = await acquire.run(_args(tmp_path))
    assert rc == 1
    state = json.loads((tmp_path / "acquire_state.json").read_text())
    assert "SSH did not become ready" in state["error"]
    assert provider.destroyed == ["i-1"] and state["row_released"] is True


@pytest.mark.asyncio
async def test_create_failure_exits_4_without_cleanup(tmp_path, monkeypatch, api) -> None:
    provider = FakeProvider(fail_create=True)
    monkeypatch.setattr(acquire, "make_provider", lambda backend: provider)
    rc = await acquire.run(_args(tmp_path))
    assert rc == 4 and provider.destroyed == [] and api == []


@pytest.mark.asyncio
async def test_missing_ssh_host_from_provider_is_a_cleanup_not_a_hang(tmp_path, monkeypatch, api) -> None:
    provider = FakeProvider(details={"ssh_host": "", "ssh_port": 22, "ssh_user": "ubuntu"})
    monkeypatch.setattr(acquire, "make_provider", lambda backend: provider)
    rc = await acquire.run(_args(tmp_path))
    assert rc == 1 and provider.destroyed == ["i-1"]
