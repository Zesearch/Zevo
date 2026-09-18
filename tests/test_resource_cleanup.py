"""Provider failures must not erase ownership or stop billing clocks."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from zevo.db import Base, Run, Ticket, InfraInstance, WorkProduct, GpuLease
from zevo.engine.run import resource_cleanup as cleanup


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(Run(id="r1", metric="accuracy", status="cancelled", gpu_provider="cloud", cloud_backend="vastai"))
        session.add(Ticket(id="infra", run_id="r1", agent_id="infrastructure", status="cancelled", payload={}))
        await session.commit()
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_failed_destroy_stays_visible_and_retries_after_confirmation(db, monkeypatch):
    run = await db.get(Run, "r1")
    row = InfraInstance(run_id=run.id, ticket_id="infra", provider="cloud", instance_id="123", meta={"backend":"vastai"})
    db.add(row)
    await db.commit()
    calls = []
    class Provider:
        gone = False
        async def list_instances(self):
            return [] if self.gone else [{"id":123,"status":"running"}]
        async def destroy_instance(self, identifier):
            calls.append(identifier)
            return False
    provider = Provider()
    monkeypatch.setattr(cleanup, "_cloud_provider", lambda _:provider)
    result = await cleanup.cleanup_run_resources(db, run, force=True)
    assert result[0]["destroyed"] is False
    assert row.released_at is None and row.meta["cleanup_error"]
    assert await cleanup.has_pending_resources(db, run.id)
    # An ordinary reconciliation pass respects durable retry backoff.
    await cleanup.cleanup_run_resources(db, run, force=True)
    assert calls == ["123"]
    provider.gone = True
    result = await cleanup.cleanup_run_resources(db, run, force=True, retry_now=True)
    assert result[0]["destroyed"] is True
    assert row.released_at is not None
    assert not await cleanup.has_pending_resources(db, run.id)


@pytest.mark.asyncio
async def test_provisioning_row_without_device_artifact_is_discovered(db, monkeypatch):
    row = InfraInstance(run_id="r1", ticket_id="infra", provider="cloud", instance_id="99", meta={"backend":"vastai"})
    db.add(row)
    await db.commit()
    seen = []
    async def released(row, run):
        seen.append(row.instance_id)
        return True
    monkeypatch.setattr(cleanup, "release_cloud", released)
    await cleanup.cleanup_run_resources(db, await db.get(Run,"r1"), force=True)
    assert seen == ["99"] and row.released_at is not None


@pytest.mark.asyncio
async def test_artifact_only_resource_gets_durable_retry_row(db, monkeypatch, tmp_path):
    artifact = tmp_path / "device.json"
    artifact.write_text(json.dumps({"run_id":"r1","ticket_id":"infra","provider":"cloud","instance_id":"88","cloud_backend":"vastai"}))
    db.add(WorkProduct(ticket_id="infra", role="device_info", path=str(artifact), meta={}))
    await db.commit()
    async def fail(row, run):
        raise TimeoutError("provider unavailable")
    monkeypatch.setattr(cleanup, "release_cloud", fail)
    results = await cleanup.cleanup_run_resources(db, await db.get(Run,"r1"), force=True)
    assert results[0]["destroyed"] is False
    assert await cleanup.has_pending_resources(db,"r1")


@pytest.mark.asyncio
async def test_slurm_failure_is_not_recorded_as_release(db, monkeypatch):
    row = InfraInstance(run_id="r1", ticket_id="infra", provider="cluster", instance_id="55", meta={})
    db.add(row)
    await db.commit()
    async def fail(*args): return False
    monkeypatch.setattr(cleanup, "release_cluster", fail)
    await cleanup.cleanup_run_resources(db, await db.get(Run,"r1"), force=True)
    assert row.released_at is None


@pytest.mark.asyncio
async def test_delete_keeps_run_when_provider_still_billing(db, monkeypatch):
    from zevo.api.routers.shared.runs import delete_run
    row = InfraInstance(run_id="r1", ticket_id="infra", provider="cloud", instance_id="77", meta={})
    db.add(row)
    await db.commit()
    async def fail(*args): return False
    monkeypatch.setattr(cleanup, "release_cloud", fail)
    with pytest.raises(HTTPException) as exc:
        await delete_run("r1", db)
    assert exc.value.status_code == 409
    assert await db.get(Run,"r1") is not None
    assert row.released_at is None


@pytest.mark.asyncio
async def test_agent_authored_remote_path_never_becomes_a_delete_command(db, monkeypatch, tmp_path):
    artifact = tmp_path / "device.json"
    artifact.write_text(json.dumps({"run_id":"r1","ticket_id":"infra","provider":"instance","instance_id":"host","instance":{"workdir":"/shared/r1/../"},"ssh":{"host":"example"}}))
    db.add(WorkProduct(ticket_id="infra",role="device_info",path=str(artifact),meta={}))
    await db.commit()
    async def unexpected(*args, **kwargs):
        raise AssertionError("untrusted remote directory must never be deleted")
    monkeypatch.setattr(cleanup.asyncio, "create_subprocess_exec", unexpected)
    await cleanup.cleanup_run_resources(db, await db.get(Run,"r1"), force=True)


@pytest.mark.asyncio
async def test_terminating_cloud_instance_counts_as_released(db, monkeypatch):
    """Lambda reports a destroyed box as 'terminating' for a while; that is a
    confirmed release, not a reason to keep re-sending terminate."""
    run = await db.get(Run, "r1")
    row = InfraInstance(run_id="r1", ticket_id="infra", provider="cloud", instance_id="123", meta={"backend": "vastai"})
    calls = []

    class Provider:
        async def list_instances(self):
            return [{"id": "123", "status": "terminating"}]

        async def destroy_instance(self, iid):
            calls.append(iid)

    monkeypatch.setattr(cleanup, "_cloud_provider", lambda *_: Provider())
    assert await cleanup.release_cloud(row, run) is True
    assert calls == []
