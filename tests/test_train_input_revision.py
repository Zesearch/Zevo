"""Do not confuse cancelled remote work with work that never started."""
from types import SimpleNamespace
from datetime import datetime, timezone
import pytest
from zevo.engine.run import train_input_revision as revision


@pytest.fixture
def probe(monkeypatch):
    async def info(*args): return SimpleNamespace(cluster=SimpleNamespace(env_setup=""))
    monkeypatch.setattr(revision, "_device_info_for_ticket", info)
    def respond(stdout, ok=True):
        async def ssh(*args, **kwargs): return {"ok": ok, "stdout": stdout}
        monkeypatch.setattr(revision, "_ssh", ssh)
    return respond


def resource(**changes):
    values = dict(released_at=datetime.now(timezone.utc), provider="cluster", instance_id="123", meta={})
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt,expected", [
    ("123|CANCELLED by 1000|Unknown|\n", True),
    ("123|CANCELLED|2026-11-20T01:00:00|\n", False),
    ("123|RUNNING|Unknown|\n", False),
    ("123|COMPLETED|Unknown|\n", False),
    ("123|CANCELLED||\n", False),
    ("456|CANCELLED|Unknown|\n", False),
    ("", False),
])
async def test_accounting_must_confirm_exact_job_never_started(probe, receipt, expected):
    probe(receipt)
    assert await revision.submission_never_started(None, None, resource()) is expected


@pytest.mark.asyncio
async def test_cancellation_request_without_release_is_not_enough(probe):
    probe("123|CANCELLED|Unknown|")
    assert not await revision.submission_never_started(None, None, resource(released_at=None))


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt,expected", [
    ('{"exit_code":130,"error":"cancelled before execution"}', True),
    ('{"exit_code":130,"error":"cancelled"}', False),
    ('{"exit_code":0}', False),
    ('not json', False),
])
async def test_retained_allocation_requires_prelaunch_cancellation_receipt(probe, receipt, expected):
    probe(receipt)
    meta = {"controller_directory":"/run/controller", "controller_request_id":"attempt",
            "controller_outcome_path":"/run/controller/outcomes/attempt.json"}
    assert await revision.submission_never_started(None, None, resource(meta=meta)) is expected


@pytest.mark.asyncio
async def test_unavailable_remote_evidence_fails_closed(probe):
    probe("123|CANCELLED|Unknown|", ok=False)
    assert not await revision.submission_never_started(None, None, resource())
