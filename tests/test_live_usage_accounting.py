"""Spend remains visible before completion and after interrupted execution."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db.models import Agent, Base, HeartbeatRun, InfraInstance, Run, Ticket
from zevo.engine.cost.budget import snapshot_for_run
from zevo.engine.cost.live_usage import LiveUsage, persist_usage
from zevo.engine.cost.pricing import estimate_cost


def test_message_snapshots_and_final_session_total_are_not_double_billed():
    usage = LiveUsage()
    usage.record({"usage_id": "m1", "usage": {"input_tokens": 100, "output_tokens": 2}})
    usage.record({"usage_id": "m1", "usage": {"input_tokens": 100, "output_tokens": 5}})
    usage.record({"usage_id": "m2", "usage": {"input_tokens": 200, "output_tokens": 10}})
    assert usage.counts["input_tokens"] == 300
    assert usage.counts["output_tokens"] == 15
    usage.record({"usage_kind": "cumulative", "usage": {"input_tokens": 310, "output_tokens": 20}})
    usage.record({"usage_id": "m2", "usage": {"input_tokens": 200, "output_tokens": 10}})
    assert usage.counts["input_tokens"] == 310
    assert usage.counts["output_tokens"] == 20


def test_missing_final_usage_preserves_interim_totals():
    usage = LiveUsage()
    usage.record({"usage": {"input_tokens": 100}})
    usage.record({"usage_kind": "cumulative", "usage": {}})
    assert usage.counts["input_tokens"] == 100


@pytest.mark.asyncio
async def test_committed_usage_is_visible_to_budget_without_finishing_heartbeat():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        db.add(Agent(id="train", name="Train", title="Train", identity_path="train.md"))
        db.add(Run(id="run", metric="accuracy", max_cost_usd=1))
        db.add(Ticket(id="ticket", run_id="run", agent_id="train"))
        db.add(HeartbeatRun(id="heartbeat", ticket_id="ticket", agent_id="train", driver="stub", model="claude-sonnet-4-6"))
        await db.commit()
    usage = LiveUsage()
    usage.record({"usage": {"input_tokens": 1_000_000, "output_tokens": 1000}})
    async with Session() as writer:
        await persist_usage(writer, "heartbeat", "claude-sonnet-4-6", usage)
        await writer.commit()
        # Retrying a successful batch must not charge it twice.
        await persist_usage(writer, "heartbeat", "claude-sonnet-4-6", usage)
        await writer.commit()
    async with Session() as observer:
        heartbeat = await observer.get(HeartbeatRun, "heartbeat")
        assert heartbeat.finished_at is None
        assert heartbeat.input_tokens == 1_000_000
        snapshot = await snapshot_for_run(observer, "run")
        assert snapshot.spent_usd == estimate_cost("claude-sonnet-4-6", usage.counts)
        assert snapshot.over_budget
    # The provider's final all-model total must survive both persistence paths.
    usage.record({"usage_kind": "cumulative", "usage": {}, "reported_cost_usd": 4.25})
    async with Session() as writer:
        await persist_usage(writer, "heartbeat", "claude-sonnet-4-6", usage)
        await persist_usage(writer, "heartbeat", "claude-sonnet-4-6", usage)
        await writer.commit()
    async with Session() as observer:
        heartbeat = await observer.get(HeartbeatRun, "heartbeat")
        assert heartbeat.estimated_cost_usd == 4.25
        assert heartbeat.input_tokens == 1_000_000
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["cancelled", "failed", "success"])
async def test_gpu_billing_continues_after_terminal_run_until_confirmed_release(status):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(timezone.utc)
    async with Session() as db:
        db.add(Run(id="run", metric="accuracy", status=status, finished_at=now - timedelta(hours=2)))
        instance = InfraInstance(id="infra", instance_id="remote", provider="cloud", run_id="run", dph=2, created_at=now - timedelta(hours=3), status="ready")
        db.add(instance)
        await db.commit()
        snapshot = await snapshot_for_run(db, "run")
        assert snapshot.gpu_cost_usd == pytest.approx(6, abs=.01)
        instance.released_at = now - timedelta(hours=1)
        instance.status = "released"
        await db.commit()
        snapshot = await snapshot_for_run(db, "run")
        assert snapshot.gpu_cost_usd == pytest.approx(4)
    await engine.dispose()


def test_claude_cache_ttls_and_auxiliary_cost_reconcile_without_double_billing():
    from zevo.engine.agent.drivers.claude_cli import _normalise_claude_event
    raw = {"input_tokens": 100, "output_tokens": 200,
           "cache_read_input_tokens": 1000000, "cache_creation_input_tokens": 300000,
           "cache_creation": {"ephemeral_5m_input_tokens": 100000,
                              "ephemeral_1h_input_tokens": 200000}}
    usage = LiveUsage()
    interim = _normalise_claude_event({"type": "assistant", "message": {"id": "m1", "usage": raw}})[0]["payload"]
    usage.record(interim)
    usage.record(interim)
    # 0.0005 input + 0.005 output + 0.5 read + 0.625 5m write + 2 1h write.
    assert usage.values("claude-opus-5")["estimated_cost_usd"] == 3.1305
    result = _normalise_claude_event({"type": "result", "usage": raw,
        "total_cost_usd": 3.2305,
        "modelUsage": {"claude-opus-5": {"costUSD": 3.1305},
                       "claude-haiku-4-5-20251001": {"costUSD": .1}}})[0]["payload"]
    usage.record(result)
    usage.record(result)
    usage.record(interim)
    assert usage.values("claude-opus-5")["estimated_cost_usd"] == 3.2305
    assert usage.counts["cache_write_1h_tokens"] == 200000
    assert usage.values("claude-opus-5")["input_tokens"] == 1300100
    assert "cache_write_1h_tokens" not in usage.values("claude-opus-5")


@pytest.mark.parametrize("bad", [None, -1, float('nan'), float('inf'), 'invalid', True])
def test_invalid_provider_cost_falls_back_to_token_estimate(bad):
    usage = LiveUsage()
    usage.record({"usage_kind": "cumulative", "usage": {"input_tokens": 1000000},
                  "reported_cost_usd": bad})
    assert usage.values("claude-opus-5")["estimated_cost_usd"] == 5


def test_final_cost_without_usage_keeps_tokens_and_includes_auxiliary_models():
    usage = LiveUsage()
    usage.record({"usage": {"input_tokens": 1000000}})
    usage.record({"usage_kind": "cumulative", "usage": {},
                  "model_usage": {"opus": {"costUSD": 5}, "haiku": {"costUSD": .1}}})
    usage.record({"usage": {"input_tokens": 1000000}})
    assert usage.values("claude-opus-5")["estimated_cost_usd"] == 5.1
    assert usage.counts["input_tokens"] == 1000000


def test_zero_provider_cost_is_not_replaced_by_estimate():
    usage = LiveUsage()
    usage.record({"usage_kind": "cumulative", "usage": {"input_tokens": 1000000},
                  "reported_cost_usd": 0})
    assert usage.values("claude-opus-5")["estimated_cost_usd"] == 0


def test_run_1d436226_usage_matches_verified_provider_total():
    from zevo.engine.agent.drivers.claude_cli import _normalise_usage
    raw = {"input_tokens": 4060, "output_tokens": 2635963,
           "cache_read_input_tokens": 219883140, "cache_creation_input_tokens": 12518250,
           "cache_creation": {"ephemeral_1h_input_tokens": 12518250}}
    usage = LiveUsage()
    usage.record({"usage": _normalise_usage(raw)})
    assert usage.values("claude-opus-5")["estimated_cost_usd"] == 301.043445
    usage.record({"usage_kind": "cumulative", "usage": _normalise_usage(raw),
                  "reported_cost_usd": 306.049208})
    assert usage.values("claude-opus-5")["estimated_cost_usd"] == 306.049208
