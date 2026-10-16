from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from zevo.engine.run.outcome import terminal_outcome, unresolved_ticket_issues


NOW = datetime.now(timezone.utc)


def _ticket(
    ticket_id: str,
    *,
    agent: str,
    status: str,
    iteration: int = 1,
    lane: str = "optimization",
    created_offset: int = 0,
    error: str = "",
):
    created = NOW + timedelta(seconds=created_offset)
    return SimpleNamespace(
        id=ticket_id,
        agent_id=agent,
        status=status,
        iteration=iteration,
        lane=lane,
        payload={"operation": "train" if agent == "train" else agent},
        summary="",
        error_message=error,
        created_at=created,
        updated_at=created,
    )


def _run(*, lifecycle=None):
    return SimpleNamespace(
        lifecycle=lifecycle or {},
        registry_version_tag="M-run",
        iterations_completed=1,
    )


def test_successful_replacement_resolves_historical_failure() -> None:
    tickets = [
        _ticket("train-run-001", agent="train", status="failed"),
        _ticket("train-run-002", agent="train", status="succeeded", created_offset=1),
    ]
    assert unresolved_ticket_issues(tickets) == []


def test_unreplaced_held_out_failure_is_a_terminal_issue() -> None:
    tickets = [
        _ticket(
            "holdout-infer-run-002",
            agent="inference",
            lane="held_out_test",
            status="failed",
        ),
    ]
    issues = unresolved_ticket_issues(tickets)
    assert [item["ticket_id"] for item in issues] == ["holdout-infer-run-002"]
    assert issues[0]["message"].startswith("Held-out Test inference did not complete")


def test_iteration_limit_is_a_stop_trigger_not_an_issue() -> None:
    outcome = terminal_outcome(
        _run(lifecycle={"finalization": {"reason": "iterations 1 >= budget 1"}}),
        [],
    )
    assert outcome["issues"] == []
    assert outcome["stop_trigger"] == {
        "code": "iteration_limit",
        "current": 1,
        "limit": 1,
        "message": "Stopped after completing 1 of 1 iterations.",
    }


def test_finalization_timeout_replaces_synthetic_limit_ticket_failure() -> None:
    outcome = terminal_outcome(
        _run(lifecycle={
            "finalization": {"reason": "iterations 1 >= budget 1"},
            "rescue_terminal_status": "failed",
        }),
        [
            _ticket(
                "registry-run-001",
                agent="registry",
                status="failed",
                error="Run limit reached; iterations 1 >= budget 1",
            ),
        ],
    )
    assert [item["code"] for item in outcome["issues"]] == ["finalization_timeout"]
    assert outcome["issues"][0]["message"] == (
        "Finalization did not complete before its deadline."
    )


def _replacement_run():
    return _run(lifecycle={"cancelled_ticket_replacements": {
        "old": {"instruction_id": "user-change", "replacement_ticket_id": "new",
                "reason": "User requested a revised data plan"},
    }})


def test_explicit_cross_iteration_replacement_resolves_only_cancelled_work():
    old = _ticket("old", agent="train", status="cancelled", iteration=6)
    new = _ticket("new", agent="train", status="succeeded", iteration=7, created_offset=1)
    assert unresolved_ticket_issues([old, new])  # No inference from later success alone.
    assert unresolved_ticket_issues([old, new], run=_replacement_run()) == []
    assert terminal_outcome(_replacement_run(), [old, new])["issues"] == []
    old.status = "failed"
    assert unresolved_ticket_issues([old, new], run=_replacement_run())[0]["ticket_id"] == "old"


def test_cross_iteration_replacement_does_not_hide_other_failures():
    tickets = [
        _ticket("old", agent="train", status="cancelled", iteration=6),
        _ticket("new", agent="train", status="succeeded", iteration=7, created_offset=1),
        _ticket("test", agent="inference", status="failed", lane="held_out_test", iteration=7),
        _ticket("registry", agent="registry", status="failed", iteration=7),
    ]
    assert {i["ticket_id"] for i in unresolved_ticket_issues(tickets, run=_replacement_run())} == {"test", "registry"}


@pytest.mark.parametrize("change", [
    {"status": "running"}, {"status": "failed"}, {"status": "cancelled"},
    {"status": "degraded"}, {"status": "skipped"}, {"lane": "held_out_test"},
    {"agent_id": "data"}, {"run_id": "different-run"}, {"iteration": 5},
    {"created_at": NOW}, {"payload": {"operation": "different"}},
])
def test_unfinished_or_unrelated_replacement_does_not_resolve_cancellation(change):
    old = _ticket("old", agent="train", status="cancelled", iteration=6)
    new = _ticket("new", agent="train", status="succeeded", iteration=7, created_offset=1)
    for key, value in change.items():
        setattr(new, key, value)
    assert "old" in {i["ticket_id"] for i in unresolved_ticket_issues([old, new], run=_replacement_run())}


def test_audited_user_stop_is_success_eligible_without_hiding_other_failures():
    stopped = _ticket("data-stop", agent="data", status="cancelled")
    failed = _ticket("infer-failure", agent="inference", status="failed")
    run = _run(lifecycle={"user_requested_stage_cancellations": {
        stopped.id: {"instruction_id": "user-stop", "reason": "register champion", "recorded_at": NOW.isoformat()}
    }})
    assert unresolved_ticket_issues([stopped], run=run) == []
    assert [i["ticket_id"] for i in unresolved_ticket_issues([stopped, failed], run=run)] == [failed.id]


def test_user_stop_of_retry_does_not_erase_original_execution_failure():
    failed = _ticket("data-failed", agent="data", status="failed")
    retry = _ticket("data-retry", agent="data", status="cancelled", created_offset=1)
    run = _run(lifecycle={"user_requested_stage_cancellations": {
        retry.id: {"instruction_id": "stop", "reason": "stop retry", "recorded_at": NOW.isoformat()}
    }})
    assert [i["ticket_id"] for i in unresolved_ticket_issues([failed, retry], run=run)] == [failed.id]


def test_generic_cancel_text_does_not_prove_user_intent():
    ticket = _ticket("data-abnormal", agent="data", status="cancelled", error="cancelled by user")
    assert unresolved_ticket_issues([ticket], run=_run())[0]["code"] == "ticket_cancelled"
