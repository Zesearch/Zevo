from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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
