from __future__ import annotations

from zevo.api.routers.shared.tickets import _external_stage_message_body


def test_pending_slurm_stage_cannot_post_done() -> None:
    assert _external_stage_message_body(
        "Done: job submitted", scheduler_state="PENDING", ticket_status="running",
    ) == "Waiting: job submitted"


def test_running_or_collecting_slurm_stage_cannot_post_done() -> None:
    assert _external_stage_message_body(
        "Done: outputs copied", scheduler_state="RUNNING", ticket_status="running",
    ) == "Running: outputs copied"
    assert _external_stage_message_body(
        "Done: outputs copied", scheduler_state="COMPLETED", ticket_status="running",
    ) == "Running: outputs copied"


def test_engine_validated_terminal_ticket_keeps_done() -> None:
    assert _external_stage_message_body(
        "Done: outputs validated",
        scheduler_state="COMPLETED",
        ticket_status="succeeded",
    ) == "Done: outputs validated"
