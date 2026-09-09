"""Auto mode: the scoping stage and its settlement onto the Run.

An ``auto`` Run is created without a user-supplied scoring contract. Instead of the supervisor
Ticket, ``POST /runs`` creates ONE engine-owned Data Ticket,
``scope-<run8>-001`` (operation ``scope_problem``), and queues it.  The Data
agent derives the whole scoring contract -- metric, direction, a materialized
public benchmark or a decontaminated synthesized held-out, answer fields and a
submission template -- and writes it as a :class:`ScopingResult`.

When that Ticket finishes, the runner hands it to :func:`complete_scoping`,
which is the engine-side twin of the tail of ``POST /runs``:

1. validate the ScopingResult and move its Test assets behind the private
   held-out boundary (``holdout_root()/runs/<run>/_scoping``);
2. build the complete ``UserRequest`` the user would otherwise have sent and
   run the SAME checks (``scoring_asset_errors``) and the SAME split
   settlement (``settle_splits``) a full_pipeline Run gets at creation;
3. write the contract onto the Run (`metric`, `metric_direction`,
   `validation_metric*`, `holdout`, provenance) and mark
   ``scoring_settled=True``;
4. create the supervisor Ticket with a ``run_created`` trigger and queue the
   Orchestrator -- from here on the Run is indistinguishable from a
   full_pipeline Run that was just created.

A scoping Ticket that ends failed (after the ordinary repair attempts) fails the
Run with the reason: there is nothing an Orchestrator could do without a
scoring contract, so it is never woken.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.contracts.data import DataResult
from zevo.contracts.orchestrator import (
    AutoUserRequest,
    UserRequest,
    inherit_test_validation_contract,
    scoring_asset_errors,
)
from zevo.contracts.scoping import ScopingResult, load_scoping_result
from zevo.contracts.tickets import validate_stored_payload
from zevo.db import Run, Ticket


SCOPING_OPERATION = "scope_problem"


def scoping_ticket_id(run: Run) -> str:
    return f"scope-{run.id[:8]}-001"


def is_scoping_ticket(ticket: Ticket) -> bool:
    """The engine-owned Auto-mode scoping work order, by its stored payload."""
    return (
        ticket.agent_id == "data"
        and str((ticket.payload or {}).get("operation") or "") == SCOPING_OPERATION
    )


def scoping_payload(request: AutoUserRequest) -> dict[str, Any]:
    """The validated stored payload of the scope_problem Ticket.

    This Ticket receives only facts needed to define evaluation. Training-side
    pins and hints live on the Run and are attached after scoping; withholding
    them here prevents the held-out choice from being tailored to a proposed
    training source, model, or method.
    """
    return validate_stored_payload(
        agent_id="data",
        input_format="typed",
        payload={
            "operation": SCOPING_OPERATION,
            "task_objective": request.task_objective,
            "test_query": request.test_query,
            "constraints": list(request.constraints),
        },
    )


def new_scoping_ticket(run: Run, request: AutoUserRequest) -> Ticket:
    """The queued scoping Ticket; the caller adds and commits it.

    It runs on the ordinary optimization lane (the Data agent's scheduler) at
    iteration 0. No Orchestrator exists yet to see its result, and settlement
    moves the Test bytes it produced out of that lane before one is created.
    """
    return Ticket(
        id=scoping_ticket_id(run), run_id=run.id, agent_id="data",
        status="queued", input_format="typed", lane="optimization", iteration=0,
        payload=scoping_payload(request), customization={}, inputs={},
        summary="scoping · derive the scoring contract from the objective",
    )


class ScopingSettlementError(ValueError):
    """The scoping output cannot become a Run scoring contract."""


def _private_scoping_dir(run: Run) -> Path:
    from zevo.paths import holdout_root
    return Path(holdout_root()) / "runs" / run.id / "_scoping"


def _move_private(source: str, dest_dir: Path) -> str:
    """Move one scoping artifact behind the held-out boundary.

    The scoping agent had to see the held-out rows to define them -- that is
    what an agent-defined held-out means -- but no LATER optimization-lane
    Ticket may. Moving (not copying) leaves nothing readable at the path the
    scoping Ticket recorded.
    """
    src = Path(source)
    if not src.is_file():
        raise ScopingSettlementError(f"scoping artifact is not a file: {source}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    return str(dest)


def user_request_from_scoping(
    run: Run, *, result: ScopingResult, payload: dict[str, Any],
    test_set: str, test_sample_submission: str,
    evaluation_script: str = "", evaluator_sha256: str = "",
) -> UserRequest:
    """The full_pipeline-shaped request settlement feeds to the split logic.

    Scoring comes only from the ScopingResult. Training-side values come only
    from the immutable Auto request snapshot on the Run, so Auto uses the same
    L1–L4 ownership ladder as Standard without exposing those choices to the
    scoping work order.
    """
    pins = dict(run.decision_pins or {})
    return UserRequest(
        task_objective=run.task_objective,
        metric=result.metric,
        metric_direction=result.metric_direction,
        metric_type=result.metric_type,
        evaluation_script=evaluation_script,
        evaluator_sha256=evaluator_sha256,
        training_method=str(pins.get("training_method") or ""),
        method_query=str(pins.get("method_query") or ""),
        method_config=dict(pins.get("method_config") or {}),
        dataset=str(pins.get("dataset") or ""),
        dataset_split=str(pins.get("dataset_split") or ""),
        dataset_config=str(pins.get("dataset_config") or ""),
        data_query=str(pins.get("data_query") or ""),
        base_model=str(pins.get("base_model") or ""),
        model_query=str(pins.get("model_query") or ""),
        test_set=test_set,
        test_answer_fields=list(result.test_answer_fields),
        test_sample_submission=test_sample_submission,
        constraints=[str(c) for c in (payload.get("constraints") or [])],
    )


async def _fail_run(
    session: AsyncSession, run_id: str, ticket_id: str, reason: str,
) -> None:
    """An auto Run without a scoring contract cannot proceed. Say why and stop.

    Written as one core UPDATE keyed by id: this runs after a failed settlement
    may have rolled the session back, so no ORM instance is trusted here.
    """
    from sqlalchemy import select, update

    from zevo.contracts.tickets import TERMINAL_RUN_STATUSES

    current = (await session.execute(
        select(Run.status).where(Run.id == run_id)
    )).scalar_one_or_none()
    if current is None or current in TERMINAL_RUN_STATUSES:
        return
    await session.execute(
        update(Run).where(Run.id == run_id).values(
            status="failed",
            halted_reason=(
                f"scoping ({ticket_id}) did not settle a scoring contract: {reason}"
            )[:2000],
            summary=(
                "Auto mode could not derive a scoring contract from the objective, "
                "so no Validation/Test measurement was possible and no training "
                "was started."
            ),
            finished_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


async def settle_scoping(
    session: AsyncSession, run: Run, ticket: Ticket, result: ScopingResult,
) -> Ticket:
    """Write the derived contract onto the Run and create the supervisor.

    Raises :class:`ScopingSettlementError` (or the split's own error) with a
    complete message when the contract cannot be settled. Commits on success
    and returns the queued supervisor Ticket (the wakeup is queued too).
    """
    from zevo.engine.run.split_settlement import SplitSettlementError, settle_splits
    from zevo.engine.run.supervisor import (
        initial_supervisor_payload, new_supervisor_ticket, supervisor_ticket_id,
    )
    from zevo.engine.run.wakeup import queue_wakeup

    payload = dict(ticket.payload or {})
    private_dir = _private_scoping_dir(run)
    test_set = _move_private(result.test_set_path, private_dir)
    sample_submission = _move_private(result.test_sample_submission_path, private_dir)

    evaluation_script, evaluator_sha256 = "", ""
    if result.metric_type == "custom":
        from zevo.evaluator_storage import freeze_evaluator
        try:
            evaluation_script, evaluator_sha256 = freeze_evaluator(result.evaluation_script)
        except ValueError as exc:
            raise ScopingSettlementError(str(exc)) from exc

    user_request = user_request_from_scoping(
        run, result=result, payload=payload,
        test_set=test_set, test_sample_submission=sample_submission,
        evaluation_script=evaluation_script, evaluator_sha256=evaluator_sha256,
    )
    # Validation is carved from the held-out population, so it inherits the Test
    # contract exactly -- the same rule Run creation applies.
    user_request = inherit_test_validation_contract(user_request)
    errors = scoring_asset_errors(user_request)
    if errors:
        raise ScopingSettlementError("; ".join(errors))

    try:
        agent_request, holdout, _note = await settle_splits(run, user_request)
    except SplitSettlementError as exc:
        raise ScopingSettlementError(str(exc)) from exc

    # Provenance travels with the private snapshot: which benchmark (or which
    # teacher) defined this Run's goal is part of what its score means.
    holdout["eval_source"] = result.eval_source
    holdout["scoping_ticket_id"] = ticket.id
    holdout["scoping"] = result.provenance_summary()

    run.metric = user_request.metric
    run.metric_direction = user_request.metric_direction
    run.validation_metric = user_request.validation_metric
    run.validation_metric_direction = user_request.validation_metric_direction
    run.holdout = holdout
    run.scoring_settled = True
    # decision_pins already holds the immutable optimization-side Auto request;
    # scoping is not allowed to add or rewrite any of those decisions.

    sup_payload = initial_supervisor_payload(
        run, agent_request=agent_request, holdout=holdout, dataset_profile={},
    )
    sup = new_supervisor_ticket(run, sup_payload)
    session.add(sup)
    run.supervisor_ticket_id = supervisor_ticket_id(run)
    run.status = "running"
    await session.commit()
    await session.refresh(sup)

    await queue_wakeup(
        session, agent_id="orchestrator", ticket_id=sup.id,
        source="handoff",
        trigger_detail=f"scoping:{ticket.id}",
        reason=(
            f"scoping settled the scoring contract ({result.eval_source}, "
            f"{result.metric}/{result.metric_direction}); starting the pipeline"
        ),
    )
    return sup


async def complete_scoping(
    session: AsyncSession, ticket: Ticket, output: BaseModel | None,
) -> None:
    """React to the scoping Ticket's terminal outcome. Never raises.

    Success settles the contract and wakes the Orchestrator; a terminal failure
    (or a success without a usable ScopingResult) fails the Run with the
    reason. A non-terminal status (still repairing/queued) is left alone.
    """
    from sqlalchemy import select

    run = (await session.execute(select(Run).where(Run.id == ticket.run_id))).scalar_one_or_none()
    if run is None:
        return
    await session.refresh(run)
    from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
    if run.status in TERMINAL_RUN_STATUSES or run.scoring_settled:
        return
    if ticket.status in ("queued", "running", "repairing", "awaiting_input", "waiting_external"):
        return
    run_id, ticket_id = run.id, ticket.id
    if ticket.status not in ("succeeded", "degraded"):
        await _fail_run(
            session, run_id, ticket_id,
            (ticket.error_message or ticket.summary or f"ticket ended {ticket.status}")[:500],
        )
        return
    if not isinstance(output, DataResult) or output.operation != SCOPING_OPERATION:
        await _fail_run(
            session, run_id, ticket_id,
            "the Ticket did not return a scope_problem DataResult",
        )
        return
    try:
        result = load_scoping_result(output.scoping_result_path)
        await settle_scoping(session, run, ticket, result)
    except Exception as exc:  # noqa: BLE001 - every failure must fail the Run, not strand it
        reason = str(exc)[:1500]
        await session.rollback()
        await _fail_run(session, run_id, ticket_id, reason)
