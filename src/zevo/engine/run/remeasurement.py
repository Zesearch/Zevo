"""Engine-owned, transactional replacement of a lane's measurement cohort.

Only token allowances change. Prompts, questions, sampling and runtime context
stay frozen. Results remain private until every checkpoint has been rescored.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from uuid import uuid4
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.contracts.configuration import InferenceRunConfig, load_inference_config
from zevo.contracts.inference import GenerationDiagnostics
from zevo.db.models import Run, ScoreEvent, Ticket, TicketNotice, WorkProduct

KEY = "token_budget_remeasurements"
THRESHOLD = 0.05
MAX_REVISIONS = 2
Batch = dict[str, Any]
SUCCESS = {"succeeded", "degraded"}
TERMINAL = SUCCESS | {"failed", "cancelled", "skipped"}


def batches(run: Run) -> list[Batch]:
    return (run.lifecycle or {}).get(KEY, [])


def pending(run: Run) -> Batch | None:
    return next((b for b in batches(run) if b["status"] == "running"), None)


def replay_ids(batch: Batch) -> set[str]:
    return {v for item in batch.get("cohort", []) for v in
            (item.get("inference_id"), item.get("evaluation_id")) if v}


def is_replay(run: Run, ticket: Ticket) -> bool:
    return any(ticket.id in replay_ids(b) for b in batches(run))


def is_unpublished_replay(run: Run, ticket: Ticket) -> bool:
    return any(ticket.id in replay_ids(b) and b["status"] != "published" for b in batches(run))


def is_superseded_prediction(run: Run, ticket: Ticket) -> bool:
    source_id = str(((ticket.inputs or {}).get("predictions") or {}).get("source_ticket_id") or ticket.id)
    return any(b["status"] == "published" and source_id in {c["source_id"] for c in b["cohort"]} for b in batches(run))


def configuration_override(run: Run, ticket: Ticket, member_name: str) -> str:
    """Staged contracts are visible only to their own engine-created tickets."""
    for batch in reversed(batches(run)):
        if batch["lane"] != ticket.lane or batch["base_model"] != (ticket.payload or {}).get("base_model"):
            continue
        if ticket.id in replay_ids(batch) or batch["status"] == "published":
            return batch["configs"].get(member_name, "")
    return ""


def question_truncation(diagnostics: GenerationDiagnostics) -> dict[str, int | float]:
    rows = {r.row_index for r in diagnostics.records}
    limited = {r.row_index for r in diagnostics.records if r.finish_reason.strip().lower() == "length"}
    return {"questions": len(rows), "truncated_questions": len(limited),
            "fraction": len(limited) / len(rows)}


def enlarged_config(config: InferenceRunConfig, pins: dict) -> InferenceRunConfig:
    if "max_new_tokens" in (pins.get("decoding_config") or {}):
        raise ValueError("max_new_tokens is user-pinned; the pin must be revised first")
    body = config.model_dump(mode="json")
    old = config.measurement.max_new_tokens
    # Keep the already planned context/KV memory allocation unchanged. Raising
    # the context itself needs a new resource plan, not a guessed memory budget.
    context = (body["implementation_config"].get("llm_kwargs") or {}).get("max_model_len")
    if not isinstance(context, int) or isinstance(context, bool) or context <= old:
        raise ValueError("no verified context headroom; a new context/resource plan is required")
    body["measurement"]["max_new_tokens"] = min(old * 2, context)
    return InferenceRunConfig.model_validate(body)


def _store(run: Run, batch: Batch) -> None:
    items = deepcopy(batches(run))
    items = [deepcopy(batch) if b["id"] == batch["id"] else b for b in items]
    if not any(b["id"] == batch["id"] for b in items):
        items.append(deepcopy(batch))
    run.lifecycle = {**(run.lifecycle or {}), KEY: items}


async def _products(session: AsyncSession, ticket_id: str, role: str) -> list[WorkProduct]:
    return list((await session.scalars(select(WorkProduct).where(
        WorkProduct.ticket_id == ticket_id, WorkProduct.role == role,
    ).order_by(WorkProduct.created_at.desc()))).all())


def _suite(run: Run, lane: str) -> list[dict]:
    from zevo.engine.run.runner import _heldout_test_sets, _validation_sets
    return _heldout_test_sets(run) if lane == "held_out_test" else _validation_sets(run)


async def _evidence(session: AsyncSession, run: Run, ticket: Ticket) -> dict[str, dict]:
    suite = _suite(run, ticket.lane)
    primary = str(suite[0]["name"]) if suite else ""
    evidence = {}
    for product in await _products(session, ticket.id, "generation_diagnostics"):
        name = str((product.meta or {}).get("suite_member_name") or primary)
        if name not in evidence:
            evidence[name] = question_truncation(GenerationDiagnostics.model_validate_json(
                Path(product.path).read_text(encoding="utf-8")))
    return evidence


async def _locked_run(session: AsyncSession, run_id: str) -> Run | None:
    return (await session.execute(select(Run).where(Run.id == run_id)
                                 .with_for_update().execution_options(populate_existing=True))).scalar_one_or_none()


async def _fail(session: AsyncSession, run: Run, batch: Batch, reason: str) -> None:
    batch["status"] = "failed"
    batch["error"] = reason
    _store(run, batch)
    # No partial scores/configs are promoted. Stop optimization rather than
    # silently proceeding with a known censored comparison or private signal.
    run.status = "degraded"
    run.halted_reason = "Measurement budget revision could not complete; previous scores retained."
    session.add(TicketNotice(ticket_id=batch["trigger_id"], severity="warning",
                             code="token_budget_remeasurement_failed", body=reason))
    await session.commit()


async def _start(session: AsyncSession, run: Run, trigger: Ticket, evidence: dict) -> bool:
    affected = [name for name, item in evidence.items() if item["fraction"] > THRESHOLD]
    model = str((trigger.payload or {}).get("base_model") or "")
    revision = {"id": uuid4().hex[:16], "status": "running", "lane": trigger.lane,
                "base_model": model, "trigger_id": trigger.id, "evidence": evidence,
                "configs": {}, "cohort": []}
    try:
        previous = [b for b in batches(run) if b["lane"] == trigger.lane and b["base_model"] == model]
        if len(previous) >= MAX_REVISIONS:
            raise ValueError("automatic token-budget revisions exhausted; inspect EOS/stopping or context limits")
        all_replays = set().union(*(replay_ids(b) for b in previous)) if previous else set()
        published_replays = set().union(*(replay_ids(b) for b in previous if b["status"] == "published")) if previous else set()
        candidates = list((await session.scalars(select(Ticket).where(
            Ticket.run_id == run.id, Ticket.agent_id == "inference",
            Ticket.lane == trigger.lane, Ticket.status.in_(SUCCESS),
        ).order_by(Ticket.created_at))).all())
        latest = {}
        for candidate in candidates:
            payload = candidate.payload or {}
            if payload.get("base_model") != model:
                continue
            if candidate.id in all_replays - published_replays:
                continue
            # Legacy one-member mirrors cannot represent a complete suite.
            if trigger.lane == "optimization" and payload.get("test_set_name"):
                continue
            latest[(payload.get("model_source"), candidate.iteration)] = candidate
        cohort = sorted(latest.values(), key=lambda t: ((t.payload or {}).get("model_source") != "base_model", t.iteration))
        baseline = next((t for t in cohort if (t.payload or {}).get("model_source") == "base_model"), None)
        if baseline is None or trigger.id not in {t.id for t in cohort}:
            raise ValueError("complete baseline/current inference cohort is unavailable")
        suite = _suite(run, trigger.lane)
        names = [str(item["name"]) for item in suite]
        if not names or set(evidence) != set(names):
            raise ValueError("complete per-question suite diagnostics are required")
        # Every published point for this lineage must be represented, including
        # checkpoints whose files may subsequently fail to resolve remotely.
        identities = {(int(t.iteration), "baseline" if (t.payload or {}).get("model_source") == "base_model" else "trained") for t in cohort}
        for row in run.history or []:
            if isinstance(row, dict) and row.get("base_model") == model:
                if (int(row["iteration"]), row["source"]) not in identities:
                    raise ValueError("a previously compared checkpoint is missing from the replay cohort")
        products = await _products(session, baseline.id, "inference_config")
        for index, name in enumerate(names):
            path = configuration_override(run, baseline, name)
            if not path:
                product = next((p for p in products if (p.meta or {}).get("suite_member_name", names[0]) == name), None)
                if product is None:
                    raise ValueError(f"missing frozen baseline configuration for {name}")
                path = product.path
            config = load_inference_config(path)
            if name in affected:
                for candidate in cohort:
                    if "max_new_tokens" in ((candidate.payload or {}).get("configuration_pins", {}).get("decoding_config") or {}):
                        raise ValueError("max_new_tokens is user-pinned; update the pin before remeasurement")
                config = enlarged_config(config, {})
            destination = Path(path).parent / "budget-revisions" / revision["id"] / f"{index:03d}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
            revision["configs"][name] = str(destination)
        revision["cohort"] = [{"source_id": t.id, "inference_id": f"remeasure-{revision['id']}-{i:03d}"} for i, t in enumerate(cohort)]
        _store(run, revision)
        session.add(TicketNotice(ticket_id=trigger.id, code="token_budget_remeasurement",
                                 severity="info", body="High question-level truncation detected. Replaying baseline and all compared checkpoints under one revised token budget; scores remain unchanged until the cohort completes."))
        await session.commit()
        return True
    except (ValueError, OSError) as exc:
        await _fail(session, run, revision, str(exc))
        return True


async def _new_inference(session: AsyncSession, run: Run, batch: Batch, item: dict) -> Ticket:
    source = await session.get(Ticket, item["source_id"])
    ticket = Ticket(id=item["inference_id"], run_id=run.id, agent_id="inference",
                    lane=batch["lane"], status="queued", input_format="typed",
                    iteration=source.iteration, payload=deepcopy(source.payload),
                    customization=deepcopy(source.customization or {}), inputs=deepcopy(source.inputs or {}),
                    summary="Remeasure checkpoint with the cohort's revised token budget")
    session.add(ticket)
    await session.commit()
    return ticket


async def _enqueue(session: AsyncSession, ticket: Ticket) -> None:
    from zevo.engine.run.wakeup import queue_wakeup
    await queue_wakeup(session, agent_id=ticket.agent_id, ticket_id=ticket.id,
                       source="handoff", reason="token budget cohort remeasurement")


async def _scores(session: AsyncSession, ticket: Ticket, suite: list[dict]) -> dict[str, dict]:
    products = await _products(session, ticket.id, "metrics")
    components = {}
    for product in products:
        meta = product.meta or {}
        name = str(meta.get("suite_member_name") or "")
        if not name and len(suite) == 1:
            name = str(suite[0]["name"])
        if name in {str(s["name"]) for s in suite} and name not in components:
            score = meta.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                raise ValueError("remeasurement returned a non-finite/missing component score")
            specification = next(s for s in suite if s["name"] == name)
            components[name] = {"score": float(score), "metric": specification.get("metric", ""),
                                "metric_direction": specification.get("metric_direction", "max"),
                                "evaluation_ticket_id": ticket.id}
    if set(components) != {str(s["name"]) for s in suite}:
        raise ValueError("remeasurement did not produce the complete evaluation suite")
    return components


async def _publish(session: AsyncSession, run: Run, batch: Batch) -> None:
    from zevo.engine.run.runner import _record_validation_score, _record_holdout_score
    suite = _suite(run, batch["lane"])
    split = "test" if batch["lane"] == "held_out_test" else "validation"
    old_events = []
    batch["previous_history"] = deepcopy(run.history or [])
    for item in batch["cohort"]:
        inference = await session.get(Ticket, item["inference_id"])
        evaluation = await session.get(Ticket, item["evaluation_id"])
        components = await _scores(session, evaluation, suite)
        source = "baseline" if (inference.payload or {}).get("model_source") == "base_model" else "trained"
        # Archive superseded points outside the live ScoreEvent series. Every
        # existing consumer therefore sees exactly one coherent revision.
        events = list((await session.scalars(select(ScoreEvent).where(
            ScoreEvent.run_id == run.id, ScoreEvent.split == split,
            ScoreEvent.iteration == inference.iteration, ScoreEvent.source == source,
        ))).all())
        for event in events:
            old_events.append({"id": event.id, "iteration": event.iteration, "source": event.source,
                               "split": split, "score": event.score, "metric_name": event.metric_name,
                               "extras": event.extras, "notes": event.notes, "ts": event.ts.isoformat()})
            await session.delete(event)
        holdout = deepcopy(run.holdout or {})
        key = f"{source}|{inference.iteration}|{batch['base_model']}"
        prefix = "suite" if split == "test" else "validation_suite"
        holdout[f"{prefix}_recorded"] = [k for k in holdout.get(f"{prefix}_recorded", []) if k != key]
        holdout.setdefault(f"{prefix}_results", {}).pop(key, None)
        run.holdout = holdout
        if split == "validation":
            await _record_validation_score(session, run, evaluation,
                sum(c["score"] for c in components.values()) / len(components),
                components=components, commit=False)
            holdout = deepcopy(run.holdout or {})
            holdout.setdefault("validation_suite_results", {})[key] = components
            holdout.setdefault("validation_suite_recorded", []).append(key)
            run.holdout = holdout
        else:
            for name, component in components.items():
                await _record_holdout_score(session, run, evaluation, component["score"], member_name=name, commit=False)
        current_events = (await session.scalars(select(ScoreEvent).where(
            ScoreEvent.run_id == run.id, ScoreEvent.split == split,
            ScoreEvent.iteration == inference.iteration, ScoreEvent.source == source,
        ))).all()
        for event in current_events:
            event.extras = {**(event.extras or {}), "measurement_revision": batch["id"]}
    batch["previous_score_events"] = old_events
    batch["status"] = "published"
    batch["handoff_pending"] = True
    _store(run, batch)
    await session.commit()


async def _handoff(session: AsyncSession, run: Run, batch: Batch) -> None:
    from zevo.engine.run.runner import _spawn_holdout_infer, _maybe_wake_supervisor, _release_deferred_wake, _holdout_settled
    item = next(c for c in batch["cohort"] if c["source_id"] == batch["trigger_id"])
    inference = await session.get(Ticket, item["inference_id"])
    evaluation = await session.get(Ticket, item["evaluation_id"])
    if batch["lane"] == "held_out_test":
        if await _holdout_settled(session, run.id):
            await _release_deferred_wake(session, run.id)
    else:
        evaluation.supervisor_wake_deferred = True
        await session.commit()
        await _spawn_holdout_infer(session, run, inference)
        if await _holdout_settled(session, run.id):
            evaluation.supervisor_wake_deferred = False
            await session.commit()
            await _maybe_wake_supervisor(session, evaluation)
        else:
            evaluation.supervisor_wake_deferred = True
    run = await _locked_run(session, run.id)
    batch["handoff_pending"] = False
    _store(run, batch)
    await session.commit()


async def advance(session: AsyncSession, run_id: str) -> None:
    """Idempotent recovery entry point, also called by the reconciler."""
    from zevo.engine.run.runner import _spawn_validation_eval, _spawn_holdout_eval
    run = await _locked_run(session, run_id)
    if run is None or run.status not in {"planning", "running"} or run.cancel_requested_at is not None:
        return
    batch = deepcopy(pending(run))
    if not batch:
        for completed in batches(run):
            if completed.get("handoff_pending"):
                await _handoff(session, run, deepcopy(completed))
        return
    try:
        for item in batch["cohort"]:
            inference = await session.get(Ticket, item["inference_id"])
            if inference is None:
                inference = await _new_inference(session, run, batch, item)
            if inference.status not in TERMINAL:
                if inference.status == "queued":
                    await _enqueue(session, inference)
                return
            if inference.status not in SUCCESS:
                raise ValueError(f"checkpoint replay failed: {inference.id}; previous scores retained")
            evidence = await _evidence(session, run, inference)
            if set(evidence) != set(batch["configs"]):
                raise ValueError("checkpoint replay lacks complete question-level diagnostics")
            # Never publish a still-censored cohort. A second complete revision
            # may be attempted, but never unbounded doubling on one checkpoint.
            if any(e["fraction"] > THRESHOLD for e in evidence.values()):
                batch["status"] = "superseded"
                _store(run, batch)
                # The next plan must start from the attempted budgets while
                # selecting the same original cohort, not partial replay rows.
                return await _retry(session, run, batch, evidence)
            evaluation = await session.get(Ticket, item["evaluation_id"]) if item.get("evaluation_id") else None
            if evaluation is None:
                name = str(_suite(run, batch["lane"])[0]["name"])
                item["evaluation_id"] = f"{item['inference_id']}-eval"
                _store(run, batch)
                products = await _products(session, inference.id, "predictions")
                primary = next((p for p in products if (p.meta or {}).get("suite_member_name", name) == name), None)
                if primary is None:
                    raise ValueError("checkpoint replay lacks primary predictions")
                kwargs = dict(enqueue=False, predictions_product=primary, ticket_id=item["evaluation_id"])
                if batch["lane"] == "held_out_test":
                    evaluation = await _spawn_holdout_eval(session, run, inference, test_set_name=name, **kwargs)
                else:
                    evaluation = await _spawn_validation_eval(session, run, inference, member_name=name, **kwargs)
                if evaluation is None:
                    raise ValueError("could not create deterministic cohort evaluation")
                item["evaluation_id"] = evaluation.id
                _store(run, batch)
                await session.commit()
            if evaluation.status not in TERMINAL:
                if evaluation.status == "queued":
                    await _enqueue(session, evaluation)
                return
            if evaluation.status not in SUCCESS:
                raise ValueError(f"cohort evaluation failed: {evaluation.id}; previous scores retained")
        await _publish(session, run, batch)
        await _handoff(session, run, batch)
    except (ValueError, OSError) as exc:
        await session.rollback()
        run = await _locked_run(session, run_id)
        if batch["status"] == "published":
            raise  # promotion committed; retry only the durable handoff
        await _fail(session, run, batch, str(exc))


async def _retry(session: AsyncSession, run: Run, batch: Batch, evidence: dict) -> None:
    # Keep the original cohort and all settings, changing only newly limited
    # members. The bounded revision count includes superseded attempts.
    if len([b for b in batches(run) if b["lane"] == batch["lane"] and b["base_model"] == batch["base_model"]]) >= MAX_REVISIONS:
        await _fail(session, run, batch, "Truncation persists after bounded budget increases; inspect stopping behavior/context. Previous scores retained.")
        return
    new = deepcopy(batch)
    new.update(id=uuid4().hex[:16], status="running", evidence=evidence, configs={})
    for index, (name, path) in enumerate(batch["configs"].items()):
        config = load_inference_config(path)
        if evidence[name]["fraction"] > THRESHOLD:
            config = enlarged_config(config, {})
        destination = Path(path).parent / new["id"] / f"{index:03d}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
        new["configs"][name] = str(destination)
    new["cohort"] = [{"source_id": c["source_id"], "inference_id": f"remeasure-{new['id']}-{i:03d}"} for i, c in enumerate(batch["cohort"])]
    _store(run, new)
    await session.commit()


async def intercept_completion(session: AsyncSession, ticket: Ticket) -> bool:
    run = await _locked_run(session, ticket.run_id)
    if run is None or run.status not in {"planning", "running"} or run.cancel_requested_at is not None:
        return False
    if is_replay(run, ticket):
        await session.commit()
        await advance(session, run.id)
        await session.refresh(ticket)
        return True
    if pending(run):
        return True
    # Delayed duplicate callbacks for old predictions must not restore stale
    # scores after a revision was published.
    if is_superseded_prediction(run, ticket):
        return True
    if ticket.agent_id != "inference" or ticket.status not in SUCCESS or (ticket.payload or {}).get("model_source") != "checkpoint":
        return False
    evidence = await _evidence(session, run, ticket)
    if not any(item["fraction"] > THRESHOLD for item in evidence.values()):
        return False
    await _start(session, run, ticket, evidence)
    await advance(session, run.id)
    await session.refresh(ticket)
    return True


async def reconcile(session: AsyncSession) -> int:
    runs = list((await session.scalars(select(Run).where(Run.status.in_(("planning", "running"))))).all())
    count = 0
    for run in runs:
        if pending(run) or any(b.get("handoff_pending") for b in batches(run)):
            await advance(session, run.id)
            count += 1
    return count
