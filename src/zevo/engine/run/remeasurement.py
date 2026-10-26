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
from zevo.contracts.length_calibration import prediction_outputs, repetition_evidence
from zevo.db.models import Run, ScoreEvent, Ticket, TicketNotice, WorkProduct

KEY = "token_budget_remeasurements"
THRESHOLD = 0.05
NO_IMPROVEMENT_LIMIT = 2
MIN_RELATIVE_TRUNCATION_DROP = 0.10
Batch = dict[str, Any]
SUCCESS = {"succeeded", "degraded"}
TERMINAL = SUCCESS | {"failed", "cancelled", "skipped"}


def batches(run: Run) -> list[Batch]:
    return (getattr(run, "lifecycle", None) or {}).get(KEY, [])


def pending(run: Run) -> Batch | None:
    return next((b for b in batches(run) if b["status"] in {"running", "paused"}), None)


def replay_ids(batch: Batch) -> set[str]:
    return {v for item in batch.get("cohort", []) for v in
            (item.get("inference_id"), item.get("evaluation_id")) if v}


def is_replay(run: Run, ticket: Ticket) -> bool:
    return any(ticket.id in replay_ids(b) for b in batches(run))


def is_unpublished_replay(run: Run, ticket: Ticket) -> bool:
    latest = next((b for b in reversed(batches(run)) if ticket.id in replay_ids(b)), None)
    return latest is not None and latest["status"] != "published"


def is_superseded_prediction(run: Run, ticket: Ticket) -> bool:
    source_id = str(((ticket.inputs or {}).get("predictions") or {}).get("source_ticket_id") or ticket.id)
    return any(b["status"] == "published" and source_id in {c["source_id"] for c in b["cohort"]} for b in batches(run))


def replay_item(run: Run, ticket_id: str) -> dict:
    for batch in reversed(batches(run)):
        for item in batch["cohort"]:
            if ticket_id in (item.get("inference_id"), item.get("evaluation_id")):
                return item
    return {}


def needs_expansion(item: dict) -> bool:
    return item["fraction"] > THRESHOLD and not item.get("repetition", {}).get("dominated", False)


def active_attempt(run: Run, ticket_id: str) -> dict:
    batch = pending(run)
    if not batch or not batch.get("reuse_tickets"):
        return {}
    for item in batch["cohort"]:
        for key in ("inference", "evaluation"):
            if item.get(f"{key}_id") == ticket_id:
                return item.get(f"{key}_attempt", {})
    return {}


def restarts_workload(run: Run, ticket_id: str, resource_id: str) -> bool:
    attempt = active_attempt(run, ticket_id)
    return bool(resource_id and attempt.get("previous_resource_id") == resource_id)


def attempt_work_dir(run: Run, ticket_id: str, work_dir: str) -> str:
    attempt = active_attempt(run, ticket_id)
    return str(Path(work_dir) / "measurement-attempts" / attempt["revision"]) if attempt else work_dir


def _original_ticket_id(run: Run, ticket_id: str) -> str:
    # Follow legacy remeasure-* provenance without rewriting existing history.
    seen = set()
    while ticket_id not in seen:
        seen.add(ticket_id)
        source = next((c["source_id"] for b in reversed(batches(run)) if not b.get("reuse_tickets") for c in b["cohort"]
                       if c["inference_id"] == ticket_id and c["source_id"] != ticket_id), None)
        if not source:
            break
        ticket_id = source
    return ticket_id


def configuration_override(run: Run, ticket: Ticket, member_name: str) -> str:
    """Staged contracts are visible only to their own engine-created tickets."""
    for batch in reversed(batches(run)):
        if batch["lane"] != ticket.lane or batch["base_model"] != (ticket.payload or {}).get("base_model"):
            continue
        if ticket.id in replay_ids(batch) and (batch["status"] == "published" or
                (batch["status"] == "running" and (not batch.get("reuse_tickets") or active_attempt(run, ticket.id)))):
            item = next(c for c in batch["cohort"] if ticket.id in (c.get("inference_id"), c.get("evaluation_id")))
            retained = item.get("retained", {}).get(member_name)
            return retained["config_path"] if retained else batch["configs"].get(member_name, "")
        if batch["status"] == "published":
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
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        raise ValueError("no verified context headroom; a new context/resource plan is required")
    if old >= context:
        return config
    body["measurement"]["max_new_tokens"] = min(old * 2, context)
    return InferenceRunConfig.model_validate(body)


def _store(run: Run, batch: Batch) -> None:
    items = deepcopy(batches(run))
    items = [deepcopy(batch) if b["id"] == batch["id"] else b for b in items]
    if not any(b["id"] == batch["id"] for b in items):
        items.append(deepcopy(batch))
    run.lifecycle = {**(run.lifecycle or {}), KEY: items}


async def _products(session: AsyncSession, ticket_id: str, role: str, run: Run | None = None) -> list[WorkProduct]:
    rows = list((await session.scalars(select(WorkProduct).where(
        WorkProduct.ticket_id == ticket_id, WorkProduct.role == role,
    ).order_by(WorkProduct.created_at.desc()))).all())
    previous = set(active_attempt(run, ticket_id).get("previous_product_ids", [])) if run else set()
    return [p for p in rows if p.id not in previous]


def _suite(run: Run, lane: str) -> list[dict]:
    from zevo.engine.run.runner import _heldout_test_sets, _validation_sets
    return _heldout_test_sets(run) if lane == "held_out_test" else _validation_sets(run)


async def _evidence(session: AsyncSession, run: Run, ticket: Ticket) -> dict[str, dict]:
    suite = _suite(run, ticket.lane)
    primary = str(suite[0]["name"]) if suite else ""
    evidence = {}
    for product in await _products(session, ticket.id, "generation_diagnostics", run):
        name = str((product.meta or {}).get("suite_member_name") or primary)
        if name not in evidence:
            diagnostics = GenerationDiagnostics.model_validate_json(Path(product.path).read_text(encoding="utf-8"))
            evidence[name] = question_truncation(diagnostics)
            records = [r.model_dump() for r in diagnostics.records]
            predictions = await _products(session, ticket.id, "predictions", run)
            configs = await _products(session, ticket.id, "inference_config", run)
            pred = next((p for p in predictions if (p.meta or {}).get("suite_member_name", primary) == name), None)
            cfg = next((p for p in configs if (p.meta or {}).get("suite_member_name", primary) == name), None)
            if pred and cfg:
                try:
                    config = load_inference_config(cfg.path)
                    column = config.implementation_config.get("raw_output_column", "complete_prediction")
                    outputs = prediction_outputs(pred.path, records, column)
                    evidence[name]["repetition"] = repetition_evidence(records, outputs)
                    evidence[name]["max_new_tokens"] = config.measurement.max_new_tokens
                    from hashlib import sha256
                    digest = sha256()
                    with open(pred.path, "rb") as handle:
                        for chunk in iter(lambda: handle.read(1 << 20), b""):
                            digest.update(chunk)
                    evidence[name]["predictions_sha256"] = digest.hexdigest()
                    product.meta = {**(product.meta or {}), "truncation_analysis": evidence[name]}
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # Missing/malformed raw outputs never justify an exemption.
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
    affected = [name for name, item in evidence.items() if needs_expansion(item)]
    model = str((trigger.payload or {}).get("base_model") or "")
    revision = {"id": uuid4().hex[:16], "status": "running", "lane": trigger.lane,
                "base_model": model, "trigger_id": trigger.id, "evidence": evidence,
                "configs": {}, "cohort": [], "affected": affected, "reuse_tickets": True}
    try:
        previous = [b for b in batches(run) if b["lane"] == trigger.lane and b["base_model"] == model]
        all_replays = set().union(*(replay_ids(b) for b in previous)) if previous else set()
        published_replays = set().union(*(replay_ids(b) for b in previous if b["status"] == "published")) if previous else set()
        candidates = list((await session.scalars(select(Ticket).where(
            Ticket.run_id == run.id, Ticket.agent_id == "inference",
            Ticket.lane == trigger.lane, Ticket.status.in_(SUCCESS),
        ).order_by(Ticket.updated_at, Ticket.created_at))).all())
        latest = {}
        for candidate in candidates:
            payload = candidate.payload or {}
            if payload.get("base_model") != model:
                continue
            if candidate.id in all_replays - published_replays:
                continue
            # Legacy one-member mirrors cannot represent a complete suite.
            if trigger.lane == "optimization" and payload.get("test_set_name") and candidate.id not in all_replays:
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
        changed = []
        for index, name in enumerate(names):
            # A retained baseline member may still use its old cap. New revisions
            # must start from the latest published cohort budget, not that exemption.
            path = next((b["configs"].get(name, "") for b in reversed(previous)
                         if b["status"] == "published"), "")
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
                expanded = enlarged_config(config, {})
                if expanded.measurement.max_new_tokens > config.measurement.max_new_tokens:
                    changed.append(name)
                config = expanded
            destination = Path(path).parent / "budget-revisions" / revision["id"] / f"{index:03d}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
            revision["configs"][name] = str(destination)
        if not changed:
            session.add(TicketNotice(ticket_id=trigger.id, code="token_budget_ceiling", severity="info",
                                     body="Available token ceiling reached; continuing evaluation with the existing outputs."))
            await session.commit()
            return False
        affected = changed
        revision["affected"] = changed
        revision["expanded_members"] = changed
        cohort.sort(key=lambda t: t.id != trigger.id)
        revision["cohort"] = [{"source_id": t.id, "inference_id": _original_ticket_id(run, t.id),
                               "request_payload": deepcopy(t.payload), "request_inputs": deepcopy(t.inputs)} for t in cohort]
        await _plan_retention(session, run, revision, affected)
        _store(run, revision)
        session.add(TicketNotice(ticket_id=trigger.id, code="token_budget_remeasurement",
                                 severity="info", body="High question-level truncation detected. Replaying baseline and previous checkpoints under revised budgets, retaining benchmark scores with verified repetition exemptions; scores remain unchanged until the cohort completes."))
        await session.commit()
        return True
    except (ValueError, OSError) as exc:
        await _fail(session, run, revision, str(exc))
        return True


async def _plan_retention(session: AsyncSession, run: Run, batch: Batch, affected: list[str]) -> None:
    """Reuse unchanged predictions; reuse scores separately when available."""
    names = list(batch["configs"])
    for item in batch["cohort"]:
        source = await session.get(Ticket, item["source_id"])
        evidence = await _evidence(session, run, source)
        kind = "baseline" if source.payload.get("model_source") == "base_model" else "trained"
        key = f"{kind}|{source.iteration}|{batch['base_model']}"
        prefix = "suite" if batch["lane"] == "held_out_test" else "validation_suite"
        scores = (run.holdout or {}).get(f"{prefix}_results", {}).get(key, {})
        previous = next((b for b in reversed(batches(run)) if source.id in replay_ids(b)), None)
        if previous and previous["status"] == "superseded" and any(
            c.get("inference_id") == source.id and c.get("inference_attempt") for c in previous["cohort"]
        ):
            # The previous budget trial produced new predictions but never
            # published scores. Existing affected-member scores describe the
            # old predictions, even when this member's budget now stays fixed.
            scores = {name: score for name, score in scores.items() if name not in affected}
        retained = {}
        for name in names:
            ev = evidence.get(name, {})
            exempt = name in affected and ev.get("repetition", {}).get("dominated") and name in scores
            assets = {}
            for role in ("predictions", "generation_diagnostics", "inference_config"):
                product = next((p for p in await _products(session, source.id, role)
                                if (p.meta or {}).get("suite_member_name", names[0]) == name), None)
                if product is None or not Path(product.path).is_file():
                    break
                assets[role] = {"path": product.path, "meta": deepcopy(product.meta)}
            if len(assets) != 3:
                continue
            try:
                unchanged = (load_inference_config(assets["inference_config"]["path"]).model_dump()
                             == load_inference_config(batch["configs"][name]).model_dump())
            except (OSError, ValueError):
                unchanged = False
            if unchanged or exempt:
                retained[name] = {"reason": "unchanged_configuration" if unchanged else "repetition_dominated_truncation",
                                  "evidence": ev, "assets": assets,
                                  "config_path": assets["inference_config"]["path"]}
                if name in scores:
                    retained[name]["score"] = deepcopy(scores[name])
        item["retained"] = retained
        item["retained_only"] = (set(names) <= set(scores) and
                                 set(affected) <= {n for n, v in retained.items() if "score" in v})
        # If every changed member is exempt, preserve the entire existing point.
        item["active_members"] = [] if item["retained_only"] else [n for n in names if n not in retained]


def retained_scores(item: dict) -> dict:
    return {name: value for name, value in item.get("retained", {}).items() if "score" in value}


def evaluation_members(batch: Batch, item: dict) -> list[str]:
    return [name for name in batch["configs"] if name not in retained_scores(item)]


async def _begin_attempt(session: AsyncSession, run: Run, batch: Batch, item: dict,
                         ticket: Ticket, kind: str) -> None:
    from zevo.engine.run.runner import _latest_slurm_resource_request
    products = list((await session.scalars(select(WorkProduct).where(WorkProduct.ticket_id == ticket.id))).all())
    resource = await _latest_slurm_resource_request(session, ticket) if kind == "inference" else None
    item[f"{kind}_attempt"] = {
        "revision": batch["id"], "previous_status": ticket.status,
        "previous_summary": ticket.summary, "previous_payload": deepcopy(ticket.payload),
        "previous_inputs": deepcopy(ticket.inputs),
        "previous_product_ids": [p.id for p in products],
        "previous_resource_id": resource.id if resource else "",
    }
    ticket.status = "queued"
    ticket.error_message = ""
    ticket.summary = f"Remeasurement attempt {batch['id']}"
    ticket.repair_attempts = 0
    ticket.repair_route = ""
    session.add(TicketNotice(ticket_id=ticket.id, code="token_budget_attempt", severity="info",
                             body=f"Starting measurement attempt {batch['id']}; previous artifacts and scores are retained."))
    _store(run, batch)


async def _new_inference(session: AsyncSession, run: Run, batch: Batch, item: dict) -> Ticket:
    source = await session.get(Ticket, item["source_id"])
    payload = deepcopy(item.get("request_payload", source.payload))
    inputs = deepcopy(item.get("request_inputs", source.inputs) or {})
    ticket = await session.get(Ticket, item["inference_id"]) if batch.get("reuse_tickets") else None
    if ticket is not None:
        if item.get("inference_attempt"):
            return ticket
        await _begin_attempt(session, run, batch, item, ticket, "inference")
        ticket.payload, ticket.inputs = payload, inputs
    else:
        ticket = Ticket(id=item["inference_id"], run_id=run.id, agent_id="inference",
                    lane=batch["lane"], status="queued", input_format="typed",
                    iteration=source.iteration, payload=deepcopy(source.payload),
                    customization=deepcopy(source.customization or {}), inputs=deepcopy(source.inputs or {}),
                    summary="Remeasure checkpoint with the cohort's revised token budget")
        session.add(ticket)
    if item.get("retained") or (item.get("active_members") and batch.get("reuse_tickets")):
        primary = (item.get("active_members") or evaluation_members(batch, item))[0]
        member = next(m for m in _suite(run, ticket.lane) if m["name"] == primary)
        payload = deepcopy(ticket.payload)
        payload.update(test_set_name=primary if ticket.lane == "held_out_test" else "validation:" + primary,
                       scoring_set=member.get("public") or payload.get("scoring_set", ""), sample_submission=member["sample_submission"])
        pins = deepcopy(payload.get("configuration_pins") or {})
        if member.get("inference_query"):
            pins["inference_config"] = {**pins.get("inference_config", {}),
                                        "inference_query": member["inference_query"]}
        payload["configuration_pins"] = pins
        ticket.payload = payload
        if member.get("inference_data_profile"):
            ticket.inputs = {**ticket.inputs, "inference_data_profile": {
                "artifact_role": "inference_data_profile", "path": member["inference_data_profile"],
            }}
    await session.flush()
    for name, retained in item.get("retained", {}).items():
        for role, asset in retained["assets"].items():
            session.add(WorkProduct(ticket_id=ticket.id, role=role, path=asset["path"],
                meta={**asset["meta"], "suite_member_name": name, "suite_primary": name == primary}))
    if item.get("retained") and not item.get("active_members"):
        ticket.status = "succeeded"  # Only missing scores need work; no GPU replay.

    await session.commit()
    return ticket


async def _enqueue(session: AsyncSession, ticket: Ticket) -> None:
    from zevo.engine.run.wakeup import queue_wakeup
    await queue_wakeup(session, agent_id=ticket.agent_id, ticket_id=ticket.id,
                       source="handoff", reason="token budget cohort remeasurement")


async def _scores(session: AsyncSession, ticket: Ticket, suite: list[dict], run: Run | None = None) -> dict[str, dict]:
    products = await _products(session, ticket.id, "metrics", run)
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
        if item.get("retained_only"):
            continue
        inference = await session.get(Ticket, item["inference_id"])
        evaluation = await session.get(Ticket, item["evaluation_id"])
        retained = retained_scores(item)
        components = await _scores(session, evaluation, [m for m in suite if m["name"] not in retained], run)
        components.update({name: deepcopy(value["score"]) for name, value in retained.items()})
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
            event.extras = {**(event.extras or {}), "measurement_revision": batch["id"],
                            "repetition_exemptions": {n: {"reason": v["reason"], "evidence": v["evidence"]}
                                for n, v in retained.items() if v["reason"] == "repetition_dominated_truncation"},
                            "reused_members": {n: v["reason"] for n, v in item.get("retained", {}).items()}}
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
    if batch and batch["status"] == "paused":
        return  # Maintenance pauses advancement, not the cohort's loop barrier.
    if not batch:
        for completed in batches(run):
            if completed.get("handoff_pending"):
                await _handoff(session, run, deepcopy(completed))
        return
    try:
        for item in batch["cohort"]:
            if item.get("retained_only"):
                continue
            inference = await session.get(Ticket, item["inference_id"])
            if inference is None or (batch.get("reuse_tickets") and not item.get("inference_attempt")):
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
            # Settle the triggering checkpoint's budget before historical replay.
            # History is measured at that frozen budget, even when still truncated.
            expandable = []
            if item["source_id"] == batch["trigger_id"]:
                _record_budget_progress(batch, inference.id, evidence,
                                        floor=settled_budget_floor(run, batch))
                for name, ev in evidence.items():
                    if name in batch.get("stalled_members", {}):
                        continue
                    if name in item.get("retained", {}) or not needs_expansion(ev):
                        continue
                    config = load_inference_config(batch["configs"][name])
                    if enlarged_config(config, {}).measurement.max_new_tokens > config.measurement.max_new_tokens:
                        expandable.append(name)
                    else:
                        batch.setdefault("ceiling_evidence", {})[name] = ev
                if not expandable:
                    batch["budget_finalized"] = True
                    _store(run, batch)
                    await session.commit()
            if expandable:
                batch["status"] = "superseded"
                _store(run, batch)
                # The next plan must start from the attempted budgets while
                # selecting the same original cohort, not partial replay rows.
                return await _retry(session, run, batch, evidence, expandable)
            evaluation = await session.get(Ticket, item["evaluation_id"]) if item.get("evaluation_id") else None
            if evaluation is None or (batch.get("reuse_tickets") and not item.get("evaluation_attempt")):
                name = evaluation_members(batch, item)[0]
                item["evaluation_id"] = f"{item['inference_id']}-eval"
                _store(run, batch)
                products = await _products(session, inference.id, "predictions", run)
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
                if batch.get("reuse_tickets"):
                    await _begin_attempt(session, run, batch, item, evaluation, "evaluation")
                    evaluation.inputs = {**(evaluation.inputs or {}), "predictions": {
                        "source_ticket_id": inference.id, "artifact_role": "predictions",
                        "work_product_id": primary.id, "path": primary.path,
                    }}
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


def settled_budget_floor(run: Run, batch: Batch) -> int:
    """Largest token allowance already published for this model in any lane.

    Output length is a property of the model, not of the lane measuring it. A
    checkpoint whose Validation cohort settled at 512 tokens cannot be scored
    on Test at 128 just because the Test lane's smaller starting budget showed
    a flat truncation curve: below the settled floor the stall rule says
    nothing about futility, only that the doublings have not yet reached the
    answer length the other lane already measured.
    """
    floor = 0
    for other in batches(run):
        if other["status"] != "published" or other["base_model"] != batch["base_model"]:
            continue
        for record in other.get("budget_progress", []):
            floor = max(floor, int(record.get("max_new_tokens") or 0))
    return floor


def _record_budget_progress(batch: Batch, inference_id: str, evidence: dict, *, floor: int = 0) -> None:
    """Count actual budget increases once, independently for each benchmark.

    A member stalls after NO_IMPROVEMENT_LIMIT doublings without a meaningful
    truncation drop, but never while its allowance is still below `floor`, the
    budget another lane already settled for the same model.
    """
    if batch.get("progress_inference_id") == inference_id:
        return
    streaks = batch.setdefault("no_improvement_streaks", {})
    history = batch.setdefault("budget_progress", [])
    for name in batch.get("expanded_members", []):
        previous = batch["evidence"][name]["fraction"]
        current = evidence[name]["fraction"]
        relative_drop = (previous - current) / previous if previous > 0 else 0.0
        improved = relative_drop >= MIN_RELATIVE_TRUNCATION_DROP or math.isclose(
            relative_drop, MIN_RELATIVE_TRUNCATION_DROP, rel_tol=1e-9, abs_tol=1e-12)
        streaks[name] = 0 if improved else streaks.get(name, 0) + 1
        config = load_inference_config(batch["configs"][name])
        below_floor = config.measurement.max_new_tokens < floor
        record = {"member": name, "inference_id": inference_id, "revision": batch.get("id", ""),
                  "max_new_tokens": config.measurement.max_new_tokens,
                  "previous_fraction": previous, "fraction": current,
                  "relative_drop": relative_drop, "meaningful_improvement": improved,
                  "no_improvement_streak": streaks[name],
                  "settled_budget_floor": floor, "below_settled_floor": below_floor}
        history.append(record)
        if streaks[name] >= NO_IMPROVEMENT_LIMIT and not below_floor:
            batch.setdefault("stalled_members", {})[name] = record
    batch["progress_inference_id"] = inference_id


async def _retry(session: AsyncSession, run: Run, batch: Batch, evidence: dict, expandable: list[str]) -> None:
    # Each retry strictly increases a budget within its verified context ceiling.
    # Historical checkpoints are not launched until the current budget settles.
    new = deepcopy(batch)
    new.update(id=uuid4().hex[:16], status="running", evidence=evidence, configs={},
               expanded_members=expandable, budget_finalized=False)
    new.pop("progress_inference_id", None)
    for index, (name, path) in enumerate(batch["configs"].items()):
        config = load_inference_config(path)
        if name in expandable:
            config = enlarged_config(config, {})
        destination = Path(path).parent / new["id"] / f"{index:03d}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
        new["configs"][name] = str(destination)
    new["cohort"] = [{**{k: deepcopy(c[k]) for k in ("request_payload", "request_inputs") if k in c},
        "source_id": c["source_id"], "inference_id": (
        c["inference_id"] if new.get("reuse_tickets") else f"remeasure-{new['id']}-{i:03d}"
    )} for i, c in enumerate(batch["cohort"])]
    affected = sorted(set(batch.get("affected", [])) | {n for n, e in evidence.items() if needs_expansion(e)})
    new["affected"] = affected
    await _plan_retention(session, run, new, affected)
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
    if not any(needs_expansion(item) for item in evidence.values()):
        await session.commit()  # Persist repetition evidence even when no replay is needed.
        return False
    if not await _start(session, run, ticket, evidence):
        return False
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
