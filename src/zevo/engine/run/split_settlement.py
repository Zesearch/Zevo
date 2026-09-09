"""Settle a Run's Validation/Test split from a complete scoring contract.

This is the one place that turns a complete ``UserRequest`` (metric, Test set,
answer fields, submission template, optional explicit Validation) into the two
lanes every Run runs on:

* the Agent-visible ``UserRequest`` whose Validation and Test fields are BLANK;
* the Run's private ``holdout`` snapshot with both lanes' paths.

It used to live inside ``POST /runs`` as ``_settle_splits`` and could therefore
only run at Run creation.  Auto mode settles AFTER creation -- once the Data
agent's ``scope_problem`` Ticket has derived the contract -- so the logic lives
here, engine-side, and raises :class:`SplitSettlementError` instead of an HTTP
error.  The router wrapper converts that to a 400 exactly as before; the
post-scoping path (``zevo.engine.run.scoping``) fails the Run with the message.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from zevo.contracts.orchestrator import UserRequest, inherit_test_validation_contract
from zevo.db import Run


class SplitSettlementError(ValueError):
    """The scoring contract cannot be split into Validation and Test lanes.

    The message is user-facing and complete; callers surface it verbatim.
    """


async def settle_splits(
    run: Run, user_request: UserRequest, *, work_dir_root: str = "",
) -> tuple[UserRequest, dict[str, Any], str]:
    """Settle scoring populations without handing either one to selection.

    Returns `(agent_request, holdout, note)`:

      * `agent_request` — the UserRequest the orchestrator receives. Its Test
        and Validation paths, fields, submissions, and evaluators are BLANK.
        Run setup still derives a deterministic 20% Validation split from Test
        when needed (at least 200 rows). The
        orchestrator plans every infer and eval ticket through
        the API-owned Run specification, so blanking the Test fields
        creates the API capability boundary: no Agent-visible Run/Ticket payload
        contains a test-set path or score. The dashboard remains trusted and
        can observe the private lane; this is deliberately not described as a
        kernel/filesystem sandbox.
      * `holdout` — the test paths, stored on the Run for the harness's own
        held-out lane (see runner._spawn_holdout_infer).
      * `note` — what happened, or what initial Data will do, for the Run plan.

    Neither scoring population becomes visible to Orchestrator or optimization
    Data. After Data fixes the Training artifact, the engine prepares the
    answer-free Validation view for Inference and binds the full population only
    to deterministic Evaluation and trainer-side evaluation.
    """
    user_request = inherit_test_validation_contract(user_request)

    from zevo.engine.remote_datasets import (
        MaterializeError, looks_like_hub_id, lookup, materialize,
    )
    from zevo.engine.method.validation_split import (
        SplitError, copy_sample_submission, resolve,
    )
    from zevo.holdout_storage import resolve_asset
    from zevo.paths import holdout_root, work_dir_root as default_work_dir_root

    root = work_dir_root or default_work_dir_root()
    out_dir = str(Path(root) / run.id / "_splits")
    private_out_dir = str(Path(holdout_root()) / "runs" / run.id / "_splits")

    # Explicit remote Validation is materialized during Run creation. Training
    # sources remain Data-stage inputs and are never consumed by split setup.
    validation_set = user_request.validation_set
    validation_fetched_note = ""
    if validation_set and looks_like_hub_id(validation_set):
        split = user_request.validation_split
        config = user_request.validation_config
        if not split:
            # The catalogue may already say which slice this repo is listed for.
            declared = lookup(validation_set, role="validation")
            if declared is not None:
                split, config = split or declared.split, config or declared.config
        try:
            # No limit: the split the user named is the validation set they
            # asked for, all of it.
            validation_set, _cols, _n, validation_fetched_note = await materialize(
                hub_id=validation_set, split=split, config=config, out_dir=out_dir,
            )
        except MaterializeError as e:
            raise SplitSettlementError(f"cannot fetch the validation set: {e}") from e

    if validation_set:
        try:
            carved = resolve(
                validation_set=validation_set,
                validation_answer_fields=user_request.validation_answer_fields,
                dataset=user_request.dataset,
                test_set=user_request.test_set,
                test_answer_fields=user_request.test_answer_fields,
                out_dir=out_dir,
            )
        except SplitError as e:
            raise SplitSettlementError(
                f"cannot settle the validation split: {e}"
            ) from e
        validation_path = carved.validation_set
        validation_fields = carved.validation_answer_fields
        validation_source = "huggingface" if validation_fetched_note else carved.source
        validation_needs_metric_binding = bool(carved.needs_metric_binding)
        notes = list(carved.notes)
    else:
        try:
            from zevo.engine.method.validation_split import carve
            carved = carve(
                dataset=user_request.dataset,
                test_set=resolve_asset(user_request.test_set),
                test_answer_fields=user_request.test_answer_fields,
                out_dir=out_dir,
                remaining_out_dir=private_out_dir,
            )
            sample_source = resolve_asset(user_request.test_sample_submission)
            sample_suffix = Path(sample_source).suffix or ".csv"
            validation_sample, remaining_sample = copy_sample_submission(
                source=sample_source,
                validation_out=str(Path(out_dir) / f"validation_submission{sample_suffix}"),
                remaining_out=str(Path(private_out_dir) / f"test_submission{sample_suffix}"),
            )
        except (SplitError, OSError, ValueError) as e:
            raise SplitSettlementError(
                f"cannot derive Validation from Test: {e}"
            ) from e
        validation_path = carved.validation_set
        validation_fields = carved.validation_answer_fields
        validation_source = "test_split"
        validation_needs_metric_binding = False
        notes = list(carved.notes)
    if validation_fetched_note:
        notes.insert(0, validation_fetched_note)

    # Store opaque semantic identities rather than Test rows in the ordinary
    # Run snapshot. The post-Data engine transform can remove accidental
    # benchmark overlap without exposing Test paths or examples to an Agent.
    try:
        from zevo.engine.artifact_validation import semantic_record_fingerprints

        test_fingerprint_source = (
            carved.test_set
            if validation_source == "test_split"
            else resolve_asset(user_request.test_set)
        )
        test_semantic_fingerprints = sorted(
            semantic_record_fingerprints(test_fingerprint_source)
        )
    except (OSError, ValueError) as exc:
        raise SplitSettlementError(
            f"cannot fingerprint the held-out Test population: {exc}"
        ) from exc

    # How many rows the loop is judged on each iteration. The orchestrator is
    # told this so "has it plateaued?" can be a judgement about evidence rather
    # than about a number someone picked: on a 150-row set one row is 0.67
    # points, so a 1-point move is a single row changing its mind.
    validation_rows = 0
    try:
        from zevo.engine.method.validation_split import _read_rows

        _, _rows = _read_rows(Path(validation_path))
        validation_rows = len(_rows)
    except Exception:
        validation_rows = 0

    holdout = {
        # Only the remaining 80% is the final private Test population.
        "test_set": carved.test_set if validation_source == "test_split" else user_request.test_set,
        "test_answer_fields": list(user_request.test_answer_fields or []),
        "test_semantic_fingerprints": test_semantic_fingerprints,
        # Engine-owned only. Neither Orchestrator nor optimization Data receives
        # this path; the runner creates the answer-free view after Data returns.
        "validation_set": validation_path,
        "validation_answer_fields": validation_fields,
        # Filled in by the runner's deterministic post-Data transform.
        "validation_public": "",
        # Closed, answer-free aggregate/schema evidence produced beside the
        # Validation public copy and consumed only by Run Setup planning/checks.
        "inference_data_profile": "",
        "test_public": "",
        "test_sample_submission": (
            remaining_sample if validation_source == "test_split"
            else user_request.test_sample_submission
        ),
        # Test-derived Validation reuses this Test contract exactly. Only a
        # separately supplied Validation set may use an independent scorer.
        "test_metric_type": user_request.metric_type,
        "test_evaluation_script": user_request.evaluation_script,
        "test_evaluator_sha256": user_request.evaluator_sha256,
        "validation_metric_type": user_request.validation_metric_type,
        "validation_evaluation_script": user_request.validation_evaluation_script,
        "validation_evaluator_sha256": user_request.validation_evaluator_sha256,
        # Derived Validation receives Test's authoritative submission
        # schema/example; explicit Validation retains its own template.
        "validation_sample_submission": (
            validation_sample if validation_source == "test_split"
            else user_request.validation_sample_submission
        ),
        "validation_needs_metric_binding": validation_needs_metric_binding,
        "validation_policy": "supplied",
        "validation_fraction": 0.0,
        # Which slice of a hub training repo to pull. Stamped onto the data
        # ticket by the runner: the orchestrator plans what to try, and which
        # rows a repo id refers to is not one of its decisions.
        "dataset_split": user_request.dataset_split,
        "dataset_config": user_request.dataset_config,
        "dataset_source_split": user_request.dataset_split,
        "dataset_source_config": user_request.dataset_config,
        "validation_source": validation_source,
        "validation_rows": validation_rows,
        "note": "; ".join(notes),
    }
    agent_request = user_request.model_copy(update={
        # The optimization loop speaks the existing generic metric vocabulary,
        # but those values now explicitly come from the Validation contract.
        "metric_type": user_request.validation_metric_type,
        "metric": user_request.validation_metric,
        "metric_direction": user_request.validation_metric_direction,
        # Scoring populations are not a supervisor capability. Later pipeline
        # tickets receive only the exact engine-owned assets required for their
        # own operation.
        "validation_set": "",
        "validation_answer_fields": [],
        "test_set": "",
        "test_answer_fields": [],
        "test_sample_submission": "",
        "evaluation_script": "",
        "evaluator_sha256": "",
        "validation_evaluation_script": "",
        "validation_evaluator_sha256": "",
        "validation_sample_submission": "",
    })
    note = "; ".join(notes)
    return agent_request, holdout, note
