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

import csv
from pathlib import Path
from typing import Any

from zevo.code_benchmarks import code_execution_adapter_for
from zevo.contracts.orchestrator import (
    TaskTestSet,
    UserRequest,
    effective_test_suite,
    inherit_test_validation_contract,
)
from zevo.db import Run
from zevo.holdout_storage import resolve_scoring_file


class SplitSettlementError(ValueError):
    """The scoring contract cannot be split into Validation and Test lanes.

    The message is user-facing and complete; callers surface it verbatim.
    """


def _checked_scoring_file(
    path: str, *, lane: str, member: str, field: str, private: bool,
) -> str:
    try:
        resolved = resolve_scoring_file(path, private=private)
        if field == "sample submission":
            with Path(resolved).open("r", encoding="utf-8-sig", newline="") as handle:
                columns = list(next(csv.reader(handle), []))
            if not columns:
                raise ValueError("CSV header is missing")
            if len(columns) != len(set(columns)):
                raise ValueError("CSV header contains duplicate columns")
        return resolved
    except (OSError, ValueError, csv.Error) as exc:
        raise SplitSettlementError(
            f"{lane} set {member!r} {field} is invalid: {exc}"
        ) from exc


async def settle_splits(
    run: Run, user_request: UserRequest, *, work_dir_root: str = "",
) -> tuple[UserRequest, dict[str, Any], str]:
    """Settle scoring populations without handing either one to selection.

    Returns `(agent_request, holdout, note)`:

      * `agent_request` — the UserRequest the orchestrator receives. Its Test
        and Validation paths, fields, submissions, and evaluators are BLANK.
        Run setup still derives a deterministic 20% Validation member from
        every sufficiently large Test-suite member. Small benchmarks remain
        final-test-only. The
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
        FRACTION,
        MIN_VALIDATION_ROWS,
        InsufficientValidationRows,
        SplitError,
        _read_rows,
        carve,
        copy_sample_submission,
        resolve,
    )
    from zevo.holdout_storage import resolve_asset
    from zevo.paths import holdout_root, work_dir_root as default_work_dir_root

    root = work_dir_root or default_work_dir_root()
    out_dir = str(Path(root) / run.id / "_splits")
    private_out_dir = str(Path(holdout_root()) / "runs" / run.id / "_splits")

    # A hub id is a direct Test-set reference, not an instruction for the user
    # to download and upload it. Materialize every suite member behind the
    # held-out boundary before deriving the Validation suite.
    suite = effective_test_suite(user_request)
    # Freeze local Test references to their protected, absolute execution
    # paths before any remote fetch. The Task's saved logical paths stay as-is.
    suite = [
        item.model_copy(update={
            "test_set": (
                item.test_set if looks_like_hub_id(item.test_set)
                else _checked_scoring_file(
                    item.test_set, lane="Test", member=item.name,
                    field="data file", private=True,
                )
            ),
            "sample_submission": _checked_scoring_file(
                item.sample_submission, lane="Test", member=item.name,
                field="sample submission", private=True,
            ),
        })
        for item in suite
    ]
    code_execution_adapters = {
        item.name.casefold(): code_execution_adapter_for(item.test_set)
        for item in suite
    }
    from zevo.engine.run.setup_progress import update as setup_progress

    materialized_suite = []
    for index, item in enumerate(suite):
        setup_progress(
            phase="benchmarks",
            completed=index,
            total=len(suite),
            label=item.name,
        )
        test_set = item.test_set
        if looks_like_hub_id(test_set):
            # The suite member is the scoring contract. Its explicit location
            # must win over catalogue discovery: the same Hub repository may
            # carry several configs/splits, and deployments need not keep a
            # local Files catalogue at all. The catalogue is only a fallback
            # for fields the member left blank.
            split = item.split
            config = item.config
            if not split or not config:
                declared = lookup(test_set, role="test")
                if declared is not None:
                    split = split or declared.split
                    config = config or declared.config
            try:
                test_set, _columns, _rows, fetched_note = await materialize(
                    hub_id=test_set,
                    split=split,
                    config=config,
                    out_dir=str(Path(private_out_dir) / f"suite-{index:03d}"),
                    answer_scope="test",
                )
            except MaterializeError as exc:
                raise SplitSettlementError(
                    f"cannot fetch Test set {item.name!r}: {exc}"
                ) from exc
            item = item.model_copy(update={"test_set": test_set})
            # Keep the note engine-side; it contains no rows or answers.
            materialized_note = f"{item.name}: {fetched_note}"
        else:
            materialized_note = ""
        materialized_suite.append((item, materialized_note))
        setup_progress(
            phase="benchmarks",
            completed=index + 1,
            total=len(suite),
            label=item.name,
        )
    if not materialized_suite:
        raise SplitSettlementError("at least one Test set is required")
    suite = [item for item, _note in materialized_suite]
    primary = suite[0]
    user_request = user_request.model_copy(update={
        "test_sets": suite,
        "test_set": primary.test_set,
        "test_answer_fields": list(primary.answer_fields),
        "test_sample_submission": primary.sample_submission,
        "metric": primary.metric,
        "metric_direction": primary.metric_direction,
    })

    validation_suite: list[dict[str, Any]] = []
    final_test_only: list[dict[str, Any]] = []
    notes: list[str] = []
    explicit_validation_suite = list(user_request.validation_sets)
    if not explicit_validation_suite and user_request.validation_set.strip():
        # Preserve the existing one-upload form as a one-member suite.
        # model_construct keeps settlement's useful error ordering for an
        # incomplete legacy draft: remote fetch errors surface before local
        # field validation. Normal API creation rejects incomplete assets.
        explicit_validation_suite = [TaskTestSet.model_construct(
            name="validation",
            test_set=user_request.validation_set,
            split=user_request.validation_split,
            config=user_request.validation_config,
            inference_query=primary.inference_query,
            sample_submission=user_request.validation_sample_submission,
            metric_type=user_request.validation_metric_type or "builtin",
            metric=user_request.validation_metric,
            metric_direction=user_request.validation_metric_direction or "max",
            answer_fields=list(user_request.validation_answer_fields),
            evaluation_script=user_request.validation_evaluation_script,
            evaluator_sha256=user_request.validation_evaluator_sha256,
        )]
    has_explicit_validation = bool(explicit_validation_suite)

    if has_explicit_validation:
        # Materialize and validate every independent member. Test paths are not
        # rewritten in this branch: an independent suite leaves 100% of every
        # original benchmark for final held-out evaluation.
        total = len(explicit_validation_suite)
        for index, item in enumerate(explicit_validation_suite):
            setup_progress(
                phase="validation", completed=index, total=total, label=item.name,
            )
            member_out = str(Path(out_dir) / f"validation-suite-{index:03d}")
            sample_submission = _checked_scoring_file(
                item.sample_submission, lane="Validation", member=item.name,
                field="sample submission", private=False,
            )
            source = item.test_set
            fetched_note = ""
            materialized_rows = 0
            if looks_like_hub_id(source):
                split = item.split
                config = item.config
                if not split or not config:
                    declared = lookup(source, role="validation")
                    if declared is not None:
                        split = split or declared.split
                        config = config or declared.config
                try:
                    source, _columns, materialized_rows, fetched_note = (
                        await materialize(
                            hub_id=source,
                            split=split,
                            config=config,
                            out_dir=member_out,
                            limit=item.max_rows,
                            answer_scope="validation",
                        )
                    )
                except MaterializeError as exc:
                    raise SplitSettlementError(
                        f"cannot fetch Validation set {item.name!r}: {exc}"
                    ) from exc
            try:
                resolved = resolve(
                    validation_set=source,
                    validation_answer_fields=list(item.answer_fields),
                    dataset=user_request.dataset,
                    test_set=user_request.test_set,
                    test_answer_fields=user_request.test_answer_fields,
                    out_dir=member_out,
                )
            except SplitError as exc:
                raise SplitSettlementError(
                    f"cannot settle Validation set {item.name!r}: {exc}"
                ) from exc
            # A Setting may point at a repository-bundled local asset. Store
            # its absolute path, since downstream fingerprinting and artifact
            # validation deliberately reject relative scoring paths.
            validation_source_path = str(Path(resolved.validation_set).resolve())
            if not materialized_rows:
                try:
                    _columns, explicit_rows = _read_rows(
                        Path(validation_source_path),
                    )
                    materialized_rows = len(explicit_rows)
                except (OSError, ValueError) as exc:
                    raise SplitSettlementError(
                        f"cannot count Validation set {item.name!r}: {exc}"
                    ) from exc
            if materialized_rows < MIN_VALIDATION_ROWS:
                raise SplitSettlementError(
                    f"Validation set {item.name!r} has {materialized_rows} "
                    f"row(s); each Validation set requires at least "
                    f"{MIN_VALIDATION_ROWS}. Use the full split or choose a "
                    "larger independent dataset."
                )
            validation_suite.append({
                "name": item.name,
                "validation_set": validation_source_path,
                "inference_query": item.inference_query,
                "sample_submission": sample_submission,
                "metric_type": item.metric_type,
                "metric": item.metric,
                "metric_direction": item.metric_direction,
                "answer_fields": list(resolved.validation_answer_fields),
                "evaluation_script": item.evaluation_script,
                "evaluator_sha256": item.evaluator_sha256,
                "public": "",
                "inference_data_profile": "",
                "source": "huggingface" if fetched_note else resolved.source,
                "n_rows": materialized_rows,
                "code_execution_adapter": code_execution_adapter_for(item.test_set),
            })
            notes.extend(f"{item.name}: {note}" for note in resolved.notes)
            if fetched_note:
                notes.append(f"{item.name}: {fetched_note}")
            setup_progress(
                phase="validation",
                completed=index + 1,
                total=total,
                label=item.name,
            )
        validation_path = str(validation_suite[0]["validation_set"])
        validation_fields = list(validation_suite[0]["answer_fields"])
        validation_source = (
            str(validation_suite[0]["source"])
            if len(validation_suite) == 1 else "independent_validation_suite"
        )
        validation_needs_metric_binding = False
    else:
        # Each benchmark keeps its own prompt, schema, metric and evaluator.
        # Deriving one Validation member at a time preserves those contracts;
        # concatenating heterogeneous files would lose exactly the information
        # the Task suite was introduced to represent.
        settled_suite = list(suite)
        for index, item in enumerate(suite):
            setup_progress(
                phase="validation",
                completed=index,
                total=len(suite),
                label=item.name,
            )
            member_out = str(Path(out_dir) / f"suite-{index:03d}")
            member_private = str(Path(private_out_dir) / f"suite-{index:03d}")
            try:
                member = carve(
                    dataset=user_request.dataset,
                    test_set=resolve_asset(item.test_set),
                    test_answer_fields=list(item.answer_fields),
                    out_dir=member_out,
                    remaining_out_dir=member_private,
                )
            except InsufficientValidationRows as exc:
                final_test_only.append({
                    "name": item.name,
                    "reason": str(exc),
                })
                notes.append(f"{item.name}: final-test-only ({exc})")
                setup_progress(
                    phase="validation",
                    completed=index + 1,
                    total=len(suite),
                    label=item.name,
                )
                continue
            except (SplitError, OSError, ValueError) as exc:
                raise SplitSettlementError(
                    f"cannot derive Validation from Test set {item.name!r}: {exc}"
                ) from exc

            try:
                sample_source = resolve_asset(item.sample_submission)
                sample_suffix = Path(sample_source).suffix or ".csv"
                validation_sample, remaining_sample = copy_sample_submission(
                    source=sample_source,
                    validation_out=str(
                        Path(member_out) / f"validation_submission{sample_suffix}"
                    ),
                    remaining_out=str(
                        Path(member_private) / f"test_submission{sample_suffix}"
                    ),
                )
            except (SplitError, OSError, ValueError) as exc:
                raise SplitSettlementError(
                    f"cannot copy scoring format for Test set {item.name!r}: {exc}"
                ) from exc

            settled_suite[index] = item.model_copy(update={
                "test_set": member.test_set,
                "sample_submission": remaining_sample,
            })
            validation_suite.append({
                "name": item.name,
                "validation_set": member.validation_set,
                "inference_query": item.inference_query,
                "sample_submission": validation_sample,
                "metric_type": item.metric_type,
                "metric": item.metric,
                "metric_direction": item.metric_direction,
                "answer_fields": list(item.answer_fields),
                "evaluation_script": item.evaluation_script,
                "evaluator_sha256": item.evaluator_sha256,
                "public": "",
                "inference_data_profile": "",
                "source": "test_split",
                "n_rows": member.n_validation,
                "code_execution_adapter": code_execution_adapters.get(
                    item.name.casefold(), "",
                ),
            })
            notes.extend(f"{item.name}: {note}" for note in member.notes)
            setup_progress(
                phase="validation",
                completed=index + 1,
                total=len(suite),
                label=item.name,
            )

        if not validation_suite:
            names = ", ".join(item.name for item in suite)
            raise SplitSettlementError(
                "none of the Task's Test sets is large enough to construct a "
                f"reliable Validation signal ({names}). Add a larger benchmark, "
                "an official development set, or an independent Validation set."
            )
        suite = settled_suite
        validation_path = str(validation_suite[0]["validation_set"])
        validation_fields = list(validation_suite[0]["answer_fields"])
        validation_source = (
            "test_split" if len(validation_suite) == 1 else "test_suite_split"
        )
        validation_needs_metric_binding = False
    notes[:0] = [note for _item, note in materialized_suite if note]

    primary_validation = validation_suite[0]
    validation_sample = str(primary_validation["sample_submission"])

    # Store opaque semantic identities rather than Test rows in the ordinary
    # Run snapshot. The post-Data engine transform can remove accidental
    # benchmark overlap without exposing Test paths or examples to an Agent.
    try:
        from zevo.engine.artifact_validation import semantic_record_fingerprints

        # Independent Validation must not reuse a final Test question. The
        # comparison is input-only, so different answer schemas cannot hide
        # an overlap. The same identities then decontaminate Training.
        final_test_fingerprints = {
            fingerprint
            for source, answers in (
                (resolve_asset(item.test_set), list(item.answer_fields))
                for item in suite
            )
            for fingerprint in semantic_record_fingerprints(
                source, excluded_fields=answers,
            )
        }
        validation_fingerprints = {
            fingerprint
            for item in validation_suite
            for fingerprint in semantic_record_fingerprints(
                str(item["validation_set"]),
                excluded_fields=list(item.get("answer_fields") or []),
            )
        }
        shared = final_test_fingerprints & validation_fingerprints
        if shared:
            raise SplitSettlementError(
                f"Validation and final Test share {len(shared)} question(s); "
                "choose an independent Validation set or remove the duplicates"
            )
        test_semantic_fingerprints = sorted(
            final_test_fingerprints | validation_fingerprints
        )
    except SplitSettlementError:
        raise
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

        validation_rows = sum(int(item.get("n_rows") or 0) for item in validation_suite)
        if not validation_rows:
            for item in validation_suite:
                _, rows = _read_rows(Path(str(item["validation_set"])))
                validation_rows += len(rows)
    except Exception:
        validation_rows = 0

    holdout = {
        "test_sets": [
            {
                **item.model_dump(mode="json"),
                "code_execution_adapter": code_execution_adapters.get(
                    item.name.casefold(), "",
                ),
                "public": "",
                "inference_data_profile": "",
            }
            for item in suite
        ],
        "suite_results": {},
        "suite_recorded": [],
        "validation_sets": validation_suite,
        "validation_suite_results": {},
        "validation_suite_recorded": [],
        "validation_final_test_only": final_test_only,
        "validation_aggregation": "unweighted_mean",
        # Scalar fields remain a projection for Train and older dashboard
        # readers. Inference/Evaluation model selection uses validation_sets.
        "validation_inference_query": str(primary_validation["inference_query"]),
        "test_set": suite[0].test_set,
        "test_answer_fields": list(suite[0].answer_fields),
        "test_semantic_fingerprints": test_semantic_fingerprints,
        # Engine-owned only. Neither Orchestrator nor optimization Data receives
        # this path; the runner creates the answer-free view after Data returns.
        "validation_set": validation_path,
        "validation_answer_fields": list(primary_validation["answer_fields"]),
        # Filled in by the runner's deterministic post-Data transform.
        "validation_public": "",
        # Closed, answer-free aggregate/schema evidence produced beside the
        # Validation public copy and consumed only by Run Setup planning/checks.
        "inference_data_profile": "",
        "test_public": "",
        "test_sample_submission": suite[0].sample_submission,
        # Test-derived Validation reuses this Test contract exactly. Only a
        # separately supplied Validation set may use an independent scorer.
        "test_metric_type": user_request.metric_type,
        "test_evaluation_script": user_request.evaluation_script,
        "test_evaluator_sha256": user_request.evaluator_sha256,
        "validation_metric_type": str(primary_validation["metric_type"]),
        "validation_evaluation_script": str(primary_validation["evaluation_script"]),
        "validation_evaluator_sha256": str(primary_validation["evaluator_sha256"]),
        # Derived Validation receives Test's authoritative submission
        # schema/example; explicit Validation retains its own template.
        "validation_sample_submission": validation_sample,
        "validation_needs_metric_binding": validation_needs_metric_binding,
        # Legacy scalar Data contract remains "supplied": settlement has
        # already performed the split before Data runs. The suite-specific
        # fields describe how those supplied artifacts were constructed.
        "validation_policy": "supplied",
        "validation_fraction": 0.0,
        "validation_suite_policy": (
            "supplied" if has_explicit_validation else "benchmark_suite_20_percent"
        ),
        "validation_suite_fraction": 0.0 if has_explicit_validation else FRACTION,
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
    aggregate_validation = len(validation_suite) > 1
    agent_request = user_request.model_copy(update={
        # The optimization loop speaks the existing generic metric vocabulary,
        # but those values now explicitly come from the Validation contract.
        "metric_type": (
            "builtin" if aggregate_validation else str(primary_validation["metric_type"])
        ),
        "metric": (
            "suite_average" if aggregate_validation else str(primary_validation["metric"])
        ),
        "metric_direction": str(primary_validation["metric_direction"]),
        "validation_metric_type": (
            "builtin" if aggregate_validation else str(primary_validation["metric_type"])
        ),
        "validation_metric": (
            "suite_average" if aggregate_validation else str(primary_validation["metric"])
        ),
        "validation_metric_direction": str(primary_validation["metric_direction"]),
        # Scoring populations are not a supervisor capability. Later pipeline
        # tickets receive only the exact engine-owned assets required for their
        # own operation.
        "validation_set": "",
        "validation_sets": [],
        "validation_answer_fields": [],
        "test_set": "",
        "test_sets": [],
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
