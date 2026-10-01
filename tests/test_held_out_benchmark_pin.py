"""Repeatable Auto runs: the user can pin the held-out benchmark and cap its rows.

Three smoke runs of one request chose three different held-out sets (TriviaQA
7,993 rows; trivia_qa_verified 3,381 rows, which failed the carve; SciQ 1,000
rows). Scope took 16.8 min when it materialized 7,993 rows and 5.8 min for
1,000. `test_benchmark` and `max_test_rows` make the population a stated input;
settlement enforces both against what scoping actually wrote.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from zevo.contracts.data import DataTaskInput
from zevo.contracts.orchestrator import AutoUserRequest, HeldOutBenchmarkPin
from zevo.contracts.scoping import BenchmarkProvenance, ScopedTestSet, ScopingResult
from zevo.contracts.tickets import DataPayload
from zevo.db.models import Run
from zevo.engine.run.runner import _build_data_input
from zevo.engine.run.scoping import (
    ScopingSettlementError,
    check_held_out_pins,
    new_scoping_ticket,
    scoping_payload,
)


def _result(tmp_path: Path, **overrides) -> ScopingResult:
    test_set = tmp_path / "bench.csv"
    test_set.write_text("id,q,answer\n1,x,A\n", encoding="utf-8")
    sample = tmp_path / "sample.csv"
    sample.write_text("id,prediction\n1,A\n", encoding="utf-8")
    body = dict(
        metric="accuracy", metric_direction="max", eval_source="public_benchmark",
        test_set_path=str(test_set), test_answer_fields=["answer"],
        test_sample_submission_path=str(sample), test_rows=1000,
        rationale="SciQ test measures closed-book science QA.",
        benchmark=BenchmarkProvenance(hub_id="allenai/sciq", split="test", rows=1000),
    )
    body.update(overrides)
    return ScopingResult(**body)


PIN = {"hub_id": "allenai/sciq", "config": "", "split": "test", "revision": ""}


def test_request_accepts_a_pin_and_a_cap_and_rejects_bad_shapes() -> None:
    req = AutoUserRequest(task_objective="o", test_benchmark=PIN, max_test_rows=500)
    assert req.test_benchmark == HeldOutBenchmarkPin(hub_id="allenai/sciq", split="test")
    assert req.max_test_rows == 500
    with pytest.raises(ValidationError, match="owner/name"):
        AutoUserRequest(task_objective="o", test_benchmark={"hub_id": "sciq", "split": "test"})
    with pytest.raises(ValidationError):
        AutoUserRequest(task_objective="o", test_benchmark={"hub_id": "allenai/sciq"})
    with pytest.raises(ValidationError):
        AutoUserRequest(task_objective="o", max_test_rows=-5)


def test_pin_and_cap_ride_the_scoping_work_order_and_nowhere_else() -> None:
    run = Run(id="auto-run-2", metric="", mode="auto", scoring_settled=False)
    request = AutoUserRequest(task_objective="o", test_benchmark=PIN, max_test_rows=500)
    payload = scoping_payload(request)
    assert payload["test_benchmark"] == PIN and payload["max_test_rows"] == 500
    ticket = new_scoping_ticket(run, request)
    inp = _build_data_input(ticket, ticket.payload, {}, "/w", run, {})
    assert isinstance(inp, DataTaskInput)
    assert inp.test_benchmark == PIN and inp.max_test_rows == 500
    # An unpinned request carries the empty defaults, not a guess.
    plain = scoping_payload(AutoUserRequest(task_objective="o"))
    assert plain["test_benchmark"] == {} and plain["max_test_rows"] == 0
    with pytest.raises(ValidationError, match="scoping fields"):
        DataPayload(
            operation="prepare_run_data", dataset="org/train", training_method="sft",
            test_benchmark=PIN,
        )
    with pytest.raises(ValidationError, match="scoping fields"):
        DataPayload(
            operation="prepare_run_data", dataset="org/train", training_method="sft",
            max_test_rows=10,
        )


def test_settlement_accepts_the_pinned_benchmark(tmp_path: Path) -> None:
    check_held_out_pins(_result(tmp_path), {"test_benchmark": PIN, "max_test_rows": 1000})
    # A pin without a revision accepts whatever revision scoping recorded.
    check_held_out_pins(
        _result(tmp_path, benchmark=BenchmarkProvenance(
            hub_id="allenai/sciq", split="test", rows=1000, revision="abc123",
        )),
        {"test_benchmark": PIN},
    )
    # No pin, no cap: nothing to enforce.
    check_held_out_pins(_result(tmp_path), {})


def test_settlement_rejects_a_different_benchmark(tmp_path: Path) -> None:
    other = _result(tmp_path, benchmark=BenchmarkProvenance(
        hub_id="mandarjoshi/trivia_qa", config="rc.nocontext", split="validation", rows=1000,
    ), rationale="TriviaQA")
    with pytest.raises(ScopingSettlementError, match="different benchmark"):
        check_held_out_pins(other, {"test_benchmark": PIN})
    with pytest.raises(ScopingSettlementError, match="not a public_benchmark"):
        check_held_out_pins(
            _result(tmp_path, eval_source="synthesized", benchmark=None,
                    rationale="synth", **_synth(1000)),
            {"test_benchmark": PIN},
        )


def test_settlement_rejects_more_rows_than_the_cap(tmp_path: Path) -> None:
    with pytest.raises(ScopingSettlementError, match="exceed max_test_rows=500: test=1000"):
        check_held_out_pins(_result(tmp_path), {"max_test_rows": 500})
    member_set = tmp_path / "m.csv"; member_set.write_text("id,q,answer\n1,x,A\n", encoding="utf-8")
    member_sample = tmp_path / "ms.csv"; member_sample.write_text("id,prediction\n1,A\n", encoding="utf-8")
    with_member = _result(tmp_path, test_rows=400, benchmark=BenchmarkProvenance(
        hub_id="allenai/sciq", split="test", rows=400,
    ), test_sets=[ScopedTestSet(
        name="extra", inference_query="answer", metric="accuracy", metric_direction="max",
        eval_source="public_benchmark", test_set_path=str(member_set),
        test_answer_fields=["answer"], test_sample_submission_path=str(member_sample),
        test_rows=700, rationale="second member",
        benchmark=BenchmarkProvenance(hub_id="org/other", split="test", rows=700),
    )])
    with pytest.raises(ScopingSettlementError, match="extra=700"):
        check_held_out_pins(with_member, {"max_test_rows": 500})


def _synth(rows: int) -> dict:
    from zevo.contracts.scoping import DecontaminationEvidence, SynthesisProvenance
    return dict(
        synthesis=SynthesisProvenance(
            teacher_model="org/teacher", generation_params={"temperature": 0.7},
            seed_sources=["syllabus"], rows_generated=1500, rows_verified=1100, rows_kept=rows,
        ),
        decontamination=DecontaminationEvidence(
            method="exact", compared_against=["org/train"], overlap_removed=10,
        ),
    )
