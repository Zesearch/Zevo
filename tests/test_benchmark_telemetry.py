from zevo.engine.run.benchmark_telemetry import (
    benchmark_progress_phase,
    canonical_benchmark_marker,
)


def test_equal_steps_from_different_benchmarks_have_distinct_progress_keys() -> None:
    math = benchmark_progress_phase("complete", "Math")
    code = benchmark_progress_phase("complete", "Coding")
    assert math != code
    assert math == benchmark_progress_phase("complete", "Math")
    assert len(math) <= 64
    assert benchmark_progress_phase("complete", "") == "complete"


def test_canonical_marker_uses_id_or_ordered_slot_not_name() -> None:
    holdout = {"validation_sets": [
        {"name": "Math · MATH-Hard"}, {"name": "Chat · Dolly"},
    ]}
    marker = canonical_benchmark_marker({
        "benchmark_id": "validation:0",
        "benchmark_name": "Math · MATH Hard",
        "active_benchmark_ids": ["validation:0", "validation:1"],
    }, holdout, "optimization")
    assert marker["benchmark_name"] == "Math · MATH-Hard"
    assert marker["active_benchmarks"] == [
        "Math · MATH-Hard", "Chat · Dolly",
    ]

    old = canonical_benchmark_marker({
        "benchmark_name": "Math · MATH Hard", "benchmark_index": 1,
        "benchmark_total": 2,
    }, holdout, "optimization")
    assert old["benchmark_id"] == "validation:0"
    assert old["benchmark_name"] == "Math · MATH-Hard"

    invalid = canonical_benchmark_marker({
        "benchmark_id": "validation:99", "benchmark_name": "Math · MATH-Hard",
        "benchmark_index": 1, "benchmark_total": 2,
    }, holdout, "optimization")
    assert invalid["benchmark_id"] == "validation:99"

    # A legacy evaluator can score a non-first primary member first. Its
    # exact name is safer than treating that evaluation order as suite order.
    heldout = {"test_sets": [{"name": "A"}, {"name": "B"}]}
    reordered = canonical_benchmark_marker({
        "benchmark_name": "B", "benchmark_index": 1, "benchmark_total": 2,
    }, heldout, "held_out_test")
    assert reordered["benchmark_id"] == "test:1"
