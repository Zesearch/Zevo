from zevo.engine.run.benchmark_telemetry import benchmark_progress_phase


def test_equal_steps_from_different_benchmarks_have_distinct_progress_keys() -> None:
    math = benchmark_progress_phase("complete", "Math")
    code = benchmark_progress_phase("complete", "Coding")
    assert math != code
    assert math == benchmark_progress_phase("complete", "Math")
    assert len(math) <= 64
    assert benchmark_progress_phase("complete", "") == "complete"
