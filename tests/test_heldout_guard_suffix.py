"""The held-out access guard must not confuse two files that share a name and
a numeric suite index in different lanes."""
from pathlib import Path

from zevo.engine.run.runner import _distinctive_suffix, _referenced_paths

HELD_OUT = "/app/data/runs/r1/holdout-data-r1-001/suite/000/validation_questions.csv"
VALIDATION = "/app/data/runs/r1/data-r1-001/validation-suite/000/validation_questions.csv"


def _call(cmd: str) -> dict:
    return {"type": "tool_call", "payload": {"input": {"command": cmd}}}


INFER_COPY = "/home/ubuntu/zevo/r1/infer-r1-002/suite/000/validation_questions.csv"
TEST_SPLIT = "/app/data/.holdout/runs/_t/t1/r1/_splits/suite-000/test_minus_validation.csv"


def test_suffix_climbs_past_numeric_and_layout_components():
    assert _distinctive_suffix(Path(HELD_OUT)) == "holdout-data-r1-001/suite/000/validation_questions.csv"
    assert _distinctive_suffix(Path(TEST_SPLIT)) == "r1/_splits/suite-000/test_minus_validation.csv"
    assert _distinctive_suffix(Path("/x/data/test.csv")) == "x/data/test.csv"
    assert _distinctive_suffix(Path("/x/results/test.csv")) == "results/test.csv"
    assert _distinctive_suffix(Path("test.csv")) == "test.csv"


def test_inference_staging_its_own_validation_copy_is_not_a_held_out_hit():
    # Run 8996a7da (2026-10-17): the optimization-lane Inference staged the
    # Validation questions under its own work dir's suite/000/ and was failed
    # as a leak because the suffix stopped at the shared layout name.
    events = [_call(f"python predict.py --questions {INFER_COPY}")]
    assert _referenced_paths([HELD_OUT], events) == []


def test_reading_own_validation_questions_is_not_a_held_out_hit():
    events = [_call(f"python predict.py --questions {VALIDATION}")]
    assert _referenced_paths([HELD_OUT], events) == []


def test_reading_the_held_out_copy_is_still_caught():
    events = [_call(f"head {HELD_OUT}")]
    assert _referenced_paths([HELD_OUT], events) == ["validation_questions.csv"]


def test_suffix_match_is_anchored_at_a_path_boundary():
    events = [_call("cat validation-suite/000/validation_questions.csv")]
    assert _referenced_paths([HELD_OUT], events) == []
    events = [_call("cd holdout && cat holdout-data-r1-001/suite/000/validation_questions.csv")]
    assert _referenced_paths([HELD_OUT], events) == ["validation_questions.csv"]
