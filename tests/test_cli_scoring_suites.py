import json

from zevo.cli.zevo import _setting_json, _task_test_sets


def test_cli_suite_parser_accepts_multiple_validation_members() -> None:
    members = [
        {"name": "math", "test_set": "org/math", "sample_submission": "math.csv"},
        {"name": "code", "test_set": "org/code", "sample_submission": "code.csv"},
    ]

    assert _task_test_sets(json.dumps(members), flag="--validation-sets") == members


def test_setting_json_keeps_validation_suite(tmp_path) -> None:
    members = [{"name": "math"}, {"name": "code"}]
    path = tmp_path / "setting.json"
    path.write_text(json.dumps({"name": "s1", "validation_sets": members}))

    assert _setting_json(str(path))["validation_sets"] == members
