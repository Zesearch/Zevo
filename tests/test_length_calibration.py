from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from zevo.contracts.length_calibration import (
    REPORT_NAME, check, main, run_measured, sample_rows, validate,
)


def inputs(root: Path, n: int = 100, truncated: int = 0):
    root.mkdir(parents=True, exist_ok=True)
    questions = root / "questions.csv"
    with questions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question"])
        writer.writeheader()
        writer.writerows({"id": i, "question": "x" * (i + 1)} for i in range(n))
    config = root / "inference_config.yaml"
    config.write_text("measurement:\n  max_new_tokens: 512\n")
    diagnostics = root / "diagnostics.json"
    diagnostics.write_text(json.dumps({"records": [
        {"request_index": i, "row_index": i, "generated_tokens": 512 if i < truncated else 30,
         "finish_reason": "length" if i < truncated else "stop"}
        for i in range(min(64, n))
    ]}))
    return config, questions, diagnostics


def test_sample_is_repeatable_and_covers_length_tails(tmp_path):
    _, q, _ = inputs(tmp_path)
    _, rows, indices = sample_rows(q)
    assert len(rows) == len(set(indices)) == 64
    assert indices[0] == 0 and indices[-1] == 99
    assert sample_rows(q) == sample_rows(q)
    target = tmp_path / "sample.csv"
    assert main(["prepare", "--questions", str(q), "--output", str(target)]) == 0
    with target.open() as handle:
        assert list(csv.DictReader(handle)) == rows


@pytest.mark.parametrize("n,truncated,passes", [(100, 3, True), (100, 4, False),
    (20, 1, True), (20, 2, False), (1, 0, True), (1, 1, False)])
def test_five_percent_boundary_including_small_sets(tmp_path, n, truncated, passes):
    c, q, d = inputs(tmp_path, n, truncated)
    if passes:
        check(c, q, d)
        validate(c, q)
    else:
        with pytest.raises(ValueError, match="maximum is 5%"):
            check(c, q, d)
        assert not c.with_name(REPORT_NAME).exists()


def test_failed_recheck_invalidates_previous_pass(tmp_path):
    c, q, d = inputs(tmp_path, truncated=3)
    check(c, q, d)
    inputs(tmp_path, truncated=4)
    with pytest.raises(ValueError):
        check(c, q, d)
    with pytest.raises(FileNotFoundError):
        validate(c, q)


@pytest.mark.parametrize("change", ["config", "questions", "indices", "missing_row", "false_pass"])
def test_stale_or_fabricated_summary_cannot_pass(tmp_path, change):
    c, q, d = inputs(tmp_path)
    report = check(c, q, d)
    data = json.loads(report.read_text())
    if change == "config":
        c.write_text(c.read_text().replace("512", "1024"))
    elif change == "questions":
        q.write_text(q.read_text().replace("x", "y"))
    elif change == "indices":
        data["sample_row_indices"] = list(range(8))
    elif change == "missing_row":
        data["records"] = data["records"][:-1]
    else:
        data["passed"] = True
        for row in data["records"][:4]:
            row["finish_reason"] = "length"
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate(c, q)


def test_multiturn_counts_questions_not_requests(tmp_path):
    c, q, d = inputs(tmp_path, n=20)
    data = json.loads(d.read_text())
    for i in range(20, 40):
        data["records"].append({"request_index": i, "row_index": 0,
                                "finish_reason": "length", "generated_tokens": 512})
    d.write_text(json.dumps(data))
    check(c, q, d)  # one affected question out of twenty: exactly 5%
    data["records"][-1]["row_index"] = 1
    d.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        check(c, q, d)


def test_suite_gate_blocks_all_generation_until_every_member_passes(tmp_path):
    a = inputs(tmp_path / "a", truncated=0)
    b = inputs(tmp_path / "b", truncated=4)
    check(*a)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"members": [
        {"config": str(c), "questions": str(q)} for c, q, _ in [a, b]
    ]}))
    marker = tmp_path / "started"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)]
    with pytest.raises(FileNotFoundError):
        run_measured(suite, command)
    assert not marker.exists()
    b = inputs(tmp_path / "b", truncated=3)
    check(*b)
    assert run_measured(suite, command) == 0
    assert marker.exists()


def waiver_inputs(tmp_path):
    import hashlib

    c, q, d = inputs(tmp_path, truncated=8)
    data = json.loads(d.read_text())
    _, _, indices = sample_rows(q)
    data['output_behavior_waiver'] = {
        'previous_attempts': [
            {'config_sha256': hashlib.sha256(str(cap).encode()).hexdigest(),
             'questions_sha256': hashlib.sha256(q.read_bytes()).hexdigest(),
             'sample_row_indices': indices, 'records': data['records'],
             'adjustment': f'Increased budget from {cap}; checked EOS'}
            for cap in [128, 256]
        ],
        'reason': 'Model repeats the answer and continues despite increased budgets',
        'stop_controls_checked': 'Verified EOS token and stop IDs against tokenizer',
        'examples': [{'request_index': 0, 'output_excerpt': 'Answer A. Answer A. Answer A.',
                      'observation': 'Repeated answer until length limit'}],
    }
    d.write_text(json.dumps(data))
    return c, q, d, data


def test_documented_output_behavior_can_pass_above_target(tmp_path):
    c, q, d, _ = waiver_inputs(tmp_path)
    report = check(c, q, d)
    validate(c, q)
    result = json.loads(report.read_text())
    assert result['acceptance'] == 'accepted_output_behavior'
    assert result['truncated_questions'] == 8
    assert result['sample_size'] == 64


@pytest.mark.parametrize('invalid', ['one_attempt', 'same_config', 'different_sample',
                                    'no_examples', 'not_truncated', 'no_reason'])
def test_retry_count_alone_cannot_waive_truncation(tmp_path, invalid):
    c, q, d, data = waiver_inputs(tmp_path)
    waiver = data['output_behavior_waiver']
    if invalid == 'one_attempt':
        waiver['previous_attempts'].pop()
    elif invalid == 'same_config':
        waiver['previous_attempts'][1]['config_sha256'] = waiver['previous_attempts'][0]['config_sha256']
    elif invalid == 'different_sample':
        waiver['previous_attempts'][0]['sample_row_indices'] = [0]
    elif invalid == 'no_examples':
        waiver['examples'] = []
    elif invalid == 'not_truncated':
        waiver['examples'][0]['request_index'] = 63
    else:
        waiver['reason'] = ''
    d.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        check(c, q, d)
    assert not c.with_name(REPORT_NAME).exists()
