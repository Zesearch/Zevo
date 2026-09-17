"""The model-agnostic suite launcher preserves benchmark measurement order."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / "playbook/runners/parallel_inference.py"

FAKE_PREDICT = r'''
import argparse
import csv
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
for option in ("model", "suite", "summary", "ticket-id"):
    parser.add_argument("--" + option, required=True)
args = parser.parse_args()
members = json.load(open(args.suite, encoding="utf-8"))["members"]
summaries = []
print("MASK=" + os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
if os.environ.get("FAKE_FAIL_MASK") == os.environ["CUDA_VISIBLE_DEVICES"]:
    time.sleep(0.2)
    sys.exit(5)
for index, member in enumerate(members, start=1):
    with open(member["questions"], newline="", encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))
    with open(member["output"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction"])
        writer.writeheader()
        for row in questions:
            writer.writerow({"id": row["id"], "prediction": "answer-" + row["id"]})
    turns = int(os.environ.get("FAKE_TURNS", "1"))
    records = [{"row_index": j, "request_index": j * turns + turn,
                "generated_tokens": 2, "finish_reason": "stop", "turn": turn + 1}
               for j in range(len(questions)) for turn in range(turns)]
    with open(member["diagnostics"], "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "records": records}, handle)
    progress = {"step": len(questions), "total": len(questions),
                "benchmark_name": member["name"], "benchmark_index": index,
                "benchmark_total": len(members)}
    print("__PROGRESS__:" + args.ticket_id + ":" + json.dumps(progress), flush=True)
    summaries.append({"name": member["name"], "n_rows": len(questions),
                      "n_requests": len(records), "n_unparseable": 0})
    time.sleep(0.01)
with open(args.summary, "w", encoding="utf-8") as handle:
    json.dump({"members": summaries}, handle)
'''


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _suite(tmp_path: Path, row_counts: list[int]) -> Path:
    members = []
    for index, count in enumerate(row_counts):
        root = tmp_path / f"benchmark-{index}"
        root.mkdir()
        questions = root / "questions.csv"
        _write_csv(questions, ["id", "question"], [
            {"id": str(j), "question": f"question {j}"} for j in range(count)
        ])
        sample = root / "sample.csv"
        _write_csv(sample, ["id", "prediction"], [{"id": "0", "prediction": ""}])
        config = root / "inference_config.yaml"
        config.write_text(
            "implementation_config:\n  llm_kwargs:\n    tensor_parallel_size: 1\n",
            encoding="utf-8",
        )
        members.append({
            "name": f"benchmark {index}", "config": str(config),
            "questions": str(questions), "sample_submission": str(sample),
            "output": str(root / "predictions.csv"),
            "diagnostics": str(root / "diagnostics.json"),
        })
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"members": members}), encoding="utf-8")
    return suite


def _run(tmp_path: Path, row_counts: list[int], turns: int = 1) -> tuple[str, list[dict]]:
    predict = tmp_path / "predict.py"
    predict.write_text(FAKE_PREDICT, encoding="utf-8")
    suite = _suite(tmp_path, row_counts)
    summary = tmp_path / "summary.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", TMPDIR="/tmp", FAKE_TURNS=str(turns))
    result = subprocess.run(
        [sys.executable, str(HELPER), "--predict", str(predict),
         "--model", "unused", "--suite", str(suite),
         "--summary", str(summary), "--ticket-id", "test-ticket",
         "--allocated-gpus", "2", "--gpus-per-worker", "1"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MASK=0" in result.stdout and "MASK=1" in result.stdout
    assert len(json.loads(summary.read_text())["members"]) == len(row_counts)
    return result.stdout, json.loads(suite.read_text())["members"]


def test_large_single_benchmark_is_sharded_and_merged_in_original_order(tmp_path: Path) -> None:
    output, members = _run(tmp_path, [2400])
    assert "2 independent replica(s)" in output
    with open(members[0]["output"], newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    assert [row["id"] for row in predictions] == [str(j) for j in range(2400)]
    records = json.loads(Path(members[0]["diagnostics"]).read_text())["records"]
    assert [row["row_index"] for row in records] == list(range(2400))
    assert [row["request_index"] for row in records] == list(range(2400))
    assert '"benchmark_name":"benchmark 0"' in output


def test_separate_benchmarks_share_one_job_but_use_distinct_gpus(tmp_path: Path) -> None:
    output, members = _run(tmp_path, [1400, 1300])
    assert "2 independent replica(s)" in output
    for index, member in enumerate(members):
        with open(member["output"], newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == [1400, 1300][index]
        assert rows[-1]["prediction"] == f"answer-{len(rows) - 1}"


def test_shard_merge_keeps_multiple_real_requests_per_row(tmp_path: Path) -> None:
    _, members = _run(tmp_path, [2100], turns=2)
    records = json.loads(Path(members[0]["diagnostics"]).read_text())["records"]
    assert len(records) == 4200
    assert [row["request_index"] for row in records] == list(range(4200))
    assert [row["row_index"] for row in records] == [
        index for index in range(2100) for _ in range(2)
    ]


def test_completed_benchmark_survives_another_worker_failure(tmp_path: Path) -> None:
    predict = tmp_path / "predict.py"
    predict.write_text(FAKE_PREDICT, encoding="utf-8")
    suite = _suite(tmp_path, [1300, 1400])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", TMPDIR="/tmp", FAKE_FAIL_MASK="1")
    result = subprocess.run(
        [sys.executable, str(HELPER), "--predict", str(predict),
         "--model", "unused", "--suite", str(suite),
         "--summary", str(tmp_path / "summary.json"), "--ticket-id", "test-ticket",
         "--allocated-gpus", "2", "--gpus-per-worker", "1"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    members = json.loads(suite.read_text())["members"]
    assert Path(members[0]["output"]).is_file(), result.stdout + result.stderr
    assert Path(members[0]["diagnostics"]).is_file()
    assert not Path(members[1]["output"]).exists()
