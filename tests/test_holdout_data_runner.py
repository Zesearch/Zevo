from __future__ import annotations

import asyncio
import csv
from pathlib import Path

from zevo.contracts.data import DataTaskInput, HoldoutDataSuiteMemberInput
from zevo.engine.agent.drivers.holdout_data_runner import HoldoutDataRunnerDriver


def _csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_one_private_data_ticket_prepares_the_whole_suite(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    sample = tmp_path / "sample.csv"
    _csv(first, [{"id": "1", "question": "One?", "answer": "A"}])
    _csv(second, [{"id": "2", "question": "Two?", "answer": "B"}])
    _csv(sample, [{"id": "1", "prediction": "<answer>"}])
    inp = DataTaskInput(
        ticket_id="holdout-data-001",
        operation="prepare_holdout_data",
        run_id="run-001",
        test_set_name="first",
        data_recipe_schema={},
        data_recipe_validation_command="unused",
        artifacts_validation_command="unused",
        scoring_set=str(first),
        answer_fields=["answer"],
        sample_submission=str(sample),
        metric="accuracy",
        work_dir=str(tmp_path / "work"),
        suite_members=[HoldoutDataSuiteMemberInput(
            name="second",
            scoring_set=str(second),
            answer_fields=["answer"],
            sample_submission=str(sample),
        )],
    )
    run = asyncio.run(HoldoutDataRunnerDriver().run_agent(
        blueprint=None, input_payload=inp, workspace_dir=inp.work_dir,
    ))
    assert run.output.status == "succeeded"
    assert len(run.output.suite_members) == 1
    prepared = [run.output.scoring_public_path, run.output.suite_members[0].scoring_public_path]
    for path in prepared:
        assert "answer" not in Path(path).read_text(encoding="utf-8")
