from __future__ import annotations

import csv
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.routers.ui.tasks import TaskBody, TaskPatch, create_task, update_task
from zevo.contracts.prompting import validate_inference_config
from zevo.contracts.task_protocol import (
    TaskInferenceProtocol,
    default_task_inference_protocol,
)
from zevo.engine.method.task_defaults import resolve_dataset_task_defaults
from zevo.db.models import Base


def test_math_protocol_separates_semantics_from_submission_shape() -> None:
    protocol = default_task_inference_protocol(
        "Solve mathematical problems and report the correct answer."
    )
    assert protocol.task_type == "math_reasoning"
    assert protocol.response_format == "boxed_answer"
    assert protocol.answer_parser == "boxed"
    mapping = validate_inference_config(protocol.inference_mapping())
    assert mapping["user_prompt_template"] == "{input}"
    assert mapping["answer_regex"] == r"\\boxed\{([^{}]+)\}"
    rendered = protocol.render_user_content({"problem": "What is 2 + 2?"})
    assert "What is 2 + 2?" in rendered
    assert "\\boxed{answer}" in rendered


def test_protocol_rejects_an_unavailable_template_field() -> None:
    protocol = TaskInferenceProtocol(
        source="user",
        task_type="generation",
        instruction="Answer the question.",
        user_prompt_template="Question: {question}\nContext: {context}",
        output_instruction="Return the answer.",
    )
    with pytest.raises(ValueError, match="context"):
        protocol.render_user_content({"question": "Why?"})


def test_inference_mapping_rejects_inconsistent_parser_and_format() -> None:
    with pytest.raises(ValueError, match="requires response_format"):
        validate_inference_config({
            "task_instruction": "Solve the problem.",
            "user_prompt_template": "{input}",
            "output_instruction": "Return the result.",
            "response_format": "plain_text",
            "answer_parser": "boxed",
        })


def test_low_input_task_defaults_infer_answer_metric_and_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_set = tmp_path / "math.csv"
    test_set.write_text(
        "id,problem,answer\nq1,What is 2 + 2?,4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZEVO_UPLOAD_ROOT", str(tmp_path / "uploads"))
    resolved = resolve_dataset_task_defaults(
        task_name="math-eval",
        objective="Solve math problems.",
        test_set=str(test_set),
    )
    assert resolved.answer_fields == ["answer"]
    assert resolved.metric == "exact_match"
    assert resolved.metric_direction == "max"
    assert resolved.inference_protocol.answer_parser == "boxed"
    with Path(resolved.sample_submission).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"id": "q1", "prediction": ""}]


def test_low_input_task_defaults_refuse_ambiguous_answer_column(tmp_path: Path) -> None:
    test_set = tmp_path / "ambiguous.csv"
    test_set.write_text("id,prompt,value\n1,hello,world\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could not identify"):
        resolve_dataset_task_defaults(
            task_name="ambiguous",
            objective="Respond to the prompt.",
            test_set=str(test_set),
        )


@pytest.mark.asyncio
async def test_create_task_accepts_only_name_objective_and_test_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "id,question,label\n1,good or bad?,positive\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZEVO_UPLOAD_ROOT", str(tmp_path / "uploads"))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        created = await create_task(
            TaskBody(
                name="sentiment",
                task_objective="Classify the sentiment.",
                test_set=str(test_set),
            ),
            db=db,
        )
    assert created.test_answer_fields == ["label"]
    assert created.metric == "accuracy"
    assert created.inference_protocol.task_type == "classification"
    assert created.test_sample_submission
    await engine.dispose()


@pytest.mark.asyncio
async def test_editing_an_auto_task_refreshes_its_inference_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "id,question,answer\n1,What is two plus two?,4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZEVO_UPLOAD_ROOT", str(tmp_path / "uploads"))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        await create_task(
            TaskBody(
                name="editable",
                task_objective="Answer each question.",
                test_set=str(test_set),
            ),
            db=db,
        )
        updated = await update_task(
            "editable",
            TaskPatch(task_objective="Solve each math problem."),
            db=db,
        )
    assert updated.inference_protocol.task_type == "math_reasoning"
    assert updated.inference_protocol.answer_parser == "boxed"
    await engine.dispose()
