"""Evaluation executor with an immutable, task-owned scoring contract.

Evaluation is a system stage, not an LLM agent.  For a task-owned scorer this
driver writes an auditable ``evaluate.sh`` and lets Bash invoke the scorer's
fixed file protocol directly.  When no scorer is supplied, it calls Zevo's
deterministic built-in metric implementation. A task-owned custom scorer may
itself call a model judge; the runner does not select a model or silently fall
back to another metric.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from zevo.contracts.evaluation import (
    EvaluationResult, EvaluationSuiteMemberResult, EvaluationTaskInput,
)
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.method.eval_metrics import score_default
from zevo.engine.run import process_registry


def _phase(
    sink: Callable[[dict[str, Any]], None] | None,
    ticket_id: str,
    name: str,
) -> None:
    if sink is not None:
        sink({"type": "phase", "payload": {"owner": ticket_id, "phase": name}})


def _failed(ticket_id: str, message: str, notes: str = "") -> EvaluationResult:
    return EvaluationResult(
        status="failed",
        ticket_id=ticket_id,
        metrics_path="",
        error_message=message[:2000],
        notes=notes,
    )


def _validated_score(metrics_path: Path) -> tuple[dict[str, Any], float]:
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise ValueError("scorer did not create a non-empty metrics.json")
    try:
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"metrics.json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("metrics.json must contain one top-level JSON object")
    raw_score = value.get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise ValueError("metrics.json must contain a numeric top-level `score`")
    score = float(raw_score)
    if not math.isfinite(score):
        raise ValueError("metrics.json top-level `score` must be finite")
    return value, score


def _timeout_seconds(*, code_execution: bool = False) -> float:
    key = (
        "ZEVO_CODE_EVALUATION_TIMEOUT_SECONDS"
        if code_execution else "ZEVO_EVALUATION_TIMEOUT_SECONDS"
    )
    raw = os.environ.get(key, "14400" if code_execution else "3600")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 3600.0


def _judge_progress(path: Path, ticket_id: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    completed, total, cached = (
        value.get("completed"), value.get("total"), value.get("cached")
    )
    if not all(type(number) is int for number in (completed, total, cached)):
        return None
    if not (0 < total <= 1_000_000 and 0 <= cached <= completed <= total):
        return None
    model = value.get("model")
    if not isinstance(model, str) or not model or len(model) > 128:
        return None
    return {
        "owner": ticket_id, "phase": "model_judge",
        "step": completed, "total": total,
        "model": model, "cached": cached,
    }


class EvaluationRunnerDriver:
    """Run the fixed evaluator without making a runner-owned LLM call."""

    name = "evaluation_runner"

    async def run_agent(
        self,
        *,
        blueprint: Any,
        input_payload: BaseModel,
        workspace_dir: str,
        stdout_sink: Callable[[str], None] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        conversation: list[dict[str, Any]] | None = None,
        max_turns: int = 60,
        model_override: str = "",
        sandbox_mode: str = "",
    ) -> DriverRunResult:
        del blueprint, conversation, max_turns, model_override, sandbox_mode
        if not isinstance(input_payload, EvaluationTaskInput):
            raise TypeError("evaluation_runner accepts typed EvaluationTaskInput only")

        inp = input_payload
        if inp.suite_members:
            # Each member retains its own immutable scorer and ground truth.
            # The parent ticket publishes only after all members succeed.
            entries = [
                (
                    inp.test_set_name, inp.benchmark_id,
                    inp.model_copy(update={"suite_members": []}),
                ),
                *[
                    (
                        member.name,
                        member.benchmark_id,
                        inp.model_copy(update={
                            "suite_members": [],
                            "test_set_name": member.name,
                            "benchmark_id": member.benchmark_id,
                            "predictions_path": member.predictions_path,
                            "scoring_set": member.scoring_set,
                            "sample_submission": member.sample_submission,
                            "metric": member.metric,
                            "evaluation_script": member.evaluation_script,
                            "evaluator_sha256": member.evaluator_sha256,
                            "answer_fields": member.answer_fields,
                            "evaluation_config": member.evaluation_config,
                            "code_execution_adapter": member.code_execution_adapter,
                        }),
                    )
                    for member in inp.suite_members
                ],
            ]
            suite_results: list[EvaluationSuiteMemberResult] = []
            for index, (name, identity, member_input) in enumerate(entries):
                if event_sink is not None:
                    event_sink({
                        "type": "phase",
                        "payload": {
                            "owner": inp.ticket_id,
                            "phase": f"scoring {index + 1}/{len(entries)}: {name}",
                        },
                    })
                member_dir = Path(workspace_dir).resolve() / "suite" / f"{index:03d}"

                def member_event(event: dict[str, Any]) -> None:
                    if event_sink is None:
                        return
                    if (event.get("type") == "progress"
                            and event.get("payload", {}).get("phase") == "model_judge"):
                        event = {**event, "payload": {
                            **event["payload"],
                            "benchmark_name": name,
                            "benchmark_id": identity,
                            "benchmark_index": index + 1,
                            "benchmark_total": len(entries),
                        }}
                    event_sink(event)

                child = await self.run_agent(
                    blueprint=None,
                    input_payload=member_input,
                    workspace_dir=str(member_dir),
                    stdout_sink=stdout_sink,
                    event_sink=member_event if event_sink is not None else None,
                )
                child_output = child.output
                if not isinstance(child_output, EvaluationResult) or child_output.status != "succeeded":
                    detail = (
                        child_output.error_message
                        if isinstance(child_output, EvaluationResult)
                        else "invalid evaluator result"
                    )
                    return DriverRunResult(
                        output=_failed(inp.ticket_id, f"{name}: {detail}"),
                        exit_code=child.exit_code or 1,
                        driver=self.name,
                    )
                try:
                    _metrics, member_score = _validated_score(Path(child_output.metrics_path))
                except (OSError, ValueError) as exc:
                    return DriverRunResult(
                        output=_failed(inp.ticket_id, f"{name}: {exc}"),
                        exit_code=1,
                        driver=self.name,
                    )
                suite_results.append(EvaluationSuiteMemberResult(
                    name=name,
                    metrics_path=child_output.metrics_path,
                    score=member_score,
                ))
                if event_sink is not None:
                    event_sink({
                        "type": "progress",
                        "payload": {
                            "owner": inp.ticket_id,
                            "phase": "benchmark_complete",
                            "step": 1,
                            "total": 1,
                            "benchmark_name": name,
                            "benchmark_id": identity,
                            "benchmark_index": index + 1,
                            "benchmark_total": len(entries),
                        },
                    })
            aggregate = sum(member.score for member in suite_results) / len(suite_results)
            root = Path(workspace_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps({
                "score": aggregate,
                "metric": "suite_average",
                "components": {
                    member.name: member.score for member in suite_results
                },
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return DriverRunResult(
                output=EvaluationResult(
                    ticket_id=inp.ticket_id,
                    status="succeeded",
                    metrics_path=str(metrics_path),
                    error_message="",
                    suite_members=suite_results,
                    notes=f"Scored {len(suite_results)} benchmarks in one ticket",
                ),
                exit_code=0,
                driver=self.name,
            )
        work_dir = Path(workspace_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        predictions = Path(inp.predictions_path).resolve()
        scoring_set = Path(inp.scoring_set).resolve()
        sample_submission = Path(inp.sample_submission).resolve()
        metrics_path = work_dir / "metrics.json"
        judge_progress_path = work_dir / "judge-progress.json"
        log_path = work_dir / "evaluate.log"
        script_path = Path(inp.evaluation_script).resolve() if inp.evaluation_script else None

        _phase(event_sink, inp.ticket_id, "validating_inputs")
        problems: list[str] = []
        for label, path in (
            ("predictions", predictions),
            ("scoring set", scoring_set),
            ("sample submission", sample_submission),
        ):
            if not path.is_file():
                problems.append(f"{label} is not a readable file: {path}")
        if predictions == scoring_set:
            problems.append("predictions and scoring set must be different files")
        if script_path is not None and not script_path.is_file():
            problems.append(f"evaluation script is not a readable file: {script_path}")
        if problems:
            output = _failed(inp.ticket_id, "; ".join(problems))
            return DriverRunResult(output=output, exit_code=1, driver=self.name)

        from zevo.engine.artifact_validation import (
            validate_scoring_prediction_artifacts,
        )
        try:
            binding = validate_scoring_prediction_artifacts(
                predictions=str(predictions),
                scoring_set=str(scoring_set),
                sample_submission=str(sample_submission),
                answer_fields=inp.answer_fields,
            )
        except (OSError, ValueError) as exc:
            output = _failed(
                inp.ticket_id,
                f"predictions violate sample submission contract: {exc}",
            )
            return DriverRunResult(output=output, exit_code=1, driver=self.name)

        if script_path is not None:
            from zevo.evaluator_storage import verify_evaluator
            try:
                verify_evaluator(script_path, inp.evaluator_sha256)
            except ValueError as exc:
                output = _failed(inp.ticket_id, str(exc))
                return DriverRunResult(output=output, exit_code=1, driver=self.name)
        elif inp.evaluator_sha256:
            output = _failed(
                inp.ticket_id, "built-in evaluation must not carry evaluator_sha256",
            )
            return DriverRunResult(output=output, exit_code=1, driver=self.name)
        if script_path is not None and inp.code_execution_adapter:
            output = _failed(
                inp.ticket_id,
                "code execution evaluation cannot carry a custom evaluator",
            )
            return DriverRunResult(output=output, exit_code=1, driver=self.name)
        if inp.code_execution_adapter and inp.metric != "pass_at_1":
            output = _failed(
                inp.ticket_id,
                "a code execution adapter requires the pass_at_1 metric",
            )
            return DriverRunResult(output=output, exit_code=1, driver=self.name)
        if inp.metric == "pass_at_1" and not inp.code_execution_adapter:
            output = _failed(
                inp.ticket_id,
                "pass_at_1 requires a registered code execution adapter",
            )
            return DriverRunResult(output=output, exit_code=1, driver=self.name)

        try:
            metrics_path.unlink(missing_ok=True)
            judge_progress_path.unlink(missing_ok=True)
        except OSError as exc:
            output = _failed(inp.ticket_id, f"cannot clear stale evaluation output: {exc}")
            return DriverRunResult(output=output, exit_code=1, driver=self.name)

        if script_path is not None or inp.code_execution_adapter:
            wrapper = work_dir / "evaluate.sh"
            if script_path is not None:
                route = "custom_script"
                command = [
                    "python3", str(script_path), str(predictions), str(scoring_set),
                    str(metrics_path),
                ]
            else:
                route = "code_execution"
                code_columns = [
                    column for column in binding.prediction_columns
                    if column.casefold() in {
                        "prediction", "code", "completion", "solution",
                    }
                ]
                if len(code_columns) == 1:
                    prediction_column = code_columns[0]
                elif len(binding.prediction_columns) == 1:
                    prediction_column = binding.prediction_columns[0]
                else:
                    output = _failed(
                        inp.ticket_id,
                        "pass_at_1 requires sample_submission to identify "
                        "one generated-code column named prediction, code, "
                        "completion, or solution",
                        f"route={route}",
                    )
                    return DriverRunResult(
                        output=output, exit_code=1, driver=self.name,
                    )
                command = [
                    sys.executable, "-m", "zevo.engine.code_execution",
                    "--adapter", inp.code_execution_adapter,
                    "--predictions", str(predictions),
                    "--scoring-set", str(scoring_set),
                    "--prediction-column", prediction_column,
                    "--metrics-out", str(metrics_path),
                ]
            wrapper.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nexec "
                + " ".join(shlex.quote(part) for part in command)
                + "\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o700)
            if event_sink is not None:
                event_sink({
                    "type": "config",
                    "payload": {
                        "owner": inp.ticket_id,
                        "route": route,
                        "metric": inp.metric,
                        "evaluation_script": str(script_path or ""),
                        "code_execution_adapter": inp.code_execution_adapter,
                    },
                })
                event_sink({
                    "type": "tool_call",
                    "payload": {
                        "tool": "run_bash",
                        "item_id": f"evaluation-{inp.ticket_id}",
                        "command": "bash evaluate.sh",
                        "input": {
                            "command": "bash evaluate.sh",
                            "description": "Run task evaluator",
                        },
                        "description": "Run task evaluator",
                        "description_source": "system",
                    },
                })
            _phase(event_sink, inp.ticket_id, "scoring")
            proc = await asyncio.create_subprocess_exec(
                "/bin/bash",
                str(wrapper),
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            process_registry.register(inp.ticket_id, proc)
            last_judge_progress: dict[str, Any] | None = None
            last_judge_emit = 0.0

            def publish_judge_progress(*, force: bool = False) -> None:
                nonlocal last_judge_progress, last_judge_emit
                if event_sink is None:
                    return
                payload = _judge_progress(judge_progress_path, inp.ticket_id)
                if payload is None or payload == last_judge_progress:
                    return
                now = time.monotonic()
                threshold = max(1, payload["total"] // 50)
                if (not force and last_judge_progress is not None
                        and payload["total"] == last_judge_progress["total"]
                        and payload["model"] == last_judge_progress["model"]
                        and payload["step"] < payload["total"]
                        and payload["step"] - last_judge_progress["step"] < threshold
                        and now - last_judge_emit < 5):
                    return
                event_sink({"type": "progress", "payload": payload})
                last_judge_progress = payload
                last_judge_emit = now

            async def watch_judge_progress() -> None:
                while True:
                    publish_judge_progress()
                    await asyncio.sleep(1)

            progress_task = (
                asyncio.create_task(watch_judge_progress())
                if event_sink is not None else None
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=_timeout_seconds(
                        code_execution=bool(inp.code_execution_adapter),
                    ),
                )
            except asyncio.TimeoutError:
                process_registry.cancel(inp.ticket_id)
                await proc.wait()
                message = (
                    "evaluation script timed out after "
                    f"{_timeout_seconds(code_execution=bool(inp.code_execution_adapter)):g} "
                    "seconds"
                )
                log_path.write_text(message + "\n", encoding="utf-8")
                output = _failed(inp.ticket_id, message, f"route={route}")
                return DriverRunResult(output=output, exit_code=124, driver=self.name)
            finally:
                if progress_task is not None:
                    progress_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await progress_task
                publish_judge_progress(force=True)
                process_registry.unregister(inp.ticket_id, proc)

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            log_path.write_text(
                "[stdout]\n" + stdout + "\n[stderr]\n" + stderr,
                encoding="utf-8",
            )
            if stdout_sink is not None:
                if stdout:
                    stdout_sink(stdout)
                if stderr:
                    stdout_sink(stderr)
            if event_sink is not None:
                transcript_output = (
                    stdout + (("\n[stderr] " + stderr) if stderr else "")
                )[-20000:]
                event_sink({
                    "type": "tool_result",
                    "payload": {
                        "tool": "run_bash",
                        "item_id": f"evaluation-{inp.ticket_id}",
                        "output": transcript_output,
                        "exit_code": int(proc.returncode or 0),
                    },
                })
            if proc.returncode != 0:
                evidence = (stderr.strip() or stdout.strip() or "no diagnostics")[-1200:]
                output = _failed(
                    inp.ticket_id,
                    f"evaluation script exited {proc.returncode}: {evidence}",
                    f"route={route}; log={log_path}",
                )
                return DriverRunResult(
                    output=output,
                    raw_stdout=stdout,
                    raw_stderr=stderr,
                    exit_code=int(proc.returncode or 1),
                    driver=self.name,
                )
        else:
            route = "builtin"
            allowed = {"prediction_column", "answer_column", "strict", "f1_average"}
            unknown = sorted(set(inp.evaluation_config) - allowed)
            if unknown:
                output = _failed(
                    inp.ticket_id,
                    f"unknown evaluation_config keys: {unknown}",
                    f"route={route}",
                )
                return DriverRunResult(output=output, exit_code=1, driver=self.name)
            for key in ("prediction_column", "answer_column"):
                value = inp.evaluation_config.get(key, "")
                if not isinstance(value, str):
                    output = _failed(
                        inp.ticket_id,
                        f"evaluation_config.{key} must be a string",
                        f"route={route}",
                    )
                    return DriverRunResult(output=output, exit_code=1, driver=self.name)
            strict = inp.evaluation_config.get("strict", False)
            if not isinstance(strict, bool):
                output = _failed(
                    inp.ticket_id,
                    "evaluation_config.strict must be a boolean",
                    f"route={route}",
                )
                return DriverRunResult(output=output, exit_code=1, driver=self.name)
            f1_average = inp.evaluation_config.get("f1_average", "micro")
            if not isinstance(f1_average, str) or f1_average not in ("micro", "macro"):
                output = _failed(
                    inp.ticket_id,
                    "evaluation_config.f1_average must be 'micro' or 'macro'",
                    f"route={route}",
                )
                return DriverRunResult(output=output, exit_code=1, driver=self.name)
            prediction_column = inp.evaluation_config.get("prediction_column", "")
            answer_column = inp.evaluation_config.get("answer_column", "")
            if not answer_column and len(inp.answer_fields) == 1:
                answer_column = inp.answer_fields[0]
            if not answer_column:
                output = _failed(
                    inp.ticket_id,
                    "built-in evaluation requires exactly one answer field or "
                    "evaluation_config.answer_column",
                    f"route={route}",
                )
                return DriverRunResult(output=output, exit_code=1, driver=self.name)
            if not prediction_column:
                if len(binding.prediction_columns) != 1:
                    output = _failed(
                        inp.ticket_id,
                        "built-in evaluation requires sample_submission to "
                        "identify exactly one prediction column; use a custom "
                        "evaluator for multi-output submissions",
                        f"route={route}",
                    )
                    return DriverRunResult(
                        output=output, exit_code=1, driver=self.name,
                    )
                prediction_column = binding.prediction_columns[0]
            elif prediction_column not in binding.prediction_columns:
                output = _failed(
                    inp.ticket_id,
                    "evaluation_config.prediction_column is not a prediction "
                    "column in sample_submission",
                    f"route={route}",
                )
                return DriverRunResult(output=output, exit_code=1, driver=self.name)
            if inp.answer_fields and answer_column not in inp.answer_fields:
                output = _failed(
                    inp.ticket_id,
                    "evaluation_config.answer_column conflicts with answer_fields",
                    f"route={route}",
                )
                return DriverRunResult(output=output, exit_code=1, driver=self.name)
            if event_sink is not None:
                event_sink({
                    "type": "config",
                    "payload": {
                        "owner": inp.ticket_id,
                        "route": route,
                        "metric": inp.metric,
                        "prediction_column": prediction_column,
                        "answer_column": answer_column,
                    },
                })
            _phase(event_sink, inp.ticket_id, "scoring")
            metrics = score_default(
                predictions,
                scoring_set,
                prediction_col_hint=prediction_column,
                gold_col_hint=answer_column,
                metric=inp.metric,
                strict=strict,
                f1_average=f1_average,
            )
            metrics_path.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if metrics.get("status") != "succeeded":
                output = _failed(
                    inp.ticket_id,
                    "built-in evaluation failed; inspect metrics.json alignment issues",
                    f"route={route}; metrics={metrics_path}",
                )
                return DriverRunResult(output=output, exit_code=1, driver=self.name)

        _phase(event_sink, inp.ticket_id, "validating_metrics")
        try:
            metrics, score = _validated_score(metrics_path)
        except (OSError, ValueError) as exc:
            output = _failed(inp.ticket_id, str(exc), f"route={route}")
            return DriverRunResult(output=output, exit_code=1, driver=self.name)

        _phase(event_sink, inp.ticket_id, "completed")
        output = EvaluationResult(
            status="succeeded",
            ticket_id=inp.ticket_id,
            metrics_path=str(metrics_path),
            error_message="",
            notes=(
                f"route={route}; metric={inp.metric}; score={score:g}; "
                f"fields={','.join(sorted(metrics))}"
            ),
        )
        return DriverRunResult(output=output, exit_code=0, driver=self.name)
