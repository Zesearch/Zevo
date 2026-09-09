"""Deterministic Evaluation executor.

Evaluation is a system stage, not an LLM agent.  For a task-owned scorer this
driver writes an auditable ``evaluate.sh`` and lets Bash invoke the scorer's
fixed file protocol directly.  When no scorer is supplied, it calls Zevo's
deterministic built-in metric implementation.  No prompt, model, or provider is
involved in either route.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from zevo.contracts.evaluation import EvaluationResult, EvaluationTaskInput
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


def _timeout_seconds() -> float:
    raw = os.environ.get("ZEVO_EVALUATION_TIMEOUT_SECONDS", "3600")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 3600.0


class EvaluationRunnerDriver:
    """Run the fixed evaluator without making an LLM call."""

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
        work_dir = Path(workspace_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        predictions = Path(inp.predictions_path).resolve()
        scoring_set = Path(inp.scoring_set).resolve()
        sample_submission = Path(inp.sample_submission).resolve()
        metrics_path = work_dir / "metrics.json"
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

        try:
            metrics_path.unlink(missing_ok=True)
        except OSError as exc:
            output = _failed(inp.ticket_id, f"cannot clear stale metrics.json: {exc}")
            return DriverRunResult(output=output, exit_code=1, driver=self.name)

        if script_path is not None:
            route = "custom_script"
            wrapper = work_dir / "evaluate.sh"
            command = [
                "python3", str(script_path), str(predictions), str(scoring_set),
                str(metrics_path),
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
                        "evaluation_script": str(script_path),
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
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=_timeout_seconds(),
                )
            except asyncio.TimeoutError:
                process_registry.cancel(inp.ticket_id)
                await proc.wait()
                message = (
                    "evaluation script timed out after "
                    f"{_timeout_seconds():g} seconds"
                )
                log_path.write_text(message + "\n", encoding="utf-8")
                output = _failed(inp.ticket_id, message, f"route={route}")
                return DriverRunResult(output=output, exit_code=124, driver=self.name)
            finally:
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
