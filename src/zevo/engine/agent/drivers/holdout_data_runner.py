"""Deterministic, private preparation of a complete held-out Test suite."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from zevo.contracts.data import DataResult, DataTaskInput, HoldoutDataSuiteMemberResult
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.artifact_validation import materialize_system_scoring_artifacts


class HoldoutDataRunnerDriver:
    name = "holdout_data_runner"

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
        if not isinstance(input_payload, DataTaskInput) or input_payload.operation != "prepare_holdout_data":
            raise TypeError("holdout_data_runner accepts private DataTaskInput only")
        inp = input_payload
        members = [
            (inp.test_set_name, inp.scoring_set, inp.answer_fields, inp.sample_submission),
            *[
                (member.name, member.scoring_set, member.answer_fields, member.sample_submission)
                for member in inp.suite_members
            ],
        ]
        prepared = []
        try:
            for index, (name, scoring_set, answer_fields, submission) in enumerate(members):
                if event_sink is not None:
                    event_sink({
                        "type": "phase",
                        "payload": {
                            "owner": inp.ticket_id,
                            "phase": f"preparing {index + 1}/{len(members)}: {name}",
                        },
                    })
                out_dir = Path(workspace_dir).resolve() / "suite" / f"{index:03d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                artifacts = materialize_system_scoring_artifacts(
                    scoring_source=scoring_set,
                    answer_fields=answer_fields,
                    sample_submission=submission,
                    out_dir=str(out_dir),
                )
                prepared.append(artifacts)
                if stdout_sink is not None:
                    stdout_sink(f"Prepared held-out Test set {index + 1}/{len(members)}: {name}\n")
        except (OSError, ValueError) as exc:
            return DriverRunResult(
                output=DataResult(
                    ticket_id=inp.ticket_id,
                    status="failed",
                    operation="prepare_holdout_data",
                    error_message=str(exc)[:2000],
                ),
                exit_code=1,
                driver=self.name,
            )
        primary = prepared[0]
        result = DataResult(
            ticket_id=inp.ticket_id,
            status="succeeded",
            operation="prepare_holdout_data",
            error_message="",
            scoring_public_path=primary.questions_path,
            inference_data_profile_path=primary.profile_path,
            sample_submission_path=primary.sample_submission_path,
            suite_members=[
                HoldoutDataSuiteMemberResult(
                    name=members[index][0],
                    scoring_public_path=artifacts.questions_path,
                    inference_data_profile_path=artifacts.profile_path,
                    sample_submission_path=artifacts.sample_submission_path,
                )
                for index, artifacts in enumerate(prepared[1:], start=1)
            ],
            notes=f"Prepared {len(prepared)} held-out Test sets in one ticket",
        )
        return DriverRunResult(output=result, exit_code=0, driver=self.name)
