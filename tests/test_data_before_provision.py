"""Every training-data source requires Infrastructure before preparation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from zevo.contracts.data import DataTaskInput
from zevo.contracts.tickets import DataPayload
from zevo.db.models import Run, Ticket
from zevo.engine.run.runner import _build_data_input


def _ticket(payload: dict) -> Ticket:
    return Ticket(id="data-r1-001", run_id="r1", agent_id="data", status="queued",
                  input_format="typed", lane="optimization", iteration=0, payload=payload)


def _run() -> Run:
    return Run(id="r1", metric="accuracy", validation_metric="accuracy", gpu_provider="cloud",
               decision_pins={"max_training_rows": 500})


@pytest.mark.parametrize("source", [
    {"data_query": "small trivia QA"}, {"dataset": "/uploaded/trivia.jsonl"},
    {"data_query": "generate verified arithmetic"},
])
def test_all_training_sources_require_device_binding(source) -> None:
    payload = DataPayload(operation="prepare_run_data", training_method="sft", **source).model_dump()
    with pytest.raises((KeyError, ValueError, ValidationError), match="device_info"):
        _build_data_input(_ticket(payload), payload, {}, "/w", _run(), {})


def test_hub_sourced_data_requires_the_device_binding() -> None:
    payload = DataPayload(operation="prepare_run_data", dataset="org/trivia-train",
                          training_method="sft").model_dump()
    with pytest.raises((KeyError, ValueError, ValidationError), match="device_info"):
        _build_data_input(_ticket(payload), payload, {}, "/w", _run(), {})
