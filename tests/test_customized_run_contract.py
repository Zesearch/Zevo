from __future__ import annotations

import pytest
from pydantic import ValidationError

from zevo.contracts.customizations import AgentCustomization, RunCustomizations


def test_customized_run_accepts_only_documented_agent_parameters() -> None:
    custom = RunCustomizations(agents={
        "data": AgentCustomization(parameters={
            "method_ids": ["acquire_hf", "reformat_jsonl"],
            "target_size": 5_000,
        }),
        "train": AgentCustomization(parameters={
            "num_epochs": 2,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "max_seq_len": 2_048,
        }),
    })
    assert custom.agents["data"].parameters["target_size"] == 5_000

    with pytest.raises(ValidationError, match="unsupported train"):
        RunCustomizations(agents={
            "train": AgentCustomization(parameters={"device": "cuda:7"}),
        })
    with pytest.raises(ValidationError, match="integer >= 1"):
        RunCustomizations(agents={
            "train": AgentCustomization(parameters={"batch_size": 1.5}),
        })
    with pytest.raises(ValidationError, match="non-empty unique"):
        RunCustomizations(agents={
            "data": AgentCustomization(parameters={"method_ids": ["acquire_hf", "acquire_hf"]}),
        })
