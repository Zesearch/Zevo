"""The Run is the single source of truth for the GPU maximum."""
from __future__ import annotations

from zevo.engine.run.runner import _num_gpus


class _Run:
    def __init__(self, n: int) -> None:
        self.num_gpus = n


def test_run_value_is_the_gpu_maximum():
    assert _num_gpus(_Run(2)) == 2


def test_missing_or_zero_run_value_means_unlimited():
    assert _num_gpus(_Run(0)) == 0


def test_ticket_payload_cannot_override_gpu_maximum():
    from zevo.contracts.tickets import InfrastructureProvisionPayload
    from pydantic import ValidationError
    import pytest

    with pytest.raises(ValidationError):
        InfrastructureProvisionPayload(
            operation="provision", purpose="train", num_gpus=8,
        )
