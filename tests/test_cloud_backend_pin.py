"""A Run's cloud_backend pin is binding for Infrastructure.

Run 0869b205 (2026-10-15) was pinned to AWS; a one-AZ capacity error there
made the agent fall back to Lambda on the same member's other key. The pin
must leave exactly one rentable backend, and a pin nobody holds a credential
for must fail at input time, not become a fallback."""
from __future__ import annotations

import pytest

from zevo.engine.run.runner import _pin_cloud_backend


def test_no_pin_leaves_the_catalogue_alone() -> None:
    assert _pin_cloud_backend(["vastai", "lambda", "aws"], "") == ["vastai", "lambda", "aws"]
    assert _pin_cloud_backend([], "") == []


def test_pin_restricts_to_exactly_that_backend() -> None:
    assert _pin_cloud_backend(["lambda", "aws"], "aws") == ["aws"]
    assert _pin_cloud_backend(["vastai", "lambda", "aws"], "lambda") == ["lambda"]


def test_pin_without_a_credential_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="pins cloud_backend='aws' but no credential"):
        _pin_cloud_backend(["lambda"], "aws")
    with pytest.raises(ValueError, match="unsupported cloud_backend pin"):
        _pin_cloud_backend(["lambda"], "gcp")
