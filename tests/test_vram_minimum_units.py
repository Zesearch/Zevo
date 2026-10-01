"""`min_vram_gb` is nameplate gigabytes; the driver reports MiB.

Smoke run 83f87bfa: the plan said 24 (from the offer "A10 (24 GB PCIe)"), the
probe reported 23028 MiB, the floor made that 22 GiB, the per-device check
rejected a healthy card, and the agent destroyed the instance it had just
rented. One conversion, used by the engine's device check and the lease API.
"""
from __future__ import annotations

import pytest

from zevo.contracts.infrastructure import (
    meets_vram_minimum,
    meets_vram_minimum_from_gib_floor,
    nameplate_gb_to_mib,
)


@pytest.mark.parametrize("card, vram_mb, nameplate", [
    ("A10 24GB", 23028, 24),
    ("T4 16GB", 15360, 16),
    ("L40S 48GB", 46068, 48),
    ("A100 80GB", 81920, 80),
    ("RTX 4090 24GB", 24564, 24),
])
def test_every_common_card_meets_its_own_nameplate(card: str, vram_mb: int, nameplate: int) -> None:
    assert meets_vram_minimum(vram_mb=vram_mb, min_vram_gb=nameplate), card


def test_a_smaller_card_still_fails_a_larger_minimum() -> None:
    assert not meets_vram_minimum(vram_mb=15360, min_vram_gb=24)   # T4 for a 24 GB plan
    assert not meets_vram_minimum(vram_mb=23028, min_vram_gb=40)   # A10 for a 40 GB plan
    assert not meets_vram_minimum(vram_mb=46068, min_vram_gb=80)   # L40S for an 80 GB plan
    assert meets_vram_minimum(vram_mb=0, min_vram_gb=0)            # no floor, nothing to meet


def test_conversion_is_decimal_gigabytes_over_mebibytes() -> None:
    assert nameplate_gb_to_mib(24) == 22889
    assert nameplate_gb_to_mib(1) == 954
    assert nameplate_gb_to_mib(80) == 76294


@pytest.mark.parametrize("vram_gb_floor, nameplate, expected", [
    (22, 24, True),    # A10 probed as 22 GiB satisfies a 24 GB plan
    (15, 16, True),    # T4
    (44, 48, True),    # L40S
    (80, 80, True),
    (15, 24, False),
    (44, 80, False),
    (24, 40, False),
    (0, 24, False),    # unknown VRAM never proves a floor
    (0, 0, True),
])
def test_gib_floor_rule_matches_the_measured_rule(vram_gb_floor: int, nameplate: int, expected: bool) -> None:
    assert meets_vram_minimum_from_gib_floor(vram_gb=vram_gb_floor, min_vram_gb=nameplate) is expected
