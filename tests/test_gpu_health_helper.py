"""The shipped GPU health gate: unsupported sections are missing evidence, not
defects; only explicit driver evidence condemns a device.

Smoke run 4bd98bf5 destroyed a healthy A10 because an agent-written gate ran
`nvidia-smi -q -d ECC,ROW_REMAPPER,RETIRED_PAGES` in one call, driver 570
rejected the section name (rc=2), and the exit code was read as a defect.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "playbook" / "runners" / "gpu_health.py"
spec = importlib.util.spec_from_file_location("zevo_gpu_health", HELPER)
gpu_health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_health)  # type: ignore[union-attr]

HEALTHY_ROW_REMAPPER = """
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
"""
DEFECTIVE_ROW_REMAPPER = """
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 8
        Pending                           : Yes
        Remapping Failure Occurred        : Yes
"""
HEALTHY_PAGES = "    Retired Pages\n        Pending Page Blacklist            : No\n"
BLACKLIST_PAGES = "    Retired Pages\n        Pending Page Blacklist            : Yes\n"
ECC_OK = "    ECC Mode\n        Current                           : Enabled\n"


def _sections(**over):
    base = {
        "ECC": (0, ECC_OK),
        "ROW_REMAPPER": (0, HEALTHY_ROW_REMAPPER),
        "PAGE_RETIREMENT": (0, HEALTHY_PAGES),
        "RETIRED_PAGES": (2, ""),  # driver 570 rejects this name
    }
    base.update(over)
    return base


def test_the_exact_a10_shape_that_misfired_is_healthy() -> None:
    verdict = gpu_health.evaluate(
        index=0, sections=_sections(), csv_query=(0, "0, No\n"), dmesg=(0, ""),
    )
    assert verdict["ok"] is True and verdict["defects"] == []
    assert "dmesg_xid=none" in verdict["evidence"]


def test_unsupported_sections_everywhere_are_evidence_not_defects() -> None:
    verdict = gpu_health.evaluate(
        index=0,
        sections={name: (2, "") for name in gpu_health.SECTIONS},
        csv_query=(2, ""), dmesg=(1, ""),
    )
    assert verdict["ok"] is True
    assert any("unsupported" in e for e in verdict["evidence"])


@pytest.mark.parametrize("sections, csv, dmesg, expect", [
    (_sections(ROW_REMAPPER=(0, DEFECTIVE_ROW_REMAPPER)), (0, "0, No"), (0, ""), "Remapping Failure Occurred"),
    (_sections(), (0, "248781, No"), (0, ""), "uncorrected volatile ECC"),
    (_sections(), (0, "0, Yes"), (0, ""), "pending retired pages"),
    (_sections(PAGE_RETIREMENT=(0, BLACKLIST_PAGES)), (0, "0, No"), (0, ""), "Pending Page Blacklist"),
    (_sections(), (0, "0, No"), (0, "[Mon] NVRM: Xid (PCI:0000:0a:00): 79, GPU has fallen off the bus"), "dmesg Xid"),
    (_sections(), (0, "0, No"), (0, "NVRM: Xid 48 (PCI:0000:0a:00): double bit error"), "dmesg Xid"),
])
def test_explicit_driver_evidence_condemns(sections, csv, dmesg, expect) -> None:
    verdict = gpu_health.evaluate(index=0, sections=sections, csv_query=csv, dmesg=dmesg)
    assert verdict["ok"] is False
    assert any(expect in d for d in verdict["defects"]), verdict


def test_benign_xids_do_not_condemn() -> None:
    verdict = gpu_health.evaluate(
        index=0, sections=_sections(), csv_query=(0, "0, No"),
        dmesg=(0, "NVRM: Xid (PCI:0000:0a:00): 13, Graphics Exception"),
    )
    assert verdict["ok"] is True


def test_gate_queries_each_section_separately_per_index() -> None:
    commands: list[str] = []

    def run(command: str):
        commands.append(command)
        if "-d RETIRED_PAGES" in command:
            return 2, ""
        if "-d ROW_REMAPPER" in command:
            return 0, HEALTHY_ROW_REMAPPER
        if "--query-gpu" in command:
            return 0, "0, No\n"
        if "dmesg" in command:
            return 0, ""
        return 0, ECC_OK

    verdict = gpu_health.gate(run, [0, 1])
    assert verdict["ok"] is True
    assert sum(1 for c in commands if "-q -d" in c) == 2 * len(gpu_health.SECTIONS)
    assert not any("," in c.split("-d ")[-1] for c in commands if "-q -d" in c)
    assert sum(1 for c in commands if "dmesg" in c) == 1
