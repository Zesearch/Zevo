"""GPU health gate for a freshly provisioned direct host (cloud / instance).

System-owned helper: the engine copies it into the Infrastructure work dir as
``zevo_gpu_health.py`` and names it in ``gpu_health_helper_path``. The agent
runs it; it does not write its own gate. Smoke run 4bd98bf5 (2026-10-01)
destroyed a healthy A10 because an agent-written gate combined three
``nvidia-smi -q -d`` sections in one call, the driver rejected one section
name (``RETIRED_PAGES`` is ``PAGE_RETIREMENT`` on driver 570) with rc=2, and
the non-zero exit was read as a hardware defect.

Rules, in one place:

* every section is queried separately; a section the driver does not support
  (or a command that fails) is *missing evidence*, recorded under
  ``evidence`` and never a defect;
* a device is defective only on an explicit pattern in a section that did
  return: ``Remapping Failure Occurred : Yes``; row-remapper ``Pending : Yes``
  together with a non-zero uncorrectable remap count; a non-zero
  ``ecc.errors.uncorrected.volatile.total``; ``Pending Page Blacklist : Yes``
  or ``retired_pages.pending == Yes``; ``Xid 48/63/64/79/94/95`` in dmesg;
* the verdict is JSON on stdout: ``{"ok": bool, "defects": [...], "evidence": [...]}``.

Usage (over SSH, from the Infrastructure work dir)::

    python3 zevo_gpu_health.py --host H --port 22 --user ubuntu --key-path K --index 0 [--index 1]

Exit status: 0 healthy, 1 defective, 3 could not reach the host.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Callable

Runner = Callable[[str], tuple[int, str]]

SECTIONS = ("ECC", "ROW_REMAPPER", "PAGE_RETIREMENT", "RETIRED_PAGES")
_XID = re.compile(r"\bXid\b[^\n]*?\b(48|63|64|79|94|95)\b", re.I)
_REMAP_FAILURE = re.compile(r"Remapping Failure Occurred\s*:\s*Yes", re.I)
_REMAP_PENDING = re.compile(r"Pending\s*:\s*Yes", re.I)
_REMAP_UNCORR = re.compile(r"Uncorrectable Error\s*:\s*(\d+)", re.I)
_PAGE_BLACKLIST = re.compile(r"Pending Page Blacklist\s*:\s*Yes", re.I)


def _section_returned(rc: int, text: str) -> bool:
    """A section counts as evidence only when nvidia-smi answered for it."""
    if rc != 0:
        return False
    lowered = text.lower()
    return bool(text.strip()) and "not supported" not in lowered and "n/a" != text.strip().lower()


def evaluate(
    *, index: int, sections: dict[str, tuple[int, str]],
    csv_query: tuple[int, str], dmesg: tuple[int, str],
) -> dict:
    """Pure verdict from captured command outputs. Tested directly."""
    defects: list[str] = []
    evidence: list[str] = []
    tag = f"idx{index}"

    rc, text = sections.get("ROW_REMAPPER", (2, ""))
    if _section_returned(rc, text):
        if _REMAP_FAILURE.search(text):
            defects.append(f"{tag}: Remapping Failure Occurred : Yes")
        uncorrectable = sum(int(n) for n in _REMAP_UNCORR.findall(text))
        if _REMAP_PENDING.search(text) and uncorrectable > 0:
            defects.append(f"{tag}: row remapper Pending : Yes with {uncorrectable} uncorrectable remaps")
    else:
        evidence.append(f"{tag}: ROW_REMAPPER section unsupported (rc={rc})")

    page_rc, page_text = sections.get("PAGE_RETIREMENT", (2, ""))
    if not _section_returned(page_rc, page_text):
        page_rc, page_text = sections.get("RETIRED_PAGES", (2, ""))
    if _section_returned(page_rc, page_text):
        if _PAGE_BLACKLIST.search(page_text):
            defects.append(f"{tag}: Pending Page Blacklist : Yes")
    else:
        evidence.append(f"{tag}: PAGE_RETIREMENT section unsupported (rc={page_rc})")

    ecc_rc, ecc_text = sections.get("ECC", (2, ""))
    if not _section_returned(ecc_rc, ecc_text):
        evidence.append(f"{tag}: ECC section unsupported (rc={ecc_rc})")

    rc, text = csv_query
    if rc == 0 and text.strip():
        fields = [f.strip() for f in text.strip().splitlines()[-1].split(",")]
        uncorrected = fields[0] if fields else ""
        pending = fields[1] if len(fields) > 1 else ""
        if uncorrected.isdigit() and int(uncorrected) > 0:
            defects.append(f"{tag}: nonzero uncorrected volatile ECC total: {uncorrected}")
        if pending.lower() == "yes":
            defects.append(f"{tag}: pending retired pages: Yes")
        evidence.append(f"{tag}: ecc.uncorrected.volatile={uncorrected or 'N/A'} retired_pages.pending={pending or 'N/A'}")
    else:
        evidence.append(f"{tag}: --query-gpu ECC/retired-pages fields unsupported (rc={rc})")

    rc, text = dmesg
    if rc == 0:
        hit = _XID.search(text or "")
        if hit:
            line = next((ln for ln in text.splitlines() if hit.group(0) in ln), hit.group(0))
            defects.append(f"dmesg Xid: {line.strip()[:200]}")
        else:
            evidence.append("dmesg_xid=none")
    else:
        evidence.append(f"dmesg unavailable (rc={rc})")

    return {"ok": not defects, "defects": defects, "evidence": evidence}


def gate(run: Runner, indices: list[int]) -> dict:
    """Run every query separately per index and merge the verdicts."""
    verdict = {"ok": True, "defects": [], "evidence": []}
    dmesg = run("(dmesg -T 2>/dev/null || sudo -n dmesg -T 2>/dev/null || true) | grep -E 'NVRM|Xid' | tail -n 50")
    for index in indices:
        sections = {name: run(f"nvidia-smi -i {index} -q -d {name}") for name in SECTIONS}
        csv_query = run(
            f"nvidia-smi -i {index} --query-gpu=ecc.errors.uncorrected.volatile.total,"
            "retired_pages.pending --format=csv,noheader"
        )
        one = evaluate(index=index, sections=sections, csv_query=csv_query, dmesg=dmesg)
        verdict["ok"] = verdict["ok"] and one["ok"]
        verdict["defects"].extend(one["defects"])
        verdict["evidence"].extend(one["evidence"])
    # dmesg evidence is host-wide; keep one copy.
    seen: set[str] = set()
    verdict["evidence"] = [e for e in verdict["evidence"] if not (e in seen or seen.add(e))]
    verdict["defects"] = [d for d in verdict["defects"] if not (d in seen or seen.add(d))]
    return verdict


def _ssh_runner(host: str, port: int, user: str, key_path: str, timeout: int) -> Runner:
    base = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={min(timeout, 30)}"]
    if key_path:
        base += ["-i", key_path]

    def run(command: str) -> tuple[int, str]:
        try:
            proc = subprocess.run(base + [f"{user}@{host}", command], capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, ""
        if proc.returncode == 255:
            raise ConnectionError(proc.stderr.strip()[:300] or "ssh exit 255")
        return proc.returncode, proc.stdout
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU health gate over SSH")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--key-path", default="")
    parser.add_argument("--index", type=int, action="append", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        verdict = gate(_ssh_runner(args.host, args.port, args.user, args.key_path, args.timeout), args.index)
    except ConnectionError as exc:
        print(json.dumps({"ok": False, "defects": [], "evidence": [f"ssh unreachable: {exc}"]}))
        return 3
    print(json.dumps(verdict))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
