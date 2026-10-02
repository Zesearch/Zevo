"""The shipped remote environment builder and its hand-off from acquisition.

Every agent activation used to write its own install sequence on the GPU host
(run a676d8ac: three conflicting numpy pins in Train, a 10-minute venv build
on a cold host before candidate inference). One idempotent helper owns the
pinned profiles; the acquisition helper starts the build in the background so
Inference/Train usually find it ready.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "playbook" / "runners" / "remote_env.sh"
ACQUIRE = Path(__file__).resolve().parents[1] / "playbook" / "runners" / "cloud_acquire.py"


def _run(args: list[str], root: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "ZEVO_ENV_ROOT": str(root)}
    return subprocess.run(["bash", str(HELPER), *args], capture_output=True, text=True, env=env, timeout=60)


def _manifest(profile: str) -> str:
    return _run(["manifest", profile], Path("/x")).stdout


def _seed_ready(root: Path, profile: str, *, manifest: str) -> Path:
    d = root / profile
    (d / "bin").mkdir(parents=True)
    py = d / "bin" / "python"; py.write_text("#!/bin/sh\necho fake\n"); py.chmod(0o755)
    (d / ".ready").write_text(hashlib.sha256(manifest.encode()).hexdigest()[:16] + "\nVERSIONS {}\n")
    return d


def test_profiles_pin_the_stack_the_smoke_runs_proved() -> None:
    infer, train = _manifest("infer"), _manifest("train")
    assert "vllm==0.10.2" in infer and "transformers==4.56.1" in infer
    assert "torch==2.8.0" in train and "trl==0.21.0" in train and "transformers==4.56.1" in train
    assert "trl" not in infer                       # TRL only where training happens
    assert _run(["python", "infer"], Path("/x")).stdout.strip() == "/x/infer/bin/python"


def test_status_is_json_and_empty_root_is_not_ready(tmp_path: Path) -> None:
    out = _run(["status"], tmp_path)
    assert out.returncode == 0
    assert json.loads(out.stdout) == {"infer": {"ready": False}, "train": {"ready": False}}


def test_ensure_returns_at_once_when_the_profile_is_ready(tmp_path: Path) -> None:
    d = _seed_ready(tmp_path, "infer", manifest=_manifest("infer"))
    out = _run(["ensure", "infer"], tmp_path)
    assert out.returncode == 0 and out.stdout.strip() == f"READY infer {d}/bin/python"
    assert json.loads(_run(["status"], tmp_path).stdout)["infer"] == {"ready": True, "python": f"{d}/bin/python"}


def test_a_changed_manifest_makes_a_cached_profile_stale(tmp_path: Path) -> None:
    _seed_ready(tmp_path, "train", manifest=_manifest("train") + "extra==1\n")
    assert json.loads(_run(["status"], tmp_path).stdout)["train"] == {"ready": False}


def test_a_build_in_progress_is_reported_not_duplicated(tmp_path: Path) -> None:
    d = tmp_path / "infer"; d.mkdir()
    (d / ".building").write_text(str(os.getpid()))      # a live pid: someone is building
    out = _run(["build", "infer"], tmp_path)
    assert out.returncode == 3 and out.stdout.startswith("BUILDING infer")
    assert json.loads(_run(["status"], tmp_path).stdout)["infer"] == {"ready": False, "building": True}


def test_unknown_profile_and_usage_fail_fast(tmp_path: Path) -> None:
    assert _run(["ensure", "gpu"], tmp_path).returncode == 2
    assert _run([], tmp_path).returncode == 2


def test_acquisition_starts_the_build_detached_and_records_it(monkeypatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("zevo_cloud_acquire", ACQUIRE)
    acquire = importlib.util.module_from_spec(spec); spec.loader.exec_module(acquire)  # type: ignore[union-attr]
    (tmp_path / "zevo_cloud_acquire.py").write_text("# copy\n")
    (tmp_path / "zevo_remote_env.sh").write_text(HELPER.read_text())
    monkeypatch.setattr(acquire, "__file__", str(tmp_path / "zevo_cloud_acquire.py"))
    ssh_cmds: list[str] = []
    monkeypatch.setattr(acquire, "_ssh", lambda route, key, cmd, timeout=60: (ssh_cmds.append(cmd) or (0, "STARTED\n", "")))
    scp_calls: list[list[str]] = []
    monkeypatch.setattr(acquire.subprocess, "run",
                        lambda cmd, **k: scp_calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    route = {"host": "203.0.113.5", "port": 22, "user": "ubuntu"}
    out = acquire.start_remote_env_build(route, "/tmp/k")
    assert out["started"] is True and out["profiles"]["infer"].endswith("/infer/bin/python")
    assert scp_calls and scp_calls[0][-1] == "ubuntu@203.0.113.5:~/zevo/env/remote_env.sh"
    assert any("setsid nohup bash ~/zevo/env/remote_env.sh build all" in c for c in ssh_cmds)


def test_acquisition_without_the_helper_beside_it_reports_and_continues(monkeypatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("zevo_cloud_acquire2", ACQUIRE)
    acquire = importlib.util.module_from_spec(spec); spec.loader.exec_module(acquire)  # type: ignore[union-attr]
    monkeypatch.setattr(acquire, "__file__", str(tmp_path / "zevo_cloud_acquire.py"))
    out = acquire.start_remote_env_build({"host": "h", "port": 22, "user": "u"}, "")
    assert out["started"] is False and "not shipped" in out["reason"]
