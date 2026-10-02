"""Cloud GPU acquisition for the Infrastructure agent, as one shipped sequence.

System-owned helper: the engine copies it into the Infrastructure work dir as
``zevo_cloud_acquire.py`` (next to ``zevo_gpu_health.py``) and names it in
``cloud_acquire_helper_path``. The agent still owns the plan (backend, offer,
region, image, disk); this helper owns the part that keeps going wrong when it
is rewritten per activation:

  create -> bookkeeping row -> wait_for_ready -> SSH route -> wait_for_ssh
  -> nvidia-smi probe -> GPU health gate -> state file; on any failure after
  creation: destroy, verify, release the row, exit non-zero.

Three agent-written versions of this sequence each destroyed a healthy rental
on 2026-10-01: a GiB/GB unit slip (run 83f87bfa), a mis-spelled nvidia-smi
section read as a defect (run 4bd98bf5), and ``ssh["host"]`` where the client
returns ``ssh_host`` (run a676d8ac).

Usage (from the Infrastructure work dir)::

    python3 zevo_cloud_acquire.py --backend lambda --offer-id gpu_1x_a10 --region us-east-1 \
        --ssh-key-path /app/data/.ssh/id_ed25519 --name zevo-<run8> \
        --run-id R --ticket-id T --dph 1.29 --state acquire_state.json
    python3 zevo_cloud_acquire.py --backend vastai --offer-id 1234567 --docker-image IMG --disk-gb 100 ...

Output: the state file (JSON) and the same JSON on stdout. Exit 0 when the
device is ready, probed and healthy; 1 when provisioning failed and cleanup
ran (``destroyed`` and ``row_released`` say how far it got); 2 for a defective
GPU (destroyed, released, ``gpu_health`` carries the evidence); 4 when the
instance could not even be created (nothing to clean up).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = os.environ.get("ZEVO_API_BASE", "").rstrip("/")
AUTH_HEADER = os.environ.get("ZEVO_API_AUTH_HEADER", "").strip()
_PROBE_RE = re.compile(r"^\s*(\d+)\s*,\s*(.+?)\s*,\s*(\d+)\s*(?:MiB)?\s*$", re.I)


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- bookkeeping
def _api(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    if not API_BASE:
        raise RuntimeError("ZEVO_API_BASE is not set")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(API_BASE + path, method=method, data=data)
    req.add_header("Content-Type", "application/json")
    if AUTH_HEADER and ":" in AUTH_HEADER:
        name, value = AUTH_HEADER.split(":", 1)
        req.add_header(name.strip(), value.strip())
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def register_row(*, instance_id: str, run_id: str, ticket_id: str, dph: float,
                 backend: str, offer_id: str, region: str) -> str:
    row = _api("POST", "/api/infra/instances", {
        "instance_id": instance_id, "provider": "cloud", "status": "provisioning",
        "run_id": run_id, "ticket_id": ticket_id, "dph": float(dph),
        "meta": {"backend": backend, "auto_release": True, "offer_id": offer_id, "region": region},
    })
    return str(row["id"])


def patch_row(row_id: str, body: dict[str, Any]) -> None:
    _api("PATCH", f"/api/infra/instances/{row_id}", body)


# ------------------------------------------------------------------- provider
def make_provider(backend: str):
    if backend == "lambda":
        from zevo.providers.lambda_labs import LambdaCloudProvider
        return LambdaCloudProvider()
    if backend == "vastai":
        from zevo.providers.vastai import VastAIProvider
        return VastAIProvider()
    raise ValueError(f"unsupported cloud backend {backend!r}")


async def create(provider, backend: str, args) -> str:
    if backend == "lambda":
        created = await provider.create_instance(
            args.offer_id, args.region, ssh_key_path=args.ssh_key_path, name=args.name,
        )
    else:
        created = await provider.create_instance(
            int(args.offer_id), args.docker_image, int(args.disk_gb or 50),
        )
    instance_id = str((created or {}).get("instance_id") or (created or {}).get("id") or "")
    if not instance_id:
        raise RuntimeError(f"provider returned no instance id: {created!r}")
    return instance_id


def ssh_route(details: dict[str, Any], backend: str) -> dict[str, Any]:
    """The client's keys are ssh_host / ssh_port / ssh_user. Nothing else."""
    host = str(details.get("ssh_host") or "").strip()
    port = int(details.get("ssh_port") or (22 if backend == "lambda" else 0) or 0)
    user = str(details.get("ssh_user") or ("ubuntu" if backend == "lambda" else "root"))
    if not host or port <= 0:
        raise RuntimeError(f"provider SSH details incomplete: {details!r}")
    return {"host": host, "port": port, "user": user}


# ---------------------------------------------------------------------- probe
def _ssh(route: dict[str, Any], key_path: str, command: str, timeout: int = 60) -> tuple[int, str, str]:
    base = ["ssh", "-p", str(route["port"]), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if key_path:
        base += ["-i", key_path]
    try:
        proc = subprocess.run(base + [f"{route['user']}@{route['host']}", command],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    return proc.returncode, proc.stdout, proc.stderr


def parse_probe(text: str) -> list[dict[str, Any]]:
    devices = []
    for line in text.splitlines():
        m = _PROBE_RE.match(line)
        if m:
            devices.append({"index": int(m.group(1)), "name": m.group(2).strip(), "vram_mb": int(m.group(3))})
    return devices


def probe(route: dict[str, Any], key_path: str) -> dict[str, Any]:
    rc, out, err = _ssh(route, key_path,
                        "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits")
    if rc != 0:
        raise RuntimeError(f"nvidia-smi probe failed rc={rc}: {err.strip()[:200]}")
    devices = parse_probe(out)
    if not devices:
        raise RuntimeError(f"nvidia-smi returned no devices: {out.strip()[:200]!r}")
    rc, drv, _ = _ssh(route, key_path, "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    rc2, cuda, _ = _ssh(route, key_path, "nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1")
    return {
        "devices": devices,
        "driver_version": drv.strip() if rc == 0 else "",
        "cuda_version": cuda.strip().replace("CUDA Version: ", "") if rc2 == 0 else "",
    }


def health(route: dict[str, Any], key_path: str, indices: list[int]) -> dict[str, Any]:
    helper = Path(__file__).with_name("zevo_gpu_health.py")
    if not helper.is_file():
        helper = Path(__file__).with_name("gpu_health.py")
    cmd = [sys.executable, str(helper), "--host", route["host"], "--port", str(route["port"]),
           "--user", route["user"], "--key-path", key_path]
    for index in indices:
        cmd += ["--index", str(index)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    try:
        verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        verdict = {"ok": False, "defects": [], "evidence": [f"gate produced no verdict (rc={proc.returncode})"]}
    verdict["exit"] = proc.returncode
    return verdict


# ------------------------------------------------------------ environment
REMOTE_ENV_DIR = "~/zevo/env"


def start_remote_env_build(route: dict[str, Any], key_path: str) -> dict[str, Any]:
    """Upload the shipped environment builder and start `build all` detached.

    The build (vLLM, torch, TRL and friends into two pinned venvs) takes a few
    minutes; the Data stage that follows provisioning takes about as long, so
    starting it now and letting Inference/Train `ensure` their profile hides
    the install behind work that happens anyway. The process is detached on
    the HOST (setsid + nohup), not in this helper, which returns at once.
    """
    helper = Path(__file__).with_name("zevo_remote_env.sh")
    if not helper.is_file():
        helper = Path(__file__).with_name("remote_env.sh")
    if not helper.is_file():
        return {"started": False, "reason": "remote_env.sh not shipped beside the acquisition helper"}
    scp = ["scp", "-P", str(route["port"]), "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if key_path:
        scp += ["-i", key_path]
    rc, _, err = _ssh(route, key_path, f"mkdir -p {REMOTE_ENV_DIR}")
    if rc != 0:
        return {"started": False, "reason": f"mkdir failed rc={rc}: {err.strip()[:160]}"}
    proc = subprocess.run(scp + [str(helper), f"{route['user']}@{route['host']}:{REMOTE_ENV_DIR}/remote_env.sh"],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return {"started": False, "reason": f"upload failed rc={proc.returncode}: {proc.stderr.strip()[:160]}"}
    rc, out, err = _ssh(route, key_path,
                        f"chmod +x {REMOTE_ENV_DIR}/remote_env.sh && "
                        f"setsid nohup bash {REMOTE_ENV_DIR}/remote_env.sh build all "
                        f"> {REMOTE_ENV_DIR}/bootstrap.log 2>&1 < /dev/null & echo STARTED")
    if rc != 0 or "STARTED" not in out:
        return {"started": False, "reason": f"start failed rc={rc}: {err.strip()[:160]}"}
    return {"started": True, "helper": f"{REMOTE_ENV_DIR}/remote_env.sh",
            "profiles": {"infer": f"{REMOTE_ENV_DIR}/infer/bin/python", "train": f"{REMOTE_ENV_DIR}/train/bin/python"},
            "log": f"{REMOTE_ENV_DIR}/bootstrap.log"}


# ----------------------------------------------------------------- lifecycle
async def destroy_and_release(provider, instance_id: str, row_id: str, reason: str, state: dict[str, Any]) -> None:
    try:
        state["destroyed"] = bool(await provider.destroy_instance(instance_id))
    except Exception as exc:  # noqa: BLE001 - cleanup must report, not raise
        state["destroyed"] = False
        state["destroy_error"] = repr(exc)[:300]
    try:
        live = await provider.list_instances()
        state["post_destroy_live"] = sum(
            1 for i in live if str(i.get("id") or i.get("instance_id") or "") == instance_id
            and str(i.get("status") or i.get("actual_status") or "").lower() not in ("terminated", "terminating", "deleted", "exited")
        )
    except Exception:  # noqa: BLE001
        state["post_destroy_live"] = None
    if row_id:
        try:
            patch_row(row_id, {"status": "released", "release_reason": reason[:400]})
            state["row_released"] = True
        except Exception as exc:  # noqa: BLE001
            state["row_released"] = False
            state["row_release_error"] = repr(exc)[:300]


async def run(args) -> int:
    state: dict[str, Any] = {"backend": args.backend, "offer_id": args.offer_id, "region": args.region,
                             "dph": args.dph, "phase": "creating"}
    out = Path(args.state)

    def save() -> None:
        out.write_text(json.dumps(state, indent=1), encoding="utf-8")

    provider = make_provider(args.backend)
    try:
        instance_id = await create(provider, args.backend, args)
    except Exception as exc:  # noqa: BLE001
        state.update(phase="create_failed", error=repr(exc)[:400]); save()
        _log(json.dumps(state)); return 4
    state.update(instance_id=instance_id, phase="registering"); save()
    _log(f"CREATED {instance_id}")

    row_id = ""
    try:
        row_id = register_row(instance_id=instance_id, run_id=args.run_id, ticket_id=args.ticket_id,
                              dph=args.dph, backend=args.backend, offer_id=args.offer_id, region=args.region)
        state.update(row_id=row_id, phase="waiting_ready"); save()
        _log(f"BOOKKEEPING_ROW {row_id}")

        ready = await provider.wait_for_ready(instance_id, timeout=args.ready_timeout)
        state["ready"] = {k: ready.get(k) for k in ("status", "ssh_host", "ssh_port", "ssh_user", "gpu_name", "region")}
        route = ssh_route(await provider.get_ssh_details(instance_id), args.backend)
        state.update(ssh=route, phase="waiting_ssh"); save()
        _log(f"SSH_ROUTE {json.dumps(route)}")

        ok = await provider.wait_for_ssh(host=route["host"], port=route["port"], user=route["user"],
                                         key_path=args.ssh_key_path, timeout=args.ssh_timeout)
        if not ok:
            raise RuntimeError(f"SSH did not become ready within {args.ssh_timeout}s")
        state.update(ssh_ready=True, phase="probing"); save()

        probed = probe(route, args.ssh_key_path)
        state.update(probe=probed, phase="health_gate"); save()
        _log(f"PROBE {json.dumps(probed)}")

        verdict = health(route, args.ssh_key_path, [d["index"] for d in probed["devices"]])
        state["gpu_health"] = verdict; save()
        _log(f"HEALTH {json.dumps(verdict)}")
        if verdict.get("exit") == 3:
            raise RuntimeError("health gate could not reach the host over SSH")
        if not verdict.get("ok"):
            reason = "GPU hardware defect: " + "; ".join(verdict.get("defects") or ["unknown"])
            state.update(phase="defective", error=reason)
            await destroy_and_release(provider, instance_id, row_id, reason + "; instance destroyed", state)
            save(); _log(json.dumps(state)); return 2

        # Environment build runs on the host while the Run moves on to Data.
        # A failure here is not a provisioning failure: the stage helpers
        # build on demand with `ensure`; record the outcome and continue.
        try:
            state["remote_env"] = start_remote_env_build(route, args.ssh_key_path)
        except Exception as exc:  # noqa: BLE001 - best effort by design
            state["remote_env"] = {"started": False, "reason": repr(exc)[:200]}
        save(); _log(f"REMOTE_ENV {json.dumps(state['remote_env'])}")

        patch_row(row_id, {
            "status": "ready", "gpu_name": probed["devices"][0]["name"],
            "gpu_count": len(probed["devices"]),
            "vram_gb": min(d["vram_mb"] for d in probed["devices"]) // 1024,
            "ssh_host": route["host"], "ssh_port": route["port"], "ssh_user": route["user"],
            "dph": float(args.dph),
        })
        state.update(phase="ready", row_status="ready"); save()
        _log(json.dumps(state)); return 0
    except Exception as exc:  # noqa: BLE001 - every post-create failure cleans up
        reason = f"provision failure cleanup: {repr(exc)[:300]}"
        state.update(phase="failed", error=repr(exc)[:400])
        await destroy_and_release(provider, instance_id, row_id, reason + "; instance destroyed", state)
        save(); _log(json.dumps(state)); return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire one cloud GPU instance end to end")
    parser.add_argument("--backend", choices=("lambda", "vastai"), required=True)
    parser.add_argument("--offer-id", required=True)
    parser.add_argument("--region", default="")
    parser.add_argument("--docker-image", default="")
    parser.add_argument("--disk-gb", type=int, default=0)
    parser.add_argument("--ssh-key-path", required=True)
    parser.add_argument("--name", default="zevo")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ticket-id", required=True)
    parser.add_argument("--dph", type=float, default=0.0)
    parser.add_argument("--state", default="acquire_state.json")
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--ssh-timeout", type=int, default=420)
    args = parser.parse_args(argv)
    if args.backend == "lambda" and not args.region:
        parser.error("--region is required for lambda")
    if args.backend == "vastai" and not args.docker_image:
        parser.error("--docker-image is required for vastai")
    return asyncio.run(run(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
