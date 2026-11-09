"""Engine-owned supervisor for one finite workload on an existing Linux host.

Only stdlib is needed. Publication is idempotent under an attempt lock; a
separate allocation lock serializes GPU use without allocating another GPU.
The process survives SSH disconnects. Receipts are atomically replaced and
bound to an execution identity and immutable launch bytes.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_receipt(path, value):
    target = Path(path)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, target)


def alive(pid, start):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == str(start)
    except (OSError, ValueError, IndexError):
        return False


def identity():
    return {"pid": os.getpid(), "process_start":
            Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]}


def stop_workers(manifest, status, row):
    """Reap even detached workers before releasing the shared GPU lock."""
    marker = ("ZEVO_EXECUTION_ID=" + manifest["execution_id"]).encode()
    ticket = ("ZEVO_TICKET_ID=" + manifest["ticket_id"]).encode()
    started = time.monotonic()
    while True:
        workers = []
        for path in Path("/proc").glob("[0-9]*"):
            if int(path.name) == os.getpid():
                continue
            try:
                env = (path / "environ").read_bytes().split(b"\0")
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if marker in env and ticket in env and fields[0] != "Z":
                    workers.append((int(path.name), fields[19]))
            except (OSError, IndexError):
                continue
        if not workers:
            return
        elapsed = time.monotonic() - started
        if elapsed > 5:
            # An unkillable process must retain the allocation lock. Never
            # report a terminal execution while its GPU workers still exist.
            row.update(state="STOPPING", error="waiting for owned workers to stop")
            write_receipt(status, row)
        for pid, start in workers:
            if alive(pid, start):
                try:
                    os.kill(pid, signal.SIGTERM if elapsed < 1 else signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(.05)


def supervise(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest_path).parent
    status = root / "workload.json"
    # This descriptor is inherited from publication, closing the spawn race.
    attempt_lock = int(os.environ.pop("ZEVO_DIRECT_ATTEMPT_LOCK"))
    row = {"execution_id": manifest["execution_id"], "sha256": manifest["sha256"],
           "state": "PENDING", "created_at": time.time(), **identity()}
    child = None

    def stop(signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Do not call Popen.wait() from a handler interrupting Popen.wait():
        # its waitpid lock is already held by this thread. Reap after unwind.
        stop_workers(manifest, status, row)
        row.update(state="CANCELLED", exit_code=128 + signum, finished_at=time.time())
        write_receipt(status, row)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    write_receipt(status, row)
    try:
        if (root / "cancelled").exists():
            stop(signal.SIGTERM, None)
        with open(manifest["allocation_lock"], "a") as allocation:
            fcntl.flock(allocation, fcntl.LOCK_EX)
            if (root / "cancelled").exists():
                stop(signal.SIGTERM, None)
            script = Path(manifest["script"])
            if hashlib.sha256(script.read_bytes()).hexdigest() != manifest["sha256"]:
                raise ValueError("validated launch script changed before execution")
            for path, digest in manifest.get("files", {}).items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise ValueError("validated execution file changed: " + path)
            row.update(state="RUNNING", started_at=time.time())
            write_receipt(status, row)
            env = {**os.environ, "ZEVO_TICKET_ID": manifest["ticket_id"],
                   "ZEVO_EXECUTION_ID": manifest["execution_id"],
                   "CUDA_VISIBLE_DEVICES": manifest["visible_devices"]}
            with open(root / "workload.out", "ab", buffering=0) as out, open(root / "workload.err", "ab", buffering=0) as err:
                child = subprocess.Popen(["bash", str(script)], cwd=root, env=env,
                                         stdout=out, stderr=err, start_new_session=True)
                code = child.wait()
            stop_workers(manifest, status, row)
            row.pop("error", None)
            row.update(state="COMPLETED" if code == 0 else "FAILED", exit_code=code,
                       finished_at=time.time())
    except Exception as exc:
        row.update(state="FAILED", exit_code=1, error=str(exc), finished_at=time.time())
    finally:
        if child is not None:
            child.wait()
        write_receipt(status, row)
        os.close(attempt_lock)


def publish(manifest):
    root = Path(manifest["script"]).parent
    root.mkdir(parents=True, exist_ok=True)
    status = root / "workload.json"
    with open(root / "publication.lock", "a") as publication:
        fcntl.flock(publication, fcntl.LOCK_EX)
        manifest_path = root / "execution.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("execution identity already has different launch inputs")
        else:
            manifest_path.write_text(json.dumps(manifest))
        if (root / "cancelled").exists():
            row = {"execution_id": manifest["execution_id"], "sha256": manifest["sha256"],
                   "state": "CANCELLED", "exit_code": 143, "finished_at": time.time()}
            if not status.exists():
                write_receipt(status, row)
            return row
        lock = open(root / "attempt.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return {"execution_id": manifest["execution_id"], "published": True}
        try:
            if status.exists():
                return json.loads(status.read_text())
            env = {**os.environ, "ZEVO_DIRECT_ATTEMPT_LOCK": str(lock.fileno()),
                   "ZEVO_TICKET_ID": manifest["ticket_id"], "ZEVO_EXECUTION_ID": manifest["execution_id"]}
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", str(manifest_path)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True,
                             pass_fds=(lock.fileno(),), env=env)
        finally:
            lock.close()
    return {"execution_id": manifest["execution_id"], "published": True}


def observe(manifest):
    root = Path(manifest["script"]).parent
    path = root / "workload.json"
    if not path.exists():
        return {"execution_id": manifest["execution_id"], "state": "UNKNOWN"}
    row = json.loads(path.read_text())
    if row["execution_id"] != manifest["execution_id"] or row["sha256"] != manifest["sha256"]:
        raise ValueError("workload receipt does not match execution")
    if row["state"] in {"PENDING", "RUNNING", "STOPPING"} and not alive(row["pid"], row["process_start"]):
        # A disappeared supervisor does not prove its worker stopped. Require
        # cleanup before repair; never silently start a second GPU workload.
        row.update(state="LOST", error="workload supervisor disappeared")
    return row


def cancel(manifest):
    root = Path(manifest["script"]).parent
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "publication.lock", "a") as publication:
        fcntl.flock(publication, fcntl.LOCK_EX)
        (root / "cancelled").touch()
        row = observe(manifest)
        if row.get("state") in {"PENDING", "RUNNING"} and alive(row["pid"], row["process_start"]):
            try:
                os.kill(row["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    return {"execution_id": manifest["execution_id"], "cancellation_requested": True}


def main():
    if sys.argv[1] == "run":
        supervise(sys.argv[2])
    else:
        manifest = json.loads(sys.stdin.read())
        result = {"publish": publish, "observe": observe, "cancel": cancel}[sys.argv[1]](manifest)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
