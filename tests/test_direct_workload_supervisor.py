"""Real Linux process tests; no GPU, network or provider account required."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "playbook/runners/direct_workload.py"


@unittest.skipUnless(sys.platform == "linux", "requires Linux process identity and flock")
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.manifests = []

    def tearDown(self):
        for manifest in self.manifests:
            self.call("cancel", manifest)
        self.directory.cleanup()

    def manifest(self, name="a", body="echo ran >> ../executions\n"):
        root = self.root / name
        root.mkdir()
        script = root / "work.sh"
        script.write_text("#!/bin/bash\nset -eu\n" + body)
        result = {"execution_id": name, "ticket_id": name, "script": str(script),
                  "sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "files": {},
                  "allocation_lock": str(self.root / "allocation.lock"), "visible_devices": "2,3"}
        self.manifests.append(result)
        return result

    def call(self, action, manifest):
        result = subprocess.run([sys.executable, str(SOURCE), action], input=json.dumps(manifest),
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def wait(self, manifest, states):
        until = time.monotonic() + 8
        while time.monotonic() < until:
            row = self.call("observe", manifest)
            if row["state"] in states:
                return row
            time.sleep(.03)
        self.fail("workload did not reach " + str(states))

    def test_duplicate_publication_and_reconnect_execute_once(self):
        manifest = self.manifest(body="sleep .2\necho $CUDA_VISIBLE_DEVICES >> ../executions\n")
        processes = [subprocess.Popen([sys.executable, str(SOURCE), "publish"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
        for process in processes:
            _, error = process.communicate(json.dumps(manifest), timeout=10)
            self.assertEqual(process.returncode, 0, error)
        self.assertEqual(self.wait(manifest, {"COMPLETED"})["exit_code"], 0)
        self.call("publish", manifest)
        self.assertEqual((self.root / "executions").read_text().splitlines(), ["2,3"])

    def test_cancel_before_publication_is_a_durable_fence(self):
        manifest = self.manifest()
        self.call("cancel", manifest)
        self.call("publish", manifest)
        self.assertEqual(self.wait(manifest, {"CANCELLED"})["exit_code"], 143)
        self.assertFalse((self.root / "executions").exists())

    def test_cancel_waiting_workload_never_starts_on_released_lock(self):
        first = self.manifest("first", "sleep 1\necho first >> ../executions\n")
        second = self.manifest("second")
        self.call("publish", first)
        self.wait(first, {"RUNNING"})
        self.call("publish", second)
        self.wait(second, {"PENDING"})
        self.call("cancel", second)
        self.wait(second, {"CANCELLED"})
        self.wait(first, {"COMPLETED"})
        self.assertEqual((self.root / "executions").read_text().splitlines(), ["first"])

    def test_launch_bytes_are_checked_after_waiting_for_resources(self):
        import fcntl
        manifest = self.manifest()
        with open(manifest["allocation_lock"], "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.call("publish", manifest)
            self.wait(manifest, {"PENDING"})
            Path(manifest["script"]).write_text("echo changed >> ../executions")
        row = self.wait(manifest, {"FAILED"})
        self.assertIn("changed before execution", row["error"])
        self.assertFalse((self.root / "executions").exists())

    def test_failure_keeps_exit_code_and_does_not_reexecute(self):
        manifest = self.manifest(body="echo attempt >> ../executions\nexit 7\n")
        self.call("publish", manifest)
        self.assertEqual(self.wait(manifest, {"FAILED"})["exit_code"], 7)
        self.call("publish", manifest)
        self.assertEqual((self.root / "executions").read_text().splitlines(), ["attempt"])

    def detached_worker(self, *, running):
        program = ("import subprocess;from pathlib import Path;"
                   "p=subprocess.Popen(['sleep','60'],start_new_session=True);"
                   "Path('worker.pid').write_text(str(p.pid))")
        manifest = self.manifest(body=shlex.quote(sys.executable) + " -c " + shlex.quote(program)
                                 + ("\nsleep 60\n" if running else "\n"))
        self.call("publish", manifest)
        pid_file = Path(manifest["script"]).with_name("worker.pid")
        until = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < until:
            time.sleep(.03)
        self.assertTrue(pid_file.exists())
        if running:
            self.call("cancel", manifest)
        self.wait(manifest, {"CANCELLED" if running else "COMPLETED"})
        stat = Path("/proc") / pid_file.read_text() / "stat"
        if stat.exists():
            self.assertEqual(stat.read_text().rsplit(")", 1)[1].split()[0], "Z")

    def test_normal_exit_stops_detached_workers_before_releasing_resources(self):
        self.detached_worker(running=False)

    def test_cancel_running_workload_stops_detached_workers(self):
        self.detached_worker(running=True)

    def test_ticket_cleanup_confirms_exit_without_stopping_other_tickets(self):
        # Load the portable command builder without the control-plane database
        # dependencies, so this real Linux test also runs in a stdlib image.
        module = ast.parse((SOURCE.parents[2] / "src/zevo/engine/run/remote_jobs.py").read_text())
        module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)
                       and node.name in {"_safe_ticket_id", "_direct_process_kill_command"}]
        namespace = {"shlex": shlex}
        exec(compile(module, "remote_jobs.py", "exec"), namespace)
        workers = [subprocess.Popen(["sleep", "60"],
                   env={**os.environ, "ZEVO_TICKET_ID": ticket})
                   for ticket in ("cleanup-owned", "cleanup-other")]
        try:
            command = namespace["_direct_process_kill_command"]("cleanup-owned")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNotNone(workers[0].poll())
            self.assertIsNone(workers[1].poll())
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.wait()


if __name__ == "__main__":
    unittest.main()
