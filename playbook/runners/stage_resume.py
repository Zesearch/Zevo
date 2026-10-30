"""Stdlib-only durable batches and phase reporting for finite stage workloads.

Data: iterate ``run_batches`` with a deterministic, bounded process_batch callback.
Persist every result, including empty/rejected batches. Identity must include the
pinned sources, recipe, model revision, decoding, verifier and implementation hash.
Exit 75 means the allocation window is exhausted; it never means stage success.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

YIELD_EXIT_CODE = 75


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def phase(ticket, name):
    if not re.fullmatch(r'[a-zA-Z0-9_.-]{1,64}', name) or not re.fullmatch(r'[a-zA-Z0-9_-]+', ticket):
        raise ValueError('invalid phase or ticket identity')
    print(f'__PHASE__:{ticket}:{name}@{time.time()}', flush=True)


def progress(ticket, name, step, total, **extra):
    print(f'__PROGRESS__:{ticket}:' + json.dumps({**extra, 'phase': name,
          'step': step, 'total': total, 't': time.time()}), flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('w') as fh:
            json.dump(value, fh, ensure_ascii=False, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


class BatchStore:
    """One immutable identity; atomic checksummed results, no partial-file reuse."""
    def __init__(self, root, *, ticket, identity):
        if not ticket or not identity:
            raise ValueError('resume requires ticket and full workload identity')
        self.identity = digest({'version': 1, 'ticket': ticket, 'workload': identity})
        self.root = Path(root) / self.identity
        self.root.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def batch(self, key):
        key_hash = digest(key)
        with (self.root / (key_hash + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield key_hash

    def load(self, key_hash):
        path = self.root / (key_hash + '.json')
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text())
            if (envelope['identity'] != self.identity or envelope['key'] != key_hash
                    or envelope['sha256'] != digest(envelope['value'])):
                raise ValueError('batch checksum or identity mismatch')
            return envelope
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'invalid saved batch: {path}') from exc

    def commit(self, key_hash, value):
        atomic_json(self.root / (key_hash + '.json'), {
            'identity': self.identity, 'key': key_hash, 'value': value,
            'sha256': digest(value),
        })


class AllocationWindow:
    """Honor actual allocation end, not a fresh duration for each reused stage."""
    def __init__(self, *, reserve_seconds=300, deadline=None):
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError('invalid shutdown reserve')
        if deadline is not None and (not math.isfinite(deadline) or deadline <= 0):
            raise ValueError('invalid stage deadline')
        self.reserve = reserve_seconds
        self.stopping = False
        self.deadline = deadline
        end = os.environ.get('SLURM_JOB_END_TIME', '')
        value = None
        if end:
            try:
                value = float(end)
                if not math.isfinite(value) or value <= 0:
                    value = None
            except ValueError:
                pass
        if value is None and os.environ.get('SLURM_JOB_ID'):
            # Compute-node read only. No job submission, polling, or GPU work.
            result = subprocess.run(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'],
                                    capture_output=True, text=True, timeout=15, check=True)
            match = re.search(r'\bEndTime=(\S+)', result.stdout)
            if not match or match[1] in {'Unknown', 'N/A', 'None'}:
                raise ValueError('cannot determine allocation end time')
            from datetime import datetime
            value = datetime.fromisoformat(match[1]).timestamp()
        if value is not None:
            self.deadline = min(self.deadline, value) if self.deadline else value

    def check(self, ticket, estimated_seconds=0):
        if self.stopping or (self.deadline and time.time() + self.reserve + estimated_seconds >= self.deadline):
            phase(ticket, 'checkpointing')
            phase(ticket, 'waiting_for_allocation')
            raise SystemExit(YIELD_EXIT_CODE)

    @contextlib.contextmanager
    def signals(self):
        old = {}
        def stop(signum, frame):
            self.stopping = True
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            old[sig] = signal.signal(sig, stop)
        try:
            yield self
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


def run_batches(items, process_batch, *, root, ticket, identity, batch_size=128,
                phase_name='generating_data', window=None, report_progress=True):
    """Yield committed JSON values in input order, replaying completed batches.

    Callbacks must not change external state; return all results and statistics,
    including rejections. Source records are part of each batch key. At most the
    current batch is repeated after a hard kill. Keep batch layout stable on retry.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    store = BatchStore(root, ticket=ticket, identity=identity)
    window = window or AllocationWindow()
    estimated = 0.0
    phase(ticket, 'restoring_progress')
    with window.signals():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            with store.batch({'start': start, 'items': chunk}) as key:
                saved = store.load(key)
                if saved is None:
                    window.check(ticket, estimated)
                    phase(ticket, phase_name)
                    began = time.monotonic()
                    value = process_batch(chunk, start)
                    estimated = max(estimated, 1.5 * (time.monotonic() - began))
                    phase(ticket, 'checkpointing')
                    store.commit(key, value)
                else:
                    value = saved['value']
            phase(ticket, phase_name)
            if report_progress:
                progress(ticket, phase_name, min(start + batch_size, len(items)), len(items),
                         restored=saved is not None)
            yield value
