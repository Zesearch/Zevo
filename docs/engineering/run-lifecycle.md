# Run finalization and resource cleanup

Run limits stop new optimization. Existing Validation/Test measurement, the
supervisor's final decision and Registry can finish for up to 15 minutes. The
Run remains `running` for API compatibility; `lifecycle.finalization` exposes
its reason, start and deadline. This grace period and checkpoint rescue may
add cost beyond the optimization cap. After the deadline, active work stops
and checkpoint rescue runs before infrastructure cleanup.

Cancellation defaults to at most three attempts, each at most five minutes,
within a 15-minute window. The request's `attempt_timeout_seconds`,
`rescue_timeout_seconds` and `max_attempts` can narrow these bounds. All copy
and Hub-upload subprocesses are cancelled and reaped when interrupted.
Claims/deadlines survive daemon restarts; successful local copies are reused.

If preservation fails, `cancel_outcome.status = preservation_failed` keeps the
source. The response exposes `retryable`, `source_retained`, the error and
`compute_may_accrue`. A fresh explicit download/Hub cancellation retries; an
explicit discard request works during an active or failed rescue. Only an
operator who sets `discard_on_failure=true` authorizes destructive cleanup
when bounded retries fail. These provider APIs do not offer a universal,
verified pause-with-durable-disk operation; retained rented instances can keep
billing until the operator recovers or discards them. Continuous durable
checkpoint export is the recommended stronger guarantee.

Cleanup discovers both provider bookkeeping and registered device artifacts,
persists missing bookkeeping before attempting release, and records retryable
errors/backoff. Cloud deletion is confirmed against the provider inventory;
Slurm release requires the exact owning Ticket identity and an empty scheduler
query. Failed cleanup never stamps `released_at` or allows deletion of its
owning Run. Cancellation forces cleanup of explicitly manual resources, while
ordinary completion preserves `auto_release=false`.

Remote shared-host work directories are retained. An agent-authored path, even
if it contains the Run ID, is not sufficient ownership proof for recursive
deletion; no remote `rm -rf` is executed. Operator-managed garbage collection
must verify the canonical path and ownership outside the shared host root.

The new `runs.lifecycle` JSON column requires Alembic revision
`e1a2b3c4d5f6` before deploying the backend/scheduler.
