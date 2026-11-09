# Run finalization and resource cleanup

## Workload execution across compute backends

Data preparation, Train, Inference and remote Evaluation use the same
prepare → wait for resources → run → collect → validate lifecycle. Preparation
validates scripts, configurations, frozen inputs and required helpers before
publishing a GPU workload. Each execution has an immutable identity, launch
hash, log stream and attempt history. A successful Agent response alone does
not complete the stage: collection must pass the ordinary result validators.

The existing `slurm_job` contract and `slurm_script_path` result keys remain for
stored-payload compatibility. `slurm_job.transport` distinguishes `slurm`
(scheduler jobs) from `direct` (instance/cloud workloads on an existing host).
Direct preparation uses a `.sh` launcher and appears as **Prepare** in the
timeline; Slurm retains **Submit**. Both expose Collect. CPU-only local runners
continue using their existing execution path.

Direct execution persists its intent before SSH publication. An engine-owned
Linux supervisor deduplicates publication, checks staged file hashes, serializes
workloads on the allocation, and records exit receipts independently of the
Agent/SSH session. The reconciler resumes monitoring after a control-service
restart, drains progress markers and queues one collection for the exact
execution. Cancellation fences future publication of that attempt and stops
its owned workers. Detached workers are stopped before the allocation lock
is released; a lost supervisor requires confirmed cleanup before repair.
The supervisor bytes are pinned per execution, so a code update cannot
silently replace the monitor of an existing workload.

Checkpoint handoff, bounded semantic repair, frozen configuration checks and
successful launcher reuse share the existing engine policy. A continuation
must retain its verified checkpoint binding; an unchanged failed workload
cannot simply be published again without an authorized restart/resume.

Execution records reference the owning allocation and do not add another GPU
lease or infrastructure charge. This does not change resource policy:
`per_run` and `per_stage` remain available on all three backends;
`per_submission` remains Slurm-specific. Finishing one execution on a retained
cloud/instance allocation does not release or provision the machine. Allocation
release still belongs to the existing stage/Run lifecycle.

Cloud allocations now have an independent provider watcher. It uses the deployment's
provider credentials and the allocation's saved region. The hosted Console
adapter restricts this to the Run owner's workspace credentials. Explicit provider termination confirms loss. An
absent instance requires two consecutive observations in a credential/region
scope that previously saw that exact instance. Authentication errors, malformed
inventory, SSH failures, stopped instances and transitional states do not prove
that a replacement is safe. Observation errors retain the allocation and retry
with durable backoff. Newly acquired AWS instances persist their effective
region even when the caller omitted `--region`.

Confirmed loss atomically closes the affected executions and records one
resource recovery notification for the Orchestrator. It provisions through the
existing Infrastructure workflow and then uses the new device artifact. A ready
replacement is exposed in the live orchestration context to prevent another
rental for the same event. Late collection responses cannot revive an execution
already closed for allocation loss. No remote cleanup is attempted on a host
the provider has confirmed gone.

Each affected Ticket retains its 20-repair limit. Replacement Tickets for the
same stage, iteration and model inherit the count; the Run's resource recovery
counter is audit history, not an additional limit across iterations. Cancellation,
finalization and exhausted cost/time/repair limits prevent replacement. Private
evaluation failures remain terminal and isolated; the coordinator receives only
allocation facts, never private scoring diagnostics. Recovery never assumes
that files stored only on the destroyed machine survived. A verified durable
checkpoint/input is needed for reuse; otherwise work must be rebuilt through a
valid work order or stopped with the missing-artifact reason.

## Failure diagnosis and repair

Agent failures, rejected results, external workload failures and engine
post-processing failures use the same bounded recovery policy. A structured
`failure_recovery` describes the next action and its authority: `current_task`
remains on the same Ticket; changes to upstream inputs, resources, fixed
configuration or Run decisions request Orchestrator review. Error-text patterns
do not override this decision. Without a proposal, a conversational executor
gets a diagnostic repair activation with the error and existing artifacts.
Deterministic executors cannot diagnose through an LLM conversation, so they
request coordinator review rather than blindly repeating the failed command.
Held-out failures remain isolated from the optimization supervisor.

The engine records failure origin and routing evidence in `failure.context`
notices. Engine-confirmed cancellation and held-out isolation are hard
boundaries, independent of Agent prose. Existing Run-limit enforcement remains
authoritative. Stored work orders that fail validation before execution still
go to the coordinator: the specialist cannot change its own input contract.
The first input-validation failure marks the Ticket `failed` with review routed
to the coordinator, removing it from cron dispatch immediately. A versioned
notice identifies the rejected work order; duplicate callbacks do not enqueue
another review of the same input, even while its coordinator is running.
Before starting the coordinator, queued validation reviews are retired if the
input changed, the child no longer needs review, or that event was already
handled. Coalesced notifications retain all review identities and keep unrelated
new events and user instructions runnable. Corrected, explicitly requeued work
orders remain eligible for dispatch. This does not consume or reset repairs.

Each Ticket retains its artifacts and a maximum of **20 repair activations**,
including diagnostics and repairs following backend exceptions. Five consecutive
failures with unchanged execution/artifact/progress evidence stop repair early.
The engine revalidates all outputs after a repair; neither an Agent's success
claim nor its recovery proposal bypasses validation. Before a retry, owned
orphan processes are stopped while the allocation is preserved. A failure to
confirm cleanup requests coordinator review instead of overlapping executions.

## Finalization

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
