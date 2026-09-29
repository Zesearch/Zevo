## Distributed execution reliability

After configuration, input, access, and resource checks, execute the workload
on the assigned resources. Validate loading, computation, synchronization, and
persistence as execution proceeds, reusing the loaded model and completed work.
Report progress from the running workload and verify required artifacts before
reporting completion.

Monitor useful progress by phase and participating worker: completed work,
last-progress time, and the operation currently executing. Process liveness or
an active allocation alone is not evidence of progress. Set finite budgets for
initialization, data access, per-item computation, synchronization, persistence,
and shutdown, based on representative measurements, workload variability, and
the remaining runtime limit. Investigate sustained stalls rather than extending
timeouts or retrying unchanged work without evidence. A timeout fallback must
preserve experiment semantics; do not silently skip inputs or alter outputs.

On failure or a progress-budget breach, preserve the earliest available error,
worker identity, phase, last completed unit of work, relevant configuration, and
per-worker logs or stack traces where available. Distinguish an originating
worker failure from downstream waits or communication timeouts. Report uncertain
causes as hypotheses instead of treating a timeout message as a root cause.
Stop and confirm termination of the affected synchronized worker group before
retrying; do not leave peers waiting or overlap old and replacement workers.
Preserve unrelated independent work and follow the resource-lifetime rules
above rather than terminating the parent allocation yourself.

Repair according to the evidence and retry within the engine's repair limit.
Resume only from verified, consistently committed state, including the progress
and runtime state required by the method. Reuse completed outputs only when
inputs and configuration still match; avoid duplicate or missing work. If a
consistent resume is unavailable, report the limitation and use the permitted
restart policy. Do not silently change the method, data, or execution semantics.

Report planned versus completed work and why execution stopped. Distinguish a
fully completed plan from an interrupted run with usable partial artifacts in
the existing result schema and summary. A saved checkpoint, some predictions,
or successful startup does not by itself establish completion of the full plan.

