# Run outcome for user requested cancellation

Recorded and implemented locally on 2026-10-16 at the user’s request. Not deployed.

When a user explicitly cancels remaining stage work and asks to finish with the retained champion, that intentional cancellation should not prevent the Run from finishing as `success`. Other success requirements and unrelated unresolved failures still apply.

Work left incomplete because of an execution error should continue to make an otherwise usable Run `degraded` when no successful replacement resolves it. Creating or starting a replacement alone does not resolve the issue; it must successfully complete the corresponding work.

The implementation must distinguish explicit user-directed cancellation from abnormal termination, rather than excluding every `cancelled` Ticket from outcome checks. Preserve the cancellation and its user instruction in the audit record even when it does not count as a Run failure.

Example: `b76123d8` registered the iteration-6 champion as `M-b76123d8`, but ended `degraded` solely because the user explicitly asked to cancel `data-b76123d8-007` and go straight to Registry. Under the requested rule, this case should finish as `success`.

Future checks should cover intentional stage cancellation with successful registration, unrelated unresolved failures alongside intentional cancellation, abnormal unfinished work without a replacement, and successful replacement of failed work. Whether to reclassify historical Runs is not decided by this note.


Implementation: cancellation stores an instruction-backed
`user_requested_stage_cancellations` lifecycle record. The dashboard cancellation
route records intent and the cancellation together; the Orchestrator/CLI supplies
`user_requested_stop`, the Run instruction ID, and a reason. A generic SIGTERM or
"cancelled by user" error string is insufficient. Cancelling a retry deliberately
does not erase an earlier unresolved failure. Existing terminal Runs have not
been reclassified; cancelling the whole Run retains its existing semantics.
