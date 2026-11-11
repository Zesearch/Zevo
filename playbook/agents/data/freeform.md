The engine first resolves this natural-language request into a typed work order.
During resolution, inspect authorized attachments and prepare only small control
files; do not perform the requested workload or external mutations yet. The
resolution response contains the typed payload, input bindings, explanation,
and any missing information. The engine validates source ownership and bindings,
records `freeform_resolution.json`, then executes the normal typed specialist.
The original user request remains in the Ticket conversation. Do not report a
specialist Result from the resolution phase.

Input is the shared `FreeformInput`: `ticket_id`, `agent_id`, `request`,
`attachments`, `work_dir`, and `run_id`. There is no separate hints object.

Interpret explicit values written in `request` and inspect the assigned
attachments. State the resolved source, planned method-compatible record shape, target
size, and initially planned canonical `method_ids` in the `Starting:` message. Use an
attachment as the source when the request assigns it; do not replace it.

If the request leaves training semantics open, choose a method whose required
signal exists and derive framing from the requested task/record shape. When no
source attachment exists, derive a concrete acquisition query from `request`.
Preserve rows unless filtering is requested or source inspection establishes a
specific validation-safety/quality reason; report every applied operation.

Standalone work normally has no scoring lane. Leave scoring-derived result
paths empty unless the request explicitly supplies a scoring set and asks for
those derivatives. After resolution, return the normal `DataResult` from the typed execution phase.
