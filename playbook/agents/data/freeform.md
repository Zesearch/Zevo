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
those derivatives. Return the normal `DataResult`.
