Input is the shared `FreeformInput`: `ticket_id`, `agent_id`, `request`,
`attachments`, `work_dir`, and `run_id`. There is no separate hints object.

Interpret the request into the canonical Run contract once. Preserve explicit
dataset, base model, training method, runtime, scoring, iteration, target, and
cost choices. Assign attachments only to roles the request or their verified
shape establishes. A scoring set without a stated validation/test role is
validation; never expose held-out test files to supervisor state.

Post the resolved objective and constraints, then follow the same state machine
as a typed wake. If safe interpretation cannot produce a valid Run, return
`mark_failed`; do not invent required scoring or paid-resource choices.
