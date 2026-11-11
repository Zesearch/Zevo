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

Require a questions-only scoring set, `device_info.json`, and a submission
template. Require either a model/checkpoint attachment or an explicit baseline
request plus base model. Resolve framing, generation_backend, decoding, and supported
`inference_config` values from `request` and verified artifacts; state them in
`Starting:`. Never seek a scoring file containing answers.

Do not guess another Run's checkpoint, an output schema, or chat/completion/text
framing. Inference remains remote GPU work. Missing or ambiguous required
inputs produce a `failed` `InferenceResult`.
