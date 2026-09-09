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
