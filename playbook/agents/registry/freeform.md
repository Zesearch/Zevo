Input is the shared `FreeformInput`: `ticket_id`, `agent_id`, `request`,
`attachments`, `work_dir`, and `run_id`. There is no separate hints object.

Require a local model directory and Evaluation-style `metrics.json`. Resolve
`base_model`, `training_method`, `dataset_source`, `task_objective`,
`metric_direction`, and registry destination from explicit request text and
verified model metadata. Derive the tag only as `M-<run_id[:8]>`; freeform text
cannot override Run-owned model identity. Never infer a specific PEFT method from adapter presence alone
when several methods remain possible.

State the resolved provenance and destination before mutation. The runner wraps
single-stage work in a Run, so missing `run_id` is invalid. Ambiguous
provenance, invalid metrics, or a tag-prefix collision produces a `failed`
`RegisterResult`.
