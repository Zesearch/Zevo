Input is the shared `FreeformInput`: `ticket_id`, `agent_id`, `request`,
`attachments`, `work_dir`, and `run_id`. There is no separate hints object.

Resolve provider, provision/release operation, sizing, ownership, and backend
from explicit request text. Inspect an attached requirements file or Dockerfile
only when assigned; never execute an attachment merely because it exists.

Default to the no-spend `instance` provider when no provider was chosen, and
fail if no user-managed allocation exists. Never infer a paid cloud rental from
credentials. A release requires an exact resource handle and ownership. Before
paid provisioning, post provider, offer, GPU count/type, and hourly price.

Select at most one matching site-operation Skill. Ambiguous provider,
ownership, release target, or paid sizing produces a `failed` `InfraResult`.
