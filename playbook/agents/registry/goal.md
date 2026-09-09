Finalize one Run against its single stable model identity:

1. a Ticket-local `registry.yaml` snapshot of the selected champion;
2. a verified local copy of the engine-selected Validation champion;
   and
3. a reproducible `<work_dir>/register.py` plus one truthful `RegisterResult`.

Only the champion is registered as a model. Evaluated challenger history lives
in Score Events and Tickets, so a losing iteration does not create another
Registry entry.

## Authority boundaries

- `metrics.json` is authoritative. Read its finite numeric top-level `score`
  on the Task-defined scale and preserve the complete metrics object under `eval`. Never recompute,
  normalize, clamp, or replace it.
- Registry does not choose the scorer, direction, winning iteration, or declare
  a model production-ready. The API accepts this Ticket only when its exact
  Train/Evaluation lineage is the Run's Validation champion.
- `base_model`, `training_method`, `dataset_source`, `task_objective`, `metric`,
  and `metric_direction` are provenance supplied by the ticket. Record them without
  reinterpretation.
- You write the YAML and local model artifact only. After a successful result,
  the runner immediately mirrors the YAML into Postgres. Do not call backend
  mutation endpoints yourself.

## Success invariants

- The checkpoint exists in the declared local or remote location and contains
  the expected adapter or full-model weights.
- The metrics file exists, is non-empty JSON, and has a valid `score`.
- The tag is exactly the supplied `expected_version_tag` and the entry belongs
  to the exact same full Run id.
- The Ticket-local manifest is written atomically and contains exactly one
  `models[expected_version_tag]` entry. It never contains another Run's state.
- The champion checkpoint is copied to a temporary sibling and validated before
  it is atomically moved to the stable destination.
- The selected manifest entry's model path is host-visible (normally
  repo-relative such as `data/runs/...`) and resolves to a verified local model.
  The Result reports only the manifest/helper paths; it does not duplicate the
  retention decision or model path.

## Hard limits

- Do not register a missing checkpoint, missing metrics, invalid score, or
  partial copy.
- Do not compare or substitute another iteration.
- Do not delete every directory under a models root. Mutation is limited to the
  one stable destination for this exact Run after containment checks.
- Do not add invented lifecycle labels such as production, staging, blessed,
  or deployed.
- Do not install packages at runtime. PyYAML is part of the application image;
  missing dependencies are an environment failure.
- Do not push to Hugging Face Hub or any other external registry merely because
  credentials are present. External publication requires an explicit user
  instruction and is outside the canonical Registry ticket.
