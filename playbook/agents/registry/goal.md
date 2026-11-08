## Run-selected model destination

When `model_save_policy` is nonempty, prepare your artifacts FIRST. The engine
validates your result and then preserves the selected model at the Run's
chosen destination. A preservation failure becomes a normal failed activation:
you receive its diagnostic in the next auto-repair activation, fix the relevant
artifact, and return a corrected result. There is no separate upload retry loop
and you must not independently publish or handle upload credentials.

Prepare `registry.yaml` with the exact assigned `checkpoint_path` as
`model_path`, `storage=model_save_policy.weights`, and an empty `revision`.
This is a preparation manifest, not proof that publication succeeded. After a
verified save, the engine atomically replaces only these destination fields
with its verified receipt before registering the model. Preserve metrics and
provenance exactly. If supplied `saved_model_path` is nonempty, a previous
attempt already saved the model: reuse that path and `saved_model_revision`.

On repair, inspect the actual preservation error and the assigned artifacts.
For a model-card error, read the checkpoint README.md and write a corrected
copy inside `work_dir`, then return its absolute `model_card_path`. Use the
supplied provenance to resolve identifiers; do not guess a Hub ID from a
folder name. Preserve the card's other accurate information. The engine applies
this card only to an upload copy. Never mutate the measured checkpoint,
weights, tokenizer, or training configuration. Keep the corrected card path
in subsequent repair results. Errors requiring a different resource, input or
fixed configuration follow the shared failure-recovery rules.

Write the preparation and any repair in reproducible `register.py`. Report
`succeeded` when your artifacts are ready for backend validation; the engine
owns the final stage outcome, including upload failures. A saved HF/SSH URL
is valid and must not be probed with local filesystem APIs. These destination
rules supersede local-copy requirements below. Empty policy retains the
legacy local-copy contract for already launched Runs.

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
  credentials are present. External publication is performed by the engine only when the Run
  model_save_policy authorizes it; do not independently publish.
