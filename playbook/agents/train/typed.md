## Work order

| Field | Contract |
|---|---|
| `operation` | Always `train`. |
| `iteration` | 1..N; iteration 0 has no Train stage. |
| `dataset_path` / `data_signature` | Exact versioned training records and their verified identity. `data_signature` is already canonical; copy its bare 64-character lowercase SHA-256 exactly. |
| `validation_dataset_path` | Engine-bound raw Validation scoring records; use only to build trainer evaluation data after configuration is fixed. |
| `validation_answer_fields` | Exact ground-truth fields to map into the method's temporary trainer-eval representation. |
| `inference_config_path` | Baseline Inference YAML; copy its prompt, template identity, and exact `template_kwargs` and load them at runtime. |
| `expected_inference_config_sha256` | Engine-computed bare lowercase 64-character SHA-256 of the exact baseline YAML bytes. Copy it exactly into `train_config.yaml.inference_config_sha256`; do not recompute it. |
| `train_config_schema` | Exact machine-readable key/type authority for `train_config.yaml`; never infer keys from a Skill example or prior Run. |
| `training_method_contracts` | Exact per-method `method_config`, complete loss contract by prompt framing, objective-config keys/recommended starts, `recommended_training_starts` (non-binding best-practice starting values for common TrainingConfig knobs—LoRA rank/alpha/target-modules, learning_rate, effective batch, warmup, epochs, packing), required software-version keys, PEFT/LoRA state, and hash encoding. Copy the machine-readable requirements instead of inferring them; treat `recommended_training_starts` as guidance to start from and deviate from with rationale. |
| `config_validation_command` | Side-effect-free validator that must pass after the optional tokenizer/data-collator-only example check and before optimizer/model execution. Its Cluster form enforces `training.implementation_config.dataloader_num_workers=0`; preserve the supplied `--cluster` argument. |
| `telemetry_helper_path` | System-owned callback to copy/import unchanged; it forwards all numeric Trainer logs. |
| `telemetry_interval_steps` | Fixed optimizer-step logging cadence: 20. |
| `execution_contract` | Mandatory foreground streaming, 4-hour tool deadline, and exact local/remote Run/Ticket environment values. |
| `base_model` | Original model identity. |
| `model_source` | `base_model` or `checkpoint`, selected before this experiment. |
| `parent_selection_rationale` | Orchestrator's evidence-based reason for the selected branch point. |
| `parent_checkpoint_path` | Empty for a baseline branch; otherwise the selected earlier Run checkpoint. |
| `parent_train_config_path` | Empty for a baseline branch; otherwise YAML from the same parent Train Ticket. |
| `branch_transition` | Engine-validated exhausted-branch record. `none` while continuing; a Method change carries its prior branch, Validation evidence, and next branch. |
| `training_method_pin` | Binding method when non-empty. |
| `method_config_pins` | Binding method-owned values. |
| `loss_objective_pins` | Binding method-valid objective values. |
| `configuration_suggestions` | High-level Orchestrator method/direction, plus any user advisory customization. |
| `configuration_pins` | Other strict Customized Pipeline values. |
| `device_info_path` | Cloud/instance device contract or cluster access route. |
| `slurm_job` | Engine-owned finite-job path/name, job-id stdout/stderr patterns, GPU count, queue deadline, phase, backend-observed JOBID/state, bookkeeping endpoints, and exact schemas. Enabled only for cluster. `submit` creates one job and returns deferred; `collect` never duplicates it. |
| `generation_backend` | Run-owned backend for methods that generate. |
| `work_dir` | Persistent local config/script/log directory. |

`run_context.used_training_methods` and `run_context.validation_history` are
engine-owned evidence for active-branch retention, exhaustion, and direction
selection. They do not override the user pin, selected parent Train YAML, or
the current model lineage's Baseline Inference YAML.

`train_config.yaml` must validate as `TrainRunConfig` and include:

- iteration, exact parent model, `baseline`/`run_checkpoint` parent kind and parent-selection rationale,
  and the bound `data_signature`;
- training method and closed method config;
- complete semantic loss contract;
- all concrete training values: epochs, maximum sequence length, per-device
  batch size, gradient accumulation, world size, effective batch size, learning
  rate, optimizer, scheduler, warmup, weight decay, gradient clipping,
  precision, distributed strategy, gradient checkpointing, packing, seed, the
  fixed 20-step logging cadence, explicit step-based Validation evaluation
  cadence, the complete active-or-zero LoRA configuration, and an explicit
  `checkpoint_retention` policy. `strategy=none` is the normal no-intermediate
  case. Any enabled policy has a hard cap, a concrete rationale, and
  `save_only_model=true` for the branch artifacts ultimately reported in the
  Result. `training.implementation_config.save_only_model` records the distinct
  framework checkpoint behavior as an explicit boolean and is `false` for
  finite Slurm work, preserving full state for exact walltime continuation.
  For Transformers/TRL, reserve the automatic terminal checkpoint slot with
  `training.implementation_config.save_total_limit =
  checkpoint_retention.max_intermediate_checkpoints + 1`; after the final model
  is saved separately, exclude/remove its duplicate terminal checkpoint and
  compact the selected reported branch artifacts to model-only directories
  before enforcing the declared intermediate cap;
- every additional framework argument that reaches the trainer under
  `training.implementation_config`, plus the actual package versions under
  `training.software_versions`; no unrecorded library default may affect the run;
- prompt, tokenizer source, chat-template source/hash, exact `template_kwargs`, and special-token ids
  copied exactly from baseline Inference;
- Run-owned generation backend used by rollout-capable methods;
- baseline YAML path and the bare 64-character lowercase SHA-256 of its exact file bytes;
- `prompt_alignment`: measured proof from the real Train rendering/collator
  path that the Baseline synthetic prompt renders identically, is a string and
  token prefix of a synthetic completed assistant sequence, has all context
  labels ignored, and starts the unmasked target at the synthetic response;
- `training_data_example`: one method-shaped normalized record using synthetic
  placeholders, the exact tokenizer/template-rendered sequence(s), declared
  loss-target and context-only placeholders, required per-sequence
  `sequence_loss_targets`, and a concise loss explanation;
- one coherent direction;
- `method_diversity_status`: `initial`, `retained_in_branch`, `varied`, or
  `user_pinned` (`retained_for_constraints` is accepted only for an older
  compatible artifact);
- a concrete `method_selection_rationale`;
- suggestion decisions and rationale.

Build the example through the same normalization and rendering code paths used
by `train.py`, loading `template_kwargs` from baseline Inference rather than
embedding them. Use `<INPUT>`, `<TARGET_RESPONSE>`, `<CHOSEN_RESPONSE>`,
`<REJECTED_RESPONSE>`, or other clearly synthetic placeholders appropriate to
the selected method. Do not copy a real training row, Validation/Test answer,
or held-out content. Pairwise methods show both rendered branches; rollout or
teacher methods show the sequence(s) their objective actually consumes.
Every source-record value that can vary across real normalized records must be
a declared placeholder; a literal is allowed only when normalization forces
that exact value for every row. Array offsets are not semantic identifiers.
For every rendered sequence, set `sequence_loss_targets[sequence_name]` to the
ordered placeholders that are directly optimized in that sequence. In
multi-turn SFT, an earlier assistant response is a target in its own expanded
sequence but masked context in a later one; the per-sequence mapping must say
so explicitly. This metadata lets observability render arbitrary normalized
record shapes without guessing dataset column names.

Return exactly one `TrainResult`. A cluster submission returns
`status="deferred"` after the JOBID is registered; this is not success or
failure and carries no final checkpoint. It posts `Waiting:`, never `Done:`;
the engine posts `Done:` only after a later collect Result and its artifacts
pass validation. For cluster success,
`slurm_script_path` is exactly `slurm_job.script_path`; for other providers it
is empty. Success requires absolute paths for
`train_config_path`, `train_script_path`, `log_path`, and the verified
checkpoint, plus measured example count, loss, and duration. The YAML alone owns
the actual base model, parent, method, objective, hyperparameters, and rationale.
The system-owned telemetry import in `train.py` and the non-empty log are
required observability artifacts; neither can compensate for a missing config
or checkpoint.

When the execution contract enables Weights & Biases, `tracking_url` must copy
its engine-owned public URL exactly; otherwise it must be empty.

`intermediate_checkpoints` is normally empty. When the YAML enables retention,
report each actually verified branch point with its path, step and/or epoch,
concise retention reason, and `save_only_model=true`. Do not repeat the final
`checkpoint_path` in this list. Loss evidence comes only from the runner-owned
diagnostics, never from an Agent-authored checkpoint annotation.
All retained paths must be Ticket-unique siblings of the final model directory,
not directories nested inside it. These successful Result artifacts are not the
full-state recovery checkpoints used to continue a finite Slurm Train job.

The runner—not Train—derives `training_diagnostics.json` from the newest real
Trainer attempt's telemetry. It summarizes available training-loss and trainer
Validation-loss series without asserting a diagnosis. Do not invent or hand-edit
this artifact.
