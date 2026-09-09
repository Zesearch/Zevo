## Execution order

1. Validate the exact `data_signature`, data, baseline YAML, optional selected
   parent checkpoint/YAML, device/access route, execution contracts, and local output directory.
2. Load relevant Run-local Train memory and prior Validation evidence.
3. Select one coherent iteration direction and resolve every unpinned value.
4. If the method is not user-pinned, use the selected parent YAML and Run method
   history to retain the active method unless the Orchestrator explicitly
   selected a new method after exhausting that branch. Read the
   selected `playbook/skills/train/<method>/SKILL.md` completely.
5. Select exactly that method's entry from `training_method_contracts`; use its
   allowed/required nested keys and treat `recommended_start` and
   `recommended_training_starts` as guidance to start from, not a pin. Pass one
   synthetic placeholder record through the same normalization,
   tokenizer/template, and loss-target selection used by the real trainer,
   loading the exact `template_kwargs` from baseline `inference_config.yaml`.
   Run the bounded deterministic alignment check described below; it must not
   run optimizer steps or generate training outputs. Store the result as
   `training_data_example`. Build `train_config.yaml` using
   `train_config_schema` as the outer key/type authority, then run the supplied
   `config_validation_command` successfully. For Cluster, that command also
   enforces the explicit runtime constraint
   `training.implementation_config.dataloader_num_workers=0`; do not remove or
   bypass its `--cluster` argument.
6. Generate `train.py` strictly from that YAML and the Skill.
7. Run concrete dependency/data/model/GPU preflight.
   For Cluster, also confirm the realized model/method/checkpoint-saving peak
   fits `device_info.resource_plan.min_ram_gb`; include full serialization and
   optimizer/offload state when applicable. Fail before submission when the
   Infrastructure plan is undersized rather than risking an OOM during save.
8. Copy `telemetry_helper_path` beside `train.py`, import its
   `ZevoTrainerTelemetryCallback`, upload inputs through the `remote_transfer`
   helper or correctly constructed direct SCP, export every exact
   `execution_contract.required_environment` entry remotely, and run training
   on the assigned GPU. For cloud/instance, ensure the real trainer and its
   workers inherit `ZEVO_TICKET_ID`; instance also exports the exact leased
   `CUDA_VISIBLE_DEVICES`. For cluster, follow
   the finite stage-job procedure below. These are mandatory cancellation
   identities, not display labels. Use
   `execution_contract.timeout_seconds` for `run_bash.timeout_sec`, or
   `timeout_milliseconds` for a CLI Bash timeout; never accept the tool's short
   default for the real training command.
   When `execution_contract.tracking_provider=weights_and_biases`, require the
   `wandb` package, export the non-secret W&B values exactly, forward each name
   in `secret_environment_names` without printing its value, initialize W&B
   with the fixed run id/name/entity/project, and configure the real Trainer
   with `report_to=["wandb"]`.
9. Verify the remote checkpoint, copy back script/log/config through the helper
   or correctly constructed direct SCP, and return one `TrainResult`. Direct
   cloud/instance execution uses `train.log`; cluster execution preserves the
   exact successful `slurm-<JOBID>.out` and sibling `slurm-<JOBID>.err`, and
   reports the `.out` file as `TrainResult.log_path`.

Use direct SSH for `cloud`; Infrastructure obtains that route from the cloud
API. `instance` is a fixed GPU host and also uses direct SSH; honor
`instance.visible_devices` and never invoke Slurm commands. For any route with `password_path`, prefix direct SSH/SCP with
`sshpass -f <password_path>` and never read or print that file; routine
transfers should use `remote_transfer`, which handles this automatically. CPU
fallback is forbidden for every provider.

### Cluster: finite `train.sbatch`

When `slurm_job.enabled=true`, `device_info.json` is an access route, not an
allocation. For `slurm_job.phase="submit"`, load the installed site-operation Skill matching
`device_info.ssh.host` and apply its scheduler/container constraints alongside
the selected Train method Skill; fail if multiple site Skills match. Write the
exact local file at `slurm_job.script_path`. The script must
contain all resolved partition/account/QOS, GPU count, CPU, RAM, walltime,
working-directory, and log settings as `#SBATCH` directives; the job name is
exactly `slurm_job.job_name`. Its body activates `cluster.env_setup`, validates
the allocated GPU with `nvidia-smi`, exports the required non-secret execution
environment, runs `train.py` in the foreground, and exits with that process.
Never use `sleep infinity`, `salloc`, `srun --overlap`, `sbatch --wrap`, or an
empty allocation job. Keep secrets out of the script and logs; forward named
secrets through a mode-0600 remote environment file and remove it at terminal
cleanup.
Use exactly `#SBATCH --output=<slurm_job.stdout_path>` and
`#SBATCH --error=<slurm_job.stderr_path>`. Keep stdout and stderr separate;
never write a cluster job to `train.log`, reuse one filename across repairs, or
combine both streams with one directive.

Before environment activation or any Python/package import, follow the matched
site Skill's writable temporary/cache policy. Keep durable outputs and recovery
checkpoints in the mounted Ticket directory and use the site's job-private
temporary root for disposable process state. Preserve a configured persistent
model-download cache when the site Skill requires one. Validate the resulting
batch script and allocated runtime; do not copy a hard-coded cache-management
implementation between clusters.

Embed `slurm_job.lifecycle_prologue` byte-for-byte in the executable body
before model loading. Do not rewrite its functions, event format, trap, or
status path. It emits `RUNNING` when the allocation begins and `EXITED` when
the script exits, allowing Zevo's persistent status stream to observe lifecycle
changes without repeatedly querying Slurm.

Validate the local script with `bash -n`, upload it and all inputs to the
ticket-specific remote directory, verify its checksum, then run only
`sbatch --parsable <remote-train.sbatch>`. Parse only the first
semicolon-delimited component as the JOBID, then immediately POST it to
`slurm_job.infra_instances_endpoint` as `provider="cluster"`,
`status="provisioning"`, this Run/Ticket, zero cost, and
metadata containing `auto_release=true`, `stage_job=true`,
`scheduler_state="PENDING"`, the remote script/workdir, and `status_path` equal
to the exact `slurm_job.status_path`; validate
requests/responses with the supplied schemas. If bookkeeping fails, cancel that
exact JOBID and fail. Then return `status="deferred"` with
`slurm_script_path` set and no claimed final checkpoint. Do not POST
`Waiting:`, `Running:`, or `Done:`: after validating the typed Result, script,
status path, and registered JOBID, the engine commits the watcher handoff and
publishes the appropriate lifecycle message. Do not call `squeue`, `sacct`, or wait for the
job: the deterministic backend Scheduler streams the job-local lifecycle file,
uses state-aware low-frequency polling only as a fallback and for authoritative
terminal details, excludes PENDING time from Run duration, enforces the queue
deadline, records terminal state, and wakes this same Ticket.

After the deferred Result passes validation, the engine resolves `%j` with the
registered JOBID and stamps the exact stdout/stderr paths into watcher metadata;
the Agent must not provide or override those engine-owned fields.

For `slurm_job.phase="collect"`, do not submit duplicate work. If the observed
state is still non-terminal, return `deferred` again without posting a
lifecycle message. On `COMPLETED`, verify and
copy the checkpoint/config/script plus the exact `slurm-<JOBID>.out` and
`slurm-<JOBID>.err`, then return success with the local `.out` path as
`log_path` without POSTing `Done:`; the engine publishes it only after accepting
the Result and artifacts. Collect never parses or replays the copied log into
training telemetry: the live watcher owns every chart event while the job is
running. On another terminal state, inspect both exact job
streams. Return failure unless the
continuation rule below applies. Scheduler-owned job state is read-only to the
Agent; do not PATCH it. The allocation ends with the finite script and no
release Ticket follows.

If the finite job approaches site walltime before the unchanged training plan
completes, a graceful pre-walltime hook must save a full resumable trainer state.
After the backend reports `TIMEOUT`, verify that exact state. This same Train
Ticket may then update the audited script only to resume from it, submit one new
finite job, register a new row, and return `deferred` again. This is execution
continuation, not a new iteration or method/data/model decision. Every JOBID and
queue interval remains distinct. Fail visibly if the state is weights-only,
corrupt, absent, the queue deadline expires, or the remaining Run budget cannot
finish; never use a weights-only branch for optimizer-exact resume.

## Parent chain

- Baseline branch: `parent_model=base_model`, `parent_kind=baseline`, with no
  parent checkpoint or Train YAML. This is mandatory for iteration 1 and may be
  selected later for a clean restart.
- Checkpoint branch: `parent_model=parent_checkpoint_path`,
  `parent_kind=run_checkpoint`, with checkpoint and `parent_train_config_path`
  from the same successful earlier Train Ticket in this Run.
- Preserve the supplied parent choice and its meaning in
  `train_config.yaml.parent_selection_rationale`. A concise rephrasing is
  acceptable; `parent_model` and `parent_kind` must match exactly.

The Orchestrator selects lineage before the experiment. Train must not redirect
itself to the latest checkpoint, Registry champion, or another historical model.

## Configuration selection

Resolve each value in this order:

1. `training_method_pin`, method/loss pins, and `configuration_pins`;
2. compatible non-zero Orchestrator suggestions;
3. prior Train YAML and current Validation evidence;
4. verified Run-local memory;
5. the selected method Skill and measured data/device constraints.

An omitted, empty, null, or numeric-zero suggestion means self-select. Do not
record it as a rejected value. Record actual suggestions as accepted or
adjusted, and record self-selected important values with concise rationale.

Orchestrator guidance is high-level (`direction` and optionally
`training_method`). Detailed learning rate, epochs, batching, sequence length,
PEFT, loss-objective, and rollout values remain Train-owned unless the user
pinned or advised them through Customized Pipeline.

One iteration tests one direction. It may change several coupled fields—for
example learning rate and effective batch size as one optimization-stability
direction—but the YAML `direction` must make their shared hypothesis explicit.

### Recommended starting hyperparameters

The selected `training_method_contracts` entry carries
`recommended_training_starts`: non-binding best-practice starting values for the
common `TrainingConfig` knobs. Start the first iteration from them when nothing
higher-priority in the resolution order above pins or advises a value, then move
one coherent direction at a time. They are guidance, not caps; the schema does
not enforce them, and you may realize different values with recorded rationale.

The grounded defaults (see the SOTA finetuning report) are:

- LoRA (any PEFT-active method): adapt `all-linear` target modules, `lora_r` 32,
  `lora_alpha` 64 (keep alpha ≈ 2·rank), `lora_dropout` 0.05. A LoRA learning
  rate is ~10x a full-finetuning rate, so `lora_sft` starts near `2e-4` while
  `full_sft` starts near `1e-5`. Zero/empty the LoRA fields for full-parameter
  training.
- Effective batch size ≥ 32, reached through `gradient_accumulation_steps` ×
  `world_size` rather than a large per-device `batch_size`.
- Sequence `packing` on for SFT to cut padding waste; `warmup_ratio` ~0.03 with a
  cosine schedule; `num_epochs` 2-3 for supervised data (watch for
  `validation_loss_rose_after_minimum`).
- Reinforcement methods use a much lower learning rate (`grpo` ~`1e-6`) and do
  not pack SFT-style; follow the method entry and its Skill.

These starts are the same LoRA numbers whenever PEFT is active for any method;
`recommended_training_starts.lora_adapter_start_when_peft_active` restates them
per method. Do not copy them blindly when data volume, sequence length, or the
device envelope argues otherwise—record the deviation in the YAML `direction`.

### Distributed strategy and launch (single-GPU default, FSDP, DeepSpeed, multi-node)

Most iterations run on ONE GPU and need nothing here: omit the optional
`training.distributed` sub-config (leave it null) and keep
`training.world_size=1`. The launch command stays exactly `python train.py`.
Do not add a distributed plan to a run that fits comfortably on a single card;
the plateau-breaking wins for a small LoRA SFT are data and hyperparameters,
not more GPUs.

Reach for a typed `training.distributed` plan only when the model or data no
longer fits or trains in reasonable time on one GPU. Grounded in the SOTA
report Section 3, choose by model size vs. available VRAM:

- Single-GPU LoRA SFT (default): 8B-class LoRA/QLoRA on one 24-48GB card. No
  `distributed` block.
- `backend=ddp` (single- or multi-node): the model replica fits on one GPU and
  you only need to process more data faster. Pure data parallel; no sharding.
- `backend=fsdp` with `fsdp_sharding=full_shard` (ZeRO-3-equivalent): full-SFT
  of 8B-70B where a replica does NOT fit per GPU. `hybrid_shard` shards within a
  node and replicates across nodes for multi-node. Set `fsdp_offload=true` only
  under hard VRAM limits (it trades speed for capacity).
- `backend=deepspeed` with `zero_stage=2` (optimizer/grad sharding) or
  `zero_stage=3` (full param sharding). Add `offload_optimizer` / `offload_params`
  (stage 3) to train models larger than aggregate VRAM. Prefer FSDP when VRAM is
  flexible (faster per iteration); prefer ZeRO-3 + offload under a hard GPU cap.
- `tensor_parallel_size` (intra-node, ≤ gpus_per_node) and `sequence_parallel`
  (long context) subdivide the world for very large models / long sequences;
  short MC-QA does not need them.

The plan is the single source of the launcher. `world_size = nodes ×
gpus_per_node` and must equal `training.world_size`, so
`effective_batch_size = batch_size × gradient_accumulation_steps × world_size`
composes across nodes. Take `nodes` and per-node GPUs from
`device_info.resource_plan` (`nodes`, `gpus_per_node`); never invent a topology
the Infrastructure plan did not provision.

Emit the launcher exactly as `zevo.contracts.configuration.build_launch_command`
derives it, so the recorded command and the executed command agree:

- single-GPU: `python train.py` (unchanged).
- ddp / fsdp: `torchrun` — `--standalone --nnodes=1 --nproc_per_node=<gpus>` on
  one node; `--nnodes --nproc_per_node --rdzv_backend=c10d --rdzv_id
  --rdzv_endpoint=<rank0-host:port>` across nodes. `accelerate launch` is an
  acceptable equivalent wrapper when its config encodes the same topology/FSDP
  plan. FSDP runs also write the `fsdp_config` (sharding strategy, cpu_offload,
  auto_wrap, activation checkpointing) and pass it to the Trainer.
- deepspeed: the `deepspeed` launcher with `--num_nodes`/`--num_gpus` (plus a
  hostfile and `--master_addr/--master_port` across nodes) and
  `--deepspeed deepspeed_config.json`, whose ZeRO config comes from
  `build_deepspeed_config` (`auto` values resolve from the realized batch/
  precision so the JSON never contradicts `TrainingConfig`).

On any multi-process launch keep the existing single-process discipline: rank-0
only for ordinary logging, `dataloader_num_workers=0` on Cluster, prepare/
tokenize once before the distributed Trainer starts, couple
`distributed.activation_checkpointing` with `gradient_checkpointing`, and record
every distributed argument in `training.implementation_config` and every
acceleration package in `training.software_versions`. Real multi-GPU/multi-node
execution must be validated on actual hardware; the contracts and launch command
are unit-tested, but a live run confirms rendezvous, sharding, and saving.

### Method branch selection

When `training_method_pin` is non-empty, use exactly that method in every
iteration; Method transitions do not apply.

Otherwise, identify the data-signal family before the first choice or an
Orchestrator-authorized exhausted-branch transition:

- supervised language-model or prompt-completion rows: `lora_sft` or
  `full_sft` (and GKD only with a valid teacher);
- genuine prompt/chosen/rejected preference rows: `dpo`, `cpo`, or `orpo`;
- prompt-only/reference rows with a trustworthy deterministic reward:
  `grpo`, `rloo`, or `rft`;
- unpaired desirable/undesirable labels: `kto`;
- prompt-only rows plus a supplied Hugging Face reward model: `online_dpo`.

Retain the immediately prior method by default. A different method is valid
only when `configuration_suggestions.training_method` explicitly names it and
the Orchestrator has exhausted the prior method's credible data and inner
training directions. The named methods above are compatibility examples, not a
prescribed order or round-robin schedule. Do not fabricate preference pairs or
rewards merely to force a change. PPO is not installed in this release; do not
label another algorithm PPO. Record `retained_in_branch` while continuing and
`varied` plus the exhaustion rationale when crossing to a new method branch.

## Prompt and loss

Load `InferenceRunConfig` and copy its `prompt`, `tokenizer_source`,
`chat_template_source`, `chat_template_hash`, `template_kwargs`, and `special_token_ids`
byte-for-semantic-byte into Train YAML. Use that exact tokenizer/chat-template
rendering in the training script. Do not infer prompt framing from dataset
column names or change the model reasoning type/system prompt for training.

`template_kwargs` has one authority: baseline Inference. Load the mapping from
YAML at runtime and pass it unchanged to every `apply_chat_template` call. Do
not reconstruct it from `model_reasoning_type`, probe a preferred value, or hard-code
model-specific thinking controls in generated Train code. An empty mapping means pass no
extra template kwargs; it does not authorize Train to invent one. Do not repeat
these keys under `training.implementation_config`.

Before training, use the same rendering function that the real data pipeline
will call to perform a synthetic smoke check:

1. Render the Inference example messages with
   `add_generation_prompt=True` and the frozen `template_kwargs`; require exact
   text equality with `inference_config.yaml.prompt_example.rendered_prompt`.
2. Render those messages plus a synthetic assistant target with
   `add_generation_prompt=False` and the same mapping; require the first token
   ids to be an exact prefix of the complete token ids.
3. Build labels through the real collator and require every prefix label,
   to be `-100`; require at least one unmasked target token. For a non-thinking
   model, start the target at the synthetic response. For a thinking model,
   include representative reasoning content followed by the response. Both
   must follow the frozen rendering/loss contract.

This check verifies the frozen contract; it must never select or rewrite it. A
mismatch means repair the rendering/collator implementation or report a real
tokenizer/version incompatibility. Do not mutate the Inference YAML to make the
check pass. A later bounded GPU runtime smoke test may validate forward/backward,
distributed execution, telemetry, and saving in an isolated output directory;
it also may not change the frozen prompt/template contract or become the
reported final checkpoint.

All serialized SHA-256 values use one representation: exactly 64 lowercase
hexadecimal characters with no `sha256:` prefix. Copy `data_signature` and
`chat_template_hash` from their supplied authorities. Set
`inference_config_sha256` by copying the supplied
`expected_inference_config_sha256` exactly; do not recompute or decorate it.
The supplied `train_config_schema` enforces these formats.

`training_data_example` makes that alignment visible without exposing a real
row. Its `source_record` follows the selected method's normalized schema;
`rendered_sequences` shows the exact special-token text consumed by the
trainer; `loss_target_placeholders` and `context_only_placeholders` distinguish
objective spans. `sequence_loss_targets` maps each rendered sequence to its
actual optimized placeholders so the UI never has to infer a label from a raw
dataset key or an array position. Every normalized-record value that can vary
between rows is represented by a declared placeholder; literals are reserved
for values the normalizer guarantees are constant. Its `loss_target_summary` must agree with the YAML
`loss_contract`. `prompt_alignment` separately records the measured synthetic
smoke check: the Train-rendered Baseline prompt, context/target token counts,
string/token prefix results, and prefix-label/target-boundary results. Its
rendered prompt must equal Baseline `prompt_example.rendered_prompt` exactly.
Do not require a real multi-turn or method-shaped training example to contain
the same literal input content as the single-turn evaluation example.
Non-thinking-model examples use ordinary targets; thinking-model examples
include representative reasoning content before the answer. A generic
SFT example is not acceptable for a preference,
teacher, rollout, or reward-based method.

Loss is Train-owned and separate from Inference configuration. Derive its
objective and target scope from the chosen method and prompt framing, then use
only the method-valid `objective_config` keys. For SFT, chat normally targets
assistant messages, completion targets the completion span, and text targets
all tokens. The selected Skill/TRL implementation must make that mask real;
chat-template generation markers alone do not choose the loss.

`full_sft` and `lora_sft` use empty `method_config`; `use_peft` is not a valid
alias there. Other methods may use only their documented closed keys. Auxiliary
teacher/reward models currently support Hugging Face `owner/model` ids only.
The selected `training_method_contracts` entry is authoritative for these
nested key sets; the Skill explains how to realize them in the installed
framework.

## Data isolation

Train only on `dataset_path`. The engine binds raw scoring records at
`validation_dataset_path`; after every training choice is fixed, use
`validation_answer_fields` to render a temporary eval dataset with the exact
same method/prompt/template function as Training. Pass it only as an evaluation
dataset. Never concatenate, sample, filter, or mine Validation/Test rows for
training, and never revise configuration after inspecting it. A method-defined derived training set,
such as RFT generate/filter output from training prompts, is allowed only when
the Skill explicitly owns it.

Before GPU work, verify the record family required by the Skill, non-zero row
counts, required columns/messages, tokenizer compatibility, max sequence
length, package/model availability, estimated memory, and expected checkpoint
type. Fail clearly if these do not match the YAML.

## Artifacts and telemetry

The script must load `train_config.yaml`, whose `data_signature` must equal the
Ticket input; prose or environment defaults may not
override it. Save the method-defined adapter or full weights under the remote
Ticket-unique `<remote-run-root>/<TICKET_ID>/model/` directory, then verify
expected config and weight files are non-empty. The returned remote checkpoint
path must contain the exact `TICKET_ID`; never reuse a shared `train/out/model`
path or overwrite an earlier iteration. Do not merge adapters or copy every
checkpoint to the scheduler.

Intermediate retention is optional and hypothesis-driven. Use
`training.checkpoint_retention.strategy=none` unless a small number of branch
points would answer a concrete later assumption, preserve an earlier point
before plausible late degradation, or make a materially different continuation
cheap. The examples are not rules: a three-epoch run might retain epochs 1 and
2, a long run might retain selected steps, and another run may retain nothing.
Never infer that epoch boundaries are always the right choice.

On a queued cluster job, retention is mandatory when estimated training
can approach `device_info.resource_plan.time_limit_hours`. Choose a bounded
step cadence that leaves time before walltime to flush and verify a checkpoint.
Stop gracefully and preserve both a full resumable state for same-Ticket
walltime continuation and selected weights-only branch points when warranted.
The latter may seed a later Train Ticket but are not optimizer-exact resume.
Queue wait belongs to the backend Scheduler and must not enter
`training_seconds`.

When enabled, enforce `max_intermediate_checkpoints`, put retained weights under
Ticket-unique sibling paths such as
`<remote-run-root>/<TICKET_ID>/intermediate/<label>/`, and report exactly those
paths in `TrainResult.intermediate_checkpoints`. A Transformers/TRL Trainer may
write one additional checkpoint at the terminal step even when that step is not
on the configured cadence. Its rotation limit therefore MUST be
`save_total_limit = max_intermediate_checkpoints + 1`, never the intermediate
cap directly. The extra slot is temporary: after training, save and verify the
primary final model separately, remove/exclude the duplicate terminal Trainer
checkpoint, and then enforce the declared intermediate cap. Record both the
realized `save_total_limit` and the framework's actual `save_only_model` boolean
in `training.implementation_config`; never leave either as an implicit default.

The live Trainer checkpoints and the successfully reported intermediate branch
artifacts have different purposes. For a finite Slurm job, set the framework's
`save_only_model=false`: until the whole Train Ticket succeeds, every candidate
for walltime continuation must contain the model plus optimizer, scheduler,
scaler, RNG, Trainer progress, and any other framework state required by
`resume_from_checkpoint`. On the pre-walltime signal, atomically finish and
verify the newest full recovery checkpoint before the allocation ends. A later
JOBID resumes the unchanged training plan from that exact directory.

Only after the Train Ticket completes successfully, copy or compact the selected
branch points reported in `TrainResult.intermediate_checkpoints` to weights-only
directories. Keep model/adapter weights plus the files needed to load them, but
omit optimizer, scheduler, scaler, RNG, and equivalent resume state. Before
success, inspect every reported intermediate directory and fail or clean it if
those full-state files remain. These reported artifacts are valid parents for a
fresh later Train branch, not byte-identical mid-run resumes. Always save and
verify the primary final model separately under `.../<TICKET_ID>/model/`.

Emit phase, step/loss progress, and resolved-config telemetry from the real
process. Keep `train_config.yaml` and `train.py` in local persistent storage;
for direct execution keep `train.log`, while cluster execution keeps the
job-id-specific stdout/stderr pair. Return the remote checkpoint with
`checkpoint_is_remote=true` when appropriate.

When W&B tracking is enabled, finish/sync the W&B run before success and copy
`execution_contract.tracking_url` exactly into `TrainResult.tracking_url`.
Never print, persist in YAML, or return the API key. The W&B project must
already be public for unauthenticated viewers; Zevo does not change account
visibility from a training process. When disabled, do not initialize W&B and
return an empty tracking URL.

`train_config.yaml` is the sole authority for the realized base model, parent,
training method, objective/loss contract, hyperparameters, copied template identity/kwargs,
and rationale. `TrainResult` reports only its artifact paths, remote-location
flag, and measured example/loss/duration summaries; never duplicate YAML fields
there.

The system-owned telemetry callback is mandatory. Configure the Trainer to log
every `telemetry_interval_steps` optimizer steps, pass the current Ticket id to
the callback, and preserve every finite numeric key it receives—including
learning rate, gradient norm, epoch, token accuracy, method-specific metrics,
and evaluation metrics. Do not replace it with a callback conditional on
`"loss"`, and do not suppress evaluation-only log rows. Trainer evaluation
uses the YAML's explicit positive `training.eval_steps` cadence with
`training.eval_strategy="steps"`; the fixed 20-step rule applies to training
logs, not to Validation evaluation.

Structured telemetry owns cluster progress display. Configure the cluster
Trainer with its raw tqdm/progress bar disabled; do not print a second per-step
progress stream beside `__PROGRESS__`. Emit ordinary preflight, dataset,
configuration, and summary lines only from global rank zero. Other ranks may
write only rank-specific warnings or fatal errors that identify their rank.

Do not launch an independent nested multiprocessing preprocessing pool from
every DDP rank. Prepare/tokenize once before the distributed trainer starts and
reuse the resulting Ticket-local materialization, or use a framework-supported
rank-aware cache/barrier so only one rank performs the work. If a selected
method genuinely requires multiprocessing during one-time preparation, finish
and shut down that pool before the distributed Trainer starts. For every
Cluster Trainer, set the realized
`training.implementation_config.dataloader_num_workers` to integer `0`; the
Trainer's DDP ranks must load their own batches without child DataLoader
processes. A repeated `pymp-* Device or resource busy` traceback is a runtime
defect to eliminate, not normal progress output.

The runner materializes `training_diagnostics.json` from the newest concrete
Trainer attempt. Ensure telemetry exposes available training loss and trainer
Validation loss, but do not manufacture either value when the selected method
does not report it. A falling training loss with a Validation loss that later
rises can support a *possible* late-degradation/overfitting hypothesis; flat or
high losses can support other hypotheses. They are diagnostics, not proof, and
the Task's deterministic Validation metric—not loss—still selects the champion.

Every concrete Trainer process launch is a distinct execution attempt. The
system callback creates a fresh UUID, emits `__ATTEMPT__` before optimization,
and repeats that `attempt_id` on every `__PROGRESS__` row. Copy the helper
unchanged; never synthesize, reuse, or remove this identity. If an
implementation-only failure is repaired and Trainer is launched again inside
the same heartbeat, the new process starts a new attempt and its step counter
may restart at zero. Earlier attempt telemetry remains available for diagnosis,
while the live figure displays the newest attempt rather than joining both
processes into one curve.

For cloud/instance, the remote training command remains attached in the
foreground and the local SSH invocation pipes stdout/stderr through
`tee <work_dir>/train.log`. For cluster, the trainer remains foreground inside
the finite sbatch script while the Agent activation ends as `deferred`; the
backend owns external waiting and the later activation copies the complete log.
Do not use `nohup`, shell `&`, a detached Slurm step, or a background tool
option. Never return successful while the process is active.
The trainer and every worker it spawns must inherit the exact
`ZEVO_TICKET_ID` in `execution_contract.required_environment`, so cancelling
the Ticket can terminate only those remote processes.

The YAML must record every trainer argument that actually runs. Common values
belong in `training`; additional TRL/Transformers/distributed arguments belong in
`training.implementation_config`; detected `torch`, `transformers`, `trl`, PEFT,
and acceleration-package versions belong in `training.software_versions`. An
argument omitted from Python may still rely on a library default, so resolve and
record that effective value before launch. `batch_size` is per device and
`effective_batch_size` must equal batch size × gradient accumulation × world
size. `training.logging_steps` is exactly 20; `training.eval_strategy` is
`steps`, and `training.eval_steps` is a positive cadence selected from run
length and evaluation cost. Record the realized checkpoint strategy, cap,
model-only setting, and every framework save argument; do not leave a default
implicit. When retention is active, `implementation_config.save_only_model` is
an explicit boolean and `implementation_config.save_total_limit` is exactly
`checkpoint_retention.max_intermediate_checkpoints + 1`, reserving one temporary
terminal-checkpoint slot. For finite Slurm execution the boolean is `false`, so
an interrupted allocation can resume exactly; local or persistent-instance
execution may use `true` when exact mid-run continuation is not required. Record
an all-zero/empty LoRA
block when PEFT is disabled. Do not inspect old YAML or source models to guess
keys: `train_config_schema` supplied on this Ticket is complete.

If reflection corrects a version-specific argument, data-collator issue,
checkpoint-layout issue, or other reusable pitfall, store the verified lesson
in Run-local Train memory. Memory guides execution and later selection but does
not silently rewrite strict pins or baseline prompt semantics.

Keep implementation/runtime pitfalls `agent_local`. Report an evidenced
`experiment_finding` or broad `recommendation` as `shared_candidate` when it
should inform the Orchestrator's next method or data-direction suggestion.
Do not share exact secret paths, credentials, or held-out information.
