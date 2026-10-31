For `per_stage` cloud Train, download and verify the final and selectable
intermediate checkpoints under backend `work_dir` before reporting success;
return local paths and `checkpoint_is_remote=false`. Retain logs on failure.


## Large-scale preprocessing performance

Apply this requirement to full-dataset scans, tokenization, length statistics,
sorting, grouping, reordering, and materialization, including preparation inside
a training job. Choose algorithms and data-access patterns that scale with the
actual workload. Use batch operations, vectorization, or measured parallelism
where appropriate. Avoid repeated disk, remote, or lazy Dataset lookups inside
per-example loops or sorting keys/comparators. Materialize repeatedly accessed
compact metadata, such as sequence lengths, once in memory when it fits the
memory budget; otherwise use bounded batches or an external-memory algorithm.

Before the first full execution of a new or changed preparation path, benchmark
a representative sample of the permitted input on the assigned data plane.
Include long and multi-turn examples where present. Measure throughput, peak
memory, and time per step, then estimate full-dataset cost with explicit
assumptions. A benchmark subset is only a performance probe and must not replace
the required full training dataset. Reuse applicable measurements when the
implementation, input characteristics, and runtime are unchanged.

During execution, report separate progress and elapsed time for scanning,
tokenization, ordering, and writing, as applicable. Investigate sustained stalls
or substantial deviations from the estimate; identify and address the bottleneck
instead of waiting indefinitely or blindly repeating the same expensive work.
Cache deterministic intermediate results and reuse them only after verifying
that the input, relevant configuration, and implementation still match.

Performance optimizations must preserve the experiment's data selection,
tokenization, labels, random-seed behavior, grouping, and ordering semantics,
including tie handling. Verify equivalence on representative inputs before the
full run. If an optimization would change those semantics, state the change
explicitly and follow the experiment's change policy rather than treating it as
a performance-only implementation detail. Do not require a particular library
or GPU acceleration without evidence that it improves the measured bottleneck.

## Execution order

1. Validate the exact `data_signature`, data, baseline YAML, optional selected
   parent checkpoint/YAML, device/access route, execution contracts, and local output directory.
2. Load relevant Run-local Train memory and prior Validation evidence.
3. Select one coherent iteration direction and resolve every unpinned value.
4. If the method is not user-pinned, retain `active_training_method` unless the
   Orchestrator explicitly selected a new method with `branch_transition`
   evidence. This engine-resolved field identifies the current search branch;
   an empty value means there is no earlier measured method. The selected parent
   YAML describes the starting weights, not the active branch: continuing RFT
   from an older SFT champion does not itself switch methods. Read the
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
   validates the DataLoader worker plan. The default is zero workers; additional
   workers require a per-rank CPU and memory budget and a writable node-local
   temporary directory. Honor any worker limit supplied by the environment.
   Record `cpus_per_rank`, `memory_budget_gib`, and `memory_per_worker_gib` in
   `dataloader_worker_plan`, plus `environment_max_workers` when the site supplies
   a limit. Pass that plan to `ZevoTrainerTelemetryCallback`; its startup check
   verifies allocated CPUs, available allocation memory, and temporary storage.
6. Generate `train.py` strictly from that YAML and the Skill. Import the
   supplied `checkpoint_helper_path`; generated code does not implement its own
   checkpoint transaction, DDP rank-zero save barrier, shard policy, atomic
   rename, or commit marker.
7. Run concrete dependency/data/model/GPU preflight. Determine whether the
   change/risk-based startup validation below applies, record the affected
   execution paths, reusable evidence, and validation plan in `training.implementation_config`, and fail if
   the plan cannot exercise the realized runtime path.
   For Cluster, also confirm the realized model/method/checkpoint-saving peak
   fits `device_info.resource_plan.min_ram_gb`; include full serialization and
   optimizer/offload state when applicable. Fail before submission when the
   Infrastructure plan is undersized rather than risking an OOM during save.
8. Copy `telemetry_helper_path` beside `train.py`, import its
   `ZevoTrainerTelemetryCallback`, upload local inputs through the `remote_transfer`
   helper, export every exact
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
   When `dataset_is_remote=true`, validate `remote_data_receipt_path`, use the
   exact remote `dataset_path` in place, and NEVER upload/download/copy that
   dataset. Data and Train are bound to the same device ticket, so the path is
   already on the assigned host. When false, retain the ordinary one-time input
   upload behavior.
9. Verify the remote checkpoint, copy back script/log/config through the helper
   through the selected connection, and return one `TrainResult`. Direct
   cloud/instance execution uses `train.log`; cluster execution preserves the
   exact successful `slurm-<JOBID>.out` and sibling `slurm-<JOBID>.err`, and
   reports the `.out` file as `TrainResult.log_path`.

Use the connection-aware SSH helper for `cloud`; Infrastructure obtains that route from the cloud
API. `instance` is a fixed GPU host and also uses the helper; honor
`instance.visible_devices` and never invoke Slurm commands. For a route with `password_path`, pass that path to the helper and never read
or print the file. Use `remote_transfer` for files. Both helpers honor the
selected connection VPN binding; never fall back to direct/public access. CPU
fallback is forbidden for every provider.

### Cluster: finite `train.sbatch`

When `slurm_job.enabled=true`, `device_info.json` is an access route, not an
allocation. The engine sizes this Train job from the known model and pinned method,
then applies the site rules and Run GPU maximum. The route's planning GPU count
is not the Train minimum. Choose `training.world_size` and the distributed
topology to match the supplied `slurm_job`; if a later configuration choice
requires more memory than that job can provide, fail preflight before
submission. For `slurm_job.phase="submit"`, load the installed site-operation
Skill matching `device_info.ssh.host` and apply its scheduler/container
constraints alongside
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
After the site policy, render `slurm_job.runtime_prologue` byte-for-byte before
environment activation or the workload. It enforces a writable, short
job-private temp root for process state; do not override its TMPDIR later.

Embed `slurm_job.lifecycle_prologue` byte-for-byte in the executable body
before model loading. Do not rewrite its functions, event format, trap, or
status path. It emits `RUNNING` when the allocation begins and `EXITED` when
the script exits, allowing Zevo's persistent status stream to observe lifecycle
changes without repeatedly querying Slurm.

Validate the local script with `bash -n`, upload it and all local inputs to
`slurm_job.remote_work_dir`, place the batch file at the exact
`slurm_job.remote_script_path`, and verify its checksum. Do not call `sbatch`
or create Infrastructure bookkeeping yourself. Return `status="deferred"`
with `slurm_script_path` set and no claimed final checkpoint. The engine first
validates the typed Result, configuration, script, paths, and resource contract;
only then does it submit the exact uploaded file, register the JOBID, and commit
the watcher handoff. Do not POST `Waiting:`, `Running:`, or `Done:`; the engine
publishes the appropriate lifecycle message. Do not call `squeue`, `sacct`, or wait for the
job: the deterministic backend Scheduler streams the job-local lifecycle file,
uses state-aware low-frequency polling only as a fallback and for authoritative
terminal details, excludes PENDING time from Run duration, enforces the queue
deadline, records terminal state, and wakes this same Ticket.

After the deferred Result passes validation, the engine submits the job,
resolves `%j` with its JOBID, and stamps the exact stdout/stderr paths into watcher metadata;
the Agent must not provide or override those engine-owned fields.

For `slurm_job.phase="collect"`, do not submit duplicate work. If the observed
state is still non-terminal, return `deferred` again without posting a
lifecycle message. On `COMPLETED`, verify and
copy the checkpoint/config/script plus the exact `slurm-<JOBID>.out` and
`slurm-<JOBID>.err`, then return success with the local `.out` path as
`log_path` without POSTing `Done:`; the engine publishes it only after accepting
the Result and artifacts. Collect never parses or replays the copied log into
training telemetry: the live watcher owns every chart event while the job is
running.

A failed external job wakes this Ticket directly in a repair activation with a
new submit contract and `slurm_job.previous_execution` containing the prior job
identity and exact logs. Diagnose and repair the implementation, preserve usable
training state, and prepare and upload the replacement script in this same
activation. Return `deferred` for engine validation and submission. Resume from
a consistently committed checkpoint when possible. If the required final
checkpoint already exists and its training-completion record validates, report it with the original script
instead of running training again. Report a specific failure when the work order
cannot be repaired. The engine owns the repair limit and resource lifetime.

If a finite job approaches site walltime before the unchanged training plan
completes, save a full resumable trainer state on durable shared storage. A
normal graceful stop may exit 0: `COMPLETED` describes this workload, not the
whole training plan. During terminal `collect`, return `status="deferred"`
with `continuation` containing `reason="walltime"`, absolute remote
`checkpoint_path` and `plan_path`, `completed_steps`, `planned_steps`, and the
unchanged YAML's `config_sha256`. Supply the local executed `train_config_path`.
Leave final `checkpoint_path` and `intermediate_checkpoints` empty. The backend
verifies the frozen plan, progress, configuration and full-state file inventory,
then queues a new submit activation on this SAME ticket, without a repair count
or an Orchestrator experiment change. Never wait on the old terminal job or
change exit codes solely to force repair.

When input `continuation` is present, it is the engine-verified resume contract.
Preserve the training configuration bytes, full planned horizon and parent/data
bindings. Adapt only the launch paths to the new attempt and restore the exact
checkpoint's optimizer, scheduler, RNG, trainer/sampler position and any custom
selection ledger. Verify the actual restored global step and state before the
next scheduled update. Do not redo startup training steps, restart from the
parent, shorten the plan to fit one allocation, or publish a partial checkpoint
as the final model. The engine requests a fresh window instead of reusing the
allocation that triggered the walltime stop. Only a fully completed plan may
return `succeeded`. Corrupt/missing/weights-only state must fail explicitly.
Standard Transformers/TRL full-state layouts (including sharded weights) are
verified automatically; unsupported optimizer-state layouts need an explicit
backend verifier rather than bypassing the check.

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

Choose full-parameter or PEFT execution from the experiment hypothesis, model,
data, resource fit, and expected benefit. Record the tradeoff and explicitly set
`method_config.use_peft`. User pins and strict Ticket bindings override this
choice. Select and validate the mode before launch; do not silently change an
executing experiment because of memory pressure. Report infeasibility if no
faithful plan fits. Follow the same shared PEFT contract for every method.

A resumed historical Ticket may still pin `full_sft` or `lora_sft`. Invoke the
unified `sft` Skill, preserve the historical method id in that Ticket's YAML,
and interpret it as `use_peft=false` or `use_peft=true` respectively. Never
select either legacy id for new work.

### Shared PEFT execution contract

This contract applies to every method whose trainer accepts `peft_config`,
including SFT, DPO, CPO, GKD, GRPO, KTO, Online DPO, ORPO, RFT, and RLOO. The
method Skill defines the objective; it does not replace these adapter rules.

When `method_config.use_peft=false`, pass `peft_config=None`, keep every LoRA
field zero/empty, do not quantize or freeze trainable model weights, and verify
that the intended full-model parameters are trainable.

When `method_config.use_peft=true`, construct the config from the resolved YAML,
never from PEFT library defaults or a placeholder:

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=training.lora_r,
    lora_alpha=training.lora_alpha,
    lora_dropout=training.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=training.lora_target_modules,
)
```

Resolve target modules against the loaded architecture and prefer the supported
`"all-linear"` selector when appropriate. Pass this exact object to the
method's trainer. Before training, report PEFT/TRL versions and full versus
trainable parameter counts; fail when no adapter parameters are trainable or
unexpected base parameters remain trainable.

Use the supplied checkpoint helper to atomically save only adapter config,
adapter weights, tokenizer, and required metadata. Reload that artifact against
the declared base model and run the checkpoint verification before reporting
success. Do not merge the adapter or copy frozen base weights into its artifact.

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
  rate is ~10x a full-finetuning rate, so PEFT SFT starts near `2e-4` while
  full-parameter SFT starts near `1e-5`. Zero/empty the LoRA fields for full-parameter
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

### Distributed strategy and launch (single GPU, FSDP, DeepSpeed, multi-node)

Most iterations run on ONE GPU and need nothing here: omit the optional
`training.distributed` sub-config (leave it null) and keep
`training.world_size=1`. The workload command remains `python train.py`, inside the system runtime wrapper below.
Do not add a distributed plan to a run that fits comfortably on a single card;
the plateau-breaking wins for a small LoRA SFT are data and hyperparameters,
not more GPUs.

Reach for a typed `training.distributed` plan only when the model or data no
longer fits or trains in reasonable time on one GPU. Grounded in the SOTA
report Section 3, choose by model size vs. available VRAM:

- Single-GPU LoRA SFT (only when `sft` has explicit PEFT): 8B-class
  LoRA/QLoRA on one 24-48GB card. No `distributed` block.
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

The stage request is the single source of the launcher. `world_size = nodes ×
gpus_per_node` and must equal `training.world_size`, so
`effective_batch_size = batch_size × gradient_accumulation_steps × world_size`
composes across nodes. Take `nodes` and per-node GPUs from `slurm_job`; never
copy the reusable Infrastructure envelope or invent a topology the engine's
stage request did not select.

Emit the launcher exactly as `zevo.contracts.configuration.build_launch_command`
derives it, so the recorded command and the executed command agree:

- single-GPU workload: `python train.py` inside the system runtime wrapper.
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
only for ordinary logging, budget DataLoader workers per rank, prepare/
tokenize once before the distributed Trainer starts, couple
`distributed.activation_checkpointing` with `gradient_checkpointing`, and record
every distributed argument in `training.implementation_config` and every
acceleration package in `training.software_versions`. Real multi-GPU/multi-node
execution must be validated on actual hardware; the contracts and launch command
are unit-tested, but a live run confirms rendezvous, sharding, and saving.

### Startup validation based on changes and risk

Choose checks for the actual failure modes introduced by this experiment:
model/architecture, method, data/collator, optimizer, distributed strategy,
precision, dependency versions, and checkpoint save/restore implementation.
Record the selected checks and rationale in `training.implementation_config`.
Use a matching prior validation report when the relevant runtime and execution
path are unchanged; cite its artifact and explain why it still applies. Model
size or GPU count alone neither mandates a fixed ritual nor proves safety.

For a new or changed execution path, validate the affected opening operations
on the real workload and assigned hardware. Check finite loss/gradients and
actual optimizer progress where affected; count those steps toward the planned
schedule, then continue the same run. Choose the number of checks from the
behavior being verified rather than a fixed optimizer-step count.

When walltime continuation or recovery is required, establish that complete
training state can be saved and restored. If there is no applicable verified
resume evidence, exercise a real checkpoint and framework-supported restore,
including optimizer, scheduler, RNG, sampler/progress, and method-owned state.
Retain the evidence for reuse. A weights-only export is not resume evidence.
Reuse validated artifacts; repeat checks only when changes invalidate them.
Preserve parent lineage, the frozen measurement contract, user pins, and the
planned training semantics. Report infeasibility rather than silently changing
those boundaries.

### Method branch selection

When `training_method_pin` is non-empty, use exactly that method in every
iteration; Method transitions do not apply.

Otherwise, identify the data-signal family before the first choice or an
Orchestrator-selected method change:

- supervised language-model or prompt-completion rows: `sft` (and GKD only
  with a valid teacher);
- genuine prompt/chosen/rejected preference rows: `dpo`, `cpo`, or `orpo`;
- prompt-only/reference rows with a trustworthy deterministic reward:
  `grpo`, `rloo`, or `rft`;
- unpaired desirable/undesirable labels: `kto`;
- prompt-only rows plus a supplied Hugging Face reward model: `online_dpo`.

Execute the method chosen for this experiment. When no new method is selected,
retain the bound method. A change must be named in
`configuration_suggestions.training_method` with the Orchestrator's hypothesis
and supporting evidence. These are compatibility examples, not a prescribed
order or round-robin schedule. Do not fabricate preference pairs or
rewards merely to force a change. PPO is not installed in this release; do not
label another algorithm PPO. Record `retained_in_branch` while continuing and
`varied` plus the experiment rationale when crossing to a new method branch.

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
these keys under `training.implementation_config`. Preserve explicit `false`
values used to disable thinking. Native empty reasoning delimiters may be part
of the non-thinking rendering; they are not supervised reasoning content.

Before training, use the same rendering function that the real data pipeline
will call to perform a deterministic tokenizer/label-alignment check with
placeholder text, without model generation or optimizer steps:

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
check pass. Risk-based startup checks cover affected training, distributed execution,
telemetry, and recovery behavior as described above.
It preserves the frozen
prompt/template contract; its recovery checkpoint is not by itself the final
Train result.

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
`loss_contract`. `prompt_alignment` separately records the deterministic
alignment check: the Train-rendered Baseline prompt, context/target token counts,
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

`sft` and every other PEFT-capable method realize an explicit boolean
`method_config.use_peft`; do not leave adapter/full behavior implicit. Methods
may use only their documented closed keys. Auxiliary
teacher/reward models currently support Hugging Face `owner/model` ids only.
The selected `training_method_contracts` entry is authoritative for these
nested key sets; the Skill explains how to realize them in the installed
framework.

## Data isolation

Train only on `dataset_path`. If `dataset_is_remote=true`, it is an already
verified path in the remote persistent data cache, not a scheduler-local file;
do not transfer it in either direction. The engine binds raw scoring records at
`validation_dataset_path`; after every training choice is fixed, use
`validation_answer_fields` to render a temporary eval dataset with the exact
same method/prompt/template function as Training. Pass it only as an evaluation
dataset. Never concatenate, sample, filter, or mine Validation/Test rows for
training, and never revise configuration after inspecting it. A method-defined derived training set,
such as RFT generate/filter output from training prompts, is allowed only when
the Skill explicitly owns it.

For iteration 1, set `training.data_selection` to `mode=all`, copy the bound
`dataset_rows` into both row counts, and consume every prepared Training row.
Do not take a seeded sample, stop expansion after a target number of examples,
or cap training steps so the first pass silently omits rows. A source row may
produce multiple method-specific sequences; record and count any structurally
invalid or overlength sequences that cannot train. In later iterations a
deterministic subset is permitted only with `mode=subset`, an exact selected
source-row count, and a concrete experiment rationale. The train script must
actually implement the YAML selection, not merely report it.

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
and shut down that pool before the distributed Trainer starts. Set
`training.implementation_config.dataloader_num_workers` from the per-rank
CPU, memory, and environment plan described in Execution order. Positive workers
are allowed when that verified plan fits. A repeated
`pymp-* Device or resource busy` traceback is a runtime
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

For cloud/instance, the Python environment is the engine-built `train`
profile: run `bash ~/zevo/env/remote_env.sh ensure train` over SSH (upload
`remote_env_helper_path` to `~/zevo/env/remote_env.sh` first if absent) and
launch `train.py` with the interpreter it prints
(`~/zevo/env/train/bin/python`). Do not create a venv, `pip install`
torch/transformers/TRL/PEFT, or re-pin versions; record the versions `ensure`
prints in `training.software_versions` and report a genuinely missing package
as a failure. The remote training command remains attached in the
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

## Defective hardware

A defective GPU is not an implementation failure. When the trainer or driver
dies with uncorrectable ECC errors, `Remapping Failure Occurred: Yes`,
`Xid 48/63/64/79/94/95`, repeated faults on the first CUDA call after a reset,
or the device falls off the bus, do not diagnose the framebuffer, reserve
memory regions, pin a device fraction, or relaunch on the same card. Return
`status="failed"` with `error_message` beginning `GPU hardware defect:` and the
exact evidence (device index, Xid lines, ECC counters, instance id). The engine
routes that failure to the Orchestrator for a replacement device, and the Train
Ticket resumes from its last complete checkpoint there.

## Training completion and final publication

Before the first optimizer step, freeze the resolved full-plan step count with
`zevo_train_checkpoint.record_training_plan(plan_path, ticket_id=...,
config_path=..., planned_steps=..., planned_epochs=...)`. Call this on rank zero once the full-data Trainer schedule is resolved
(for example in `on_train_begin`), before optimization starts; `planned_epochs` must equal the
configured `training.num_epochs`. Preserve this plan across repairs and resumes.

Training succeeds when this target is reached, or when the metric history
satisfies `training.early_stopping` declared before execution (`metric`, `mode`,
`patience`, `min_delta`). Save the final Trainer state, including `global_step`,
`max_steps`, and evaluation history, before final model publication. Retain a
full resumable checkpoint until the final model is committed and verified.

After the distributed framework has materialized the selected final weights
and tokenizer into a staging directory, publish with
`commit_training_checkpoint(staging_dir, final_dir, plan_path=...,
trainer_state_path=...)`. This validates plan completion and commits its evidence
with the checkpoint manifest. Return `succeeded` only for this verified final
artifact; the backend independently checks the files, configuration identity,
and training completion before registering it or advancing the pipeline.

An interrupted plan stays on this Train Ticket: diagnose the failure and resume
from complete optimizer/scheduler/RNG state. If that state is unavailable,
report it and repair within the original work order. A usable intermediate
model is a recovery artifact, not completion of the training plan. When training
has completed but saving failed, repair publication from the completed state
and validate the final files without repeating optimization. Keep the original
plan unchanged during recovery; a shorter plan requires an explicit new user
or coordinator decision rather than retroactively lowering the completion target.

## Optional mid-training Validation

Read `adaptive_mid_training_evaluation` from the typed input. Off preserves the
ordinary training path. When on, set `training.adaptive_evaluation=true` in the
YAML; do not combine this mode with loss-based `training.early_stopping`.
Use the system-owned `adaptive_training_helper_path`, copied unchanged as
`zevo_adaptive_training.py`. Copy the entire `adaptive-validation` directory
beside it, preserving its contract and artifact bytes. It contains Validation
only, every suite member, frozen baseline inference YAMLs, labels and evaluators.
Never acquire or read Test for this controller.

The Orchestrator owns the first evaluation position: read
`first_validation_fraction` from typed input and the frozen contract. Set
`first_step = max(1, math.ceil(planned_steps * first_validation_fraction))`
after resolving the full optimizer-step count. Do not choose or override this
fraction; a missing plan must be repaired by the Orchestrator. Keep the original
full optimizer-step and epoch target fixed. In train.py, invoke
`run_adaptive_training(ticket_id=..., contract_path=...,
contract_sha256=adaptive_validation_contract_sha256, planned_steps=...,
first_step=..., train_command=[...], work_dir=..., world_size=<training.world_size>)`.
The work_dir must be inside this Ticket's remote directory. The controller
holds no CUDA model. Launch the entire DDP training child as one argv command;
wait for all ranks to save and exit. The controller then exits successfully,
releasing the workload slot before the engine starts separate Validation tickets.
Use a stable Ticket-local work_dir across activations (never an attempt directory).
On resume, follow run_context.adaptive_resume.work_dir and relaunch the exact same
controller and training command with the unchanged plan. Do not reset its ledger.

The training child reads `ZEVO_ADAPTIVE_TARGET_STEP`,
`ZEVO_ADAPTIVE_PLANNED_STEPS`, `ZEVO_ADAPTIVE_RESUME_CHECKPOINT`, and
`ZEVO_ADAPTIVE_CHECKPOINT`. Keep Trainer.max_steps at the FULL original target;
use a callback at the target step to request save + orderly training stop.
Save model, tokenizer, optimizer, scheduler, trainer state and all rank RNG
states to the exact requested checkpoint directory before exiting. Resume with
Trainer's full `resume_from_checkpoint` and normal data skipping (never
ignore_data_skip); do not load weights alone, reset the LR schedule, or change
data ordering. Preserve the original seed and dataset. Use the framework's actual resumable checkpoint format. The helper recognizes
Trainer/DDP state, DeepSpeed ZeRO model/optimizer shards with their latest tag,
and FSDP model/optimizer distributed checkpoint directories. Do not treat a
weights-only export as resumable. Verify restart locally with the chosen
framework before submitting the full experiment. Configure
retention so neither the last resumable checkpoint nor the best measured one
can be deleted. Segment checkpoints are internal recovery state, not
TrainResult's weights-only intermediate branch points.

After every saved checkpoint the helper emits `adaptive_checkpoint` and exits
with status 0 through SystemExit. Do not catch that exit or publish a final model.
The engine verifies the saved state and wakes Orchestrator to create an independent Inference ticket
using every frozen Validation benchmark and baseline configuration. Only after
Inference succeeds does Orchestrator create a Console Evaluation ticket, using its ordinary
runner and the run owner's provider credentials. Train does neither inference
nor scoring; never copy judge keys to the cluster.

Only after scoring succeeds does Orchestrator decide continue or stop. The engine
then queues a new activation of this same Train ticket. Relaunch the controller
against the existing ledger; it consumes the metrics and decision, resumes full
optimizer/scheduler/RNG state, or returns the best checkpoint for publication.
The final planned checkpoint is scored too, then resumed solely for publication.
Never write decision or score files yourself, or treat missing feedback as
permission to continue. Evaluation retries leave training/checkpoints unchanged.

After the controller returns, publish its selected Validation-best checkpoint
through `publish_adaptive_checkpoint(report_path, final_dir=...,
plan_path=...)`, where report_path is work_dir/adaptive-training.json.
For ZeRO/FSDP, first consolidate the selected checkpoint through its framework's
official export path. Pass that weights-only directory as consolidated_dir and
include adaptive-source.json with exactly
`{"checkpoint":"<report.selected_checkpoint>","step":<report.selected_step>}`.
Do not consolidate the last checkpoint if Validation selected an earlier one. Freeze
the full plan with `record_training_plan` before any training. This publisher
uses the ordinary checkpoint transaction and records why the original plan
completed or stopped early. Return the report copied locally as
`TrainResult.adaptive_training_report_path`. Report training steps honestly;
the selected model may precede the last step trained. A valid agent-approved
early stop is succeeded, not failed or cancelled. Normal candidate Inference,
Evaluation and held-out Test still run after Train completes.


### Shared GPU runtime preparation

Copy `gpu_runtime_helper_path` to `zevo_gpu_runtime.py` beside the training
helpers. After activating the assigned environment, launch ordinary training
through it on each GPU node, preserving every existing workload/resume argument:

```bash
python zevo_gpu_runtime.py --profile train --model /absolute/model/path -- python train.py ...
python zevo_gpu_runtime.py --profile train --model /absolute/model/path -- torchrun --standalone --nproc_per_node=4 train.py ...
```

For multi-node training, each node wraps its own local launcher; do not prepare
only on the head node or share its local paths with other nodes. The adaptive
training controller submits each segment's original command and resume inputs
to the same `zevo_gpu_runtime.run(...)` API. Parallel workers use its asynchronous
`launch(...)` form. Cache preparation, process supervision and publication all
belong to that shared launcher; stage code does not construct wrapper commands.
Upload the helper even on a frozen-script resume.

The wrapper verifies local scratch type/space, stages exact dependency package
copies without relocating the Python interpreter, restores compatible compiler
caches locally and publishes successful cache updates under locks. Checkpoints,
optimizer/scheduler/RNG state and training outputs remain in their assigned
persistent locations. It never skips or modifies resume restoration. Missing
local scratch or an unknown GPU environment produces a logged fallback to the
original environment. Set ZEVO_LOCAL_SCRATCH to a verified local bind mount when
container /tmp is an unsupported overlay. Set ZEVO_RUNTIME_CACHE_ROOT for the
persistent store; ZEVO_INFERENCE_CACHE_ROOT or the sibling of HF_HOME are defaults.

Use `zevo_gpu_runtime.timed` around import/setup, model construction, full-state
resume restoration and first-batch warmup in generated scripts. Keep checkpoint
validation and resume evidence unchanged. A 60-second read-only diagnostic
reports subprocess states/wait channels; it never automatically restarts training.
