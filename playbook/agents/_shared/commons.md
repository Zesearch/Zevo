This contract applies to every Agent and both input formats. `typed.md` or
`freeform.md` defines the invocation envelope; `platform.md` defines the
Agent-specific work and result.

## One wake, one activation

One heartbeat handles one activation of a Ticket:

1. Read the resolved input and current Ticket state.
2. POST a `Starting:` message.
3. Perform only this Ticket's work.
4. Validate every claimed output.
5. POST a `Done:` message.
6. Emit one final JSON object and exit.

Finite cluster Train and Inference jobs are the exception to step 5. Their
submission activation POSTs `Waiting:` after the JOBID is registered and
returns `deferred`; it must never say `Done:`. The backend Scheduler publishes
`Running:` when Slurm starts the job. On the collect activation, the Agent
validates and reports the artifacts without posting `Done:` itself; the runner
publishes `Done:` only after the typed Result and every required artifact have
passed engine validation. Thus `Done:` always means the stage is genuinely
complete, never merely submitted or deferred.

Every workflow Ticket is one complete execution activation. Inference selects
its baseline configuration and generates predictions in one heartbeat; Train
selects its iteration configuration and trains in one heartbeat. Durable YAML
artifacts are the single authority for realized configuration.

The Task fixes metric, direction, evaluator, answer fields, and submission
shape before Baseline. Baseline Inference fixes the complete model-side
measurement configuration. Later Inference must reuse that exact file, and
Train must copy its prompt/template identity and template kwargs. Neither suggestions nor memory
may create a second evaluation-affecting configuration.

Do not poll for another activation. The scheduler creates a new wake when the
same Ticket is ready to advance, when another Ticket has work, or when a user
message requires one. Do not repeat completed work unless explicitly reactivated.

## Runtime environment

| Variable | Meaning |
|---|---|
| `$TICKET_ID` | Opaque Ticket id. Echo it; never parse it for semantics. |
| `$AGENT_ID` | Current Agent id. |
| `$RUN_ID` | Parent Run id. |
| `$ZEVO_API_BASE` | Backend base URL. Use it without a localhost fallback. |
| `$WORK_DIR` | Persistent per-Ticket workspace. |

Every Bash call starts a fresh shell. Define variables, functions, exports, and
the command that uses them in the same call. Guard required values before side
effects. Use `python3` and `json.dumps` for JSON; do not hand-build JSON that
contains user text.

Keep inspection output bounded and textual. Use `file`, a schema-aware script,
or a byte/row-limited preview instead of unbounded `cat` on datasets, logs, model
artifacts, executables, or shell-special paths such as `$0`. Never dump a binary
or a whole large single-line file into a tool result.

Every Bash tool call that accepts a `description` must include one. Use a
concise, outcome-oriented phrase (about 3-8 words) that says why the command is
being run, such as `Inspect the assigned GPU` or `Validate the predictions
schema`. Do not repeat shell syntax, long paths, credentials, tokens, or other
secrets in it. Narration before the call does not replace this field. The
platform may derive a display-only fallback when a CLI cannot supply the field;
that fallback never changes the original command or tool input.

## Canonical Ticket records

The stored Ticket envelope has one meaning per field:

- `payload`: the Agent's requested task parameters;
- `inputs`: role-addressed `ArtifactBinding` objects for literal or upstream
  artifacts;
- `customization`: one `AgentCustomization` block;
- `input_format`: `typed` or `freeform`;
- `lane`: `optimization` or `held_out_test`;
- `iteration`: baseline `0`, then trained iterations `1..N`;
- `status`: `queued`, `running`, `repairing`, `awaiting_input`,
  `waiting_external`, `succeeded`, `degraded`, `failed`, `skipped`, or
  `cancelled`. A finite cluster job is `waiting_external` while Slurm reports
  PENDING and `running` while Slurm reports RUNNING.

`repairing` is a non-terminal system state. It means the prior activation
failed for a defect owned by this Agent and the same Ticket will be activated
again, at most three times. The system appends an exact `__REPAIR__` message to
the Ticket conversation. On that activation, inspect existing files and prior
evidence first, reuse verified expensive work, and correct only the invalid
output, generated implementation, command, or artifact. Do not blindly repeat
training/inference when a completed artifact can be verified and reported.
Upstream binding/experiment conflicts go to Orchestrator instead; cancellation,
held-out leakage, authentication, budget, and security boundaries are not
automatically retried.

The runner resolves `inputs` before invoking a worker. The resolved typed input
contains concrete paths such as `dataset_path`, `device_info_path`,
`checkpoint_path`, `predictions_path`, or `metrics_path`. Read those paths
exactly. Never derive paths from Ticket ids or directory conventions.

Optimization Specialists may also receive `run_context`: bounded, engine-owned
Validation history, used training methods, runtime, budget, and safe task
context. It is advisory. Infrastructure additionally receives user/model/data
capacity evidence for resource sizing. Execution Specialists do not receive a
second copy of user pins there: their complete stored payload is the sole
configuration authority. Held-out workers receive an empty `run_context`.

Filesystem visibility is not authorization. Read only paths explicitly assigned
through the typed input, payload, `customization.input_paths`, the current
Ticket workspace, or an applicable Playbook/Skill. Do not enumerate sibling
task directories or use `find`, recursive globbing, or repository-wide search
to discover additional datasets, scorers, submissions, checkpoints, or Run
artifacts. In the `optimization` lane, any held-out Test path is forbidden even
when a shared mount makes it technically visible. The runner fails a Ticket
that touches one, and none of that Ticket's artifacts may enter the DAG. A
`held_out_test` Ticket may read only the exact held-out paths assigned to it.

`payload`, `inputs`, and `customization` are separate. Do not put an artifact
binding in payload, copy a payload value into inputs, or flatten customization
into renamed fields.

## Run-scoped memory

Every resolved Agent input also contains `memory`. It is a bounded continuity
context selected by the runner from this exact `run_id`, `agent_id`, and lane;
it is not a Ticket payload field and it is not provider conversation history.
A new Run starts with no memory, even when it uses the same Task, model, or
Agent. Held-out workers receive no memory while a Run is active.

Treat memory as advisory evidence. The current Ticket, strict user pins,
assigned artifacts, Specialist YAML, and measured files always win. Ignore an
entry whose applicability no longer matches the current model, method,
configuration, evaluator, or data identity. Transient runtime environment is
not a memory applicability dimension. Never use memory to mutate a reused YAML
configuration.

A successful or degraded worker result may return `memory_updates`. Each update
has `kind`, stable `key`, concise `summary`, small structured `details`, and
`visibility`. `details` is a short list of strict string pairs such as
`{"key":"installed_version","value":"vLLM 0.10.2"}`—never a JSON object.
Use separate pairs for separate facts and keep complex evidence concise in the
string value:

- `agent_local`: useful only to later Tickets for this Agent in this Run;
- `shared_candidate`: potentially useful across Agents, but only as advisory
  evidence for an Orchestrator suggestion or decision. It is not a binding
  configuration store.

Use `shared_candidate` only for `verified_fact`, `experiment_finding`, or
`recommendation` entries whose evidence should influence another Agent or the
next Orchestrator direction. Keep command/version/runtime pitfalls and artifact
references `agent_local`. A measured saturation/regression finding or justified
data-direction recommendation should be shared; the shell correction that made
the experiment run should not.

Store only durable, evidenced lessons such as a verified data property, an
observed runtime pitfall, a reproducible experiment finding, or an artifact
reference. In particular, when execution encounters an error, reflection
identifies its cause, a correction is actually verified by successful execution,
and the finding may help a later Ticket of this Agent in the same Run, include
it in `memory_updates`. Record the concise symptom, cause, correction, and
verification evidence rather than the surrounding transcript. A recovered
error is not itself a failed result when the requested work subsequently
succeeds.

Do not store plans, guesses, unverified attempted fixes, routine status, copied
Ticket fields, raw transcripts/logs, credentials, secrets, or held-out Test
paths/scores. Memory is not permission to override strict pins or a reused YAML
configuration: a recovery may be remembered only when it preserves the
assigned semantics and outputs.
Use the same `kind` and `key` to supersede an earlier lesson whose applicability
is unchanged. Do not emit duplicate `kind`/`key` pairs in one result. Failed or
cancelled work must return no lessons.

## Portable shell use

Treat assigned paths as exact absolute paths; never replace one with a basename
or a working-directory guess. The portable runtime guarantees Bash, Python, and
the typed tools, not optional utilities such as `file`. Use Python's standard
library for format inspection when a utility is not explicitly supplied. Do
not use blocking `sleep` as orchestration or readiness logic: query the exact
documented status endpoint/provider and let the scheduler perform later
wakeups. Build API calls only from endpoint and schema fields supplied on the
current typed input.

Build direct SSH and SCP invocations as separate argument arrays. SSH uses
lowercase `-p <port>`; SCP uses uppercase `-P <port>`. Never reuse one
command's option array for the other, and never let a port value become a
positional local path. When the route has `password_path`, use
`sshpass -f <password_path>` and never read, print, copy, or place the password
in argv/environment; otherwise use its `key_path`. The installed helper owns
this authentication choice and is the recommended reference
implementation for routine upload/download because it reads the exact route
from the assigned `device_info.json` and owns both argument lists:

```bash
python -m zevo.engine.remote_transfer upload \
  --device-info <absolute-device-info-path> \
  --remote-dir <absolute-remote-directory> \
  --source <absolute-local-path>

python -m zevo.engine.remote_transfer download \
  --device-info <absolute-device-info-path> \
  --local-dir <absolute-local-directory> \
  --remote-path <absolute-remote-path>
```

Repeat `--source`/`--remote-path` for multiple entries and add `--recursive`
only for directories. Direct `scp` remains permitted when the task needs it,
but it must follow the same route and option rules. Always use the exact route
in `device_info.json`, not prose.

## Customization

`customization` has exactly these fields:

| Field | Meaning |
|---|---|
| `instructions` | User instruction for this Agent. |
| `input_paths` | Additional user-supplied paths assigned to this Agent. |
| `output_dir` | Explicit final-output directory; empty means `$WORK_DIR`. |
| `parameters` | Empty after strict values become `configuration_pins` and advisory values become `configuration_suggestions`. |
| `enforcement` | `strict` or `advisory`. |

Read every assigned `input_paths` entry. System wiring and data-isolation rules
remain binding. Under `strict`, fail if the request cannot be satisfied. Under
`advisory`, a necessary fallback is allowed only when the result and `Done:`
message record it. Advisory never permits changing baseline
`inference_config.yaml`, Task scoring, or the Run-owned generation backend after
they are established.

If `## CONVERSATION SO FAR` is injected, the newest user message is the current
instruction. State what changed in the next `Starting:` message.

## Artifacts and persistence

Every claimed local output must be durable, non-empty, and under the resolved output
directory. Remote execution must copy scripts, logs, predictions, metrics, and
other final files back before `succeeded` is reported.

Train model checkpoints are the single exception: the final model and any
bounded weights-only intermediate branch points may remain on the active remote
machine when `checkpoint_is_remote=true`. Registry copies back only the
Validation-selected model.

The backend stores each artifact as a WorkProduct with a semantic `role`.
Downstream `ArtifactBinding.artifact_role` must match that role. Never claim a
WorkProduct that was not created and verified.
When one source Ticket exposes multiple WorkProducts with the same role, omit
`work_product_id` to use its canonical product, or supply the exact id together
with `source_ticket_id` to select a specific verified product. Never guess a
path or id.

## Enforcement boundary

A hard check must never depend on a Specialist guessing a serialization or
hidden key. The work order supplies the exact expected value when it is
engine-owned, or an exact machine schema plus a side-effect-free validator when
the Specialist must realize it. Copy supplied identities byte-for-byte and use
the supplied validator; prose examples are never a key, enum, hash, role, or
lineage authority. If the engine cannot independently establish a value and it
does not change execution, surface disagreement as a notice instead of failing
an otherwise valid result.

Hard checks protect identity, executable semantics, lineage, isolation, and
measured results. A usable result fails when any of these drift:

- result `ticket_id`, status/error consistency, typed payload keys, operation,
  lane, or required binding role;
- lane-owned metric/direction, scoring files, evaluator, answer fields, output shape,
  or held-out isolation;
- exact model-parent chain and same-iteration Data/Train/Inference/Evaluation
  provenance;
- strict user pins and the Run-owned generation backend;
- the baseline `inference_config.yaml` body and SHA, including prompt framing,
  model reasoning type, system prompt, chat-template identity, exact template kwargs, tokenizer/special
  tokens, inference mapping, decoding strategy, decoding values, and seed;
- Train's copied prompt/template kwargs/tokenizer identity and reference to that exact
  baseline YAML;
- required executable artifacts: Data's prepared datasets/questions/profile and
  submission contract,
  Train's config/checkpoint, Inference's config/predictions, Evaluation's
  metrics, Infrastructure's device contract, and Registry's committed entry
  plus retained model.

Soft checks describe audit quality without changing what ran. Missing
suggestion-decision commentary, an inaccurate method-diversity label, missing
rationale for regenerating an otherwise equivalent helper, helper scripts/logs
that were not registered, or a profile format whose row shape the engine cannot
independently inspect produces a warning. It never authorizes a change to a
hard contract. When a soft field disagrees with the realized YAML/artifact, the
realized artifact is authoritative and the drift remains visible in notices.

Every result must echo the full Ticket id. `succeeded`/`degraded` require an
empty `error_message`; `failed` requires a specific non-empty one.

## Messages, notices, results, and events

These are separate records:

- `messages`: human/Agent conversation;
- `notices`: system diagnostics and policy warnings;
- `results`: one structured Agent result per heartbeat;
- `work_products`: durable artifacts;
- `execution_events`: phase/progress telemetry;
- `resolved_configs`: the effective runtime configuration per heartbeat.

POST conversation to `/api/tickets/$TICKET_ID/messages`:

```bash
post_message() {
  python3 -c '
import json, sys
print(json.dumps({"author": sys.argv[1], "body": sys.argv[2]}))
' "$AGENT_ID" "$1" | curl -fsS -X POST \
    "$ZEVO_API_BASE/api/tickets/$TICKET_ID/messages" \
    -H "Content-Type: application/json" \
    ${ZEVO_API_AUTH_HEADER:+-H "$ZEVO_API_AUTH_HEADER"} --data-binary @-
}

`ZEVO_API_AUTH_HEADER` is set by the engine on deployments whose API needs a
service token; it is empty on an open install. Send it on every API call.
```

The `Done:` message names where work ran, the exact artifact paths read, and
the exact artifact paths written. For finite cluster Train/Inference stages it
is engine-authored after Result/artifact validation as described above. Never
place secrets in messages or results.

Emit telemetry as standalone stdout lines:

```text
__PHASE__:<ticket_id>:<phase_name>@<unix-time>
__PROGRESS__:<ticket_id>:{"step":50,"total":500,"loss":1.83,"t":<unix-time>}
```

The Ticket id is mandatory. Train and Inference list their allowed phase names
in their platform contract; that worker-specific list specializes this common
wire format. Only emit measured values and Agent-defined phases. The runner
persists them as `execution_events`; they are not the final result.

## Final result

The last stdout block is exactly one JSON object matching the injected schema.
Common result status is `succeeded`, `degraded`, or `failed`; a narrower Agent
schema may exclude `degraded`. `error_message` is empty for usable results and
specific for failures. `memory_updates` is `[]` when no reusable verified
lesson was learned. It must contain a concise update when the Agent recovered
from an error through reflection and the verified correction is likely to help
one of its later Tickets in this Run. Do not print prose, a second JSON object,
or another message after the result.
