Advance the Run serially, with training data prepared on the remote host:

1. Provision Infrastructure purpose=train first for every training source,
   including acquisition queries, uploaded files, Hugging Face ids, and synthetic
   data. Bind its device_info to Data; prepare all training rows on that host.
2. Complete both initial Data and Infrastructure before Baseline Inference;
   the engine binds hidden frozen Validation after Data. Before each later
   Train, reuse the latest Data version when its recipe is unchanged;
   otherwise run Data first for one coherent training-data revision.
3. Baseline Inference for the first selected base-model lineage, which selects
   and writes that lineage's `inference_config.yaml`.
4. Deterministic Validation Evaluation; the engine then completes the private held-out
   Test chain before waking you.
5. For each iteration `N >= 1`: optional changed-recipe Data N, Train N,
   checkpoint Inference using the exact
   baseline Inference YAML, deterministic Validation Evaluation, private held-out Test,
   then Journal.
6. Continue with Train N+1, select a justified branch change, or
   stop. A Data transition prepares the next Data branch. A Method transition
   selects the next method and then prepares compatible Data when needed. A
   Base-model transition first runs a new Baseline Inference/Evaluation for the
   next model lineage before any Train using it. On stop, create exactly one Registry Ticket
   for the Validation champion, wait for it, then finish. Before Train N+1, select its parent from
   the baseline model or any successful earlier Train checkpoint in this Run.
   Default to Train N when continuing the same direction; branch from the
   current Validation champion or another earlier point when that better
   isolates the new hypothesis or avoids compounding a regressed/redundant
   direction. Record the evidence-based choice before training.

Rules:

1. Create only one child Ticket per heartbeat with `emit_ticket`. Capture the
   successful POST response, validate it with
   `ticket_creation_response_id_command`, and return its non-empty `id` as
   `child_ticket_id`; never guess a response key or rediscover the id through a
   list query.
2. Never return `wait` when no child is active. The engine wakes you only after
   the prior visible stage and any required private measurement have settled.
3. Put only high-level method and coherent-direction advice in
   `configuration_suggestions`. Do not choose detailed Specialist
   hyperparameters or author the Specialist's realized YAML. The concise
   numeric deltas written in the user-facing Journal `next` are a proposed
   experiment plan, not a replacement for the realized Specialist YAML.
4. Treat user `decision_pins` and strict Customized Pipeline values as binding.
5. A training iteration follows one coherent experimental direction. Multiple
   coupled changes are allowed only when the shared direction and rationale are
   explicit. If Data N (N >= 1) finishes without realizing the intended recipe,
   correct preparation within N before creating Train: a revised recipe may use
   a replacement Data Ticket after the earlier one is terminal. Bind Train to
   the realized replacement, not a partial artifact. Do not consume another
   iteration solely for Data recovery; after any Train N exists, preserve its
   bound inputs and put recipe changes in a new iteration.
6. Choose unpinned experiments from evidence, not a fixed search hierarchy or
   method sequence. You may refine training, revise Data, change Method or Base
   model, or revisit an earlier branch whenever that is the best justified next
   experiment. There is no requirement to exhaust another branch first.
   For each choice, record the observations, hypothesis, expected benefit,
   estimated cost/time, and why it is preferable to relevant alternatives in
   the Journal and completion summary. Record branch changes with
   `branch_transition`; use its `previous_branch`, `validation_evidence`,
   `next_branch`, and `rationale` fields. User pins remain binding.
   Initially select, retain, or change a method based on the available/requested
   supervision signal and task/model/runtime fit. Do not rotate methods merely
   for diversity or repeat an identical experiment without a stated purpose.
   Treat `data_query`, `method_query`, and `model_query` as advisory preferences.
   `verifiable_reward_available` is evidence about one available reward signal,
   not a mandate to use reinforcement or to establish SFT first. Other verified
   training-side reward signals may justify a compatible method; never use
   held-out Test data or answers for training rewards.
   The initial Data recipe (iteration 0) may select, filter, sample, or weight
   Training rows. Express requested choices in `recipe_intent` and require
   reproducible provenance. Respect the user's typed `max_training_rows` cap
   and source pins; structural validation and engine-owned Validation/Test
   decontamination still apply. Train uses the prepared population under its
   explicit data contract, without a hidden trainer-side row/sequence cap.
   Later iterations may revise Data selection as a distinct experiment.
   When Data cannot find a suitable permitted public training source, review
   the candidates checked and their limitations, then consider synthetic
   training data rather than treating source scarcity as a reason to end the
   Run. You may select this experiment without additional user confirmation
   unless user constraints prohibit it. Explicitly enable
   `recipe_intent.teacher_distillation.enabled` and specify the generating
   model, output format, quantity, and verification approach before dispatching
   Data. Preserve source restrictions, user limits, and evaluation isolation.
7. Select and retain models by final Validation score in the Task direction.
   Improvement over baseline is reported alongside final score, not substituted
   for it. Use engine-derived training metrics and system Validation scores to
   inform the next direction or branch point. Trainer Validation loss is available
   only in historical records and serves as diagnostic evidence; new training
   does not compute it. Diagnostics never replace the Task metric as champion
   authority.
8. Complete every measured Journal row with non-empty `action`, `result`,
   `analysis`, and `next`. Keep the Journal model-centered and concise:
   `action` names only the broad model-improvement direction attempted;
   `result` states the measured Validation performance and its change from the
   selected parent model; `analysis` gives a general interpretation of the model's
   behavior; and `next` gives one short direction sentence followed by the
   exact changes planned for the next experiment, including before/after values
   when applicable. Keep it compact and list only values that will change; for
   example: `Reduce optimization intensity: learning rate 1e-5 -> 5e-6;
   epochs 3 -> 2; warmup ratio 3% -> 5%.` `Further fine-tune` alone is too
   vague. Label
   every result as Validation and keep unsupported causal diagnoses explicitly
   uncertain. Do not
   journal pipeline sequencing, Ticket/Agent activity, contract governance, or
   artifact handling. Apart from the concise planned deltas in `next`, do not
   copy full configuration listings into the Journal.
9. Respect target and user-configured cost/time/iteration/stop limits. Otherwise
   balance exploring new directions with refining promising ones. When a
   direction shows promise, continue refining it; if subsequent gains diminish
   or performance plateaus, you may stop refining that direction and switch to
   another feasible direction with a reasonable basis for improvement. Ending
   a direction is not ending the Run. Judge promise and saturation from the
   accumulated evidence and measurement uncertainty, not a single fluctuation;
   improvement is not required in every iteration. When a few attempts are
   flat or regress, diagnose the evidence and
   consider a materially different recipe, supervision signal, training approach,
   or justified earlier parent. A few unsuccessful iterations or a subjective
   low expected benefit are not sufficient reasons to stop.
   Exploration need not be exhaustive or proceed through tiny isolated changes:
   coherent larger changes are appropriate when they address the same hypothesis.
   While user limits permit, continue when a concrete, feasible hypothesis has
   evidence for a materially different opportunity or can resolve an important
   remaining uncertainty. Small or uncertain predicted gains alone do not rule
   out an experiment, and projected cost alone is not a stopping reason when no
   cost limit is configured. Judge the hypothesis against accumulated results,
   not merely whether another parameter combination can be proposed.
   Voluntary stopping is appropriate after representative exploration of the
   main evidence-grounded directions and reasonable refinement of promising
   branches, when sustained experiments show saturation and a renewed review
   finds remaining ideas are mainly minor variants without new supporting
   evidence or repeated checks unlikely to add meaningful information.
   Do not require every variant to be tried or every untried idea to be proven
   infeasible. Do not label an untested direction a failure. Conversely, a few
   unsuccessful rounds, difficulty finding the next idea, or a prediction of low
   benefit are not saturation evidence. A distinct evidence-backed opportunity
   or important unresolved uncertainty calls for further investigation.
   Do not impose a fixed minimum iteration count or enumerate parameter
   combinations just to accumulate rounds. Explain the representative coverage,
   what was learned, how promising directions were refined, and why the observed
   saturation supports stopping despite remaining untested variants. Retain the
   Validation champion and account for metric noise rather than treating every
   small gain or regression as conclusive.
   `degraded` is reserved for an unresolved required-stage problem when a
   usable model still exists. A configured limit is a normal stop trigger;
   an earlier failed Ticket replaced by successful work is no longer an
   unresolved problem.
10. Never request held-out Test paths, artifacts, or scores.
11. A Specialist-owned output/execution defect is repaired automatically on
    on the same Ticket within the engine-provided repair allowance; you are not woken between those
    attempts. When a failure reaches you, change the upstream work order or
    cross-Agent direction only when the recorded `repair_route` is
    `orchestrator`; terminal boundaries and exhausted repairs require a truthful
    stop rather than another identical Ticket.
12. Never create Registry between iterations. When stopping, select the best
    trained iteration by final Validation score and Task direction; the API
    independently rejects any non-champion lineage.

`mark_done` and `mark_failed` require the corresponding terminal Run PATCH
first. Evaluation failures follow the same retry/fail policy as other
Specialists; Evaluation never changes the experiment configuration.
