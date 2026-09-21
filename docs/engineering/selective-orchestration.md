# Suggestion: Use the Orchestrator Only at Decision Boundaries

## Summary

Zevo currently uses a deliberately serial workflow in which almost every
visible Specialist completion returns control to the Orchestrator. This gives
the Orchestrator strong control over sequencing, but it also introduces an LLM
activation and scheduler handoff between stages whose next action is already
fully determined.

The suggested design keeps the Orchestrator responsible for experiment-level
decisions while allowing the workflow engine to perform deterministic stage
handoffs. In the common trained-candidate path, the engine would run Train,
Inference, Validation Evaluation, and the private held-out measurement as one
controlled chain. The Orchestrator would be awakened after the measurements
settle, when a new decision is actually required.

This is a medium-sized change if it is limited to deterministic handoffs. A
fully decentralized design in which Specialists invoke one another directly
would be a much larger change and would duplicate control-plane responsibilities
across Agents.

## Current Workflow

The normal trained-candidate path currently behaves approximately as follows:

```text
Orchestrator decides the experiment
  -> Train
  -> Orchestrator
  -> Inference
  -> Orchestrator
  -> Validation Evaluation
  -> engine-owned held-out measurement
  -> Orchestrator evaluates the result and decides what to do next
```

The Orchestrator activation after Train usually creates an Inference Ticket for
the checkpoint that Train just produced. The activation after Inference usually
creates an Evaluation Ticket for the predictions that Inference just produced.
Both handoffs are constrained by typed contracts and normally contain no new
experiment-level decision.

Zevo already has a partial version of the proposed pattern: validation-suite
mirrors and the private held-out Test chain are created by the engine without an
Orchestrator decision for every intermediate step.

## Proposed Workflow

```text
Orchestrator decides the experiment
  -> Train
  -> engine starts checkpoint Inference
  -> engine starts Validation Evaluation
  -> engine completes the private held-out measurement
  -> Orchestrator reviews the completed round and decides what to do next
```

Any event that requires interpretation or changes the experiment would still
return control to the Orchestrator immediately.

### Workflow Engine Responsibilities

The workflow engine should own transitions whose inputs and behavior are fully
determined by committed artifacts and Run-owned configuration:

- Successful Train to checkpoint Inference.
- Successful Inference to primary Validation Evaluation.
- Validation-suite evaluation fan-out and settlement.
- Private held-out Test inference and evaluation.
- Existing bounded self-repair for Specialist-owned generated failures.
- Exactly-once Ticket creation, artifact binding, wakeup recovery, and stage
  admission checks.

These transitions should remain separate Tickets so provenance, Timeline
visibility, cancellation, retry behavior, and cost accounting remain intact.
The backend workflow engine should create the next Ticket; Specialists should
continue to operate only within their own work order.

### Orchestrator Responsibilities

The Orchestrator should remain responsible for decisions that require task
context, evidence, or tradeoffs:

- Selecting or changing the base model, training method, or training data.
- Choosing the next experiment after Validation results settle.
- Deciding whether an inner search or a model, method, or data branch is
  exhausted.
- Interpreting and routing user instructions.
- Handling failures that require a different upstream binding, resource plan,
  or experiment direction.
- Applying budget and runtime policy before starting another complete round.
- Selecting the Validation champion, creating the Registry action, writing the
  Run Journal, and ending the Run.

## Safe Handoff Conditions

The engine should start the next deterministic stage only when all of the
following conditions hold:

1. The source Ticket finished with an accepted terminal result.
2. Its typed result and required WorkProducts passed validation.
3. The Run is still active and has not been cancelled.
4. No unresolved user instruction is holding the optimization lane.
5. No equivalent semantic Ticket already exists for the same iteration and
   lineage.
6. The next Ticket's artifact bindings can be resolved exactly.
7. Failure policy has not classified the result as requiring Orchestrator
   intervention.

If any condition fails, the engine should hold the next stage and wake the
Orchestrator or follow the existing terminal policy.

The user-instruction gate is especially important. A user may submit an
instruction after Train completes but before Inference starts. The automatic
handoff must check the gate before releasing every new Specialist activation,
so the Orchestrator can decide whether the instruction applies immediately or
to a later iteration.

## Expected Time Savings

A standard trained iteration can remove two Orchestrator activations:

1. The activation between Train and Inference.
2. The activation between Inference and Evaluation.

The approximate saving per trained iteration is therefore:

```text
2 * (Orchestrator execution latency + Orchestrator queue latency)
```

For example, if one Orchestrator activation takes one to three minutes, the
common path would save roughly two to six minutes per trained iteration. Queue
contention can make the saving larger.

The percentage reduction will be small for a training job that runs for several
hours, but the absolute delay and cost still matter when a cloud GPU instance
remains leased between stages. The improvement will be more visible for smoke
tests, short fine-tuning jobs, small evaluations, and environments where many
Runs contend for a serialized Orchestrator lane.

This change does not remove Slurm queue time or the compute time of Train,
Inference, or Evaluation. Combining those stages into one Slurm allocation
would be a separate resource-execution change.

## Change Scope

### Targeted Deterministic Handoffs

The targeted version is a medium-sized backend change. It primarily affects:

- Ticket-completion routing in the runner.
- Deterministic builders for candidate Inference and primary Evaluation
  Tickets.
- Artifact binding and lineage checks.
- Wakeup reconciliation and lost-handoff recovery.
- User-instruction admission checks between automatic stages.
- Sequential workflow, failure, cancellation, validation isolation, and
  recovery tests.

The current Ticket, WorkProduct, Timeline, and stage-status models can remain in
place. A database migration and substantial frontend redesign should not be
necessary.

### Fully Decentralized Specialist Control

Allowing Specialists to invoke one another directly would require a broader
redesign of control ownership. Budget decisions, instruction routing, failure
escalation, duplicate prevention, recovery, and lineage validation would need
to be distributed across multiple Agents. That approach has a much larger
implementation and reliability cost.

## Suggested Delivery Sequence

### Phase 1: Candidate Measurement Chain

- Automatically create checkpoint Inference after a successful Train Ticket.
- Automatically create primary Validation Evaluation after successful
  Inference.
- Wake the Orchestrator after Validation and held-out settlement.
- Preserve the current behavior for user instructions, failures, cancellation,
  and retries.

This phase removes the two most mechanical Orchestrator activations without
changing experiment planning.

### Phase 2: Baseline Measurement Chain

- Automatically create baseline Validation Evaluation after baseline
  Inference.
- Keep base-model selection and initial Inference configuration selection under
  Orchestrator control.

### Phase 3: Optional Planned Subflows

- Consider allowing one Orchestrator decision to describe an Infrastructure and
  Data subflow when the resource and data direction are already explicit.
- Keep an Orchestrator checkpoint wherever Specialist output could change the
  model, method, or data decision.

Each phase should be evaluated before expanding the automatic chain.

## Success Metrics

The change should be evaluated using production measurements rather than only
unit-test completion:

- Median and p95 time from Train completion to Inference start.
- Median and p95 time from Inference completion to Evaluation start.
- Orchestrator activations per completed trained iteration.
- End-to-end iteration duration, separated into execution, queue, and control
  time.
- Cloud lease minutes spent without an active GPU stage.
- Duplicate Ticket, lost-handoff, and stuck-Run rates.
- Frequency of user instructions arriving during an automatic handoff.
- Percentage of automatic chains escalated to the Orchestrator because a safe
  handoff condition failed.

## Recommendation

Implement selective orchestration around decision boundaries. Start with the
Train to Inference to Evaluation chain, because its contracts and artifact
lineage already make the next action deterministic. Continue to use the
Orchestrator for experiment selection, evidence-based iteration decisions,
user instructions, budget policy, exceptional recovery, and finalization.

This design removes avoidable control-plane latency while preserving the
existing safety, provenance, and recovery model.
