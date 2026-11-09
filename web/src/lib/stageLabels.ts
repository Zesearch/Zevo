export function operationCaption(t: {
  agent_id?: string;
  lane?: string;
  iteration?: number;
  operation?: string;
  model_source?: string;
  test_set_name?: string;
  payload?: Record<string, unknown>;
}, heartbeatOperation = "", activationPhase = ""): string {
  const op = heartbeatOperation || t.operation || String(t.payload?.operation || "")
    || ({train: "train", inference: "run_inference"} as Record<string, string>)[t.agent_id || ""] || "";
  const modelSource = t.model_source || String(t.payload?.model_source || "");
  const testSetName = t.test_set_name || String(t.payload?.test_set_name || "");
  const phase = activationPhase.trim().toLowerCase();
  const inferenceName = t.lane === "held_out_test"
    ? "Held-Out Inference"
    : modelSource === "base_model"
      ? "Baseline Inference"
      : "Candidate Inference";
  const stageName = ({
    infrastructure: "Infrastructure",
    data: "Data",
    train: "Train",
    inference: "Inference",
    evaluation: "Evaluation",
    registry: "Registry",
    orchestrator: "Orchestrator",
  } as Record<string, string>)[t.agent_id || ""] || t.agent_id || "Stage";

  if (op === "run_inference") {
    if (phase === "remeasure") return "Length Remeasurement";
    if (phase === "remeasure_collect") return "Collect Length Remeasurement Results";
    if (phase === "submit") return `Submit ${inferenceName}`;
    if (phase === "collect") return `Collect ${inferenceName} Results`;
    if (phase === "continue") return `Continue ${inferenceName}`;
  }
  if (op === "train") {
    if (phase === "submit") return "Submit Candidate Training";
    if (phase === "collect") return "Collect Candidate Training Results";
    if (phase === "continue") return "Continue Candidate Training";
  }
  if (phase === "submit") return `Submit ${stageName} Job`;
  if (phase === "collect") return `Collect ${stageName} Results`;
  if (phase === "repair") return `${operationCaption(t, op)} · Repair`;
  if (phase === "instruction") return "Apply Instruction";
  if (phase === "continue") return `Continue ${stageName}`;

  const labels: Record<string, string> = {
    provision: "Provision compute",
    prepare_run_data: "Prepare Run data and Validation setup",
    prepare_holdout_data: `Prepare questions-only held-out data${
      testSetName ? ` · ${testSetName}` : ""
    }`,
    train: "Select config and train candidate",
    run_inference: t.lane === "held_out_test"
      ? "Run Held-Out Inference"
      : modelSource === "base_model"
        ? "Run Baseline Inference"
        : "Run Candidate Inference",
    release: "Release resources",
  };
  if (labels[op]) return labels[op];
  if (t.agent_id === "evaluation") {
    return t.lane === "held_out_test"
      ? `Run held-out evaluator via Bash${testSetName ? ` · ${testSetName}` : ""}`
      : "Run validation evaluator via Bash";
  }
  if (t.agent_id === "registry") return "Save and Register Model";
  if (t.agent_id === "orchestrator") return "Review Run";
  if (t.agent_id === "infrastructure") return "Provision Compute";
  if (t.agent_id === "data") return "Prepare Data";
  return op ? op.replaceAll("_", " ") : "Execute stage";
}

export function evaluationAttemptLabel(phase: string, sequence: number, lane: string): string {
  const suite = lane === "held_out_test" ? "Held-Out" : "Validation";
  if (phase === "collect") return `Collect ${suite} Results`;
  if (phase === "repair") return `Evaluate ${suite} · Repair`;
  return sequence > 1 ? `Re-evaluate ${suite}` : `Evaluate ${suite}`;
}

export function withRepairLabel(label: string, heartbeat: {activation_phase: string; execution_phase?: string}): string {
  const repair = heartbeat.activation_phase === "repair" || heartbeat.execution_phase === "repair";
  return repair && !label.endsWith(" · Repair") ? `${label} · Repair` : label;
}

export function stageAttemptLabel(
  ticket: Parameters<typeof operationCaption>[0],
  heartbeat: {operation: string; activation_phase: string; execution_phase?: string; training_finalization?: boolean},
  sequence = 1, activity = "",
): string {
  let phase = ["remeasure", "remeasure_collect", "instruction", "continue"].includes(heartbeat.activation_phase)
    ? heartbeat.activation_phase : heartbeat.execution_phase || heartbeat.activation_phase;
  if (phase === "repair") phase = "";
  const label = activity || (phase === "instruction" ? "Apply Instruction"
    : ticket.agent_id === "train" && heartbeat.training_finalization ? "Finalize Candidate Training"
    : ticket.agent_id === "evaluation"
      ? evaluationAttemptLabel(phase, heartbeat.activation_phase === "repair" ? 1 : sequence, ticket.lane || "")
    : ticket.agent_id === "orchestrator" ? sequence === 1 ? "Plan Run" : "Review Run"
    : operationCaption(ticket, heartbeat.operation, phase));
  return withRepairLabel(label, heartbeat);
}
