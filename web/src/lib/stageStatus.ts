// Instruction decisions and workload execution are separate state machines.
export function instructionBlocks(status: string): boolean {
  return ["queued", "delivered", "needs_input"].includes(status);
}

export function stageStatus(
  status: string, paused: boolean, schedulerState = "", needsInput = false, activePhase = "",
): string {
  if (["succeeded", "degraded", "failed", "skipped", "cancelled"].includes(status)) return status;
  // A remote job keeps running while the control plane reviews an instruction.
  if (status === "waiting_external") {
    const state = schedulerState.toUpperCase().split(/[ +]/)[0];
    if (["PENDING", "CONFIGURING"].includes(state)) return "queued";
    if (["RUNNING", "COMPLETING"].includes(state)) return "running";
  }
  if (needsInput) return "needs_input";
  if (paused) return "paused";
  if (status === "running" && activePhase === "repair") return "repairing";
  return status === "awaiting_input" ? "needs_input" : status;
}
