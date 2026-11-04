type Artifact = { ticket_id: string; role: string };
type Ticket = { id: string; agent_id: string; lane: string; status: string };

export function completedTrainingData<T extends Artifact>(artifacts: T[], tickets: Map<string, Ticket>): T[] {
  return artifacts.filter((artifact) => {
    const ticket = tickets.get(artifact.ticket_id);
    return artifact.role === "training_dataset" && ticket?.agent_id === "data"
      && ticket.lane === "optimization" && ticket.status === "succeeded";
  });
}

/** Reused, completed Data is evidence for this round even without a new artifact. */
export function hasIterationEvidence(artifacts: Artifact[], activeData: Artifact[]): boolean {
  return activeData.some((artifact) => artifact.role === "training_dataset")
    || artifacts.some((artifact) => artifact.role === "checkpoint" || artifact.role === "predictions");
}
