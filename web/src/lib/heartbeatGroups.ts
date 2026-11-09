import type { HeartbeatDTO } from "./api";

/** A repair belongs to the preceding activation of the same ticket. */
export function heartbeatGroups(chronological: HeartbeatDTO[]): HeartbeatDTO[][] {
  const groups: HeartbeatDTO[][] = [];
  for (const heartbeat of chronological) {
    const previous = groups.at(-1);
    if (heartbeat.activation_phase === "repair" && previous?.[0].ticket_id === heartbeat.ticket_id
        && (!heartbeat.execution_phase || !previous.at(-1)?.execution_phase
          || heartbeat.execution_phase === previous.at(-1)?.execution_phase)
        && previous[0].mid_training?.request_id === heartbeat.mid_training?.request_id) {
      previous.push(heartbeat);
    } else {
      groups.push([heartbeat]);
    }
  }
  return groups;
}

export function settledHeartbeat(group: HeartbeatDTO[]): HeartbeatDTO {
  const first = group[0];
  const last = group[group.length - 1];
  return { ...last, id: first.id, started_at: first.started_at,
    action: last.action || first.action,
    child_ticket_id: last.child_ticket_id || first.child_ticket_id };
}

/** Timeline rows may omit instruction wakes; selection must use identity. */
export function heartbeatGroupFor(chronological: HeartbeatDTO[], heartbeatId?: string): HeartbeatDTO[] | undefined {
  const id = heartbeatId ?? chronological.at(-1)?.id;
  return heartbeatGroups(chronological).find((group) => group.some((heartbeat) => heartbeat.id === id));
}

/** A historical supervisor action has its own status, independent of later wakes. */
export function heartbeatStatus(heartbeat: HeartbeatDTO): string {
  if (heartbeat.is_live) return heartbeat.activation_phase === "repair" ? "repairing" : "running";
  return heartbeat.exit_code !== 0 || heartbeat.action === "mark_failed" ? "failed" : "succeeded";
}
