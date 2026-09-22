import type { HeartbeatDTO } from "./api";

/** A repair belongs to the preceding activation of the same ticket. */
export function heartbeatGroups(chronological: HeartbeatDTO[]): HeartbeatDTO[][] {
  const groups: HeartbeatDTO[][] = [];
  for (const heartbeat of chronological) {
    const previous = groups.at(-1);
    if (heartbeat.activation_phase === "repair" && previous?.[0].ticket_id === heartbeat.ticket_id) {
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
