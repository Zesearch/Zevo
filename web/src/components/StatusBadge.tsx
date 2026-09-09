// ZEVO status "lamp chip" — a glowing indicator dot + tracked-mono label.
// Used everywhere (runs, tickets, agents) so status reads like a console light.

export type StatusTone = { dot: string; text: string; ring: string; live?: boolean };

export function statusLabelFor(status: string): string {
  return status === "awaiting_input" || status === "waiting_external"
    ? "waiting"
    : status;
}

export function waitingDetailFor(status: string): string {
  if (status === "awaiting_input") return "User input";
  if (status === "waiting_external") return "Slurm job";
  return "";
}

export function statusToneFor(status: string): StatusTone {
  switch (status) {
    case "succeeded":
    case "success":
      return { dot: "bg-phosphor-400 shadow-glow-phosphor", text: "text-phosphor-300", ring: "border-phosphor-500/30 bg-phosphor-500/10" };
    // Brass, not phosphor: work IN FLIGHT must not look like work that
    // FINISHED. Sharing phosphor with done/success left the pulse as the only
    // difference, which is invisible in a screenshot or at a glance.
    case "running":
      return { dot: "bg-brass-400 shadow-glow-brass", text: "text-brass-200", ring: "border-brass-500/30 bg-brass-500/10", live: true };
    case "repairing":
      return { dot: "bg-lilac-400 shadow-glow-brass", text: "text-lilac-200", ring: "border-lilac-500/40 bg-lilac-500/10", live: true };
    case "queued":
    case "planning":
      return { dot: "bg-skyx-400 shadow-glow-brass", text: "text-skyx-300", ring: "border-skyx-500/30 bg-skyx-500/10" };
    case "failed":
    case "halted":
      return { dot: "bg-coral-400 shadow-glow-coral", text: "text-coral-300", ring: "border-coral-500/30 bg-coral-500/10" };
    case "degraded":
      return { dot: "bg-lilac-400", text: "text-lilac-300", ring: "border-lilac-500/30 bg-lilac-500/10" };
    case "awaiting_input":
    case "waiting_external":
      return { dot: "bg-skyx-400 shadow-glow-brass", text: "text-skyx-200", ring: "border-skyx-400/50 bg-skyx-500/15", live: true };
    case "cancelled":
    case "skipped":
      return { dot: "bg-dim", text: "text-slate-400", ring: "border-slate-600/40 bg-slate-800/40" };
    default:
      return { dot: "bg-slate-600", text: "text-slate-300", ring: "border-slate-700/60 bg-slate-800/40" };
  }
}

export function StatusBadge({ status }: { status: string }) {
  const t = statusToneFor(status);
  const label = statusLabelFor(status);
  const detail = waitingDetailFor(status);
  return (
    <span
      title={detail || undefined}
      aria-label={detail ? `${label}: ${detail}` : label}
      className={[
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em]",
        t.ring,
        t.text,
      ].join(" ")}
    >
      <span className={["h-1.5 w-1.5 rounded-full", t.dot, t.live ? "animate-lamp-pulse" : ""].join(" ")} />
      {label}
    </span>
  );
}
