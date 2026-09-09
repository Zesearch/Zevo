// ZEVO signature element — the model-improvement loop drawn as a living orbit.
// Six stations (infra → data → train → inference → eval → registry) sit around a
// ring; the orbit segment feeding an ACTIVE station animates with a flowing dash,
// and each station is coloured by its aggregate state, pulsing while it runs.
// Pure SVG, responsive.

import type { ReactNode } from "react";

export type StationState = "idle" | "active" | "done" | "degraded" | "failed" | "cancelled";

export type Station = {
  key: string;
  label: string;
  short: string; // 3-letter instrument code
  state: StationState;
  count?: number; // Agent heartbeats observed in this stage
};

// Ticket ownership is canonical: agent_id equals the station key.
export const LOOP_STATIONS: Omit<Station, "state" | "count">[] = [
  { key: "infrastructure", label: "Infrastructure", short: "INF" },
  { key: "data", label: "Data", short: "DAT" },
  { key: "train", label: "Train", short: "TRN" },
  { key: "inference", label: "Inference", short: "IFR" },
  { key: "evaluation", label: "Evaluation", short: "EVL" },
  { key: "registry", label: "Registry", short: "REG" },
];

/** Does this ticket belong to that station? */
export function ticketAtStation(
  t: { agent_id?: string },
  st: { key: string },
): boolean {
  return (t.agent_id || "").toLowerCase() === st.key;
}

// Active is BRASS, done is PHOSPHOR. They used to share phosphor, so a station
// mid-flight looked identical to one that had finished.
const COL = {
  idle: "#3B4C5B",
  active: "#E08D38",
  done: "#4FD1B5",
  degraded: "#A98FD6",
  failed: "#F26F58",
  cancelled: "#66788A",
};
const ACTIVE_GLOW = "rgba(224,141,56,";

export function PipelineRing({
  stations,
  centerMain = "Orchestrator",
  centerCount,
  centerState = "idle",
  countNote,
  stationNotes = {},
}: {
  stations: Station[];
  /** What sits at the hub. The ring's own name moved to the panel heading —
   *  a title belongs above the thing it names, not inside it, and the space it
   *  vacated is where the agent that drives the loop belongs: every station on
   *  the rim is woken by it, so the hub is where it actually is. */
  centerMain?: string;
  /** Wakes of the supervisor so far — the same measure the rim carries. */
  centerCount?: number;
  /** Same aggregate states as a station: the hub is an agent and can degrade or fail. */
  centerState?: StationState;
  /** Rendered right after the FIRST station count, which is the number it
   *  explains. A caption above the panel had to describe the number in words
   *  ("the number beside a stage is…") because it was nowhere near one. */
  countNote?: ReactNode;
  /** Notes tied to a specific station label, such as Evaluation's execution
   *  model. Keeping the note on the station avoids implying it describes the
   *  whole loop. */
  stationNotes?: Partial<Record<string, ReactNode>>;
}) {
  const W = 500;
  const H = 560;
  const cx = W / 2;
  const cy = H / 2;
  // The hub sits a little above the ring's centre. It carries a caption BELOW
  // it, which the rim stations at the sides do not, so centring the disc left
  // the disc-plus-label pair hanging low in the middle of the orbit.
  const hubY = cy - 24;
  const rx = 180;
  const ry = 180;
  const n = stations.length;
  // The first station that actually has a count — annotating "Data" when Data
  // was never called would explain a number that is not on screen.
  const noteAt = countNote ? stations.findIndex((st) => (st.count ?? 0) > 0) : -1;
  // Station i sits at angle; start at top (-90°) and go clockwise.
  const angle = (i: number) => (-90 + (360 / n) * i) * (Math.PI / 180);
  const pt = (i: number) => ({ x: cx + rx * Math.cos(angle(i)), y: cy + ry * Math.sin(angle(i)) });

  // Orbit segment from station i to i+1 as an elliptical arc.
  const segPath = (i: number) => {
    const a = pt(i);
    const b = pt((i + 1) % n);
    return `M ${a.x} ${a.y} A ${rx} ${ry} 0 0 1 ${b.x} ${b.y}`;
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full select-none overflow-visible" role="img" aria-label="Zevo improvement pipeline">
      {/* The ring itself is always there — it is the shape of the loop, not a
          claim about progress. */}
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="none" stroke={COL.idle} strokeWidth={2} strokeOpacity={0.35} />

      {/* Colour rides on top of it, and only where the run has actually been.
          A segment lights when its SOURCE has finished and its destination has
          begun, and glows while that destination is still running; between two
          stations that have done nothing it stays the plain grey ring. */}
      {stations.map((s, i) => {
        const dest = stations[(i + 1) % n];
        const travelled = ["done", "degraded", "failed", "cancelled"].includes(s.state);
        const arrived = dest.state !== "idle";
        if (!travelled || !arrived) return null;
        const flowing = dest.state === "active";
        return (
          <path
            key={`seg-${i}`}
            d={segPath(i)}
            fill="none"
            stroke={flowing ? COL.active : COL[dest.state]}
            strokeWidth={flowing ? 3 : 2}
            strokeDasharray={flowing ? "3 6" : undefined}
            strokeOpacity={flowing ? 1 : 0.7}
            className={flowing ? "animate-dash-flow" : ""}
            style={flowing ? { filter: `drop-shadow(0 0 4px ${ACTIVE_GLOW}0.6))` } : undefined}
          />
        );
      })}

      {/* hub — the agent that drives the ring, drawn as a station because that
          is what it is: one more agent, woken like the rest. Same disc, same
          double stroke, same code-then-label-below stack.

          One deliberate difference, because it is NOT a stop on the loop: no
          01..06 index. It is not a step in the sequence — it is what runs the
          sequence, which is also why it sits in the middle rather than on the
          rim. Its lamp reads off the same states as every station — one
          colour vocabulary for the whole diagram, so a failed supervisor looks
          like a failed anything else. */}
      <circle cx={cx} cy={hubY} r={54} fill="#0E141B" stroke={COL[centerState]} strokeWidth={2.5}
        style={centerState === "active" ? { filter: `drop-shadow(0 0 8px ${ACTIVE_GLOW}0.7))` } : undefined}>
        {centerState === "active" && (
          <animate attributeName="stroke-opacity" values="1;0.25;1" dur="1.8s" repeatCount="indefinite" />
        )}
      </circle>
      <circle cx={cx} cy={hubY} r={54} fill="none" stroke="#ffffff" strokeOpacity={0.05} strokeWidth={1} />
      <text x={cx} y={hubY + 8} textAnchor="middle" fill={COL[centerState]} fontSize={21} fontWeight={600} fontFamily="Spline Sans Mono">
        ORCH
      </text>
      {/* Name and count below the disc, exactly as the rim stations carry
          theirs — same font, size and colour, so the hub is read the same way.
          Set clear of the disc: at +78 the label sat 24px off a 54px circle and
          read as part of it rather than as its caption. */}
      <text x={cx} y={hubY + 90} textAnchor="middle" fill={centerState === "idle" ? "#66788A" : "#D3DDE5"} fontSize={20} fontWeight={500} fontFamily="Instrument Sans">
        {centerMain}
        {centerCount ? `  ·  ${centerCount}` : ""}
      </text>

      {/* stations */}
      {stations.map((s, i) => {
        const p = pt(i);
        const col = COL[s.state];
        const live = s.state === "active";
        const below = p.y > cy + 4;
        // Labels sit outside their station, but on the four side stations the
        // ring curves back under them and the long ones ("Evaluation · 5")
        // ran into the arc. Nudge those outward, along the radius. Top and
        // bottom stations have no sideways room to gain, so they stay put.
        const cos = Math.cos(angle(i));
        const labelDx = Math.abs(cos) < 0.2 ? 0 : cos * 26;
        const stationNote = stationNotes[s.key];
        return (
          <g key={s.key}>
            {/* station disc — running is the disc itself pulsing. It used to be
                a 6px lamp pinned to the disc's shoulder, which is a second
                thing to notice for a state the disc could carry on its own. */}
            <circle cx={p.x} cy={p.y} r={42} fill="#0E141B" stroke={col} strokeWidth={2.5}
              style={live ? { filter: `drop-shadow(0 0 8px ${ACTIVE_GLOW}0.7))` } : undefined}>
              {live && (
                <animate attributeName="stroke-opacity" values="1;0.25;1" dur="1.8s" repeatCount="indefinite" />
              )}
            </circle>
            <circle cx={p.x} cy={p.y} r={42} fill="none" stroke="#ffffff" strokeOpacity={0.05} strokeWidth={1} />
            {/* station code */}
            <text x={p.x} y={p.y - 2} textAnchor="middle" fill={col} fontSize={21} fontWeight={600} fontFamily="Spline Sans Mono">
              {s.short}
            </text>
            {/* station index */}
            <text x={p.x} y={p.y + 18} textAnchor="middle" fill="#8B9CAB" fontSize={13} fontFamily="Spline Sans Mono">
              {String(i + 1).padStart(2, "0")}
            </text>
            {/* label */}
            {noteAt === i || stationNote ? (
              // foreignObject rather than <text>: this one label carries the
              // note marker, and that is an HTML control with a tooltip.
              <foreignObject x={p.x + labelDx - 130} y={(below ? p.y + 60 : p.y - 84)} width={260} height={26}>
                <div className="flex h-full items-center justify-center gap-1.5 whitespace-nowrap font-sans text-[20px] font-medium leading-none text-[#D3DDE5]">
                  {stationNote}
                  <span>{s.label}{s.count ? `  ·  ${s.count}` : ""}</span>
                  {noteAt === i && countNote}
                </div>
              </foreignObject>
            ) : (
              <text
                x={p.x + labelDx}
                y={below ? p.y + 78 : p.y - 66}
                textAnchor="middle"
                fill={s.state === "idle" ? "#66788A" : "#D3DDE5"}
                fontSize={20}
                fontWeight={500}
                fontFamily="Instrument Sans"
              >
                {s.label}
                {s.count ? `  ·  ${s.count}` : ""}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
