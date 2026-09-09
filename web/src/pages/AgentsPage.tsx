import { useState } from "react";
import useSWR from "swr";
import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import type { AgentDTO } from "../lib/api";
import { PageHead, Kicker } from "../components/zevo/primitives";

// Show execution roles in pipeline order (orchestrator first, then the worker stages in
// the order they run), not whatever order the API returns.
const AGENT_ORDER = [
  "orchestrator", "infrastructure", "data", "train",
  "inference", "evaluation", "registry",
];
const orderIndex = (id: string) => {
  const i = AGENT_ORDER.indexOf(id);
  return i === -1 ? AGENT_ORDER.length : i;
};

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** A small brass/mono chip carrying a kicker label + a mono value. */
function DialChip({ label, value, missing }: { label: string; value: string; missing?: boolean }) {
  return (
    <div className="bezel-flat px-3 py-2">
      <div className="kicker text-[0.62rem]">{label}</div>
      <div className={`mt-1 truncate font-mono text-2xs ${missing ? "text-coral-300" : "text-slate-200"}`} title={value}>
        {value}
      </div>
    </div>
  );
}

function StationCard({ a, featured = false }: { a: AgentDTO; featured?: boolean }) {
  const isSystemRunner = a.id === "evaluation";
  const displayName = isSystemRunner ? "Evaluation Agent" : cap(a.name || a.id);
  const configured = !!(a.default_driver && a.default_model);
  // One rule for every agent: green once it has a driver and a model, red until
  // then. The orchestrator used to get a third colour, which made the legend
  // describe a state only one card could ever be in.
  const lampCls = configured ? "lamp-live" : "lamp-coral";

  // The pool is collapsed to a few chips so the cards stay the same height;
  // the rest are one click away rather than a "+7" you cannot open.
  const shown = featured ? 8 : 4;
  const [openSkills, setOpenSkills] = useState(false);
  const skills = openSkills ? a.skills : a.skills.slice(0, shown);
  const hidden = a.skills.length - skills.length;

  const content = (
    <>
      {/* identity: status lamp, then the name itself */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span className={`lamp ${lampCls} shrink-0`} />
          <h2 className={`truncate font-display font-semibold leading-none tracking-tight text-ink ${featured ? "text-3xl" : "text-2xl"}`}>
            {displayName}
          </h2>
        </div>
        {!configured && !isSystemRunner ? (
          <span className="mt-0.5 shrink-0 font-mono text-[0.62rem] uppercase tracking-[0.14em] text-coral-300">unconfigured</span>
        ) : !isSystemRunner ? (
          <ArrowUpRight size={15} className="mt-1 shrink-0 text-slate-600 transition group-hover:text-brass-300" />
        ) : null}
      </div>

      <div className="mt-2 min-w-0 font-mono text-2xs">
        {a.title && <span className="truncate text-ink">{a.title}</span>}
      </div>

      {isSystemRunner ? (
        <div className="mt-4 grid grid-cols-2 gap-2">
          <DialChip label="Driver" value="Deterministic runner" />
          <DialChip label="Model" value="No model call" />
        </div>
      ) : (
        <div className="mt-4 grid grid-cols-3 gap-2">
          <DialChip label="Driver" value={a.default_driver || "not set"} missing={!a.default_driver} />
          <DialChip label="Model" value={a.default_model || "not set"} missing={!a.default_model} />
          <DialChip label="Skills" value={String(a.skills.length)} />
        </div>
      )}

      {/* skills as bezel-flat chips */}
      {!isSystemRunner && a.skills.length > 0 && (
        <div className="mt-4">
          <Kicker className="text-[0.62rem]">Skills pool</Kicker>
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {skills.map((s) => (
              <span key={s} className="bezel-flat px-2 py-0.5 font-mono text-[0.72rem] text-slate-300">
                {s}
              </span>
            ))}
            {(hidden > 0 || openSkills) && (
              <button
                type="button"
                // inside a Link — keep the click from navigating to the agent
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setOpenSkills((v) => !v); }}
                className="px-1 py-0.5 font-mono text-[0.72rem] text-slate-500 underline decoration-dotted underline-offset-2 hover:text-brass-300"
              >
                {openSkills ? "show less" : `+${hidden} more`}
              </button>
            )}
          </div>
        </div>
      )}
    </>
  );

  if (isSystemRunner) {
    return <div className="animate-zevo-in block bezel p-5">{content}</div>;
  }
  return (
    <Link
      to={`/agents/${a.id}`}
      className="group animate-zevo-in block bezel p-5 transition hover:shadow-glow-brass"
    >
      {content}
    </Link>
  );
}

export function AgentsPage() {
  // Poll so runtime changes (e.g. `agent set …` in the Zevo shell) show up
  // without a manual reload — keeps this list in sync with the backend.
  const { data: agentsRaw = [] } = useSWR<AgentDTO[]>("/api/agents", {
    refreshInterval: 3000,
  });
  const agents = agentsRaw
    .sort((a, b) => orderIndex(a.id) - orderIndex(b.id));
  const director = agents.find((a) => a.id === "orchestrator");
  const stations = agents.filter((a) => a.id !== "orchestrator");

  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Agents"
        subtitle="Agent roles and their settings in Zevo"
        right={
          <div className="flex items-center gap-3 text-sm text-slate-400">
            <span className="flex items-center gap-2"><span className="lamp lamp-live" /> configured</span>
            <span className="flex items-center gap-2"><span className="lamp lamp-coral" /> unset</span>
          </div>
        }
      />

      {/* Featured flight director */}
      {director && (
        <div className="mb-5">
          <StationCard a={director} featured />
        </div>
      )}

      <div className="stagger grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {stations.map((a) => (
          <StationCard key={a.id} a={a} />
        ))}
      </div>

      {agents.length === 0 && (
        <div className="mt-6 text-sm text-slate-500">No agents registered yet.</div>
      )}
    </div>
  );
}
