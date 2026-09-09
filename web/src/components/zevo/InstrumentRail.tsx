// ZEVO left instrument rail — vertical navigation, icon + label. The label is
// always visible rather than a hover tooltip: a dock you have to point at to
// read is only navigable once you have memorised the icons.
import { NavLink } from "react-router-dom";
import {
  LayoutGrid, GitBranch, ListChecks, Users, Folder, Box, KeyRound,
} from "lucide-react";

// Labels match each page's own title, so the rail and the heading you land on
// say the same word.
//
// No Tickets entry: the list it showed is a subset of what an agent's own
// "Tickets & heartbeats" tab already shows, minus the transcripts. Individual
// tickets are still reachable at /tickets/<id> from a run or an agent.
//
// No Leaderboard or Hardware entry either. Both pages still exist and open from
// the command palette; the rail is for the pages a run passes through, and
// hardware setup is reached from Settings when a run needs it. How to run is
// explained where the choice is made: the "i" on each mode card in the launch
// dialog.
const links = [
  { to: "/", label: "Dashboard", icon: LayoutGrid, end: true },
  { to: "/runs", label: "Runs", icon: GitBranch },
  { to: "/agents", label: "Agents", icon: Users },
  { to: "/tasks", label: "Tasks", icon: ListChecks },
  { to: "/files", label: "Files", icon: Folder },
  { to: "/models", label: "Models", icon: Box },
  { to: "/settings", label: "Settings", icon: KeyRound },
];

export function InstrumentRail() {
  return (
    <nav className="flex w-44 shrink-0 flex-col gap-1 border-r border-hair bg-panel/40 px-2 py-4">
      {links.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          className={({ isActive }) =>
            [
              "relative flex h-11 items-center gap-3 rounded-xl px-3 transition",
              isActive
                ? "bg-brass-500/12 text-brass-300 shadow-[inset_0_0_0_1px_rgba(224,141,56,0.28)]"
                : "text-ink hover:bg-raised",
            ].join(" ")
          }
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <span className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-brass-400 shadow-glow-brass" />
              )}
              <Icon size={19} strokeWidth={1.8} className="shrink-0" />
              <span className="truncate text-sm">{label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}
