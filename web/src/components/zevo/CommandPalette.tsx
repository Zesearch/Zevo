// ZEVO ⌘K command palette — the primary navigator. Fuzzy-jump to any page,
// run, or agent; fire quick actions. Opens on ⌘K / Ctrl-K or the console-bar
// trigger; keyboard-driven (↑/↓/↵/esc).
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import useSWR from "swr";
import {
  LayoutGrid, GitBranch, ListChecks, Trophy, Users, Folder, Cpu, Box, KeyRound,
  Rocket, CornerDownLeft, Search, Layers,
} from "lucide-react";
import type { AgentDTO, RunSummary } from "../../lib/api";
import { fireCommand, onCommand } from "../../lib/commands";

type Item = {
  id: string;
  label: string;
  hint?: string;
  group: string;
  icon: any;
  run: () => void;
};

const PAGES: { to: string; label: string; icon: any }[] = [
  { to: "/", label: "Dashboard", icon: LayoutGrid },
  { to: "/runs", label: "Runs", icon: GitBranch },
  { to: "/agents", label: "Agents", icon: Users },
  { to: "/tasks", label: "Tasks", icon: ListChecks },
  { to: "/files", label: "Files", icon: Folder },
  { to: "/models", label: "Models", icon: Box },
  { to: "/leaderboard", label: "Leaderboard", icon: Trophy },
  { to: "/hardware", label: "Hardware", icon: Cpu },
  { to: "/settings", label: "Settings", icon: KeyRound },
  // Autonomy Level explainer — reachable in context from Task Settings, and
  // here so it is findable by search rather than only by a deep link.
  { to: "/levels", label: "Autonomy Levels", icon: Layers },
];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const nav = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const { data: runs = [] } = useSWR<RunSummary[]>(open ? "/api/runs?limit=500" : null);
  const { data: agents = [] } = useSWR<AgentDTO[]>(open ? "/api/agents" : null);

  // open/close wiring: keyboard shortcut + command bus
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    const off = onCommand((c) => c === "open-palette" && setOpen(true));
    return () => {
      window.removeEventListener("keydown", onKey);
      off();
    };
  }, []);

  useEffect(() => {
    if (open) {
      setQ("");
      setSel(0);
      setTimeout(() => inputRef.current?.focus(), 20);
    }
  }, [open]);

  const items: Item[] = useMemo(() => {
    const actions: Item[] = [
      { id: "act-new", label: "Launch a new run", hint: "action", group: "Actions", icon: Rocket, run: () => { setOpen(false); fireCommand("open-new-run"); } },
    ];
    const pages: Item[] = PAGES.map((p) => ({
      id: `pg-${p.to}`, label: p.label, hint: p.to, group: "Navigate", icon: p.icon,
      run: () => { setOpen(false); nav(p.to); },
    }));
    const runItems: Item[] = runs.slice(0, 24).map((r) => ({
      id: `run-${r.id}`,
      // Named runs are looked for by their name; the task is the fallback, and
      // rides in the hint either way so one task's runs stay findable by it.
      label: r.run_name || r.task_name || r.task_objective?.slice(0, 60) || r.id.slice(0, 8),
      hint: [r.run_name ? r.task_name : "", r.id.slice(0, 8), r.status].filter(Boolean).join(" · "),
      group: "Runs", icon: GitBranch,
      run: () => { setOpen(false); nav(`/runs/${r.id}`); },
    }));
    // Evaluation is shown as a Run stage, not as an Agent destination.
    const agentItems: Item[] = agents.filter((a) => a.id !== "evaluation").map((a) => ({
      id: `agent-${a.id}`,
      // The id is what you type to find it; the hint says what it does.
      label: a.id,
      hint: a.title || "",
      group: "Agents", icon: Users,
      run: () => { setOpen(false); nav(`/agents/${a.id}`); },
    }));
    return [...actions, ...pages, ...runItems, ...agentItems];
  }, [runs, agents, nav]);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return items;
    return items.filter((it) => (it.label + " " + (it.hint ?? "") + " " + it.group).toLowerCase().includes(s));
  }, [items, q]);

  useEffect(() => setSel(0), [q]);
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${sel}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  if (!open) return null;

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, filtered.length - 1)); }
    if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
    if (e.key === "Enter") { e.preventDefault(); filtered[sel]?.run(); }
  };

  // group in stable order
  const groups = ["Actions", "Navigate", "Runs", "Agents"];
  let idx = -1;

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center px-4 pt-[12vh]" onMouseDown={() => setOpen(false)}>
      <div className="absolute inset-0 bg-canvas/80 backdrop-blur-sm" />
      <div
        className="bezel relative w-full max-w-2xl overflow-hidden animate-zevo-in"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-hair px-4 py-3">
          <Search size={17} className="text-brass-400" />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Jump to a run, agent, page, or launch a run…"
            className="w-full bg-transparent text-base text-ink placeholder:text-slate-500 focus:outline-none"
          />
          <kbd className="rounded border border-hair bg-raised px-1.5 py-0.5 font-mono text-2xs text-slate-500">ESC</kbd>
        </div>
        <div ref={listRef} className="max-h-[52vh] overflow-y-auto p-2">
          {filtered.length === 0 && (
            <div className="px-3 py-8 text-center text-sm text-slate-500">No matches for “{q}”.</div>
          )}
          {groups.map((g) => {
            const gi = filtered.filter((it) => it.group === g);
            if (!gi.length) return null;
            return (
              <div key={g} className="mb-1">
                <div className="px-3 py-1.5">
                  <span className="kicker text-[0.65rem]">{g}</span>
                </div>
                {gi.map((it) => {
                  idx += 1;
                  const active = idx === sel;
                  const Icon = it.icon;
                  const myIdx = idx;
                  return (
                    <button
                      key={it.id}
                      data-idx={myIdx}
                      onMouseEnter={() => setSel(myIdx)}
                      onClick={() => it.run()}
                      className={[
                        "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition",
                        active ? "bg-brass-500/12 text-ink" : "text-slate-300 hover:bg-raised",
                      ].join(" ")}
                    >
                      <Icon size={16} className={active ? "text-brass-300" : "text-slate-500"} strokeWidth={1.8} />
                      <span className="flex-1 truncate text-sm">{it.label}</span>
                      {it.hint && <span className="truncate font-mono text-2xs text-slate-500">{it.hint}</span>}
                      {active && <CornerDownLeft size={13} className="text-brass-400" />}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
