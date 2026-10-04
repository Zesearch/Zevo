import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

export function RunGroupFilter({ kind, options, selected, onChange, loading, error, onRetry }: {
  kind: "tasks" | "statuses";
  options: { value: string; label: string }[];
  selected: string[];
  onChange: (values: string[]) => void;
  loading?: boolean;
  error?: boolean;
  onRetry?: () => void;
}) {
  const root = useRef<HTMLDetailsElement>(null);
  const [search, setSearch] = useState("");
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) root.current.open = false;
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);
  const visible = options.filter(option => option.label.toLowerCase().includes(search.toLowerCase()));
  return (
    <details ref={root} className="relative" onKeyDown={event => {
      if (event.key === "Escape" && root.current) {
        root.current.open = false;
        root.current.querySelector("summary")?.focus();
      }
    }}>
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-md border border-hair px-2.5 py-1.5 font-mono text-2xs text-slate-300">
        {selected.length ? `${selected.length} ${kind} selected` : `All ${kind}`}
        <ChevronDown size={12} />
      </summary>
      <div className="absolute right-0 z-30 mt-2 w-72 max-w-[85vw] rounded-md border border-hair bg-panel p-3 shadow-xl">
        <button className="mb-2 text-sm text-brass-300" onClick={() => onChange([])}>All {kind}</button>
        {kind === "tasks" && <input aria-label="Find a task" placeholder="Find a task…"
          value={search} onChange={event => setSearch(event.target.value)}
          className="mb-2 w-full rounded border border-hair bg-canvas px-2 py-1 text-sm text-slate-200" />}
        {loading && <p className="text-sm text-slate-400">Loading {kind}…</p>}
        {error ? <p role="alert" className="text-sm text-coral-300">Could not load {kind}. <button onClick={onRetry}>Retry</button></p> : (
          <fieldset aria-label={`Filter ${kind}`} className="max-h-64 overflow-y-auto">
            {visible.map(option => <label key={option.value} className="flex cursor-pointer items-start gap-2 rounded px-1 py-2 text-sm text-slate-200 hover:bg-raised">
              <input type="checkbox" className="mt-1 accent-amber-400" checked={selected.includes(option.value)}
                onChange={() => onChange(selected.includes(option.value)
                  ? selected.filter(value => value !== option.value) : [...selected, option.value])} />
              <span className="break-all">{option.label}</span>
            </label>)}
            {!loading && visible.length === 0 && <p className="text-sm text-slate-400">No {kind} found.</p>}
          </fieldset>
        )}
        <p className="mt-2 text-xs text-slate-500">Select one or more. No selection shows all.</p>
      </div>
    </details>
  );
}
