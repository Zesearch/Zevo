import { useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";

/** Just the two fields the suggestion list shows. The full task row is bigger,
 *  but this component only ever names and describes one. */
type TaskLite = { name: string; task_objective?: string };

/**
 * The task-name box: free text, with the predefined tasks offered as you type.
 *
 * A run's task name is not a fixed set — anything you type is a new custom task
 * — so this is a combobox rather than a select: type "capy" and every
 * predefined name containing it drops down, but nothing stops you naming
 * something the catalogue has never heard of.
 */
export function TaskNameInput({
  value,
  onChange,
  onPick,
  placeholder = "",
  className = "",
  id,
}: {
  value: string;
  onChange: (v: string) => void;
  /** Fired only when a suggestion was chosen, never on typing. */
  onPick?: (name: string) => void;
  placeholder?: string;
  className?: string;
  id?: string;
}) {
  // SWR dedupes this against the caller's own /tasks fetch, so asking for the
  // list here costs nothing and keeps the call sites to one line.
  const { data: tasks = [] } = useSWR<TaskLite[]>("/api/tasks");
  const [open, setOpen] = useState(false);
  // Which row the arrow keys are on. -1 = none, so Enter submits normally.
  const [cursor, setCursor] = useState(-1);
  const box = useRef<HTMLDivElement>(null);

  const matches = useMemo(() => {
    const term = value.trim().toLowerCase();
    // An empty box offers everything: opening the list is also how you find out
    // what the predefined tasks ARE.
    const hits = term
      ? tasks.filter((t) => t.name.toLowerCase().includes(term))
      : tasks;
    // An exact match has nothing left to suggest — the dropdown would just
    // cover the form with the one row you already typed.
    if (hits.length === 1 && hits[0].name.toLowerCase() === term) return [];
    return hits.slice(0, 8);
  }, [tasks, value]);

  // Clicking anywhere else closes it. Pointerdown, not click: the field below
  // should receive the click that dismissed this.
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);

  function pick(name: string) {
    onChange(name);
    onPick?.(name);
    setOpen(false);
    setCursor(-1);
  }

  const show = open && matches.length > 0;

  return (
    <div ref={box} className="relative">
      <input
        id={id}
        value={value}
        onChange={(e) => { onChange(e.target.value); setOpen(true); setCursor(-1); }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            if (!show) { setOpen(true); return; }
            e.preventDefault();
            setCursor((c) =>
              e.key === "ArrowDown"
                ? Math.min(c + 1, matches.length - 1)
                : Math.max(c - 1, -1),
            );
          } else if (e.key === "Enter" && show && cursor >= 0) {
            e.preventDefault();
            pick(matches[cursor].name);
          } else if (e.key === "Escape" && show) {
            // Swallowed, so closing the list does not also close the dialog.
            e.stopPropagation();
            setOpen(false);
          }
        }}
        placeholder={placeholder}
        className={className}
        spellCheck={false}
        autoComplete="off"
        role="combobox"
        aria-expanded={show}
      />
      {show && (
        <ul className="absolute left-0 right-0 top-full z-20 mt-1 max-h-56 overflow-y-auto rounded-md border border-hair bg-raised py-1 shadow-lg">
          {matches.map((t, i) => (
            <li key={t.name}>
              {/* Mousedown, not click: the input's blur would otherwise fire
                  first and take the row out from under the pointer. */}
              <button
                type="button"
                onMouseDown={(e) => { e.preventDefault(); pick(t.name); }}
                onMouseEnter={() => setCursor(i)}
                className={`block w-full px-3 py-1.5 text-left transition ${
                  i === cursor ? "bg-brass-500/10" : "hover:bg-canvas/60"
                }`}
              >
                <span className="font-mono text-xs text-brass-300">{t.name}</span>
                {t.task_objective && (
                  <span className="mt-0.5 block truncate text-2xs text-slate-500">
                    {t.task_objective}
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
