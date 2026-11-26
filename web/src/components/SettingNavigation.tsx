import { useEffect, useId, type ReactNode } from "react";
import { LaunchContents, LaunchStepHeading } from "./LaunchContents";

/** A settings picker or configuration stepper with a single content pane. */
export function SettingNavigation({ items, active, onSelect, label, action, children, top, numbered = true }: {
  items: { id: string; label: string; description?: ReactNode; accessory?: ReactNode; group?: "Required" | "Optional" | "Review"; complete?: boolean }[];
  numbered?: boolean;
  active: string;
  onSelect: (id: string) => void;
  label: string;
  action?: ReactNode;
  top?: ReactNode;
  children: ReactNode;
}) {
  const id = useId();
  const checklistReady = items.filter(item => item.id !== "checklist").every(item => item.complete);
  const visibleItems = numbered ? items.filter(item => item.id !== "checklist" || checklistReady) : items;
  useEffect(() => {
    if (numbered && active === "checklist" && !checklistReady) {
      const incomplete = items.find(item => item.id !== "checklist" && !item.complete);
      if (incomplete) onSelect(incomplete.id);
    }
  }, [numbered, active, checklistReady, items, onSelect]);
  const headings: Record<string, [string, string]> = {
    basics: ["Let’s start with the essentials.", label === "Task setup" ? "Name your task and describe what you want to improve." : label === "File setup" ? "Name your file set and add an optional description." : "Give this setting a name so you can reuse it later."],
    tests: ["How will you measure success?", "Define the Test sets for this task."],
    files: ["Bring your files together.", "Upload files or connect a Hugging Face dataset."],
    validation: ["Set up your validation.", "By default, sample 20% from each eligible Test set, with a minimum of 200 rows. Or add independent Validation sets."],
    train: ["How would you like to train?", "Choose your data, model and method, or leave them for Zevo to decide."],
    limits: ["A few finishing touches.", "Set optional limits for iterations, cost and validation score."],
    checklist: ["Ready for a final look?", label === "Task setup" ? "Review your configuration before creating the task." : label === "File setup" ? "Review your file set before saving." : "Review your configuration before saving the setting."],
  };
  if (numbered) return <div className="setting-navigation-layout">
    <aside className="setting-navigation">
      {top && <div className="mb-5">{top}</div>}
      <LaunchContents steps={visibleItems.map(item => ({ label: item.label, complete: item.complete ?? false, group: item.id === "checklist" ? "Review" : item.group }))}
        active={Math.max(0, visibleItems.findIndex(item => item.id === active))} onSelect={index => onSelect(visibleItems[index].id)} />
      {action}
    </aside>
    <div className="setting-navigation-panel min-w-0"><LaunchStepHeading key={active} label={active} copy={headings[active]} />{children}</div>
  </div>;
  return <div className="setting-navigation-layout">
    <aside className="setting-navigation">
      <p className="setting-navigation-label">{label}</p>
      <div role="tablist" aria-label={label} aria-orientation="vertical">
        {items.map((item, index) => <div key={item.id} className="setting-saved-item" data-active={active === item.id}>
        <button
          type="button" role="tab" id={`${id}-tab-${item.id}`}
          aria-selected={active === item.id} aria-controls={`${id}-panel`}
          tabIndex={active === item.id ? 0 : -1}
          onClick={() => onSelect(item.id)}
          onKeyDown={event => {
            const next = event.key === "ArrowDown" || event.key === "ArrowRight" ? (index + 1) % items.length
              : event.key === "ArrowUp" || event.key === "ArrowLeft" ? (index + items.length - 1) % items.length
              : event.key === "Home" ? 0 : event.key === "End" ? items.length - 1 : -1;
            if (next < 0) return;
            event.preventDefault();
            onSelect(items[next].id);
            document.getElementById(`${id}-tab-${items[next].id}`)?.focus();
          }}
        >
          {numbered && <span className="setting-navigation-number">{String(index + 1).padStart(2, "0")}</span>}
          <span className="setting-saved-name min-w-0 block">{item.label}</span>
          {item.description && <span className="setting-saved-description">{item.description}</span>}
        </button>
        {item.accessory}
        </div>)}
      </div>
      {action && <div className="setting-navigation-action">{action}</div>}
    </aside>
    <div id={`${id}-panel`} role="tabpanel" aria-labelledby={active ? `${id}-tab-${active}` : undefined} tabIndex={0} className="setting-navigation-panel min-w-0">{children}</div>
  </div>;
}
