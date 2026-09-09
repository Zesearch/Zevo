import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { RunLaunchMode } from "./NewRunModal";

/**
 * The "i" beside each mode card in the launch dialog: what that way of
 * running asks for, as the ordered walk from an empty form to Launch.
 *
 * The dialog shows one mode's form at a time and has no room to say what the
 * others want, so the comparison lives here, one popover per card. Every step
 * names a field the dialog's checklist actually requires; optional ones say
 * so.
 */

type Way = { title: string; when: string; steps: string[]; note?: string };

const WAYS: Record<RunLaunchMode, Way> = {
  auto: {
    title: "Auto",
    when: "You want Zevo to define the evaluation contract.",
    steps: [
      "Name the run and give the task a new name. Auto always starts a fresh task, so a predefined task name is refused.",
      "Write the objective in plain language: what the model should get better at and what a correct answer looks like.",
      "Optionally pin the training data, base model, or training method. Leaving them blank gives Zevo the same L1–L4 ownership ladder as Standard.",
      "Optionally set a spend cap, an iteration budget, and the GPU backend.",
      "Launch. Zevo chooses the metric and a public or synthesized held-out Test contract, then runs the normal optimization pipeline with your training-side choices.",
    ],
  },
  full_pipeline: {
    title: "Standard",
    when: "You know how success is measured and can supply the examples.",
    steps: [
      "Name the run and the task, and write the objective.",
      "Choose the test metric type, the metric, and the target direction. A custom metric needs an evaluation script.",
      "Upload the test set, name its answer fields, and give a sample submission that shows the expected output columns.",
      "Optionally add a validation set with its own answer fields and sample submission. Without one, Zevo reserves 20% of the test set for validation, which needs at least 1,000 test rows.",
      "Optionally pin the training data, base model or training method, then pick the GPU backend and the budgets.",
      "Review the checklist until it reads ready, then launch.",
    ],
  },
  customized_pipeline: {
    title: "Customized",
    when: "You want the full pipeline but with marching orders for each agent.",
    steps: [
      "Fill the same contract as Standard: names, objective, test metric and target, test set with answer fields and sample submission.",
      "For each agent, write its instructions and set its inputs: files, output location and hyperparameters.",
      "Mark each block strict, so the agent rejects any deviation, or advisory, so it may fall back only in the documented ways.",
      "Launch. The orchestrator runs the pipeline inside the blocks you set.",
    ],
  },
  single_stage: {
    title: "Single Stage",
    when: "One agent, one job. No optimization loop.",
    steps: [
      "Name the run and the task, and write the objective for that one job.",
      "Choose the agent: data, infrastructure, train, inference or registry.",
      "Supply the inputs that agent needs for the job.",
      "Launch. The ticket goes straight to the agent and the run ends when it does.",
    ],
  },
};

const POPOVER_W = 26 * 16;

export function ModeInfo({
  mode,
  open,
  onToggle,
  onClose,
}: {
  mode: RunLaunchMode;
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
}) {
  const way = WAYS[mode];
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!(e.target as Element).closest("[data-mode-info]")) onClose();
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, onClose]);

  function toggle() {
    const r = btnRef.current!.getBoundingClientRect();
    setPos({
      left: Math.min(r.right + 8, window.innerWidth - POPOVER_W - 16),
      top: Math.max(16, Math.min(r.top, window.innerHeight - 420)),
    });
    onToggle();
  }

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        data-mode-info
        onClick={toggle}
        aria-label={`How to run: ${way.title}`}
        className={`absolute right-2 top-2 rounded-md p-1 transition ${open ? "text-brass-300" : "text-slate-500 hover:text-brass-300"}`}
      >
        <span
          aria-hidden="true"
          className="flex h-3.5 w-3.5 items-center justify-center rounded-full bg-current font-sans text-[9px] font-black leading-none"
        >
          <span className="text-canvas">i</span>
        </span>
      </button>
      {open && pos && createPortal(
        // Portaled past the dialog, which clips and scrolls its body. React
        // still bubbles clicks up the component tree to the dialog's
        // close-on-backdrop handler, so stop them here.
        // `.bezel` is plain CSS with `position: relative`, which outranks the
        // layered `fixed` utility, so the fixed placement lives on a wrapper.
        <div
          data-mode-info
          className="fixed z-[110]"
          style={{ top: pos.top, left: pos.left, width: POPOVER_W, maxWidth: "calc(100vw - 2rem)" }}
          onClick={(e) => e.stopPropagation()}
        >
        <div className="bezel p-4 animate-zevo-in">
          <div className="text-sm font-semibold text-ink">{way.title}</div>
          <p className="mt-0.5 text-xs text-slate-400">{way.when}</p>
          <ol className="mt-3 flex flex-col gap-2">
            {way.steps.map((step, i) => (
              <li key={i} className="flex gap-2.5 text-xs leading-5 text-slate-300">
                <span className="readout mt-0.5 w-4 shrink-0 text-right text-2xs text-brass-400">{i + 1}</span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
          {way.note && <p className="mt-3 text-2xs leading-4 text-slate-500">{way.note}</p>}
        </div>
        </div>,
        document.body,
      )}
    </>
  );
}
