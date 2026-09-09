import { relPath } from "../lib/format";

type SettingsPanelProps = {
  config: Record<string, unknown>;
};

const HIDDEN_SETTINGS = new Set([
  "mode", "t", "resolved_framing", "chat_template_patched",
]);

const LABELS: Record<string, string> = {
  dtype: "Dtype",
  eos_override: "EOS Override",
  eval_steps: "Evaluation Steps",
  gpu: "GPU",
  lr: "Learning Rate",
  lora_alpha: "LoRA Alpha",
  lora_dropout: "LoRA Dropout",
  lora_r: "LoRA R",
  max_seq_len: "Maximum Sequence Length",
  n_generations: "Generated Turns",
  n_eval: "Evaluation Examples",
  n_examples: "Training Examples",
  peft: "PEFT",
  top_k: "Top K",
  top_p: "Top P",
  trl: "TRL",
};

function labelFor(key: string): string {
  if (LABELS[key]) return LABELS[key];
  return key
    .split("_")
    .map((part) => ({ id: "ID", hf: "HF", gpu: "GPU", trl: "TRL", peft: "PEFT", eos: "EOS" }[part] || `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`))
    .join(" ");
}

function displayValue(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  if (typeof value === "string") return relPath(value);
  return String(value);
}

function resolvedEntries(config: Record<string, unknown>) {
  const values: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(config)) {
    if (key !== "inference_config") values[key] = value;
  }
  for (const key of HIDDEN_SETTINGS) delete values[key];
  return values;
}

function SettingsRows({
  entries,
}: {
  entries: Array<[string, unknown]>;
}) {
  const isBlock = ([, value]: [string, unknown]) =>
    typeof value === "object"
    || (typeof value === "string" && (value.includes("\n") || value.length > 72));
  const ordered = [...entries.filter((entry) => !isBlock(entry)), ...entries.filter(isBlock)];
  return (
    <div className="grid grid-cols-1 gap-x-6 gap-y-2 font-mono text-[15px] sm:grid-cols-2 lg:grid-cols-3">
      {ordered.map(([key, value]) => {
        const text = displayValue(value);
        if (isBlock([key, value])) {
          return (
            <div key={key} className="col-span-full border-b border-hair/60 pb-2">
              <div className="text-dim">{labelFor(key)}</div>
              <pre className="mt-1 overflow-x-auto whitespace-pre-wrap break-words rounded bg-canvas/60 p-2 text-[13px] leading-relaxed text-slate-200">
                {text}
              </pre>
            </div>
          );
        }
        return (
          <div key={key} className="flex items-baseline gap-2 border-b border-hair/60 pb-1.5">
            <span className="shrink-0 text-dim">{labelFor(key)}</span>
            <span className="ml-auto min-w-0 break-all text-right text-slate-200">{text}</span>
          </div>
        );
      })}
    </div>
  );
}

export function SettingsPanel({ config }: SettingsPanelProps) {
  const values = resolvedEntries(config || {});
  const entries = Object.entries(values);
  if (entries.length === 0) return null;

  return (
    <div className="rounded-md border border-hair/70 bg-canvas/20 p-3">
      <SettingsRows entries={entries} />
    </div>
  );
}
