// Per-driver model catalog used by AgentConfigCard's dependent dropdown.
// Four drivers: bedrock (AWS Bedrock Converse), claude_cli,
// codex_cli, and openrouter.
// The bedrock model ids + tool-use support were verified live against this
// account's bedrock-runtime.
//
// When adding a model here, ALSO add it to src/zevo/engine/cost/pricing.py with
// $/Mtok rates so the cost meter shows a real number; otherwise the
// cost panel will silently render $0.00 for that model.

export type ModelOption = {
  id: string;            // value sent to the backend & to the CLI binary
  label: string;         // human-readable name
  tag?: string;          // short qualifier shown next to the name (e.g. "newest", "fast")
  contextK?: number;     // input context window in thousand tokens (for the hint)
  inputUsd?: number;     // $/Mtok in
  outputUsd?: number;    // $/Mtok out
  note?: string;         // single-line description (intelligence/speed/cost trade-off)
};

export type DriverSpec = {
  id: string;
  short: string;          // 1-2 word friendly name used in the heading
  defaultModel: string;
  models: ModelOption[];
};

// Curation rule:
//   * Default = newest GA model the CLI talks to today
//   * Include 1-2 prior versions for cost / behaviour comparison
//   * Always include a "fast/cheap" tier (mini, nano, haiku)
//   * Drop dated suffix variants (-20251101 etc.) — the providers also
//     accept the unsuffixed alias.

export const DRIVERS: DriverSpec[] = [
  // ───────────────────────────── bedrock ───────────────────────────────
  {
    id: "bedrock",
    short: "AWS Bedrock",
    defaultModel: "moonshotai.kimi-k2.5",
    models: [
      // Open models — verified available + tool-use on this account.
      { id: "moonshotai.kimi-k2.5",    label: "Kimi K2.5",        tag: "default", inputUsd: 0.57, outputUsd: 2.85,
        note: "Moonshot. Strong agentic / tool-use model. The default." },
      { id: "deepseek.v3.2",           label: "DeepSeek V3.2",                    inputUsd: 0.269, outputUsd: 0.4,
        note: "DeepSeek. Cheap, strong reasoning + tool use." },
      { id: "minimax.minimax-m2.5",    label: "MiniMax M2.5",     tag: "cheap",   inputUsd: 0.22, outputUsd: 0.9,
        note: "MiniMax. Cheapest of the bunch." },
      { id: "zai.glm-5",               label: "GLM 5",                            inputUsd: 0.95, outputUsd: 2.55,
        note: "Z.AI (Zhipu) flagship." },
      { id: "qwen.qwen3-coder-next",   label: "Qwen3 Coder",                      inputUsd: 0.12, outputUsd: 0.8,
        note: "Alibaba Qwen, coding-tuned." },
    ],
  },

  // ───────────────────────────── claude_cli ────────────────────────────
  {
    id: "claude_cli",
    short: "Claude Code CLI",
    defaultModel: "claude-opus-5",
    models: [
      { id: "claude-opus-5",      label: "Claude Opus 5",     tag: "default",  contextK: 1000, inputUsd: 5.0, outputUsd: 25.0,
        note: "Current Opus. Strongest on long agentic runs; the default here." },
      { id: "claude-sonnet-5",    label: "Claude Sonnet 5",                    contextK: 1000, inputUsd: 2.0, outputUsd: 10.0,
        note: "Near-Opus quality on coding at Sonnet cost. Introductory rate." },
      { id: "claude-fable-5-1",   label: "Claude Fable 5.1",  tag: "flagship", contextK: 1000, inputUsd: 10.0, outputUsd: 50.0,
        note: "Most capable Claude. Newest Fable. For the hardest reasoning only." },
      { id: "claude-fable-5",     label: "Claude Fable 5",                     contextK: 1000, inputUsd: 10.0, outputUsd: 50.0,
        note: "Previous Fable. Same price. Pin it to reproduce an older run." },
      { id: "claude-opus-4-8",    label: "Claude Opus 4.8",                    contextK: 1000, inputUsd: 5.0, outputUsd: 25.0,
        note: "Previous Opus. Same price. Pin it to reproduce an older run." },
      { id: "claude-opus-4-7",    label: "Claude Opus 4.7",                    contextK: 1000, inputUsd: 5.0, outputUsd: 25.0,
        note: "The Opus before 4.8. Same price. Pin it to reproduce an older run." },
      { id: "claude-haiku-4-5",   label: "Claude Haiku 4.5",   tag: "fast",    contextK:  200, inputUsd: 1.0, outputUsd: 5.0,
        note: "Cheap + fast. Good for orchestration." },
    ],
  },

  // ───────────────────────────── codex_cli ─────────────────────────────
  {
    id: "codex_cli",
    short: "Codex CLI",
    defaultModel: "gpt-5.5",
    models: [
      { id: "gpt-5.5",            label: "GPT-5.5",       tag: "default", contextK: 1050, inputUsd: 5.0, outputUsd: 30.0,
        note: "Recommended Codex model for complex agent work; subscription runs consume ChatGPT/Codex usage credits." },
      { id: "gpt-5.3-codex",      label: "GPT-5.3 Codex", contextK: 400, inputUsd: 1.75, outputUsd: 14.0,
        note: "Previous Codex generation, retained for reproducibility." },
      { id: "gpt-5.2-codex",      label: "GPT-5.2 Codex", contextK: 400, inputUsd: 1.75, outputUsd: 14.0,
        note: "Older Codex generation, retained for reproducibility." },
      { id: "gpt-5.1-codex-max",  label: "GPT-5.1 Codex Max",             contextK: 400, inputUsd: 1.25, outputUsd: 10.0,
        note: "Previous generation, cheaper than the 5.2/5.3 pair." },
      { id: "gpt-5.1-codex-mini", label: "GPT-5.1 Codex mini", tag: "fast", contextK: 400, inputUsd: 0.25, outputUsd: 2.0,
        note: "Cheap + fast. Good for orchestration." },
    ],
  },

  // ───────────────────────────── openrouter ────────────────────────────
  // One key, several hundred models. These are a starting set; any id
  // OpenRouter accepts works via "Custom…", it just prices at $0.00 until
  // it is added to src/zevo/engine/cost/pricing.py too.
  {
    id: "openrouter",
    short: "OpenRouter",
    defaultModel: "qwen/qwen3.8-max",
    // Anthropic and OpenAI models are deliberately absent: claude_cli and
    // codex_cli reach those directly, on a subscription rather than per call.
    // This menu is what the other two drivers cannot get you — the current
    // generation of each open-weight family, refreshed from the live
    // catalogue. Any id OpenRouter accepts still works via "Custom…".
    models: [
      { id: "qwen/qwen3.8-max", label: "Qwen3.8 Max", tag: "default", contextK: 1000, inputUsd: 2.0, outputUsd: 6.0,
        note: "Alibaba's current flagship. 1M context, strong coding + agentic." },
      { id: "moonshotai/kimi-k3", label: "Kimi K3", tag: "flagship", contextK: 1049, inputUsd: 3.0, outputUsd: 15.0,
        note: "Moonshot's newest. 1M context, strong agentic behaviour." },
      { id: "x-ai/grok-4.5", label: "Grok 4.5", contextK: 500, inputUsd: 2.0, outputUsd: 6.0,
        note: "xAI. 500K context." },
      { id: "google/gemini-3.6-flash", label: "Gemini 3.6 Flash", contextK: 1049, inputUsd: 1.5, outputUsd: 7.5,
        note: "Google's newest Flash. 1M context." },
      { id: "z-ai/glm-5.2", label: "GLM 5.2", contextK: 1049, inputUsd: 0.546, outputUsd: 1.716,
        note: "Z.AI (Zhipu). 1M context, cheaper than GLM 5 and newer." },
      { id: "minimax/minimax-m3", label: "MiniMax M3", contextK: 1049, inputUsd: 0.3, outputUsd: 1.2,
        note: "1M context at a fraction of the flagship prices." },
      { id: "deepseek/deepseek-v4-pro", label: "DeepSeek V4 Pro", contextK: 1049, inputUsd: 0.435, outputUsd: 0.87,
        note: "DeepSeek's current top model. 1M context." },
      { id: "deepseek/deepseek-v4-flash-0731", label: "DeepSeek V4 Flash", tag: "cheap", contextK: 1049, inputUsd: 0.09, outputUsd: 0.18,
        note: "Cheapest here by an order of magnitude. 1M context." },
      { id: "qwen/qwen3.7-flash", label: "Qwen3.7 Flash", tag: "fast", contextK: 1000, inputUsd: 0.03, outputUsd: 0.13,
        note: "Cheapest input of the set. Good for orchestration." },
    ],
  },
];

export function getDriver(id: string): DriverSpec | undefined {
  return DRIVERS.find((d) => d.id === id);
}

export function isKnownModel(driverId: string, modelId: string): boolean {
  const d = getDriver(driverId);
  if (!d) return false;
  return d.models.some((m) => m.id === modelId);
}
