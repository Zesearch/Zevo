/**
 * The provider-credential board: one presentation module shared by the
 * open-source Settings page and the hosted console's Settings page.
 *
 * Everything here is presentation plus the copy that makes the page double as
 * a setup guide ("how do I get this key"). It owns no data source: rows call
 * the `onSave` / `onClear` callbacks the host page supplies, so the same board
 * can write to the single-tenant `/api/settings`, or to the console's
 * `/api/tenant-settings` and `/api/user-settings`.
 *
 * A host page narrows the board with `keys` (only these env names render) and
 * fills the GPU section's two compute slots with whatever backend inventory it
 * can see. Sections and cards left empty by that filter are dropped.
 */
import { createContext, useContext, useState, type ReactNode } from "react";
import {
  KeyRound, Eye, EyeOff, Check, RotateCw, AlertTriangle,
  ExternalLink, Copy, Cloud, Server, ShieldCheck, Clock, XCircle,
} from "lucide-react";

/** Current state of one credential, as every backend reports it. */
export type KeyEntry = { name: string; present: boolean; preview: string };

type ProviderBoardContextValue = {
  entries: KeyEntry[];
  onSave: (name: string, value: string) => Promise<void>;
};

const ProviderBoardContext = createContext<ProviderBoardContextValue | null>(null);
const CLOUD_BACKEND_KEY = "ZEVO_CLOUD_BACKEND";

export type AuthDriver = {
  driver: string; mode: "subscription" | "api_key" | "either" | "none";
  has_subscription: boolean; has_api_key: boolean; ready: boolean;
  effective: string;
};

// One row per secret. Layout, copy, format hint, and CLI-vs-API
// distinction (Claude OAuth = "use the CLI to generate this") are
// per-key so the page doubles as a setup guide for new users.
export type SecretSpec = {
  key: string;
  label: string;
  driver: string;        // which driver this credential is for
  kind: "api_key" | "oauth_token" | "plain";
  format_hint: string;
  obtain: string;        // one-line "how do I get this"
  obtain_link?: string;  // optional URL
  obtain_cmd?: string;   // optional CLI command
  choices?: string[];    // fixed set of legal values -> render a picker
};

export const SECRETS: SecretSpec[] = [
  {
    key: "CLAUDE_CODE_OAUTH_TOKEN",
    label: "Claude Max OAuth token",
    driver: "claude_cli",
    kind: "oauth_token",
    format_hint: "starts with sk-ant-oat…, valid ~1 year",
    obtain: "Run `claude setup-token` on your laptop, paste the browser's code back into the terminal, then copy the sk-ant-oat… token the terminal prints (not the browser code)",
    obtain_cmd: "claude setup-token",
  },
  {
    key: "ANTHROPIC_API_KEY",
    label: "Anthropic API key",
    driver: "claude_cli",
    kind: "api_key",
    format_hint: "starts with sk-ant-…",
    obtain: "console.anthropic.com > API Keys. Usage is billed separately from a Claude subscription",
    obtain_link: "https://console.anthropic.com/settings/keys",
  },
  {
    key: "ANTHROPIC_AUTH_TOKEN",
    label: "Anthropic bearer token",
    driver: "claude_cli",
    kind: "api_key",
    format_hint: "non-empty bearer token",
    obtain: "Optional explicit bearer-token override for managed deployments",
  },
  {
    key: "OPENAI_API_KEY",
    label: "OpenAI API key",
    driver: "codex_cli",
    kind: "api_key",
    format_hint: "starts with sk-…",
    obtain: "platform.openai.com > API keys. Optional usage-billed fallback when no ChatGPT login cache is mounted",
    obtain_link: "https://platform.openai.com/api-keys",
  },
  {
    key: "OPENROUTER_API_KEY",
    label: "OpenRouter API key",
    driver: "openrouter",
    kind: "api_key",
    format_hint: "starts with sk-or-v1-…",
    obtain: "openrouter.ai > Keys. One key covers every model OpenRouter fronts",
    obtain_link: "https://openrouter.ai/keys",
  },
  {
    key: "AWS_BEARER_TOKEN_BEDROCK",
    label: "Bedrock API key (bearer token)",
    driver: "bedrock",
    kind: "api_key",
    format_hint: "base64 blob, the single key string AWS gives you",
    obtain: "AWS console > Bedrock > API keys > Generate (simplest Bedrock auth)",
    obtain_link: "https://console.aws.amazon.com/bedrock/home#/api-keys",
  },
  {
    key: "AWS_REGION",
    label: "AWS region",
    driver: "bedrock",
    kind: "api_key",
    format_hint: "e.g. us-east-1",
    obtain: "Pick a region where Bedrock has the models you need (us-east-1, us-west-2 are common)",
  },
  {
    key: "ZEVO_CLOUD_BACKEND",
    label: "Default cloud backend",
    driver: "infra",
    kind: "plain",
    format_hint: "lambda or vastai",
    obtain: "Which rental provider a Cloud run reaches for first. Whichever you pick needs its API key below",
    choices: ["lambda", "vastai"],
  },
  {
    key: "VASTAI_API_KEY",
    label: "Vast.ai API key",
    driver: "infra",
    kind: "api_key",
    format_hint: "64 hex chars",
    obtain: "Account > API Keys at cloud.vast.ai",
    obtain_link: "https://cloud.vast.ai/api-keys/",
  },
  {
    key: "LAMBDA_API_KEY",
    label: "Lambda Cloud API key",
    driver: "infra",
    kind: "api_key",
    format_hint: "starts with secret_…",
    obtain: "Dashboard > API keys at cloud.lambda.ai",
    obtain_link: "https://cloud.lambda.ai/api-keys",
  },
  {
    key: "LAMBDA_SSH_KEY_NAME",
    label: "Lambda SSH key name (optional)",
    driver: "infra",
    kind: "plain",
    format_hint: "the key's name in your Lambda account, e.g. my-lambda-key",
    obtain: "Dashboard > SSH keys at cloud.lambda.ai. Leave empty to have one generated per instance",
    obtain_link: "https://cloud.lambda.ai/ssh-keys",
  },
  {
    key: "HF_TOKEN",
    label: "Hugging Face token",
    driver: "huggingface",
    kind: "api_key",
    format_hint: "starts with hf_…",
    obtain: "Hugging Face > Settings > Access Tokens. Needed only for gated or private models and datasets",
    obtain_link: "https://huggingface.co/settings/tokens",
  },
  {
    key: "WANDB_ENTITY",
    label: "Weights & Biases entity",
    driver: "wandb",
    kind: "plain",
    format_hint: "your user or team slug",
    obtain: "The entity that owns the public training project",
  },
  {
    key: "WANDB_PROJECT",
    label: "Weights & Biases project",
    driver: "wandb",
    kind: "plain",
    format_hint: "e.g. zevo",
    obtain: "Create the project in W&B and set its visibility to Public",
    obtain_link: "https://wandb.ai/home",
  },
  {
    key: "WANDB_API_KEY",
    label: "Weights & Biases API key",
    driver: "wandb",
    kind: "api_key",
    format_hint: "W&B personal/team API key",
    obtain: "wandb.ai > User settings > API keys",
    obtain_link: "https://wandb.ai/authorize",
  },
];

const specFor = (key: string) => SECRETS.find((s) => s.key === key);

/** SECRETS entries for `keys`, in registry order, skipping unknown names. */
export function specsFor(keys: string[]): SecretSpec[] {
  return keys.map(specFor).filter((s): s is SecretSpec => Boolean(s));
}


export function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="btn !px-1.5 !py-1 text-2xs"
      title="copy"
    >
      {copied ? <Check size={11} /> : <Copy size={11} />}
    </button>
  );
}


export function SecretRow({
  spec, entry, onSave, onClear,
}: {
  spec: SecretSpec;
  entry?: KeyEntry;
  onSave: (name: string, value: string) => Promise<void>;
  onClear: (name: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(spec.choices ? spec.choices[0] : "");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const blank = () => (spec.choices ? spec.choices[0] : "");

  async function save(clear: boolean = false) {
    setBusy(true);
    setError(null);
    try {
      if (clear) await onClear(spec.key);
      else await onSave(spec.key, value.trim());
      setValue(blank());
      setEditing(false);
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  const present = entry?.present ?? false;
  const preview = entry?.preview ?? "";

  return (
    <div className="py-3.5 first:pt-1">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-display text-sm font-semibold text-ink">{spec.label}</span>
            <code className="rounded bg-canvas px-1.5 py-px font-mono text-2xs text-slate-400">
              {spec.key}
            </code>
            {present ? (
              <span className="inline-flex items-center gap-1 rounded border border-phosphor-500/40 bg-phosphor-500/10 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-phosphor-300">
                <Check size={10} /> set
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded border border-hair bg-raised/50 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-slate-500">
                not set
              </span>
            )}
          </div>

          {/* How to obtain */}
          <p className="mt-1.5 text-2xs text-slate-400">
            {spec.obtain}
            {spec.obtain_link && (
              <a href={spec.obtain_link} target="_blank" rel="noreferrer"
                 className="ml-1 inline-flex items-center gap-0.5 text-brass-300 hover:underline">
                <ExternalLink size={10} /> open
              </a>
            )}
          </p>

          {spec.obtain_cmd && (
            <div className="mt-1.5 flex items-center gap-2">
              <code className="rounded bg-canvas px-2 py-1 font-mono text-2xs text-slate-300">
                {spec.obtain_cmd}
              </code>
              <CopyBtn text={spec.obtain_cmd} />
            </div>
          )}

          {present && !editing && (
            <div className="mt-2.5 flex items-center gap-2 font-mono text-xs text-slate-500">
              <KeyRound size={11} className="text-brass-400" /> {preview || "(redacted)"}
            </div>
          )}

          {editing && (
            <div className="mt-2.5">
              <div className="flex gap-1">
                {spec.choices ? (
                  <select
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    className="w-full rounded-md border border-hair bg-canvas px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-brass-500/50"
                  >
                    {spec.choices.map((choice) => (
                      <option key={choice} value={choice}>{choice}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type={spec.kind === "plain" || show ? "text" : "password"}
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    placeholder={spec.format_hint}
                    autoComplete="off"
                    spellCheck={false}
                    className="w-full rounded-md border border-hair bg-canvas px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-brass-500/50"
                  />
                )}
                {spec.kind !== "plain" && (
                  <button
                    onClick={() => setShow((s) => !s)}
                    className="btn !px-2"
                    title={show ? "hide" : "show"}
                  >
                    {show ? <EyeOff size={12} /> : <Eye size={12} />}
                  </button>
                )}
              </div>
              <p className="mt-1.5 font-mono text-2xs text-slate-500">format: {spec.format_hint}</p>
            </div>
          )}

          {error && (
            <div className="mt-2.5 rounded-md border border-coral-500/30 bg-coral-500/10 p-2 text-2xs text-coral-300">
              {error}
            </div>
          )}
        </div>

        {/* Right-side actions */}
        <div className="flex w-[5.5rem] flex-shrink-0 flex-col items-stretch gap-1.5 text-center">
          {!editing ? (
            <>
              <button onClick={() => setEditing(true)} className="btn btn-brass justify-center !px-2 !py-1 !text-[12px] uppercase !tracking-[0.1em]">
                {present ? "replace" : "set"}
              </button>
              {present && (
                <button
                  onClick={() => save(true)}
                  disabled={busy}
                  className="btn justify-center !px-2 !py-1 !text-[12px] uppercase !tracking-[0.1em] disabled:opacity-40"
                >
                  clear
                </button>
              )}
            </>
          ) : (
            <>
              <button
                onClick={() => save(false)}
                disabled={busy || !value.trim()}
                className="btn justify-center !px-2 !py-1 !text-[12px] uppercase !tracking-[0.1em] !text-phosphor-300 disabled:opacity-40"
              >
                {busy ? <RotateCw size={11} className="inline animate-spin" /> : "save"}
              </button>
              <button
                onClick={() => { setEditing(false); setValue(blank()); setError(null); }}
                className="btn justify-center !px-2 !py-1 !text-[12px] uppercase !tracking-[0.1em]"
              >
                cancel
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}


export type CredentialGroup = {
  id: string;
  title: string;
  subtitle: string;
  specs: SecretSpec[];
};

export function CredentialCard({
  group, entries, onSave, onClear,
}: {
  group: CredentialGroup;
  entries?: KeyEntry[];
  onSave: (name: string, value: string) => Promise<void>;
  onClear: (name: string) => Promise<void>;
}) {
  return (
    <section className="min-w-0 rounded-md border border-hair bg-panel/40 p-4 sm:p-5">
      <div className="min-w-0">
        <h3 className="font-display text-base font-semibold tracking-tight text-ink">{group.title}</h3>
        <p className="mt-1 text-2xs leading-relaxed text-slate-500">{group.subtitle}</p>
      </div>
      <div className="mt-3 divide-y divide-hair/70">
        {group.specs.map((spec) => (
          <SecretRow
            key={spec.key}
            spec={spec}
            entry={entries?.find((entry) => entry.name === spec.key)}
            onSave={onSave}
            onClear={onClear}
          />
        ))}
      </div>
    </section>
  );
}

export function SectionStatusRow({
  items, loading = false,
}: {
  items: Array<{ label: string; ready: boolean }>;
  loading?: boolean;
}) {
  if (loading || items.length === 0) return null;
  const ready = items.filter((item) => item.ready);
  const missing = items.filter((item) => !item.ready);
  return (
    <div className="mb-4 flex flex-wrap items-center gap-2 font-mono text-xs">
      <div className="flex flex-wrap items-center gap-2">
        {ready.length ? ready.map((item) => (
          <span key={item.label} className="inline-flex items-center gap-1.5 rounded border border-phosphor-500/40 bg-phosphor-500/10 px-2.5 py-1 text-phosphor-200">
            <span className="lamp lamp-live" />
            <Check size={10} /> {item.label}
          </span>
        )) : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {missing.length ? missing.map((item) => (
          <span key={item.label} className="inline-flex items-center gap-1.5 rounded border border-coral-500/40 bg-coral-500/10 px-2.5 py-1 text-coral-200">
            <span className="lamp lamp-coral" />
            <AlertTriangle size={10} /> {item.label}
          </span>
        )) : null}
      </div>
    </div>
  );
}

// One compute backend as it would appear in the New Run "GPU backend" picker.
// `available` mirrors the gating there: a cloud backend needs its API key, a
// managed SSH host needs to be verified, an env SSH connection needs to be
// configured. `detail` is a short, credential-free descriptor.
export type ProviderRow = {
  key: string;
  icon: "cloud" | "server";
  name: string;
  type: string;
  detail: string;
  available: boolean;
  note: string;      // why it is / isn't offered in a run
  status?: string;   // optional verification status for SSH backends
};

export function ProviderStatusPill({ row }: { row: ProviderRow }) {
  if (row.available) {
    return (
      <span className="inline-flex items-center gap-1 rounded border border-phosphor-500/40 bg-phosphor-500/10 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-phosphor-300">
        <ShieldCheck size={10} /> available
      </span>
    );
  }
  if (row.status === "failed") {
    return (
      <span className="inline-flex items-center gap-1 rounded border border-coral-500/40 bg-coral-500/10 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-coral-300">
        <XCircle size={10} /> failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded border border-hair bg-raised/50 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-slate-500">
      <Clock size={10} /> not available
    </span>
  );
}

// Roll-up of compute providers the New Run picker can use right now. Cloud rows
// also own the default-backend choice, keeping that setting beside the usable
// providers instead of rendering it as a separate credential card.
export function ComputeProvidersPanel({
  rows, loading = false, error,
}: {
  rows: ProviderRow[];
  loading?: boolean;
  error?: string;
}) {
  const board = useContext(ProviderBoardContext);
  const [savingDefault, setSavingDefault] = useState("");
  const [defaultError, setDefaultError] = useState("");
  const availableRows = rows.filter((row) => row.available);
  const defaultBackend = (
    board?.entries.find((entry) => entry.name === CLOUD_BACKEND_KEY && entry.present)?.preview ?? ""
  ).toLowerCase();

  async function chooseDefault(backend: string) {
    if (!board || savingDefault) return;
    setSavingDefault(backend);
    setDefaultError("");
    try {
      await board.onSave(CLOUD_BACKEND_KEY, backend);
    } catch (cause) {
      setDefaultError(String((cause as Error).message || cause));
    } finally {
      setSavingDefault("");
    }
  }

  return (
    <section className="mb-4 min-w-0 rounded-md border border-hair bg-panel/40 p-4 sm:p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-display text-base font-semibold tracking-tight text-ink">
          Available compute
        </h3>
        <span className="font-mono text-2xs uppercase tracking-[0.14em] text-slate-500">
          {loading && !error ? "loading…" : `${availableRows.length} ready for a run`}
        </span>
      </div>
      <p className="mt-1 text-2xs leading-relaxed text-slate-500">
        Backends the New Run picker can offer right now. Cloud needs its API key;
        Cluster and Instance need a configured, verified SSH connection.
      </p>
      {error ? (
        <p className="mt-3 text-2xs text-coral-300">{error}</p>
      ) : loading ? (
        <p className="mt-3 font-mono text-2xs text-slate-500">Checking providers…</p>
      ) : availableRows.length === 0 ? (
        <p className="mt-3 text-2xs text-slate-500">
          No compute providers are available yet. Add a cloud API key or a
          verified Cluster or Instance SSH connection below.
        </p>
      ) : (
        <div className="mt-3 divide-y divide-hair/70">
          {availableRows.map((row) => {
            const cloudBackend = row.key.startsWith("cloud:") ? row.key.slice("cloud:".length) : "";
            const isDefault = Boolean(cloudBackend) && cloudBackend === defaultBackend;
            return (
              <div key={row.key} className="flex items-center justify-between gap-3 py-2.5 first:pt-1">
                <div className="flex min-w-0 items-center gap-2.5">
                  <span className="text-brass-400">
                    {row.icon === "cloud" ? <Cloud size={15} /> : <Server size={15} />}
                  </span>
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="truncate font-display text-sm font-semibold text-ink">{row.name}</span>
                      <span className="rounded bg-canvas px-1.5 py-px font-mono text-2xs uppercase tracking-[0.12em] text-slate-400">
                        {row.type}
                      </span>
                    </div>
                    <p className="mt-0.5 truncate font-mono text-2xs text-slate-500">
                      {row.detail}{row.note ? ` · ${row.note}` : ""}
                    </p>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {cloudBackend && board && (
                    isDefault ? (
                      <span className="inline-flex items-center gap-1 rounded border border-brass-500/40 bg-brass-500/10 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-brass-300">
                        <Check size={10} /> default
                      </span>
                    ) : (
                      <button
                        type="button"
                        onClick={() => void chooseDefault(cloudBackend)}
                        disabled={Boolean(savingDefault)}
                        className="btn !px-2 !py-1 !text-[11px] uppercase !tracking-[0.1em] disabled:opacity-40"
                      >
                        {savingDefault === cloudBackend ? "saving…" : "set default"}
                      </button>
                    )
                  )}
                  <ProviderStatusPill row={row} />
                </div>
              </div>
            );
          })}
          {defaultError && <p className="pt-2 text-2xs text-coral-300">{defaultError}</p>}
        </div>
      )}
    </section>
  );
}

export function SettingsSection({
  title, subtitle, singleLineSubtitle = false, children,
}: {
  title: string;
  subtitle: string;
  singleLineSubtitle?: boolean;
  children: ReactNode;
}) {
  return (
    <section className="border-t border-hair pt-6">
      <div className="mb-4">
        <h2 className="font-display text-lg font-semibold tracking-tight text-ink">{title}</h2>
        <p className={`mt-1 text-sm leading-relaxed text-slate-400 ${singleLineSubtitle ? "overflow-x-auto whitespace-nowrap" : "max-w-4xl"}`}>{subtitle}</p>
      </div>
      {children}
    </section>
  );
}


// --- The board itself --------------------------------------------------------

/** How a section's readiness chips are computed. */
type StatusKind = "drivers" | "integrations" | "none";

type BoardSection = {
  id: string;
  title: string;
  subtitle: string;
  singleLineSubtitle?: boolean;
  status: StatusKind;
  gridClass: string;
  /** GPU only: the section that hosts the compute panel / SSH card slots. */
  computeSlots?: boolean;
  groups: { id: string; title: string; subtitle: string; keys: string[] }[];
};

/**
 * Sections, cards, and card membership, in display order. Membership is by
 * explicit key name so a page can add a key to the registry above without
 * silently rearranging anyone's page — a host still has to list it in `keys`.
 */
export const BOARD_SECTIONS: BoardSection[] = [
  {
    id: "agent",
    title: "Agent API",
    subtitle: "Credentials used by the four supported Agent execution drivers.",
    status: "drivers",
    gridClass: "grid grid-cols-1 gap-4 xl:grid-cols-2",
    groups: [
      {
        id: "claude_cli",
        title: "Anthropic",
        subtitle: "Credential order: subscription token, subscription session, bearer token, then usage-billed API key. Lower-priority credentials are removed from the Claude child process.",
        keys: ["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
      },
      {
        id: "codex_cli",
        title: "OpenAI",
        subtitle: "Uses the mounted ChatGPT login from `codex login`; the API key is an optional usage-billed fallback.",
        keys: ["OPENAI_API_KEY"],
      },
      {
        id: "bedrock",
        title: "AWS Bedrock",
        subtitle: "Uses a Bedrock bearer API key together with the AWS region containing the selected model.",
        keys: ["AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"],
      },
      {
        id: "openrouter",
        title: "OpenRouter",
        subtitle: "One API key provides access to the models exposed by OpenRouter.",
        keys: ["OPENROUTER_API_KEY"],
      },
    ],
  },
  {
    id: "gpu",
    title: "GPU Providers",
    subtitle: "Cloud uses Vast.ai or Lambda.ai API keys · Cluster and Instance use SSH connections from Settings or .env.",
    singleLineSubtitle: true,
    status: "none",
    computeSlots: true,
    gridClass: "grid grid-cols-1 gap-4 xl:grid-cols-3",
    groups: [
      {
        id: "vastai",
        title: "Vast.ai",
        subtitle: "Rent on-demand cloud GPU instances through the Infrastructure Agent.",
        keys: ["VASTAI_API_KEY"],
      },
      {
        id: "lambda",
        title: "Lambda.ai",
        subtitle: "Rent Lambda Cloud GPU instances through the Infrastructure Agent.",
        keys: ["LAMBDA_API_KEY", "LAMBDA_SSH_KEY_NAME"],
      },
    ],
  },
  {
    id: "others",
    title: "Others",
    subtitle: "Optional external services used for model/data workflows and experiment observability.",
    status: "integrations",
    gridClass: "grid grid-cols-1 gap-4 xl:grid-cols-2",
    groups: [
      {
        id: "huggingface",
        title: "Hugging Face",
        subtitle: "Authenticate downloads of gated or private models and datasets from the Hugging Face Hub.",
        keys: ["HF_TOKEN"],
      },
      {
        id: "wandb",
        title: "Weights & Biases",
        subtitle: "Publish Train telemetry to a W&B project. Set entity, project, and API key together.",
        keys: ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_API_KEY"],
      },
    ],
  },
];

/** The four Agent drivers, in the order the readiness chips list them. */
const DRIVER_CHIPS: { driver: string; label: string; keys: string[] }[] = [
  { driver: "claude_cli", label: "Anthropic", keys: ["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"] },
  { driver: "codex_cli", label: "OpenAI", keys: ["OPENAI_API_KEY"] },
  { driver: "bedrock", label: "AWS Bedrock", keys: ["AWS_BEARER_TOKEN_BEDROCK"] },
  { driver: "openrouter", label: "OpenRouter", keys: ["OPENROUTER_API_KEY"] },
];

const isPresent = (entries: KeyEntry[], key: string) =>
  entries.some((entry) => entry.name === key && entry.present);

/**
 * Driver readiness chips. `drivers` (from an /auth-status endpoint that can see
 * the process env) wins when given; otherwise readiness is inferred from which
 * keys are stored, which is all a multi-tenant backend can honestly say.
 */
export function driverStatusItems(
  entries: KeyEntry[], drivers?: AuthDriver[],
): Array<{ label: string; ready: boolean }> {
  return DRIVER_CHIPS.map((chip) => ({
    label: chip.label,
    ready: drivers
      ? drivers.find((d) => d.driver === chip.driver)?.ready ?? false
      : chip.keys.some((key) => isPresent(entries, key)),
  }));
}

function integrationStatusItems(entries: KeyEntry[]) {
  return [
    { label: "Hugging Face", ready: isPresent(entries, "HF_TOKEN") },
    {
      label: "Weights & Biases",
      ready: ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_API_KEY"].every((key) => isPresent(entries, key)),
    },
  ];
}

export function ProviderKeyBoard({
  entries, onSave, onClear, drivers, keys, notice, compute, computeCard,
  showStatus = true, loading = false,
}: {
  /** Current state per key, as the host's settings endpoint reports it. */
  entries: KeyEntry[];
  onSave: (name: string, value: string) => Promise<void>;
  onClear: (name: string) => Promise<void>;
  /** Readiness chips; omit to derive them from which keys are present. */
  drivers?: AuthDriver[];
  /** Only render rows/cards for these key names (default: every known key). */
  keys?: string[];
  /** Rendered above the first section, e.g. a restart banner. */
  notice?: ReactNode;
  /** Full-width slot at the top of the GPU section (the compute roll-up). */
  compute?: ReactNode;
  /** Extra card in the GPU section's grid (the SSH connection list). */
  computeCard?: ReactNode;
  /** Set false when the host page renders one status row above several boards. */
  showStatus?: boolean;
  /** Hide the status chips until the host's data has arrived. */
  loading?: boolean;
}) {
  const allowed = keys ? new Set(keys) : null;
  const visible = (group: BoardSection["groups"][number]) =>
    specsFor(allowed ? group.keys.filter((key) => allowed.has(key)) : group.keys);

  const sections = BOARD_SECTIONS.map((section) => ({
    section,
    groups: section.groups
      .map((group): CredentialGroup => ({ ...group, specs: visible(group) }))
      .filter((group) => group.specs.length > 0),
  })).filter(({ section, groups }) =>
    groups.length > 0 || (section.computeSlots && (compute || computeCard)),
  );

  return (
    <ProviderBoardContext.Provider value={{ entries, onSave }}>
      <div className="space-y-6">
        {notice}
        {sections.map(({ section, groups }) => (
          <SettingsSection
            key={section.id}
            title={section.title}
            subtitle={section.subtitle}
            singleLineSubtitle={section.singleLineSubtitle}
          >
            {showStatus && section.status === "drivers" && (
              <SectionStatusRow items={driverStatusItems(entries, drivers)} loading={loading} />
            )}
            {showStatus && section.status === "integrations" && (
              <SectionStatusRow items={integrationStatusItems(entries)} loading={loading} />
            )}
            {section.computeSlots && compute}
            <div className={section.gridClass}>
              {groups.map((group) => (
                <CredentialCard
                  key={group.id}
                  group={group}
                  entries={entries}
                  onSave={onSave}
                  onClear={onClear}
                />
              ))}
              {section.computeSlots && computeCard}
            </div>
          </SettingsSection>
        ))}
      </div>
    </ProviderBoardContext.Provider>
  );
}
