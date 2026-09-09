import { useState } from "react";
import useSWR from "swr";
import {
  KeyRound, Eye, EyeOff, Check, RotateCw, AlertTriangle,
  ExternalLink, Copy, Cloud, Server, ShieldCheck, Clock, XCircle,
} from "lucide-react";
import { PageHead } from "../components/zevo/primitives";
import { SshConnections, type SshHost } from "../components/SshConnections";
import { api } from "../lib/api";

type SecretEntry = { name: string; present: boolean; preview: string };
type EnvironmentSshStatus = {
  id: string;
  label: string;
  configured: boolean;
  status: string;
  last_error: string;
  last_verified_at: string | null;
};
type SecretsResponse = {
  env_path: string;
  entries: SecretEntry[];
  ssh_connections: EnvironmentSshStatus[];
};
type AuthDriver = {
  driver: string; mode: "subscription" | "api_key" | "either" | "none";
  has_subscription: boolean; has_api_key: boolean; ready: boolean;
  effective: string;
};
type AuthResponse = { drivers: AuthDriver[] };

// One row per secret. Layout, copy, format hint, and CLI-vs-API
// distinction (Claude OAuth = "use the CLI to generate this") are
// per-key so the page doubles as a setup guide for new users.
const SECRETS: {
  key: string;
  label: string;
  driver: string;        // which driver this credential is for
  kind: "api_key" | "oauth_token" | "plain";
  format_hint: string;
  obtain: string;        // one-line "how do I get this"
  obtain_link?: string;  // optional URL
  obtain_cmd?: string;   // optional CLI command
}[] = [
  {
    key: "CLAUDE_CODE_OAUTH_TOKEN",
    label: "Claude Max OAuth token",
    driver: "claude_cli",
    kind: "oauth_token",
    format_hint: "starts with sk-ant-oat…, valid ~1 year",
    obtain: "Run `claude setup-token` on your laptop (browser-based)",
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


function CopyBtn({ text }: { text: string }) {
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


function SecretRow({
  spec, entry, onChanged,
}: {
  spec: typeof SECRETS[number];
  entry?: SecretEntry;
  onChanged: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(clear: boolean = false) {
    setBusy(true);
    setError(null);
    try {
      await api("/settings", {
        method: "POST",
        body: JSON.stringify({ values: { [spec.key]: clear ? "" : value.trim() } }),
      });
      setValue("");
      setEditing(false);
      onChanged();
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
                <input
                  type={spec.kind === "plain" || show ? "text" : "password"}
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  placeholder={spec.format_hint}
                  autoComplete="off"
                  spellCheck={false}
                  className="w-full rounded-md border border-hair bg-canvas px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-brass-500/50"
                />
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
                onClick={() => { setEditing(false); setValue(""); setError(null); }}
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


type CredentialGroup = {
  id: string;
  title: string;
  subtitle: string;
  specs: typeof SECRETS;
};

function CredentialCard({
  group, entries, onChanged,
}: {
  group: CredentialGroup;
  entries?: SecretEntry[];
  onChanged: () => void;
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
            onChanged={onChanged}
          />
        ))}
      </div>
    </section>
  );
}

function SectionStatusRow({
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
type ProviderRow = {
  key: string;
  icon: "cloud" | "server";
  name: string;
  type: string;
  detail: string;
  available: boolean;
  note: string;      // why it is / isn't offered in a run
  status?: string;   // optional verification status for SSH backends
};

function ProviderStatusPill({ row }: { row: ProviderRow }) {
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

// Read-only roll-up of every compute provider the New Run picker can offer, so
// the available backends are visible from Settings and not only mid-run.
function ComputeProvidersPanel({
  rows, loading = false,
}: {
  rows: ProviderRow[];
  loading?: boolean;
}) {
  const availableCount = rows.filter((row) => row.available).length;
  return (
    <section className="mb-4 min-w-0 rounded-md border border-hair bg-panel/40 p-4 sm:p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-display text-base font-semibold tracking-tight text-ink">
          Available compute
        </h3>
        <span className="font-mono text-2xs uppercase tracking-[0.14em] text-slate-500">
          {loading ? "loading…" : `${availableCount} of ${rows.length} ready for a run`}
        </span>
      </div>
      <p className="mt-1 text-2xs leading-relaxed text-slate-500">
        Backends the New Run picker can offer right now. Cloud needs its API key;
        Cluster and Instance need a configured, verified SSH connection.
      </p>
      {loading ? (
        <p className="mt-3 font-mono text-2xs text-slate-500">Checking providers…</p>
      ) : rows.length === 0 ? (
        <p className="mt-3 text-2xs text-slate-500">
          No compute providers configured yet. Add a cloud API key or an SSH
          connection below to make one available in a run.
        </p>
      ) : (
        <div className="mt-3 divide-y divide-hair/70">
          {rows.map((row) => (
            <div key={row.key} className="flex items-center justify-between gap-3 py-2.5 first:pt-1">
              <div className="flex min-w-0 items-center gap-2.5">
                <span className={row.available ? "text-brass-400" : "text-slate-600"}>
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
              <ProviderStatusPill row={row} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function SettingsSection({
  title, subtitle, singleLineSubtitle = false, children,
}: {
  title: string;
  subtitle: string;
  singleLineSubtitle?: boolean;
  children: React.ReactNode;
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


export function SettingsPage() {
  const { data, mutate } = useSWR<SecretsResponse>("/api/settings");
  const { data: authData, mutate: mutateAuth } = useSWR<AuthResponse>("/api/auth-status");
  const { data: sshHosts } = useSWR<SshHost[]>("/api/hardware/ssh");

  const driverGroups: CredentialGroup[] = [
    {
      id: "claude_cli",
      title: "Anthropic",
      subtitle: "Credential order: subscription token, subscription session, bearer token, then usage-billed API key. Lower-priority credentials are removed from the Claude child process.",
      specs: SECRETS.filter((s) => s.driver === "claude_cli"),
    },
    {
      id: "codex_cli",
      title: "OpenAI",
      subtitle: "Uses the mounted ChatGPT login from `codex login`; the API key is an optional usage-billed fallback.",
      specs: SECRETS.filter((s) => s.key === "OPENAI_API_KEY"),
    },
    {
      id: "bedrock",
      title: "AWS Bedrock",
      subtitle: "Uses a Bedrock bearer API key together with the AWS region containing the selected model.",
      specs: SECRETS.filter((s) => s.key.startsWith("AWS_")),
    },
    {
      id: "openrouter",
      title: "OpenRouter",
      subtitle: "One API key provides access to the models exposed by OpenRouter.",
      specs: SECRETS.filter((s) => s.key === "OPENROUTER_API_KEY"),
    },
  ];

  const gpuGroups: CredentialGroup[] = [
    {
      id: "vastai",
      title: "Vast.ai",
      subtitle: "Rent on-demand cloud GPU instances through the Infrastructure Agent.",
      specs: SECRETS.filter((s) => s.key === "VASTAI_API_KEY"),
    },
    {
      id: "lambda",
      title: "Lambda.ai",
      subtitle: "Rent Lambda Cloud GPU instances through the Infrastructure Agent.",
      specs: SECRETS.filter((s) => s.key === "LAMBDA_API_KEY"),
    },
  ];

  const integrationGroups: CredentialGroup[] = [
    {
      id: "huggingface",
      title: "Hugging Face",
      subtitle: "Authenticate downloads of gated or private models and datasets from the Hugging Face Hub.",
      specs: SECRETS.filter((s) => s.driver === "huggingface"),
    },
    {
      id: "wandb",
      title: "Weights & Biases",
      subtitle: "Publish Train telemetry to a W&B project. Set entity, project, and API key together.",
      specs: SECRETS.filter((s) => s.driver === "wandb"),
    },
  ];

  const refreshSettings = () => { void mutate(); void mutateAuth(); };
  const hasSecret = (key: string) => data?.entries.some(
    (entry) => entry.name === key && entry.present,
  ) ?? false;
  const driverStatus = (id: string) => authData?.drivers.find(
    (driver) => driver.driver === id,
  );
  const agentStatuses = [
    { label: "Anthropic", ready: driverStatus("claude_cli")?.ready ?? false },
    { label: "OpenAI", ready: driverStatus("codex_cli")?.ready ?? false },
    { label: "AWS Bedrock", ready: driverStatus("bedrock")?.ready ?? false },
    { label: "OpenRouter", ready: driverStatus("openrouter")?.ready ?? false },
  ];
  // The compute backends the New Run picker can offer, gated exactly as the
  // picker gates them: cloud by API-key presence, managed SSH by verification,
  // env SSH by being configured.
  const providerRows: ProviderRow[] = [
    {
      key: "cloud:vastai",
      icon: "cloud",
      name: "Vast.ai",
      type: "Cloud",
      detail: "On-demand cloud GPU rental",
      available: hasSecret("VASTAI_API_KEY"),
      note: hasSecret("VASTAI_API_KEY") ? "API key set" : "API key required",
    },
    {
      key: "cloud:lambda",
      icon: "cloud",
      name: "Lambda.ai",
      type: "Cloud",
      detail: "Lambda Cloud GPU rental",
      available: hasSecret("LAMBDA_API_KEY"),
      note: hasSecret("LAMBDA_API_KEY") ? "API key set" : "API key required",
    },
    ...(data?.ssh_connections ?? [])
      .filter((connection) => connection.configured)
      .map((connection): ProviderRow => ({
        key: `environment:${connection.id}`,
        icon: "server",
        name: connection.label,
        type: connection.id === "cluster" ? "Cluster" : "Instance",
        detail: "SSH connection from .env",
        available: true,
        note: connection.status === "verified" ? "verified" : "configured",
        status: connection.status,
      })),
    ...(sshHosts ?? []).map((host): ProviderRow => ({
      key: `connection:${host.id}`,
      icon: "server",
      name: host.name || host.host,
      type: host.category === "cluster" ? "Cluster" : "Instance",
      detail: `SSH · ${host.host}`,
      available: host.status === "verified",
      note:
        host.status === "verified" ? "verified"
        : host.status === "failed" ? "verification failed"
        : "not verified yet",
      status: host.status,
    })),
  ];
  const otherStatuses = [
    { label: "Hugging Face", ready: hasSecret("HF_TOKEN") },
    {
      label: "Weights & Biases",
      ready: ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_API_KEY"].every(hasSecret),
    },
  ];

  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Settings"
        subtitle="Agent drivers, GPU providers, and external integrations"
      />

      <div className="space-y-6">
        <div className="flex items-center rounded-md border border-brass-500/25 bg-brass-500/[0.06] px-4 py-3 text-brass-200">
          <span className="min-w-0 text-sm leading-relaxed">
            After saving, restart the containers to load the new values:
            <span className="mt-1 block max-w-full overflow-x-auto whitespace-nowrap font-mono text-xs font-semibold text-slate-200">
              docker compose up -d --force-recreate backend scheduler holdout-scheduler
            </span>
          </span>
        </div>

        <SettingsSection
          title="Agent API"
          subtitle="Credentials used by the four supported Agent execution drivers."
        >
          <SectionStatusRow items={agentStatuses} loading={authData === undefined} />
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            {driverGroups.map((group) => (
              <CredentialCard
                key={group.id}
                group={group}
                entries={data?.entries}
                onChanged={refreshSettings}
              />
            ))}
          </div>
        </SettingsSection>

        <SettingsSection
          title="GPU Providers"
          subtitle="Cloud uses Vast.ai or Lambda.ai API keys · Cluster and Instance use SSH connections from Settings or .env."
          singleLineSubtitle
        >
          <ComputeProvidersPanel
            rows={providerRows}
            loading={data === undefined || sshHosts === undefined}
          />
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
            {gpuGroups.map((group) => (
              <CredentialCard
                key={group.id}
                group={group}
                entries={data?.entries}
                onChanged={refreshSettings}
              />
            ))}
            <SshConnections embedded />
          </div>
        </SettingsSection>

        <SettingsSection
          title="Others"
          subtitle="Optional external services used for model/data workflows and experiment observability."
        >
          <SectionStatusRow items={otherStatuses} loading={data === undefined} />
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            {integrationGroups.map((group) => (
              <CredentialCard
                key={group.id}
                group={group}
                entries={data?.entries}
                onChanged={refreshSettings}
              />
            ))}
          </div>
        </SettingsSection>
      </div>
    </div>
  );
}
