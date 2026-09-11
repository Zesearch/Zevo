/**
 * Settings — provider credentials, GPU backends, and integrations.
 *
 * The presentation lives in `components/settings/ProviderKeyBoard`, shared with
 * the hosted console so both products' Settings pages stay identical. This page
 * only binds that board to the single-tenant data: `/api/settings` for the keys
 * themselves, `/api/auth-status` for driver readiness, and the SSH inventory
 * that fills the GPU section's compute slots.
 */
import useSWR from "swr";
import { PageHead } from "../components/zevo/primitives";
import { SshConnections, type SshHost } from "../components/SshConnections";
import {
  ProviderKeyBoard, ComputeProvidersPanel,
  type AuthDriver, type KeyEntry, type ProviderRow,
} from "../components/settings/ProviderKeyBoard";
import { api } from "../lib/api";

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
  entries: KeyEntry[];
  ssh_connections: EnvironmentSshStatus[];
};
type AuthResponse = { drivers: AuthDriver[] };

// The credentials this deployment's `/api/settings` writes into .env. Keys the
// shared registry knows but this product does not expose stay hidden.
const SETTINGS_KEYS = [
  "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
  "OPENAI_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION",
  "OPENROUTER_API_KEY", "VASTAI_API_KEY", "LAMBDA_API_KEY",
  "HF_TOKEN", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_API_KEY",
];

// Saving writes .env; the running containers keep their old env until recreated.
function RestartNotice() {
  return (
    <div className="flex items-center rounded-md border border-brass-500/25 bg-brass-500/[0.06] px-4 py-3 text-brass-200">
      <span className="min-w-0 text-sm leading-relaxed">
        Credential changes require a restart. A Default compute change applies
        to the next Run immediately.
        <span className="mt-1 block max-w-full overflow-x-auto whitespace-nowrap font-mono text-xs font-semibold text-slate-200">
          docker compose up -d --force-recreate backend scheduler holdout-scheduler
        </span>
      </span>
    </div>
  );
}

export function SettingsPage() {
  const { data, error: settingsError, mutate } = useSWR<SecretsResponse>("/api/settings");
  const { data: authData, mutate: mutateAuth } = useSWR<AuthResponse>("/api/auth-status");
  const { data: sshHosts, error: sshError } = useSWR<SshHost[]>("/api/hardware/ssh");

  const entries = data?.entries ?? [];
  const write = async (name: string, value: string) => {
    await api("/settings", {
      method: "POST",
      body: JSON.stringify({ values: { [name]: value } }),
    });
    void mutate();
    void mutateAuth();
  };

  const hasSecret = (key: string) => entries.some(
    (entry) => entry.name === key && entry.present,
  );
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

  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Settings"
        subtitle="Agent drivers, GPU providers, and external integrations"
      />

      <ProviderKeyBoard
        entries={entries}
        keys={SETTINGS_KEYS}
        drivers={authData?.drivers}
        loading={data === undefined || authData === undefined}
        onSave={write}
        onClear={(name) => write(name, "")}
        notice={<RestartNotice />}
        compute={
          <ComputeProvidersPanel
            rows={providerRows}
            loading={data === undefined || sshHosts === undefined}
            defaultSettingKey="ZEVO_DEFAULT_COMPUTE"
            defaultValueForRow={(row) => row.key}
            error={
              settingsError ? "Could not load Settings. Check the backend and try again."
              : sshError ? "Could not load SSH connections. Check the backend and try again."
              : undefined
            }
          />
        }
        computeCard={<SshConnections embedded />}
      />
    </div>
  );
}
