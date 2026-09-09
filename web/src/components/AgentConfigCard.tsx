import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { Check, RotateCw, KeyRound, Sparkles, Zap, Crown,
  AlertTriangle, TerminalSquare, ShieldCheck,
} from "lucide-react";
import { DRIVERS, getDriver, isKnownModel, type ModelOption } from "../lib/modelCatalog";
import { api } from "../lib/api";
import { Bezel } from "./zevo/primitives";
import { ThemedSelect } from "./ThemedSelect";

type AuthStatusDriver = {
  driver: string;
  mode: "subscription" | "api_key" | "either" | "none";
  has_subscription: boolean;
  has_api_key: boolean;
  ready: boolean;
};
type AuthStatusResp = { drivers: AuthStatusDriver[] };

type AgentConfig = {
  default_driver: string;
  default_model: string;
  sandbox?: "none" | "openshell";
};

const CUSTOM_MODEL_SENTINEL = "__custom__";

function TagPill({ tag }: { tag: string }) {
  const color =
    tag === "newest" || tag === "default" ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300" :
    tag === "flagship" ? "border-violet-500/40 bg-violet-500/10 text-violet-300" :
    tag === "deep" ? "border-sky-500/40 bg-sky-500/10 text-sky-300" :
    "border-amber-500/40 bg-amber-500/10 text-amber-300";
  const Icon =
    tag === "newest" || tag === "default" ? Sparkles :
    tag === "flagship" ? Crown :
    tag === "fast" || tag === "cheap" ? Zap : Sparkles;
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-1 py-px text-[13px] uppercase tracking-wider ${color}`}>
      <Icon size={9} /> {tag}
    </span>
  );
}

function AuthBadge({ status }: { status?: AuthStatusDriver }) {
  if (!status) {
    return (
      <span className="inline-flex items-center gap-1 rounded border border-slate-700 bg-slate-800/50 px-1.5 py-0.5 text-[14px] uppercase tracking-wider text-slate-500">
        <RotateCw size={10} className="animate-spin" /> auth…
      </span>
    );
  }
  if (status.has_subscription) {
    const planLabel = status.driver === "codex_cli" ? "CHATGPT plan" : "MAX plan";
    return (
      <span title="Subscription login detected; model calls use that plan rather than the API-key fallback"
        className="inline-flex items-center gap-1 rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-[14px] uppercase tracking-wider text-emerald-300">
        <Crown size={10} /> {planLabel}
      </span>
    );
  }
  if (status.has_api_key) {
    return (
      <span title="Using an API key, so per-call billing applies"
        className="inline-flex items-center gap-1 rounded border border-sky-500/40 bg-sky-500/10 px-1.5 py-0.5 text-[14px] uppercase tracking-wider text-sky-300">
        <KeyRound size={10} /> API key
      </span>
    );
  }
  return (
    <span title="No credentials detected, so agent runs with this driver will fail"
      className="inline-flex items-center gap-1 rounded border border-rose-500/40 bg-rose-500/10 px-1.5 py-0.5 text-[14px] uppercase tracking-wider text-rose-300">
      <AlertTriangle size={10} /> no auth
    </span>
  );
}

function Money({ inputUsd, outputUsd }: { inputUsd?: number; outputUsd?: number }) {
  if (inputUsd == null && outputUsd == null) return null;
  const fmt = (n?: number) => (n == null ? "?" : n < 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(2)}`);
  return (
    <span className="font-mono text-[14px] text-slate-500">
      {fmt(inputUsd)} in / {fmt(outputUsd)} out per Mtok
    </span>
  );
}

/**
 * Per-agent driver + model picker.
 *
 * - Top section: provider/driver picker as a 2x2 grid of cards (more
 *   scannable than a long select). The selected card highlights and
 *   shows its auth badge (subscription vs API key).
 * - Middle: model dropdown scoped to the chosen driver. Each option
 *   carries tags (newest/flagship/fast), a per-Mtok cost preview, and
 *   a one-line note about when to pick it.
 * - "Custom…" sentinel at the bottom of the dropdown switches to a
 *   free-text input for models not in the catalog.
 *
 * Saves via PATCH /api/agents/{id}. Takes effect on the agent's
 * NEXT heartbeat — no scheduler restart.
 */
export function AgentConfigCard({
  agentId,
  current,
  onUpdated,
}: {
  agentId: string;
  current: AgentConfig;
  onUpdated: () => void;
}) {
  // The API returns the effective value: DB override first, then identity.md.
  const initialDriver = current.default_driver || "";
  const initialModel = current.default_model || "";

  const [driver, setDriver] = useState(initialDriver);
  const [model, setModel] = useState(initialModel);
  // Per-agent terminal: 'none' = regular in-container, 'openshell' = sandbox.
  const [sandbox, setSandbox] = useState(current.sandbox || "none");
  const [customMode, setCustomMode] = useState(
    initialModel ? !isKnownModel(initialDriver, initialModel) : false,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  // Real auth state from the backend — replaces the previous
  // heuristic. Cached for 60s so we don't refetch on every key press.
  const { data: authData } = useSWR<AuthStatusResp>("/api/auth-status", {
    refreshInterval: 60000,
  });
  const authForDriver = (id: string) => authData?.drivers.find((d) => d.driver === id);

  useEffect(() => {
    const d = current.default_driver || "";
    const m = current.default_model || "";
    setDriver(d);
    setModel(m);
    setSandbox(current.sandbox || "none");
    setCustomMode(m ? !isKnownModel(d, m) : false);
    setSaved(false);
    setError(null);
  }, [agentId, current.default_driver, current.default_model, current.sandbox]);

  const driverSpec = useMemo(() => getDriver(driver), [driver]);

  function handleDriverChange(next: string) {
    if (next === driver) return;
    setDriver(next);
    if (next !== "claude_cli") setSandbox("none");
    setSaved(false);
    const spec = getDriver(next);
    if (!spec) return;
    if (customMode) return;
    if (!isKnownModel(next, model)) setModel(spec.defaultModel);
  }

  function handleModelSelect(value: string) {
    setSaved(false);
    if (value === CUSTOM_MODEL_SENTINEL) {
      setCustomMode(true);
      setModel("");
      return;
    }
    setCustomMode(false);
    setModel(value);
  }

  const dirty =
    driver !== (current.default_driver || "") ||
    model !== (current.default_model || "") ||
    sandbox !== (current.sandbox || "none");
  const unset = !(current.default_driver && current.default_model);

  async function save() {
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      await api(`/agents/${encodeURIComponent(agentId)}`, {
        method: "PATCH",
        body: JSON.stringify({ default_driver: driver, default_model: model, sandbox }),
      });
      setSaved(true);
      onUpdated();
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  }

  const focusedModel: ModelOption | undefined =
    !customMode && driverSpec
      ? driverSpec.models.find((m) => m.id === model)
      : undefined;

  return (
    <Bezel className="p-5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="section-title">Configuration</h3>
        <AuthBadge status={authForDriver(driver)} />
      </div>

      {/* Only possible when neither runtime config nor identity.md defines it. */}
      {unset && (
        <div className="mb-3 flex items-start gap-1.5 rounded-md border border-rose-500/30 bg-rose-500/10 p-2 text-[15px] text-rose-300">
          <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
          <span>
            No driver/model set for this agent, so runs will fail until you pick one and save.
          </span>
        </div>
      )}

      {/* ─── Provider / driver picker as cards ─── */}
      <div className="mb-6">
        <label className="field-label">Driver</label>
        <div className="mt-1.5 grid grid-cols-2 gap-2">
          {DRIVERS.map((d) => {
            const active = d.id === driver;
            return (
              <button
                key={d.id}
                type="button"
                onClick={() => handleDriverChange(d.id)}
                className={`rounded-lg border px-3 py-2 text-left transition ${
                  active
                    ? "border-brass-500/60 bg-brass-500/10 ring-1 ring-brass-500/40"
                    : "border-hair bg-canvas/60 hover:border-slate-600"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-medium ${active ? "text-brass-200" : "text-slate-200"}`}>
                    {d.short}
                  </span>
                  {active && <Check size={14} className="text-brass-300" />}
                </div>
                <div className="mt-0.5 font-mono text-[14px] text-slate-500">{d.id}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* ─── Model picker (scoped to selected driver) ─── */}
      <div className="mb-6">
        <label className="field-label">Model</label>
        {customMode ? (
          <div className="mt-1 flex gap-1">
            <input
              value={model}
              onChange={(e) => { setModel(e.target.value); setSaved(false); }}
              placeholder="custom model id (must be accepted by the driver)"
              className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200"
            />
            <button
              onClick={() => {
                const spec = getDriver(driver);
                setCustomMode(false);
                setModel(spec?.defaultModel || "");
                setSaved(false);
              }}
              title="Pick from catalogued models"
              className="rounded border border-slate-700 px-2 text-[14px] uppercase tracking-wider text-slate-400 hover:bg-slate-800"
            >
              list
            </button>
          </div>
        ) : (
          <ThemedSelect
            value={isKnownModel(driver, model) ? model : (driverSpec?.defaultModel || "")}
            onChange={handleModelSelect}
            options={[
              ...(driverSpec?.models.map((m) => ({
                value: m.id,
                label: `${m.label}${m.tag ? `  ·  ${m.tag}` : ""}  ·  ${m.id}${
                  m.inputUsd != null && m.outputUsd != null
                    ? `   ($${m.inputUsd}/$${m.outputUsd} per Mtok)`
                    : ""
                }`,
              })) ?? []),
              { value: CUSTOM_MODEL_SENTINEL, label: "Custom… (type model id)" },
            ]}
            ariaLabel="Model"
            className="mt-1"
            buttonClassName="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200"
          />
        )}
      </div>

      {/* ─── Focused-model preview ─── */}
      {focusedModel && (
        <div className="mb-2 flex items-start gap-2 rounded-md border border-slate-800 bg-slate-950/40 p-2">
          <div className="flex-1">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-medium text-slate-200">{focusedModel.label}</span>
              {focusedModel.tag && <TagPill tag={focusedModel.tag} />}
              {focusedModel.contextK && (
                <span className="rounded border border-slate-700 px-1 py-px font-mono text-[13px] text-slate-400">
                  {focusedModel.contextK >= 1000 ? `${focusedModel.contextK / 1000}M` : `${focusedModel.contextK}K`} ctx
                </span>
              )}
            </div>
            {focusedModel.note && (
              <p className="mt-0.5 text-[15px] text-slate-400">{focusedModel.note}</p>
            )}
          </div>
          <Money inputUsd={focusedModel.inputUsd} outputUsd={focusedModel.outputUsd} />
        </div>
      )}

      {/* ─── Terminal (execution sandbox) toggle ─── */}
      <div className="mb-6">
        <label className="field-label">Terminal</label>
        <div className="mt-1.5 grid grid-cols-2 gap-2">
          {([
            { id: "none", label: "Regular terminal", Icon: TerminalSquare },
            ...(driver === "claude_cli" && agentId === "orchestrator"
              ? [{ id: "openshell" as const, label: "OpenShell sandbox", Icon: ShieldCheck }]
              : []),
          ] as const).map((opt) => {
            const active = sandbox === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => { setSandbox(opt.id); setSaved(false); }}
                className={`rounded-lg border px-3 py-2 text-left transition ${
                  active
                    ? "border-brass-500/60 bg-brass-500/10 ring-1 ring-brass-500/40"
                    : "border-hair bg-canvas/60 hover:border-slate-600"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`flex items-center gap-1.5 text-sm font-medium ${active ? "text-brass-200" : "text-slate-200"}`}>
                    <opt.Icon size={13} /> {opt.label}
                  </span>
                  {active && <Check size={14} className="text-brass-300" />}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {error && (
        <div className="mb-2 rounded-md border border-rose-500/30 bg-rose-500/10 p-2 text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* ─── Save ─── */}
      <div className="flex items-center justify-between gap-2">
        {unset ? (
          <div className="text-[15px] text-rose-300/80">
            Pick a provider + model, then <strong>save</strong> to configure this agent.
          </div>
        ) : (
          <span />
        )}
        <button
          onClick={save}
          disabled={busy || !dirty || !model || !driver}
          className="btn btn-brass flex-shrink-0 uppercase tracking-wider disabled:opacity-40"
        >
          {busy ? (<><RotateCw size={12} className="animate-spin" /> saving…</>)
            : saved ? (<><Check size={12} /> saved</>)
            : "save"}
        </button>
      </div>
    </Bezel>
  );
}
