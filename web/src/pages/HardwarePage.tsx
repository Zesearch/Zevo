import { useEffect, useState } from "react";
import useSWR from "swr";
import { Cpu, Search, Server, Trash2 } from "lucide-react";
import { Bezel, Kicker, PageHead } from "../components/zevo/primitives";
import { StatusBadge } from "../components/StatusBadge";
import { api } from "../lib/api";

type LocalGPU = {
  has_gpu: boolean;
  gpu_count: number;
  gpu_name: string;
  vram_gb: number;
  driver_version: string;
  cuda_version: string;
  source: string;
  error: string;
};

type CloudBackend = { id: string; label: string; configured: boolean };

type CloudOffer = {
  backend: string;       // which cloud this row came from
  backendLabel: string;
  id: string;
  gpu_name: string;
  num_gpus: number;
  gpu_ram_gb: number;
  dph_total: number;
  cpu_cores: number;
  ram_gb: number;
  disk_space_gb: number;
  reliability: number;
  location: string;
  dlperf: number;
  region: string;
};

type Instance = {
  backend: string;
  backendLabel: string;
  id: string;
  gpu_name: string;
  actual_status: string;
  dph_total: number;
  ssh_host: string;
  ssh_port: number;
};

export function HardwarePage() {
  const [local, setLocal] = useState<LocalGPU | null>(null);
  const [localBusy, setLocalBusy] = useState(false);

  // Which cloud to search and rent from. The harness supports either, so the
  // page does too — otherwise it can only answer half the question.
  const { data: backendData } = useSWR<{ backends: CloudBackend[] }>("/api/hardware/cloud/backends");
  const backends = backendData?.backends ?? [];
  // "" = every configured cloud. Comparing prices across providers is the
  // question this page exists to answer, so all-at-once is the default and the
  // per-cloud pills narrow it.
  const [scope, setScope] = useState("");
  const inScope = backends.filter((b) => b.configured && (!scope || b.id === scope));
  const scopeLabel = scope ? backends.find((b) => b.id === scope)?.label ?? scope : "any cloud";

  const [search, setSearch] = useState({ min_vram_gb: 12, gpu_type: "", limit: 8 });
  const [offers, setOffers] = useState<CloudOffer[] | null>(null);
  const [searchErr, setSearchErr] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);

  const [instances, setInstances] = useState<Instance[] | null>(null);
  const [instancesErr, setInstancesErr] = useState<string | null>(null);
  // Bumped after a successful rent/destroy so the rentals table re-fetches.
  const [rentalsNonce, setRentalsNonce] = useState(0);
  const [rentErr, setRentErr] = useState<string | null>(null);
  const [renting, setRenting] = useState("");

  // Rentals are fetched per cloud and merged. Only configured clouds are asked
  // — an unset key 503s, and polling that just floods the console.
  useEffect(() => {
    if (!inScope.length) { setInstances(null); setInstancesErr(null); return; }
    let cancelled = false;
    (async () => {
      const errs: string[] = [];
      const rows: Instance[] = [];
      await Promise.all(inScope.map(async (b) => {
        try {
          const d = await api<{ instances?: Instance[] }>(`/hardware/cloud/${b.id}/instances`);
          for (const i of d.instances ?? []) rows.push({ ...i, backend: b.id, backendLabel: b.label });
        } catch (e) {
          errs.push(`${b.label}: ${String((e as Error).message || e)}`);
        }
      }));
      if (cancelled) return;
      setInstances(rows);
      setInstancesErr(errs.length ? errs.join(" · ") : null);
    })();
    return () => { cancelled = true; };
  }, [inScope.map((b) => b.id).join(","), backends.length, rentalsNonce]);

  async function detectLocal() {
    setLocalBusy(true);
    try {
      setLocal(await api<LocalGPU>("/hardware/detect-local", { method: "POST" }));
    } catch {
      /* the probe endpoint reports its own errors in the LocalGPU payload */
    } finally {
      setLocalBusy(false);
    }
  }

  async function runSearch() {
    setSearching(true);
    setSearchErr(null);
    // Every cloud in scope is queried in parallel and the offers merged, so the
    // cheapest machine wins regardless of who is renting it.
    const errs: string[] = [];
    const rows: CloudOffer[] = [];
    await Promise.all(inScope.map(async (b) => {
      try {
        const found = await api<CloudOffer[]>(`/hardware/cloud/${b.id}/search`, {
          method: "POST",
          body: JSON.stringify(search),
        });
        for (const o of found) rows.push({ ...o, backend: b.id, backendLabel: b.label });
      } catch (e) {
        errs.push(`${b.label}: ${String((e as Error).message || e)}`);
      }
    }));
    rows.sort((a, b) => a.dph_total - b.dph_total);
    setOffers(rows);
    setSearchErr(errs.length ? errs.join(" · ") : null);
    setSearching(false);
  }

  async function destroyInstance(i: Instance) {
    if (!confirm(`Destroy ${i.backendLabel} instance ${i.id}? This stops billing.`)) return;
    setRentErr(null);
    try {
      await api(
        `/hardware/cloud/${i.backend}/instances/${encodeURIComponent(i.id)}`,
        { method: "DELETE" },
      );
      setInstances((prev) => (prev ?? []).filter((x) => !(x.id === i.id && x.backend === i.backend)));
      setRentalsNonce((n) => n + 1);
    } catch (e) {
      setRentErr(String((e as Error).message || e));
    }
  }

  async function rentOffer(o: CloudOffer) {
    if (!confirm(`Rent ${o.gpu_name} on ${o.backendLabel} at $${o.dph_total.toFixed(3)}/hr?`)) return;
    setRenting(`${o.backend}:${o.id}`);
    setRentErr(null);
    try {
      await api(`/hardware/cloud/${o.backend}/rent`, {
        method: "POST",
        body: JSON.stringify({ offer_id: o.id, region: o.region }),
      });
      // The new machine belongs in the rentals table below, without a reload.
      setRentalsNonce((n) => n + 1);
    } catch (e) {
      setRentErr(String((e as Error).message || e));
    } finally {
      setRenting("");
    }
  }

  return (
    <div className="w-full px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Hardware"
        subtitle="Local and cloud GPUs available to Zevo"
      />

      {/* ── Local GPU ── */}
      <Bezel className="mb-5">
        <div className="flex items-center justify-between border-b border-hair px-5 py-4">
          <div className="flex items-center gap-2">
            <Cpu size={15} className="text-slate-500" />
            <Kicker strong className="!text-sm">Local GPU</Kicker>
          </div>
          <button onClick={detectLocal} disabled={localBusy} className="btn disabled:opacity-50">
            {localBusy ? "probing…" : "detect"}
          </button>
        </div>
        {local ? (
          <div className="p-5">
            <div className="mb-4 flex items-center gap-2.5">
              <span className={`lamp ${local.has_gpu ? "lamp-live" : "lamp-coral"}`} />
              <span className="font-display text-lg font-semibold text-ink">
                {local.has_gpu ? "GPU detected" : "No GPU detected"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Readout label="Name" value={local.gpu_name || "—"} />
              <Readout label="VRAM" value={local.vram_gb ? `${local.vram_gb}` : "—"} unit={local.vram_gb ? "GB" : undefined} tone={local.vram_gb ? "phosphor" : "ink"} />
              <Readout label="Driver" value={local.driver_version || "—"} />
              <Readout label="CUDA" value={local.cuda_version || "—"} />
              <Readout label="GPU count" value={local.gpu_count || "—"} />
              <Readout label="Source" value={local.source || "—"} />
            </div>
            {local.error && (
              <div className="mt-4 rounded-md border border-coral-500/30 bg-coral-500/10 p-3 font-mono text-2xs text-coral-300">
                {local.error}
              </div>
            )}
          </div>
        ) : (
          <div className="p-5 text-sm text-slate-500">Press detect to probe the host for an NVIDIA GPU.</div>
        )}
      </Bezel>

      {/* ── Cloud marketplace ── */}
      <Bezel className="mb-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hair px-5 py-4">
          <div className="flex items-center gap-2">
            <Search size={15} className="text-slate-500" />
            <Kicker strong className="!text-sm">Cloud marketplace</Kicker>
          </div>
          {/* An unconfigured cloud stays listed: the point is to show it exists
              and needs a key, not to hide it. */}
          <div className="flex items-center gap-1 rounded-md border border-hair bg-canvas p-1">
            <button
              onClick={() => { setScope(""); setOffers(null); setSearchErr(null); }}
              className={`rounded px-2.5 py-1 font-mono text-2xs transition ${
                scope === "" ? "bg-brass-500/15 text-brass-300" : "text-slate-500 hover:text-slate-300"
              }`}
            >
              all clouds
            </button>
            {backends.map((b) => (
              <button
                key={b.id}
                onClick={() => { setScope(b.id); setOffers(null); setSearchErr(null); }}
                title={b.configured ? `${b.label} API key is set` : `${b.label} API key is not set`}
                className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-mono text-2xs transition ${
                  b.id === scope ? "bg-brass-500/15 text-brass-300" : "text-slate-500 hover:text-slate-300"
                }`}
              >
                <span className={`lamp ${b.configured ? "lamp-live" : "lamp-coral"}`} />
                {b.label}
              </button>
            ))}
          </div>
        </div>
        <div className="p-5">
          <div className="bezel-flat grid grid-cols-1 gap-3 p-4 sm:grid-cols-4">
            <Field label="min VRAM (GB)">
              <input
                type="number"
                value={search.min_vram_gb}
                onChange={(e) => setSearch({ ...search, min_vram_gb: Number(e.target.value) })}
                className={inputCls}
              />
            </Field>
            <Field label="GPU type (optional)">
              <input
                type="text"
                placeholder="e.g. RTX 3090"
                value={search.gpu_type}
                onChange={(e) => setSearch({ ...search, gpu_type: e.target.value })}
                className={inputCls}
              />
            </Field>
            <Field label="limit">
              <input
                type="number"
                value={search.limit}
                onChange={(e) => setSearch({ ...search, limit: Number(e.target.value) })}
                className={inputCls}
              />
            </Field>
            <div className="flex items-end">
              <button onClick={runSearch} disabled={searching} className="btn btn-brass w-full justify-center disabled:opacity-50">
                {searching ? "searching…" : "search"}
              </button>
            </div>
          </div>

          {searchErr && (
            <div className="mt-4 rounded-md border border-coral-500/30 bg-coral-500/10 p-3 font-mono text-2xs text-coral-300">
              {searchErr}
            </div>
          )}

          {rentErr && (
            <div className="mt-4 rounded-md border border-coral-500/30 bg-coral-500/10 p-3 font-mono text-2xs text-coral-300">
              {rentErr}
            </div>
          )}

          {offers && offers.length === 0 && !searchErr && (
            <div className="mt-4 text-sm text-slate-500">
              No offers matched on {scopeLabel}.
            </div>
          )}

          {offers && offers.length > 0 && (
            <div className="mt-4 overflow-x-auto rounded-bezel border border-hair">
              <table className="w-full text-xs">
                <thead className="table-label bg-raised/60 text-left">
                  <tr>
                    <th className="px-3 py-2.5 font-medium">cloud</th>
                    <th className="px-3 py-2.5 font-medium">offer</th>
                    <th className="px-3 py-2.5 font-medium">gpu</th>
                    <th className="px-3 py-2.5 font-medium">vram</th>
                    <th className="px-3 py-2.5 font-medium">$/hr</th>
                    <th className="px-3 py-2.5 font-medium">CPU</th>
                    <th className="px-3 py-2.5 font-medium">reliability</th>
                    <th className="px-3 py-2.5 font-medium">location</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-hair font-mono">
                  {offers.map((o) => (
                    <tr key={`${o.backend}:${o.id}`} className="transition hover:bg-raised">
                      <td className="px-3 py-2.5 text-skyx-300">{o.backendLabel}</td>
                      <td className="px-3 py-2.5 text-slate-500">{o.id}</td>
                      <td className="px-3 py-2.5 text-slate-200">{o.num_gpus}× {o.gpu_name}</td>
                      <td className="px-3 py-2.5 text-phosphor-300">{o.gpu_ram_gb} GB</td>
                      <td className="px-3 py-2.5 text-brass-300">${o.dph_total.toFixed(3)}</td>
                      <td className="px-3 py-2.5 text-slate-400">{o.cpu_cores}c / {o.ram_gb}GB</td>
                      <td className="px-3 py-2.5 text-slate-300">{(o.reliability * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2.5 text-slate-500">{o.location}</td>
                      <td className="px-3 py-2.5 text-right">
                        <button
                          onClick={() => void rentOffer(o)}
                          disabled={!!renting}
                          className="btn !py-1 text-2xs !text-phosphor-300 disabled:opacity-50"
                        >
                          {renting === `${o.backend}:${o.id}` ? "renting…" : "rent"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Bezel>

      {/* ── Current rentals ── */}
      <Bezel>
        <div className="flex items-center gap-2 border-b border-hair px-5 py-4">
          <Server size={15} className="text-slate-500" />
          <Kicker strong className="!text-sm">Active rentals</Kicker>
        </div>
        <div className="p-5">
          {instancesErr && (
            <div className="mb-3 rounded-md border border-coral-500/30 bg-coral-500/10 p-3 font-mono text-2xs text-coral-300">
              {instancesErr}
            </div>
          )}
          {!instances || instances.length === 0 ? (
            <div className="text-sm text-slate-500">No active rentals on {scopeLabel}.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="table-label text-left">
                  <tr>
                    <th className="py-2 pr-3 font-medium">cloud</th>
                    <th className="pr-3 font-medium">id</th>
                    <th className="pr-3 font-medium">gpu</th>
                    <th className="pr-3 font-medium">status</th>
                    <th className="pr-3 font-medium">$/hr</th>
                    <th className="pr-3 font-medium">ssh</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-hair font-mono">
                  {instances.map((i) => (
                    <tr key={`${i.backend}:${i.id}`}>
                      <td className="py-2.5 pr-3 text-skyx-300">{i.backendLabel}</td>
                      <td className="pr-3 text-slate-300">{i.id}</td>
                      <td className="pr-3 text-slate-200">{i.gpu_name}</td>
                      <td className="pr-3"><StatusBadge status={i.actual_status} /></td>
                      <td className="pr-3 text-brass-300">${i.dph_total.toFixed(3)}</td>
                      <td className="pr-3 text-slate-500">{i.ssh_host}:{i.ssh_port}</td>
                      <td className="text-right">
                        <button
                          onClick={() => destroyInstance(i)}
                          className="btn !py-1 text-2xs !text-coral-300"
                        >
                          <Trash2 size={11} /> destroy
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Bezel>
    </div>
  );
}

const inputCls =
  "w-full rounded-md border border-hair bg-canvas px-2.5 py-1.5 font-mono text-sm text-slate-200 focus:border-brass-500/50";

/** Compact instrument readout for local-GPU facts (small variant of the primitive). */
function Readout({
  label, value, unit, tone = "ink",
}: {
  label: string; value: React.ReactNode; unit?: string; tone?: "ink" | "phosphor";
}) {
  const toneCls = tone === "phosphor" ? "text-phosphor-300" : "text-ink";
  return (
    <div>
      <Kicker>{label}</Kicker>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span className={`readout truncate text-lg font-semibold ${toneCls}`} title={typeof value === "string" ? value : undefined}>
          {value}
        </span>
        {unit && <span className="readout text-2xs text-slate-500">{unit}</span>}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5"><Kicker>{label}</Kicker></div>
      {children}
    </div>
  );
}
