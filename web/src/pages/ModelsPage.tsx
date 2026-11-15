import { useMemo, useState } from "react";
import useSWR from "swr";
import { Link, useSearchParams } from "react-router-dom";
import {
  X, Box, FileText, ArrowLeftRight, AlertTriangle, Search,
  TrendingUp, TrendingDown, Minus,
  FolderOpen, UploadCloud, Copy, Check,
} from "lucide-react";
import type { ModelDTO } from "../lib/api";
import { fmtScore, fmtMetric, fmtDate, isPercentageMetric, shortModel } from "../lib/format";
import { PageHead, Bezel, Gauge, Kicker } from "../components/zevo/primitives";
import { PAGE_SIZES, Pager } from "../components/zevo/Pager";
import { useRowsPerPage } from "../lib/useRowsPerPage";
import { Markdown } from "../components/Markdown";

type CompareResponse = {
  a_tag: string;
  b_tag: string;
  fields: Record<string, { a: unknown; b: unknown; differs: boolean }>;
  champion_test_score_delta: number | null;
  headline_summary: string;
};

type DrawerMode = "card" | "compare";

// One model card plus the gap below it; measured from the rendered grid.
const CARD_PX = 300;
// How many a row holds at the widest breakpoint (see the grid classes).
const CARD_COLS = 3;


export function ModelsPage() {
  const { data: allModels = [], error: modelsError, isLoading, mutate } = useSWR<ModelDTO[]>("/api/models");
  // The model endpoint owns catalog membership; a paginated run list cannot prove absence.
  const kept = allModels;

  // `?q=` like Runs and Tasks, so a filtered registry is a link you can send.
  const [params, setParams] = useSearchParams();
  const query = params.get("q") ?? "";
  const setQuery = (v: string) => {
    const next = new URLSearchParams(params);
    if (v) next.set("q", v); else next.delete("q");
    setParams(next, { replace: true });
  };
  // Everything the card puts on screen is searchable, plus the version tag it
  // hides — searching for what you were shown should never come back empty.
  const q = query.trim().toLowerCase();
  const models = useMemo(
    () => (!q ? kept : kept.filter((m) => [
      modelName(m), m.version_tag, m.run_id ?? "", m.task_name ?? "",
      m.base_model, m.training_method,
    ].some((f) => f.toLowerCase().includes(q)))),
    [kept, q],
  );

  // Every surface on this page names a model the same way, so the compare
  // button and the drawer title do not fall back to the version string.
  const nameOf = useMemo(() => {
    const byTag = new Map(allModels.map((m) => [m.version_tag, modelName(m)]));
    return (tag: string) => byTag.get(tag) ?? tag;
  }, [allModels]);

  // Paged like Tasks and Files: the catalogue grows a model per iteration, so
  // a registry that started at a dozen cards ends at a hundred.
  const [page, setPage] = useState(0);
  const [pageSizePref, setPageSizePref] = useState(PAGE_SIZES[0]);
  const [listEl, setListEl] = useState<HTMLDivElement | null>(null);
  // Model cards are the page's primary catalogue, so Auto keeps two visible
  // rows like Files instead of collapsing to a single three-card strip on the
  // normal dashboard viewport. Taller pages can still resolve to more rows.
  const fits = useRowsPerPage(listEl, CARD_PX, { min: 2, max: 40 }) * CARD_COLS;
  const pageSize = pageSizePref || fits;

  const [drawer, setDrawer] = useState<{ mode: DrawerMode; tag: string; otherTag?: string } | null>(null);
  const [compareSelection, setCompareSelection] = useState<string[]>([]);
  const total = models.length;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const current = Math.min(page, pageCount - 1);
  const shown = models.slice(current * pageSize, current * pageSize + pageSize);
  const first = total === 0 ? 0 : current * pageSize + 1;
  const last = Math.min(total, (current + 1) * pageSize);
  // Keep the first visible card visible when the page size changes.
  const resize = (n: number) => {
    setPage(Math.floor((current * pageSize) / (n || fits)));
    setPageSizePref(n);
  };
  // The page's two fetches are its only failure paths; surface whichever broke.
  const fetchError = modelsError;
  const error = fetchError ? String((fetchError as Error).message || fetchError) : null;

  function toggleCompare(tag: string) {
    setCompareSelection((s) =>
      s.includes(tag) ? s.filter((t) => t !== tag)
                      : (s.length >= 2 ? [s[1], tag] : [...s, tag])
    );
  }

  return (
    <div className="flex w-full flex-1 flex-col px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Models"
        subtitle="Saved best models trained through Zevo per run"
        right={
          compareSelection.length === 2 ? (
            <button
              onClick={() => setDrawer({ mode: "compare", tag: compareSelection[0], otherTag: compareSelection[1] })}
              className="btn"
              style={{ borderColor: "rgba(169,143,214,0.4)", color: "#C3B0E8" }}
            >
              <ArrowLeftRight size={13} /> compare {compareSelection.map(nameOf).join(" vs ")}
            </button>
          ) : (
            <span className="font-mono text-2xs text-slate-500">
              {compareSelection.length === 1 ? "select 1 more to compare" : "select 2 to compare"}
            </span>
          )
        }
      />

      {error && (
        <div className="mb-5 flex items-center gap-2 rounded-bezel border border-coral-500/30 bg-coral-500/10 px-3 py-2.5 text-xs text-coral-300">
          <AlertTriangle size={13} /> {error} <button className="btn" onClick={() => void mutate()}>Retry</button>
        </div>
      )}

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1 sm:max-w-md">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="search model, run, task, base or method"
            spellCheck={false}
            className="w-full rounded-md border border-hair bg-canvas py-2 pl-9 pr-8 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-brass-500/50"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              title="Clear"
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-500 transition hover:text-slate-200"
            >
              <X size={13} />
            </button>
          )}
        </div>
        {q && (
          <span className="font-mono text-2xs text-slate-500">
            {models.length} match{models.length === 1 ? "" : "es"}
          </span>
        )}
      </div>

      {isLoading ? <p role="status">Loading saved models…</p> : error && allModels.length === 0 ? null : models.length === 0 ? (
        <div className="rounded-bezel border border-dashed border-hair p-16 text-center text-sm text-slate-500">
          {q ? `No model matches "${query}".`
             : "No models registered yet."}
        </div>
      ) : (
        <div ref={setListEl} className="stagger grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
          {shown.map((m) => {
            // The dial is the HELD-OUT score: what this model is worth on data
            // nothing in the run could tune against. `eval.score` is the
            // validation number the registry agent was handed — it has no path
            // to the test set, by design — and showing it here reported the
            // number the loop optimized as if it were the result.
            // Only the held-out number. `eval.score` is the VALIDATION score
            // the registry agent was handed — the set the loop optimized — and
            // a page about finished models has no use for it: it says how well
            // the run fitted its own yardstick, not what the model is worth.
            const score = m.champion_test_score;
            const selected = compareSelection.includes(m.version_tag);
            return (
              <Bezel
                key={m.version_tag}
                className={`flex flex-col gap-5 p-5 transition ${selected ? "ring-1 ring-lilac-500/50 shadow-glow-brass" : ""}`}
              >
                {/* Header: tag + compare toggle */}
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    {/* The model name opens its run; the document icon opens its model card. */}
                    <div className="flex min-w-0 items-center gap-1.5">
                      <Box size={14} className="shrink-0 text-brass-400/70" />
                      {m.run_id ? <Link to={`/runs/${m.run_id}`} className="readout truncate text-sm font-semibold text-ink hover:text-brass-300" title="Open producing run">
                        {modelName(m)}
                      </Link> : <span className="readout truncate text-sm font-semibold text-ink">{modelName(m)}</span>}
                      <button
                        onClick={() => setDrawer({ mode: "card", tag: m.version_tag })}
                        className="shrink-0 rounded p-0.5 text-slate-500 transition hover:text-brass-300"
                        title="model card"
                      >
                        <FileText size={13} />
                      </button>
                    </div>
                  </div>
                  <label
                    className={`flex shrink-0 cursor-pointer items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] transition ${
                      selected
                        ? "border-lilac-500/40 bg-lilac-500/10 text-lilac-300"
                        : "border-hair text-slate-500 hover:border-lilac-500/40 hover:text-lilac-300"
                    }`}
                    title="select up to 2 to compare"
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleCompare(m.version_tag)}
                      className="sr-only"
                    />
                    <ArrowLeftRight size={11} /> compare
                  </label>
                </div>

                {/* Body: the paired held-out outcomes + labelled fields. */}
                <div className="flex items-center gap-4">
                  <div className="w-28 shrink-0 text-center">
                    <div
                      title={score == null
                        ? "the held-out measurement for this iteration did not complete"
                        : m.baseline_test_score == null
                          ? undefined
                          : `Held-out baseline: ${fmtScore(m.baseline_test_score, m.metric)}`}
                    >
                      <div className="font-mono text-[0.62rem] uppercase tracking-[0.14em] text-slate-500">
                        Test score
                      </div>
                      <Gauge
                        value={score ?? 0}
                        display={fmtScore(score, m.metric)}
                        tone="phosphor"
                        size={104}
                        displayScale={0.15}
                        bounded={isPercentageMetric(m.metric)}
                      />
                    </div>
                    <div className="mt-2 border-t border-hair pt-2">
                      <div className="font-mono text-[0.62rem] uppercase tracking-[0.14em] text-slate-500">
                        Improvement
                      </div>
                      <div className={`mt-0.5 font-mono text-base ${
                        (m.improvement ?? 0) < 0 ? "text-coral-300" : "text-phosphor-300"
                      }`}>
                        {m.improvement == null
                          ? "—"
                          : `${m.improvement >= 0 ? "+" : ""}${fmtScore(m.improvement, m.metric)}`}
                      </div>
                    </div>
                  </div>
                  {/* Which run made it, when, and what it was made from. The
                      rest of the recipe is a click away in the model card, and
                      lived here only by naming things twice. */}
                  <div className="grid min-w-0 flex-1 grid-cols-2 gap-x-4 gap-y-2.5">
                    <Field label="run" value={m.run_id ? m.run_id.slice(0, 8) : "—"} mono
                      to={m.run_id ? `/runs/${m.run_id}` : undefined} title={m.run_id || ""} />
                    <Field label="created" value={fmtDate(m.registered_at)} />
                    {/* Custom-objective runs have no task; the row still shows,
                        with a dash, so the grid reads the same on every card. */}
                    <Field label="task" value={m.task_name || "—"} mono
                      to={m.task_name ? `/tasks?q=${encodeURIComponent(m.task_name)}` : undefined} />
                    <Field label="base" value={shortModel(m.base_model)} title={m.base_model} />
                    <Field label="training method" value={m.training_method} mono />
                    <Field
                      label="metric · target"
                      value={`${fmtMetric(m.metric)} · ${m.metric_direction === "min" ? "Min" : "Max"}`}
                      mono
                    />
                  </div>
                </div>

                {/* The commands sit at the bottom of the card; nothing between
                    them and the fields now that the footer strip is gone. */}
                <div className="mt-auto">
                  <Commands tag={m.version_tag} />
                </div>
              </Bezel>
            );
          })}
        </div>
      )}

      <Pager
        total={total} first={first} last={last} page={current} pageCount={pageCount}
        pageSizePref={pageSizePref} onSize={resize} onPage={setPage}
      />

      {drawer && (
        <Drawer
          mode={drawer.mode}
          tag={drawer.tag}
          otherTag={drawer.otherTag}
          nameOf={nameOf}
          onClose={() => setDrawer(null)}
        />
      )}
    </div>
  );
}

type ModelStorage = {
  location: { kind: "hf" | "remote" | "local" | "missing"; machine: string; path: string; url: string };
  hf_url: string;
  can_upload: boolean;
  upload: { status?: string; error?: string; repo_id?: string };
};

function Commands({ tag }: { tag: string }) {
  const [open, setOpen] = useState<"locate" | "upload" | null>(null);
  const [copied, setCopied] = useState(false);
  const [repo, setRepo] = useState("");
  const [isPrivate, setPrivate] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [missingToken, setMissingToken] = useState(false);
  const endpoint = `/api/models/${encodeURIComponent(tag)}`;
  const { data, error: storageError, mutate } = useSWR<ModelStorage>(`${endpoint}/storage`, {
    refreshInterval: (current) => current?.upload.status === "uploading" ? 3000 : 0,
  });
  const { data: settings, error: settingsError } = useSWR<{ entries: { name: string; present: boolean }[] }>(
    open === "upload" ? "/api/settings" : null,
  );
  const hasToken = settings?.entries.some(e => e.name === "HF_TOKEN" && e.present);
  const uploading = submitting || data?.upload.status === "uploading";
  const needsToken = missingToken || (settings !== undefined && !hasToken);
  const validRepo = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+$/.test(repo.trim());

  async function upload() {
    setSubmitting(true);
    setError("");
    setMissingToken(false);
    try {
      const response = await fetch(`${endpoint}/upload-hf`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ weights: "hf", hf_repo_id: repo.trim(), hf_private: isPrivate }),
      });
      const body = await response.json();
      if (!response.ok) {
        if (body.detail?.code === "hf_token_missing") setMissingToken(true);
        throw new Error(typeof body.detail === "string" ? body.detail : body.detail?.message || "Could not start upload.");
      }
      await mutate();
    } catch (e) { setError(e instanceof Error ? e.message : "Could not start upload."); }
    finally { setSubmitting(false); }
  }

  return <div className="pt-1">
    <div className="flex flex-wrap gap-1.5">
      <button className="btn" onClick={() => { setOpen(open === "locate" ? null : "locate"); setCopied(false); }}>
        <FolderOpen size={12} /> where it is
      </button>
      {data?.hf_url ? <a className="btn" href={data.hf_url} target="_blank" rel="noreferrer">View on Hugging Face</a>
        : data?.can_upload && <button className="btn" onClick={() => { setOpen(open === "upload" ? null : "upload"); setMissingToken(false); }}>
          <UploadCloud size={12} /> {uploading ? "Uploading…" : "Upload to Hugging Face"}
        </button>}
    </div>
    {storageError && <p role="alert" className="mt-2 text-xs text-coral-300">Could not load model location. <button onClick={() => void mutate()}>Retry</button></p>}
    {open === "locate" && !data && !storageError && <p role="status" className="mt-2 text-xs text-slate-400">Loading location…</p>}
    {open === "locate" && data && <div className="mt-2 rounded-bezel border border-hair bg-canvas/60 p-2.5 text-xs">
      <p className="text-slate-400">{data.location.machine || "No saved model location recorded."}</p>
      <div className="mt-1 flex items-start gap-2">
        {data.location.kind === "hf"
          ? <a className="min-w-0 break-all text-brass-300" href={data.location.url} target="_blank" rel="noreferrer">{data.location.path}</a>
          : <code className="min-w-0 flex-1 break-all text-slate-300">{data.location.path}</code>}
        {data.location.path && <button title="Copy model location" onClick={() => {
          const value = data.location.kind === "remote" ? `${data.location.machine}:${data.location.path}` : data.location.path;
          void navigator.clipboard.writeText(value).then(() => setCopied(true)).catch(() => setError("Could not copy location."));
        }}>{copied ? <Check size={12} /> : <Copy size={12} />}</button>}
      </div>
      {data.hf_url && data.location.kind !== "hf" && <a className="mt-2 block break-all text-brass-300" href={data.hf_url} target="_blank" rel="noreferrer">Hugging Face: {data.hf_url}</a>}
    </div>}
    {open === "upload" && !data?.hf_url && <div className="mt-2 space-y-2 text-xs">
      {needsToken ? <p>Set a Hugging Face write token in <Link className="text-brass-300 underline" to="/settings">Settings</Link> before uploading.</p>
        : <>
          <label className="block"><span className="field-label">Hugging Face repository</span>
            <input aria-label="Hugging Face repository" value={repo} onChange={e => setRepo(e.target.value)} disabled={uploading}
              placeholder={`owner/${tag}`} className="mt-1 w-full rounded border border-hair bg-canvas p-2 text-slate-200" />
          </label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={isPrivate} disabled={uploading} onChange={e => setPrivate(e.target.checked)} />Private repository</label>
          <p className="text-slate-500">Uses your Hugging Face write token from Settings. Keeps the original model.</p>
          <button className="btn" disabled={uploading || !validRepo || !hasToken} onClick={() => void upload()}>{uploading ? "Uploading…" : "Upload"}</button>
          {settingsError && <p role="alert">Could not check your token. Open <Link className="text-brass-300 underline" to="/settings">Settings</Link> to check it.</p>}
        </>}
    </div>}
    {uploading && <p role="status" className="mt-2 text-xs text-slate-400">Upload in progress. You can leave this page.</p>}
    {(error || data?.upload.error) && <p role="alert" className="mt-2 text-xs text-coral-300">{error || data?.upload.error}</p>}
  </div>;
}

/** The Registry tag is both the stable key and the user-facing model name. */
function modelName(m: ModelDTO): string {
  return m.version_tag;
}

/** One labelled line. With `to`, the value becomes a link to that page —
 *  same line, same weight, so a linked field doesn't read as a different kind
 *  of thing from the ones beside it. */
function Field({
  label, value, mono = false, to, title,
}: { label: string; value: string; mono?: boolean; to?: string; title?: string }) {
  const cls = `block truncate text-sm text-slate-200 ${mono ? "font-mono" : ""}`;
  return (
    <div className="min-w-0">
      <div className="font-mono text-[0.72rem] uppercase tracking-[0.14em] text-slate-500">{label}</div>
      {to ? (
        <Link to={to} className={`${cls} transition hover:text-brass-300`} title={title || value}>
          {value}
        </Link>
      ) : (
        <span className={cls} title={title || value}>{value}</span>
      )}
    </div>
  );
}


function Drawer({
  mode, tag, otherTag, nameOf, onClose,
}: {
  mode: DrawerMode; tag: string; otherTag?: string;
  nameOf: (tag: string) => string; onClose: () => void;
}) {
  const kicker = mode === "card" ? "Model card" : "Compare";
  // The heading uses the same stable Registry tag as the cards and leaderboard.
  const title = mode === "compare"
    ? `${nameOf(tag)} vs ${nameOf(otherTag || "")}`
    : nameOf(tag);
  return (
    <div className="fixed inset-0 z-50 bg-canvas/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="absolute right-0 top-0 flex h-full w-[48rem] max-w-full animate-zevo-in flex-col border-l border-hair bg-panel p-6 shadow-bezel"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-start justify-between gap-3 border-b border-hair pb-4">
          <div className="min-w-0">
            <Kicker>{kicker}</Kicker>
            <div className="readout mt-1.5 truncate text-lg font-semibold text-ink" title={title}>{title}</div>
          </div>
          <button className="rounded-md p-1.5 text-slate-500 transition hover:text-ink" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="mt-5 min-h-0 flex-1 overflow-auto">
          {mode === "card" && <CardView tag={tag} />}
          {mode === "compare" && otherTag && <CompareView a={tag} b={otherTag} nameOf={nameOf} />}
        </div>
      </div>
    </div>
  );
}


function CardView({ tag }: { tag: string }) {
  const { data, error } = useSWR<string>(
    `/api/models/${encodeURIComponent(tag)}/card`,
    async (url: string) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return r.text();
    },
  );
  if (error) return <div className="text-coral-300 text-xs">failed to load card</div>;
  if (!data) return <div className="text-slate-500 text-xs">loading…</div>;
  return (
    <div className="rounded-bezel border border-hair bg-canvas/60 p-5">
      <Markdown>{data}</Markdown>
    </div>
  );
}


function CompareView({ a, b, nameOf }: { a: string; b: string; nameOf: (tag: string) => string }) {
  const { data, error } = useSWR<CompareResponse>(
    `/api/models/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`,
  );
  if (error) return <div className="text-coral-300 text-xs">failed to compare</div>;
  if (!data) return <div className="text-slate-500 text-xs">loading…</div>;

  const delta = data.champion_test_score_delta;
  const metricA = typeof data.fields.metric?.a === "string" ? data.fields.metric.a : "";
  const metricB = typeof data.fields.metric?.b === "string" ? data.fields.metric.b : metricA;
  const DeltaIcon =
    delta != null && delta > 0 ? TrendingUp :
    delta != null && delta < 0 ? TrendingDown :
    Minus;
  const tone: "phosphor" | "coral" | "ink" =
    delta != null && delta > 0 ? "phosphor" :
    delta != null && delta < 0 ? "coral" :
    "ink";
  const deltaColor = { phosphor: "text-phosphor-300", coral: "text-coral-300", ink: "text-slate-400" }[tone];

  return (
    <>
      {/* Cross-model difference is distinct from each model's Improvement
          over its own baseline, which appears as a separate row below. */}
      <Bezel flat className="mb-5 flex items-center gap-3 p-4">
        <DeltaIcon size={18} className={deltaColor} />
        <div>
          <Kicker>Test Score Difference</Kicker>
          <div className={`readout mt-1 text-sm font-semibold ${deltaColor}`}>
            {delta == null
              ? data.headline_summary
              : `${nameOf(data.b_tag)} ${delta >= 0 ? "+" : ""}${fmtScore(delta, metricB)} vs ${nameOf(data.a_tag)} on the held-out test set`}
          </div>
        </div>
      </Bezel>

      {/* Two-column readout diff */}
      <div className="overflow-hidden rounded-bezel border border-hair bezel-flat">
        <div className="table-label grid grid-cols-[1fr_1fr_1fr] gap-px border-b border-hair bg-raised">
          <div className="px-3 py-2">model</div>
          <div className="px-3 py-2 truncate">{nameOf(data.a_tag)}</div>
          <div className="px-3 py-2 truncate">{nameOf(data.b_tag)}</div>
        </div>
        <div className="divide-y divide-hair">
          {Object.entries(data.fields).map(([k, v]) => (
            <div
              key={k}
              className={`grid grid-cols-[1fr_1fr_1fr] gap-px text-2xs ${v.differs ? "bg-brass-500/[0.06]" : ""}`}
            >
              <div className="px-3 py-2 font-mono text-slate-400">{k}</div>
              <div className={`px-3 py-2 font-mono ${v.differs ? "text-brass-200" : "text-slate-300"}`}>{fmtCompare(k, v.a, metricA)}</div>
              <div className={`px-3 py-2 font-mono ${v.differs ? "text-brass-200" : "text-slate-300"}`}>{fmtCompare(k, v.b, metricB)}</div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function fmtCompare(key: string, value: unknown, metric: string): string {
  if (key === "metric" && typeof value === "string") return fmtMetric(value);
  if ((key === "eval.score" || key === "test score" || key === "improvement" || key.endsWith("_score")) && typeof value === "number") {
    return fmtScore(value, metric);
  }
  return fmt(value);
}
