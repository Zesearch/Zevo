import { useMemo, useState } from "react";
import useSWR from "swr";
import { Upload, ChevronRight, Pencil, Search, Trash2, X, Folder } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { FileSetView } from "../components/FileSetView";
import { NewFileSetModal } from "../components/NewFileSetModal";
import { PageHead, Bezel, Kicker } from "../components/zevo/primitives";
import { bytes, fmtDate, groupByFolder } from "../lib/format";
import { PAGE_SIZES, Pager } from "../components/zevo/Pager";
import { useRowsPerPage } from "../lib/useRowsPerPage";
import { api } from "../lib/api";
import type { FileSetDTO, RemoteFileDTO } from "../lib/api";

// One dataset card plus the gap below it; measured from the rendered grid.
const CARD_PX = 250;
// How many a row holds at the widest breakpoint (see the grid classes).
const CARD_COLS = 3;
// How many filenames a card lists before it offers the rest. Three keeps the
// card short enough that a page of them does not fill the window on its own.
const FILES_SHOWN = 3;

/** A dataset's files, four at a time.
 *
 *  A bundle of five and a bundle of one made cards of two different heights,
 *  and the list is a reference, not something read top to bottom — so it is
 *  capped, and says how many it is holding back.
 */
function FileList({ files }: { files: { name: string; remote: RemoteFileDTO | null }[] }) {
  const [all, setAll] = useState(false);
  // Grouped by subfolder, then flattened back into rows so the collapsed height
  // stays a count of ROWS: a folder heading takes a line like any file does,
  // and reserving space for files only would make the cards ragged again.
  const rows = useMemo(() => {
    const local = files.filter((f) => !f.remote);
    const remote = files.filter((f) => f.remote);
    const out: ({ kind: "folder"; folder: string }
      | { kind: "file"; label: string; remote: RemoteFileDTO | null; indented: boolean })[] = [];
    for (const g of groupByFolder(local.map((f) => f.name))) {
      if (g.label) out.push({ kind: "folder", folder: g.label });
      for (const f of g.files) {
        out.push({ kind: "file", label: f.leaf, remote: null, indented: g.indent });
      }
    }
    for (const r of remote) {
      out.push({ kind: "file", label: r.name, remote: r.remote, indented: false });
    }
    return out;
  }, [files]);
  const shown = all ? rows : rows.slice(0, FILES_SHOWN);
  return (
    <div className="mt-1 font-mono text-2xs leading-5 text-slate-400">
      {/* A constant FILES_SHOWN rows tall while collapsed, whatever it is
          holding: an evaluation set lists four files and a training set often
          lists one, so without the reservation everything below them — the
          description, the created date — started at a different height on the
          two kinds of card, which is exactly what makes a grid unreadable. */}
      <div className={all ? "" : "h-[3.75rem]"}>
        {shown.map((r, i) => (
          r.kind === "folder" ? (
            <div key={`f:${r.folder}:${i}`} className="text-[0.9rem] leading-5 text-brass-300">{r.folder}</div>
          ) : (
          // One hub id can appear twice under two splits, so the name alone
          // is not a key.
          <div
            key={`${r.label}:${r.remote?.split ?? ""}:${i}`}
            className={`flex items-center gap-1.5 ${r.indented ? "pl-3" : ""}`}
          >
            <span className="min-w-0 truncate text-slate-300">{r.label}</span>
            {/* Local is the default, so only the fetched ones are marked. */}
            {/* The split is part of the identity of a remote file: the same hub
                id listed twice IS two different files. Shown even when the
                entry left it blank, because blank does not mean "unspecified"
                to the loader, it means `train` — and a row that says nothing
                reads as the former. */}
            {r.remote && (
              <span className="shrink-0 font-mono text-[0.58rem] text-slate-500">
                {r.remote.config ? `${r.remote.config}/` : ""}{r.remote.split || "train"}
              </span>
            )}
            {r.remote && (
              <span className="shrink-0 rounded border border-hair px-1 py-px text-[0.58rem] uppercase tracking-[0.1em] text-slate-400">
                {r.remote.kind === "huggingface" ? "hf" : "url"}
              </span>
            )}
          </div>
          )
        ))}
      </div>
      {/* `mb-1`: expanded, this button is the last thing before the card's
          next rule, and it sat right against it. */}
      <div className="mb-3 mt-1 h-4">
        {rows.length > FILES_SHOWN && (
          <button
            onClick={(e) => { e.stopPropagation(); setAll((v) => !v); }}
            className="text-slate-500 transition hover:text-brass-300"
          >
            {all ? "show less" : `show ${rows.length - FILES_SHOWN} more`}
          </button>
        )}
      </div>
    </div>
  );
}

export function FilesPage() {
  const { data: fileSets = [], mutate, error: loadError } = useSWR<FileSetDTO[]>("/api/files");
  const [busy, setBusy] = useState("");

  async function deleteFileSet(name: string) {
    // Say what survives. A task's runs keep their name as plain text; a
    // dataset's files are the thing a finished run POINTED at, so deleting one
    // is not the same kind of safe and the prompt should not imply it is.
    if (!window.confirm(
      `Delete "${name}" and every file in it? A run that used it keeps its ` +
      `results, but its inputs will no longer be on disk.`
    )) return;
    setBusy(name);
    setError(null);
    try {
      await api(`/files/${encodeURIComponent(name)}`, { method: "DELETE" });
      await mutate();
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setBusy("");
    }
  }
  const [params, setParams] = useSearchParams();
  const open = params.get("open");
  const setOpen = (name: string | null) => {
    const next = new URLSearchParams(params);
    if (name) next.set("open", name);
    else next.delete("open");
    setParams(next, { replace: true });
  };
  // Which file of the open dataset is on screen. "" = let the server pick the
  // most informative one, which is what opening the card should show first.
  const [query, setQuery] = useState("");
  // "" = both halves. Filtering hides the other half's files AND any dataset
  // left with nothing to show, so the page answers "where is my training data"
  // rather than making you read past the scoring files in every card.
  const [page, setPage] = useState(0);
  // 0 = "as many as fit". Held in state rather than a ref: the hook re-measures
  // when the grid mounts, and a ref never changes identity.
  const [pageSizePref, setPageSizePref] = useState(PAGE_SIZES[0]);
  const [listEl, setListEl] = useState<HTMLDivElement | null>(null);
  const fits = useRowsPerPage(listEl, CARD_PX, { min: 1, max: 40 }) * CARD_COLS;
  const pageSize = pageSizePref || fits;
  // The create/manage form. `manage` names the dataset being edited; null =
  // creating a new one.
  const [formOpen, setFormOpen] = useState(false);
  const [manage, setManage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const q = query.trim().toLowerCase();
  const matches = fileSets.filter(
    (d) => (!q || d.name.toLowerCase().includes(q)
      || (d.source?.note ?? "").toLowerCase().includes(q)
      || d.files.some((f) => f.toLowerCase().includes(q))
      || (d.source?.remote ?? []).some((r) => r.id.toLowerCase().includes(q))),
  );
  const total = matches.length;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const current = Math.min(page, pageCount - 1);
  const shown = matches.slice(current * pageSize, current * pageSize + pageSize);
  const first = total === 0 ? 0 : current * pageSize + 1;
  const last = Math.min(total, (current + 1) * pageSize);
  // Keep the first visible card visible when the page size changes.
  const resize = (n: number) => {
    setPage(Math.floor((current * pageSize) / (n || fits)));
    setPageSizePref(n);
  };


  return (
    <div className="flex w-full flex-1 flex-col px-[max(1.5rem,1.5vw)] py-8">
      <PageHead
        title="Files"
        subtitle="Files you uploaded, stored in Zevo and ready to use"
        right={
          <button onClick={() => { setManage(null); setFormOpen(true); }} className="btn btn-brass">
            <Upload size={14} /> New file
          </button>
        }
      />

      {(error || loadError) && (
        <div className="mb-5 flex items-center gap-2 rounded-bezel border border-coral-500/30 bg-coral-500/10 px-3 py-2.5 text-xs text-coral-300">
          <span className="lamp lamp-coral shrink-0" />
          {error ?? `Could not load file sets: ${loadError instanceof Error ? loadError.message : loadError}`}
        </div>
      )}

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1 sm:max-w-md">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={query}
            onChange={(e) => { setQuery(e.target.value); setPage(0); }}
            placeholder="search name, file or source"
            spellCheck={false}
            className="w-full rounded-md border border-hair bg-canvas py-2 pl-9 pr-8 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:border-brass-500/50"
          />
          {query && (
            <button
              onClick={() => { setQuery(""); setPage(0); }}
              title="Clear"
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-500 transition hover:text-slate-200"
            >
              <X size={13} />
            </button>
          )}
        </div>
        {query && (
          <span className="font-mono text-2xs text-slate-500">
            {matches.length} match{matches.length === 1 ? "" : "es"}
          </span>
        )}
      </div>

      <div ref={setListEl} className="stagger grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {shown.map((d) => (
          <button
            key={d.name}
            onClick={() => setOpen(d.name)}
            className="group text-left"
          >
            <Bezel className="flex h-full flex-col p-5 transition group-hover:border-brass-500/40 group-hover:shadow-glow-brass">
              <div className="flex items-start justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2">
                  <Folder size={14} className="shrink-0 text-brass-400/70" />
                  <span className="readout truncate text-sm text-ink" title={d.name}>{d.name}</span>
                  {/* What this dataset is FOR, said on the dataset rather than
                      only on its file groups: most hold both halves, and the
                      ones that hold only one are the ones worth spotting. */}
                </div>
                {/* The same pair as a task card, in the same corner: these
                    are the same two actions, and a written `manage` pill beside
                    a bare chevron made them look like different kinds of
                    thing. Opening the card moved to the footer, where it is
                    named. */}
                <span className="flex shrink-0 items-center gap-1">
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => { e.stopPropagation(); setManage(d.name); setFormOpen(true); }}
                    onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); setManage(d.name); setFormOpen(true); } }}
                    title="Add or remove files, edit the description"
                    className="rounded-md p-1 text-slate-600 transition hover:text-brass-300"
                  >
                    <Pencil size={13} />
                  </span>
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => { e.stopPropagation(); void deleteFileSet(d.name); }}
                    onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); void deleteFileSet(d.name); } }}
                    title="Delete these files"
                    className={`rounded-md p-1 text-slate-600 transition hover:text-coral-300 ${busy ? "pointer-events-none opacity-40" : ""}`}
                  >
                    <Trash2 size={13} />
                  </span>
                </span>
              </div>
              {/* size on the left, the files on the right — the names are what
                  you scan for, so they get the room, not a count. The files are
                  split the way they are used: trained on, or scored by. */}
              <div className="mt-4 flex items-start justify-between gap-5">
                <div className="shrink-0">
                  <Kicker strong>size</Kicker>
                  <div className="readout mt-1 text-lg font-semibold text-phosphor-300">{bytes(d.size_bytes)}</div>
                </div>
                {/* The files get the remaining card width. A fixed 15rem column
                    collided with the size block on narrower cards. */}
                <div className="min-w-0 max-w-60 flex-1 text-left">
                  {/* No "training files" / "evaluation files" heading: the label
                      beside the name already says which of the two this is. */}
                  <Kicker strong>files</Kicker>
                  <FileList
                    files={[
                      ...d.files.map((f) => ({ name: f, remote: null as RemoteFileDTO | null })),
                      ...(d.source?.remote ?? []).map((r) => ({ name: r.id, remote: r })),
                    ]}
                  />
                </div>
              </div>

              {/* Optional, and written by whoever uploaded it — a dataset
                  without one simply ends at its files. */}
              {d.source?.note && (
                <div className="mt-4">
                  <Kicker>description</Kicker>
                  <p
                    title={d.source.note}
                    className="mt-1 line-clamp-3 break-words text-sm leading-relaxed text-slate-400"
                  >
                    {d.source.note}
                  </p>
                </div>
              )}

              {/* mt-auto, so `created` sits on the floor of the card whatever is
                  above it — the cards in a row are stretched to one height, so
                  that puts every card's date on the same line. */}
              <div className="mt-auto flex items-center justify-between border-t border-hair pt-2.5 font-mono text-2xs text-slate-300">
                <span>created {fmtDate(d.created_at)}</span>
                {/* Named, not just an arrow. The chevron alone sat next to
                    `manage` and read as a second way to do the same thing;
                    here it says what it opens. */}
                <span className="flex items-center gap-0.5 transition group-hover:text-brass-300">
                  details
                  <ChevronRight size={13} />
                </span>
              </div>
            </Bezel>
          </button>
        ))}
        {matches.length === 0 && !loadError && (
          <div className="col-span-full rounded-bezel border border-dashed border-hair p-12 text-center text-sm text-slate-500">
            {fileSets.length === 0
              ? "No files yet. Upload a CSV or JSONL to get started."
              : query
              ? `No file set matches "${query}".`
              : "Nothing here yet."}
          </div>
        )}
      </div>

      <Pager
        total={total} first={first} last={last} page={current} pageCount={pageCount}
        pageSizePref={pageSizePref} onSize={resize} onPage={setPage}
      />

      <NewFileSetModal
        open={formOpen}
        name={manage}
        onClose={() => setFormOpen(false)}
        onChanged={() => void mutate()}
      />

      {/* Preview drawer */}
      {open && (
        <div className="fixed inset-0 z-50 bg-canvas/70 backdrop-blur-sm" onClick={() => setOpen(null)}>
          <div
            className="absolute right-0 top-0 flex h-full w-[42rem] max-w-full animate-zevo-in flex-col border-l border-hair bg-panel p-6 shadow-bezel"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex shrink-0 items-start justify-between gap-3 border-b border-hair pb-4">
              <div className="min-w-0">
                <Kicker strong>Files</Kicker>
                <div className="readout mt-1.5 truncate text-lg font-semibold text-ink" title={open}>{open}</div>
              </div>
              <button className="shrink-0 rounded-md p-1.5 text-slate-500 transition hover:text-ink" onClick={() => setOpen(null)}>
                <X size={16} />
              </button>
            </div>

            <div className="mt-4 min-h-0 flex-1 overflow-auto">
              <FileSetView name={open} onLeave={() => setOpen(null)} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
