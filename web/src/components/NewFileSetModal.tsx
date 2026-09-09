import { useEffect, useRef, useState } from "react";
import { Trash2, Upload } from "lucide-react";
import { Modal } from "./Modal";
import { Kicker } from "./zevo/primitives";
import { ROOT_LABEL, groupByFolder } from "../lib/format";
import { api } from "../lib/api";
import type { FileSetDTO } from "../lib/api";

/**
 * Create a dataset, or manage one that exists.
 *
 * Uploading used to be a single file-picker in the page header: one file became
 * one dataset named after it, with no description and no way to add a second
 * file or remove a wrong one. A dataset is a named bundle — several files that
 * belong together — so it gets a form: a name, what it is, and the files, each
 * removable.
 *
 * Opened with `name`, the same form manages an existing dataset instead.
 */

const field =
  "w-full rounded-md border border-hair bg-canvas p-2.5 text-sm leading-relaxed text-slate-200 placeholder:text-slate-600 placeholder:opacity-100 focus:border-brass-500/50 focus:outline-none";
const head = "section-title mb-2 block !text-brass-300";

type Remote = NonNullable<FileSetDTO["source"]>["remote"][number];

/** Where a file of this dataset comes from. Both are marked here — this is the
 *  form where you add them, so which route a row came in by is the point. The
 *  card outside marks only the fetched ones. */
function SourceTag({ hf }: { hf: boolean }) {
  return (
    <span className={`shrink-0 rounded border px-1.5 py-px font-mono text-[0.58rem] uppercase tracking-[0.1em] ${
      hf ? "border-hair text-slate-400" : "border-phosphor-500/40 bg-phosphor-500/10 text-phosphor-300"
    }`}>
      {hf ? "hf" : "local"}
    </span>
  );
}

export function NewFileSetModal({
  open, name: existing, onClose, onChanged,
}: {
  open: boolean;
  /** null = create a new dataset; a name = manage that one. */
  name: string | null;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const [remote, setRemote] = useState<Remote[]>([]);
  const [hubId, setHubId] = useState("");
  // Which slice of the hub repo, and which named subset. Both go into
  // `load_dataset` when the Data Agent acquires — see zevo.engine.remote_datasets.
  const [hubSplit, setHubSplit] = useState("");
  const [hubConfig, setHubConfig] = useState("");
  const [busy, setBusy] = useState("");
  const [progress, setProgress] = useState("");
  // Depth-counted: dragging over a child fires dragleave on the parent, so a
  // boolean flickers the highlight off halfway across the zone.
  const [dragDepth, setDragDepth] = useState(0);
  // Which subfolder the next upload lands in. "" is the dataset root.
  const [folder, setFolder] = useState("");
  // Whether the add-files panel is open. Closed by default: the list of what
  // the dataset already holds is what this dialog is mostly for.
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const managing = !!existing;

  useEffect(() => {
    if (!open) return;
    setError(null);
    setBusy("");
    if (!existing) {
      setName("");
      setNote("");
      setFiles([]);
      setRemote([]);
      setHubId("");
      setHubSplit("");
      setHubConfig("");
      return;
    }
    void (async () => {
      const all = await api<FileSetDTO[]>("/files");
      const d = all.find((x) => x.name === existing);
      setName(existing);
      setNote(d?.source?.note ?? "");
      setFiles(d?.files ?? []);
      setRemote(d?.source?.remote ?? []);
      setHubId("");
      setHubSplit("");
      setHubConfig("");
    })();
  }, [open, existing]);

  /** Upload into `name`, creating the dataset on the first file.
   *
   *  One at a time rather than one request with all of them: the endpoint takes
   *  a single file, and a partial failure then names the file that failed
   *  instead of failing the whole drop. Whatever landed stays landed.
   */
  async function upload(picked: FileList | File[] | null | undefined) {
    const chosen = Array.from(picked ?? []);
    if (!chosen.length) return;
    const target = name.trim();
    if (!target) {
      setError("Name this set of files first, because that is the folder they go into.");
      return;
    }
    setBusy("upload");
    setError(null);
    const failed: string[] = [];
    let latest: string[] | null = null;
    for (const [i, f] of chosen.entries()) {
      setProgress(chosen.length > 1 ? `${i + 1}/${chosen.length}` : "");
      try {
        const fd = new FormData();
        fd.append("file", f);
        fd.append("name", target);
        if (folder) fd.append("folder", folder);
        const d = await api<{ files?: string[] }>("/files", { method: "POST", body: fd });
        latest = d.files ?? [];
      } catch (e) {
        failed.push(`${f.name}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    if (latest) setFiles(latest);
    setProgress("");
    setBusy("");
    if (failed.length) setError(failed.join("\n"));
    onChanged();
  }

  /** Name a hub dataset as one of this dataset's files. */
  async function addRemote() {
    const ident = hubId.trim();
    const target = name.trim();
    if (!ident) return;
    if (!target) {
      setError("Name this set of files first.");
      return;
    }
    setBusy("remote");
    setError(null);
    try {
      const d = await api<FileSetDTO>(`/files/${encodeURIComponent(target)}/remote`, {
        method: "POST",
        body: JSON.stringify({
          id: ident, split: hubSplit.trim(), config: hubConfig.trim(),
        }),
      });
      setRemote(d.source?.remote ?? []);
      setHubId("");
      setHubSplit("");
      setHubConfig("");
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  async function removeRemote(id: string, split: string) {
    setBusy(`${id}:${split}`);
    setError(null);
    try {
      // The split is part of which entry this is: one repo listed twice, once
      // per split, is the ordinary way to say "train on this, validate on that".
      const d = await api<FileSetDTO>(
        `/files/${encodeURIComponent(name)}/remote/${encodeURIComponent(id)}`
        + `?split=${encodeURIComponent(split)}`,
        { method: "DELETE" },
      );
      setRemote(d.source?.remote ?? []);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  async function removeFile(f: string) {
    if (!window.confirm(`Delete "${f}" from ${name}? The file is removed from disk.`)) return;
    setBusy(f);
    setError(null);
    try {
      const d = await api<{ files?: string[] }>(
        `/files/${encodeURIComponent(name)}/contents/${encodeURIComponent(f)}`,
        { method: "DELETE" },
      );
      setFiles(d.files ?? []);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  async function saveNote() {
    setBusy("note");
    setError(null);
    try {
      await api(`/files/${encodeURIComponent(name)}/note`, {
        method: "PUT",
        body: JSON.stringify({ note }),
      });
      onChanged();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  return (
    <Modal
      open={open}
      title={managing ? `Manage ${existing}` : "New file"}
      onClose={onClose}
      width="max-w-2xl"
    >
      <div className="max-h-[70vh] space-y-4 overflow-y-auto pr-1">
        <div>
          <label className={head}>File set name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={managing}
            spellCheck={false}
            placeholder="Name this file set, e.g. medqa-usmle"
            className={`${field} font-mono disabled:opacity-60`}
          />
        </div>

        <div>
          <label className={head}>Description</label>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={2}
            placeholder="What this file set contains and how it will be used"
            className={field}
          />
        </div>

        <div>
          <label className={head}>Files</label>
          <div className="space-y-1.5 rounded-md border border-hair bg-canvas/50 p-3">
            {groupByFolder(files).map((g) => (
              <div key={g.folder} className="space-y-1.5">
                {g.label && (
                  <div className="font-mono text-2xs text-brass-300">{g.label}</div>
                )}
                {g.files.map((f) => (
                  <div key={f.path} className={`flex items-center justify-between gap-3 ${g.indent ? "pl-3" : ""}`}>
                    <span className="flex min-w-0 items-center gap-2">
                      <span className="min-w-0 truncate font-mono text-2xs text-slate-300">{f.leaf}</span>
                      <SourceTag hf={false} />
                    </span>
                    <button
                      onClick={() => void removeFile(f.path)}
                      disabled={!!busy}
                      title="Delete this file"
                      className="shrink-0 rounded p-1 text-slate-500 transition hover:bg-coral-500/10 hover:text-coral-300 disabled:opacity-40"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                ))}
              </div>
            ))}
            {remote.map((r) => (
              <div key={`${r.id}:${r.split ?? ""}`} className="flex items-center justify-between gap-3">
                <span className="flex min-w-0 items-center gap-2">
                  <a
                    href={r.url}
                    target="_blank"
                    rel="noreferrer"
                    className="min-w-0 truncate font-mono text-2xs text-slate-300 hover:text-brass-300"
                  >
                    {r.id}
                  </a>
                  {/* Which slice, and what it is here. The same repo can be
                      listed twice, so without these two the rows are
                      indistinguishable. */}
                  <span className="shrink-0 font-mono text-2xs text-slate-500">
                    {r.config ? `${r.config} · ` : ""}{r.split || "train"}
                  </span>
                  <SourceTag hf />
                </span>
                <button
                  onClick={() => void removeRemote(r.id, r.split ?? "")}
                  disabled={!!busy}
                  title="Stop listing this"
                  className="shrink-0 rounded p-1 text-slate-500 transition hover:bg-coral-500/10 hover:text-coral-300 disabled:opacity-40"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
            <div className="space-y-2 pt-1">
              {/* Two routes in, and nothing else to answer. What a file IS gets
                  decided where it is USED (a task names its test set, a setting
                  names its validation set), so asking again here was asking the
                  same question in the wrong place. */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => setAdding((v) => !v)}
                  disabled={!!busy}
                  className="btn !text-[13px] disabled:opacity-50"
                >
                  <Upload size={12} />{" "}
                  {busy === "upload"
                    ? `uploading${progress ? ` ${progress}` : ""}…`
                    : "Upload files"}
                </button>
                {/* On the same line as the two routes in, because it qualifies
                    both: a dropped file and a picked one land in the same
                    place, and that place is not something this dialog can
                    change afterwards. */}
                <span className="font-mono text-2xs text-slate-500">into</span>
                {[["", ROOT_LABEL], ["train", "train"], ["validation", "validation"], ["test", "test"]]
                  .map(([value, label]) => (
                  <button
                    key={label}
                    onClick={() => setFolder(value)}
                    className={`rounded border px-1.5 py-0.5 font-mono text-2xs transition ${
                      folder === value
                        ? "border-brass-500/50 bg-brass-500/15 text-brass-300"
                        : "border-hair text-slate-500 hover:text-slate-300"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {/* Opened, not assumed: the button used to go straight to the
                  system file dialog, which is one way in and hides the other.
                  Both are here, named, and the folder they land in is chosen
                  before either — moving a file afterwards is not something this
                  dialog can do. */}
              {adding && (
                <div
                  onDragEnter={(e) => { e.preventDefault(); setDragDepth((d) => d + 1); }}
                  onDragOver={(e) => e.preventDefault()}
                  onDragLeave={() => setDragDepth((d) => Math.max(0, d - 1))}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragDepth(0);
                    void upload(e.dataTransfer.files);
                  }}
                  className="space-y-2 rounded-md border border-hair bg-raised/40 p-3"
                >
                  <div
                    className={`flex flex-col items-center justify-center gap-1 rounded-md border border-dashed py-6 font-mono text-2xs transition ${
                      dragDepth > 0
                        ? "border-brass-500/60 bg-brass-500/10 text-brass-300"
                        : "border-hair/70 text-slate-500"
                    }`}
                  >
                    <Upload size={16} />
                    {/* Named only when there is a name: "into others" reads as
                        a place, and it is not one — it is the absence of one. */}
                    {dragDepth > 0 ? "Release to upload" : "Drag files here"}
                  </div>
                  <div className="flex items-center justify-center gap-2">
                    <span className="font-mono text-2xs text-slate-600">or</span>
                    <input
                      ref={picker}
                      type="file"
                      multiple
                      className="hidden"
                      onChange={(e) => {
                        void upload(e.target.files);
                        e.target.value = "";
                      }}
                    />
                    <button
                      onClick={() => picker.current?.click()}
                      disabled={!!busy}
                      className="btn !text-[13px] disabled:opacity-50"
                    >
                      Select from your computer
                    </button>
                  </div>
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-2xs text-slate-600">or</span>
                <input
                  value={hubId}
                  onChange={(e) => setHubId(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void addRemote(); } }}
                  placeholder="Hugging Face dataset repository, e.g. trl-lib/Capybara"
                  spellCheck={false}
                  className={`${field} min-w-0 flex-1 font-mono`}
                />
                <button
                  onClick={() => void addRemote()}
                  disabled={!!busy || !hubId.trim()}
                  className="btn !text-[13px] disabled:opacity-40"
                >
                  {busy === "remote" ? "adding…" : "add"}
                </button>
              </div>

              {/* Only once there is a repo to ask about. An upload has no split
                  and no config, so on that route the two boxes were furniture
                  you had to read past to find out they did not apply. */}
              {hubId.trim() && (
                <div className="flex flex-wrap items-center gap-2">
                  <input
                    value={hubSplit}
                    onChange={(e) => setHubSplit(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void addRemote(); } }}
                    placeholder="Dataset split to import; blank uses train"
                    spellCheck={false}
                    list="hf-splits"
                    className={`${field} !w-32 font-mono !p-1.5 !text-2xs`}
                  />
                  <datalist id="hf-splits">
                    <option value="train" />
                    <option value="validation" />
                    <option value="test" />
                  </datalist>
                  <input
                    value={hubConfig}
                    onChange={(e) => setHubConfig(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void addRemote(); } }}
                    placeholder="Dataset subset/config; blank uses the only one"
                    spellCheck={false}
                    className={`${field} !w-32 font-mono !p-1.5 !text-2xs`}
                  />
                </div>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="whitespace-pre-line rounded-md border border-coral-500/30 bg-coral-500/10 p-2.5 text-2xs text-coral-300">
            {error}
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between gap-2 border-t border-hair pt-4">
        <Kicker className="!text-slate-600">
          {files.length + remote.length} file{files.length + remote.length === 1 ? "" : "s"}
        </Kicker>
        <div className="flex gap-2">
          <button onClick={onClose} className="btn">close</button>
          <button
            onClick={() => void saveNote()}
            disabled={!!busy || !name.trim() || (files.length === 0 && remote.length === 0)}
            title={files.length === 0 && remote.length === 0 ? "Add at least one file first" : ""}
            className="btn btn-brass uppercase tracking-[0.14em] disabled:opacity-30"
          >
            {busy === "note" ? "saving…" : "save"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
