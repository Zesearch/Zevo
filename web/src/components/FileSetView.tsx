import { useEffect, useState } from "react";
import useSWR from "swr";
import { Link } from "react-router-dom";
import { Kicker } from "./zevo/primitives";
import { FILES_ROOT, groupByFolder } from "../lib/format";
import type { FileSetDTO } from "../lib/api";

/**
 * One dataset, read from the inside: its files, whichever one you pick, and the
 * tasks that train on it.
 *
 * Lives on its own so the Datasets drawer and the Tasks popup show the same
 * thing — a task's data cell should not have to send you to another page to
 * answer "what is in that dataset".
 */

type Preview = {
  name: string;
  file: string;
  kind: "csv" | "jsonl" | "json" | "pdf" | "txt" | "script" | "parquet" | "binary" | "empty";
  n_columns: number;
  columns: string[];
  n_rows: number;
  preview_rows: Record<string, unknown>[];
  sample_text: string;
  notes: string;
};

type TaskRef = {
  name: string;
  test_set: string;
  evaluation_script: string;
  test_sample_submission: string;
};

export function FileSetView({
  name, file: initialFile = "", split = "", onLeave,
}: {
  name: string;
  /** Open on this file instead of the dataset's default. Callers that
   *  arrive from a named file — a run's input list, say — mean that file,
   *  not whichever one the dataset would show first.
   *
   *  A hub id is accepted here too: it is a file of the dataset like any
   *  other, it just has no preview, so it is marked rather than opened. */
  file?: string;
  /** Which slice, when `file` is a hub id listed under more than one. */
  split?: string;
  onLeave?: () => void;
}) {
  // A hub id is not a previewable file: asking the preview endpoint for
  // `trl-lib/Capybara` gets nothing. It still names which entry to MARK, which
  // is what `initialFile` is for.
  //
  // "Has a slash" used to be the test, and stopped being one the moment a
  // dataset could hold `test/test.csv`: that is a local file with a slash in
  // it, and treating it as a repo silently skipped the preview and marked
  // nothing. A hub id is `owner/name` — exactly two segments, and no suffix,
  // because a file has one and a repo does not.
  const looksLikeHubId = /^[^/\s]+\/[^/\s]+$/.test(initialFile)
    && !initialFile.includes(".");
  const asLocal = looksLikeHubId ? "" : initialFile;
  const [file, setFile] = useState(asLocal);
  useEffect(() => setFile(asLocal), [name, asLocal]);
  // Arrived for a hub entry, and nothing has been clicked since. The bundle's
  // default local file is NOT what you asked to see, so previewing it would
  // put a second gold chip on a file you never named — two answers to "which
  // one did I click", one of them wrong.
  const onRemote = !!initialFile && !asLocal && !file;

  const { data: datasets = [] } = useSWR<FileSetDTO[]>("/api/files");
  const { data: tasks = [] } = useSWR<TaskRef[]>("/api/tasks");
  const { data: preview, error: previewError } = useSWR<Preview>(
    onRemote ? null
      : `/api/files/${encodeURIComponent(name)}/preview?n=10`
        + (file ? `&file=${encodeURIComponent(file)}` : ""),
  );

  const ds = datasets.find((d) => d.name === name);
  // Task owns held-out scoring assets only. Training/validation dataset usage
  // belongs to Settings and is displayed there, never inferred from Task.
  const usedBy = tasks.filter((t) =>
    [t.test_set, t.evaluation_script, t.test_sample_submission]
      .some((p) => (p || "").startsWith(`${FILES_ROOT}/${name}/`)));
  const tabular = preview && ["csv", "jsonl", "json", "parquet"].includes(preview.kind);

  return (
    <>
      {ds && (
        <div>
          <Kicker strong>Files</Kicker>
          {/* One row per subfolder, the folder named once on the left instead
              of repeated inside every chip. A dataset with no subfolders has a
              single unlabelled row and looks exactly as it always did. */}
          <div className="mt-2 space-y-1.5">
            {groupByFolder(ds.files).map((g) => (
              // Two columns, not one wrapping row: with the heading inside the
              // same wrap, a group whose chips spill onto a second line had
              // that line start at the far left, under the heading instead of
              // beside it.
              <div key={g.folder} className="flex items-start gap-2">
                {g.label && (
                  <span className="w-32 shrink-0 whitespace-nowrap pt-1 font-mono text-[0.9rem] text-brass-300">
                    {g.label}
                  </span>
                )}
                <div className="flex min-w-0 flex-wrap gap-1.5">
                {g.files.map((f) => (
                  <button
                    key={f.path}
                    onClick={() => setFile(f.path)}
                    title={f.path}
                    className={`rounded-md border px-2.5 py-1 font-mono text-2xs transition ${
                      !onRemote && (file || preview?.file) === f.path
                        ? "border-brass-500/40 bg-brass-500/15 text-brass-300"
                        : "border-hair bg-raised text-slate-300 hover:text-ink"
                    }`}
                  >
                    {f.leaf}
                  </button>
                ))}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-1.5 grid min-w-0 gap-1.5 sm:grid-cols-2">
            {/* Not on disk, but still a file of this dataset. Marked the same
                way a selected local file is when that is the one you came in
                for: arriving at a bundle of six and having to work out which
                of them you had just clicked is the thing the mark prevents. */}
            {ds.source?.remote.map((r) => {
              const picked = !!initialFile && r.id === initialFile
                && (r.split || "") === (split || "");
              return (
              <a
                key={`${r.id}:${r.split ?? ""}`}
                href={r.url}
                target="_blank"
                rel="noreferrer"
                title={`${r.role} data: the ${r.split || "train"} split${
                  r.config ? ` of the ${r.config} config` : ""
                }, fetched from ${r.kind === "huggingface" ? "HuggingFace" : "the web"} at run time`}
                // Exactly a file chip. That this one is fetched rather than
                // stored is already said by the HF/URL mark it carries; saying
                // it again in the border only made one row of one list look
                // like a different kind of control.
                className={`flex min-w-0 max-w-full items-center gap-1.5 rounded-md border px-2.5 py-1 font-mono text-2xs transition ${
                  picked
                    ? "border-brass-500/40 bg-brass-500/15 text-brass-300"
                    : "border-hair bg-raised text-slate-300 hover:text-ink"
                }`}
              >
                <span className="min-w-0 truncate">{r.id}</span>
                {/* The split is what tells two entries of one repo apart. */}
                <span className="text-slate-500">
                  {r.config ? `${r.config}/` : ""}{r.split || "train"}
                </span>
                <span className="rounded border border-hair px-1 py-px text-[0.58rem] uppercase tracking-[0.1em]">
                  {r.kind === "huggingface" ? "HF" : "URL"}
                </span>
              </a>
              );
            })}
          </div>
        </div>
      )}

      {/* What the selected file is — under the chips, where it was chosen. A
          dataset with nothing stored locally has no file to describe, so it
          says nothing here and lets the note below speak; it used to read
          "(empty) · empty". */}
      {onRemote ? (
        <p className="mt-3 text-2xs leading-relaxed text-slate-400">
          Fetched from {initialFile.includes("http") ? "the web" : "HuggingFace"} when a run
          needs it, so there is nothing stored here to preview. Open the link above to read it,
          or pick one of this file set&apos;s own files.
        </p>
      ) : (previewError || !preview || preview.kind !== "empty") && (
        <div className="mt-3 font-mono text-sm text-slate-400">
          {previewError
            ? <span className="text-coral-400">failed to load preview</span>
            : !preview
            ? "loading…"
            : tabular
            ? `${preview.file} · ${preview.kind} · ${preview.n_columns} cols · ${preview.n_rows} rows`
            : `${preview.file} · ${preview.kind}`}
        </div>
      )}

      {/* Plain text: it is a sentence about the dataset, not a callout. */}
      {preview && preview.notes && (
        <p className="mt-3 text-2xs leading-relaxed text-slate-400">
          {preview.kind === "empty" && (ds?.source?.remote.length ?? 0) > 0
            ? `${ds?.source?.remote.length} remote sources. They are fetched at run time; choose a source above to open it.`
            : preview.notes}
        </p>
      )}

      {tabular && preview && (
        <>
          <div className="mt-4">
            <Kicker>columns</Kicker>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {preview.columns.map((c) => (
                <span key={c} className="rounded-md border border-hair bg-raised px-2 py-0.5 font-mono text-2xs text-slate-300">
                  {c}
                </span>
              ))}
            </div>
          </div>
          <div className="bezel-flat mt-3 max-h-[45vh] overflow-auto rounded-bezel border border-hair">
            <table className="w-full text-2xs">
              <thead className="table-label bg-raised text-left">
                <tr>
                  {preview.columns.map((c) => <th key={c} className="px-2.5 py-2 font-normal">{c}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-hair font-mono">
                {preview.preview_rows.map((r, i) => (
                  <tr key={i} className="transition hover:bg-raised/60">
                    {preview.columns.map((c) => (
                      <td key={c} className="max-w-xs truncate px-2.5 py-1.5 text-slate-300">
                        {typeof r[c] === "object" && r[c] !== null ? JSON.stringify(r[c]) : String(r[c] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Scripts get line numbers and no wrapping — an eval script is read by
          structure, and a soft-wrapped line hides its indentation. */}
      {preview && preview.kind === "script" && (
        <div className="mt-4 max-h-[45vh] overflow-auto rounded-bezel border border-hair bg-canvas/60">
          <table className="w-full border-collapse font-mono text-2xs leading-relaxed">
            <tbody>
              {(preview.sample_text || "").split("\n").map((ln, i) => (
                <tr key={i}>
                  <td className="w-10 select-none border-r border-hair px-2 py-px text-right align-top text-slate-600">{i + 1}</td>
                  <td className="whitespace-pre px-3 py-px text-slate-300">{ln || " "}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {preview && (preview.kind === "pdf" || preview.kind === "txt") && (
        <pre className="mt-4 max-h-[45vh] overflow-auto whitespace-pre-wrap break-words rounded-bezel border border-hair bg-canvas/60 p-3 font-mono text-2xs text-slate-300">
          {preview.sample_text || "(no extractable text)"}
        </pre>
      )}

      {usedBy.length > 0 && (
        <div className="mt-5 border-t border-hair pt-4">
          <Kicker strong>Used by following tasks</Kicker>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {usedBy.map((t) => (
              <Link
                key={t.name}
                to={`/tasks?q=${encodeURIComponent(t.name)}`}
                onClick={onLeave}
                className="rounded-md border border-hair bg-raised px-2.5 py-1 font-mono text-2xs text-slate-300 transition hover:border-brass-500/40 hover:text-brass-300"
              >
                {t.name}
              </Link>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
