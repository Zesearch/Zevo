import { useRef, useState } from "react";
import { Paperclip, X, Upload, Loader2 } from "lucide-react";
import { api } from "../lib/api";
import { bytes } from "../lib/format";

export type Attachment = {
  path: string;
  name: string;
  size_bytes: number;
  mime: string;
};

export function AttachmentDropzone({
  attachments,
  onChange,
}: {
  attachments: Attachment[];
  onChange: (next: Attachment[]) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function uploadOne(file: File): Promise<Attachment> {
    const fd = new FormData();
    fd.append("file", file);
    return await api<Attachment>("/attachments", { method: "POST", body: fd });
  }

  async function ingestFiles(files: FileList | File[]) {
    setError(null);
    setUploading(true);
    const list = Array.from(files);
    const uploaded: Attachment[] = [];
    try {
      for (const f of list) {
        uploaded.push(await uploadOne(f));
      }
      onChange([...attachments, ...uploaded]);
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setUploading(false);
    }
  }

  function remove(idx: number) {
    onChange(attachments.filter((_, i) => i !== idx));
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
        className="btn !text-[13px] disabled:opacity-50"
      >
        {uploading ? <Loader2 size={13} className="animate-spin" /> : <Upload size={13} />}
        {uploading ? "Uploading…" : "Upload files"}
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) void ingestFiles(e.target.files);
          e.target.value = "";
        }}
      />
      {error && (
        <div className="mt-2 flex items-center gap-2 rounded-bezel border border-coral-500/30 bg-coral-500/10 px-3 py-2 text-2xs text-coral-300">
          <span className="lamp lamp-coral shrink-0" /> {error}
        </div>
      )}
      {attachments.length > 0 && (
        <ul className="stagger mt-2 space-y-1.5">
          {attachments.map((a, i) => (
            <li
              key={a.path}
              className="bezel-flat flex items-center justify-between gap-2 px-3 py-2 text-2xs"
            >
              <div className="flex min-w-0 items-center gap-2">
                <Paperclip size={13} className="shrink-0 text-brass-400/70" />
                <span className="truncate font-mono text-slate-300">{a.name}</span>
                <span className="readout text-slate-600">{bytes(a.size_bytes)}</span>
              </div>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  remove(i);
                }}
                className="rounded p-0.5 text-slate-500 transition hover:text-coral-300"
                title="remove"
              >
                <X size={13} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
