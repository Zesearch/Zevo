import { useRef, useState } from "react";
import useSWR from "swr";
import {
  Server, RotateCw, Trash2, Plus, ShieldCheck, ShieldAlert, Clock, Pencil,
  ChevronRight, FileText, Upload, X,
} from "lucide-react";
import { api } from "../lib/api";
import { Modal } from "./Modal";

// One row as GET /hardware/ssh returns it (never includes credential content).
export type SshHost = {
  id: string;
  name: string;
  category: "cluster" | "instance";
  host: string;
  port: number;
  username: string;
  private_key_path: string;
  private_key_uploaded: boolean;
  authentication: "private_key" | "password" | "none";
  remote_dir: string;
  env_setup: string;
  container_image: string;
  skill_path: string;
  status: string; // "verified" | "unverified" | "failed"
  gpu_info: unknown;
  last_error: string;
  created_at: string;
  last_verified_at: string | null;
};

type SkillFile = { name: "SKILL.md"; markdown: string };

function SkillFilePicker({
  selected,
  currentPath = "",
  onChange,
}: {
  selected: SkillFile | null;
  currentPath?: string;
  onChange: (file: SkillFile | null) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState("");

  async function choose(file?: File) {
    if (!file) return;
    setError("");
    if (file.name.toLowerCase() !== "skill.md") {
      setError("Choose a file named SKILL.md.");
      return;
    }
    if (file.size > 50_000) {
      setError("SKILL.md must be 50 KB or smaller.");
      return;
    }
    const markdown = await file.text();
    if (!markdown.trim()) {
      setError("SKILL.md cannot be empty.");
      return;
    }
    onChange({ name: "SKILL.md", markdown });
  }

  return (
    <div className="mt-0.5">
      <div className="flex min-w-0 items-center gap-2">
        <button type="button" onClick={() => inputRef.current?.click()} className="btn !px-2.5 !py-1.5 !text-[12px]">
          <Upload size={12} /> {currentPath ? "replace SKILL.md" : "upload SKILL.md"}
        </button>
        <div className="flex min-w-0 items-center gap-1.5 font-mono text-2xs text-slate-500">
          <FileText size={12} className="shrink-0" />
          <span className="truncate" title={selected?.name || currentPath}>
            {selected ? selected.name : currentPath || "No file selected"}
          </span>
        </div>
        {selected && (
          <button type="button" onClick={() => onChange(null)} title="Keep the current Skill" className="shrink-0 text-slate-500 hover:text-coral-300">
            <X size={13} />
          </button>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".md,text/markdown,text/plain"
        className="hidden"
        onChange={(event) => {
          void choose(event.target.files?.[0]);
          event.target.value = "";
        }}
      />
      {error && <p className="mt-1 text-2xs text-coral-300">{error}</p>}
    </div>
  );
}

type KeyFile = { name: string; contents: string };

function PrivateKeyPicker({
  selected, onChange, replace = false,
}: {
  selected: KeyFile | null;
  onChange: (file: KeyFile | null) => void;
  replace?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState("");

  async function choose(file?: File) {
    if (!file) return;
    setError("");
    if (file.name.toLowerCase().endsWith(".pub")) {
      setError("That is the public key. Choose the private key file (e.g. id_ed25519).");
      return;
    }
    if (file.size > 20_000) {
      setError("Private key must be 20 KB or smaller.");
      return;
    }
    const contents = await file.text();
    if (!contents.includes("PRIVATE KEY")) {
      setError("This does not look like an OpenSSH or PEM private key.");
      return;
    }
    onChange({ name: file.name, contents });
  }

  return (
    <div className="mt-0.5">
      <div className="flex min-w-0 items-center gap-2">
        <button type="button" onClick={() => inputRef.current?.click()} className="btn !px-2.5 !py-1.5 !text-[12px]">
          <Upload size={12} /> {replace ? "replace private key" : "upload private key"}
        </button>
        <div className="flex min-w-0 items-center gap-1.5 font-mono text-2xs text-slate-500">
          <FileText size={12} className="shrink-0" />
          <span className="truncate" title={selected?.name}>{selected ? selected.name : "No file selected"}</span>
        </div>
        {selected && (
          <button type="button" onClick={() => onChange(null)} title="Remove" className="shrink-0 text-slate-500 hover:text-coral-300">
            <X size={13} />
          </button>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        onChange={(event) => {
          void choose(event.target.files?.[0]);
          event.target.value = "";
        }}
      />
      {error && <p className="mt-1 text-2xs text-coral-300">{error}</p>}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "verified")
    return <span className="inline-flex items-center gap-1 rounded border border-phosphor-500/40 bg-phosphor-500/10 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-phosphor-300"><ShieldCheck size={10} /> verified</span>;
  if (status === "failed")
    return <span className="inline-flex items-center gap-1 rounded border border-coral-500/40 bg-coral-500/10 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-coral-300"><ShieldAlert size={10} /> failed</span>;
  return <span className="inline-flex items-center gap-1 rounded border border-hair bg-raised/50 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.14em] text-slate-500"><Clock size={10} /> unverified</span>;
}

function AddForm({ onAdded }: { onAdded: () => void }) {
  const [open, setOpen] = useState(false);
  const empty = {
    name: "",
    category: "" as "" | "cluster" | "instance",
    host: "",
    port: "22",
    username: "",
    password: "",
    remote_parent_dir: "",
    env_setup: "",
    container_image: "",
  };
  const [f, setF] = useState(empty);
  const [skillFile, setSkillFile] = useState<SkillFile | null>(null);
  const [keyFile, setKeyFile] = useState<KeyFile | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof f) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setF((s) => ({ ...s, [k]: e.target.value }));

  async function submit() {
    setBusy(true); setError(null);
    try {
      await api("/hardware/ssh", {
        method: "POST",
        body: JSON.stringify({
          name: f.name.trim(), category: f.category, host: f.host.trim(),
          port: Number(f.port) || 22, username: f.username.trim(),
          private_key: keyFile?.contents ?? "",
          password: f.password,
          remote_parent_dir: f.remote_parent_dir.trim(),
          env_setup: f.env_setup.trim(),
          container_image: f.category === "cluster" ? f.container_image.trim() : "",
          ...(skillFile ? {
            skill_filename: skillFile.name,
            skill_markdown: skillFile.markdown,
          } : {}),
        }),
      });
      setF(empty);
      setSkillFile(null);
      setOpen(false);
      onAdded();
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(false); }
  }

  const inputCls = "w-full rounded-md border border-hair bg-canvas px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-brass-500/50";
  const close = () => { setOpen(false); setKeyFile(null); setError(null); };
  return (
    <>
      <button onClick={() => setOpen(true)} className="btn btn-brass !px-2.5 !py-1 !text-[12px] uppercase !tracking-[0.1em]">
        <Plus size={12} className="mr-1 inline" /> add connection
      </button>
      <Modal open={open} onClose={close} title="Add SSH connection" width="max-w-2xl">
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        <label className="text-2xs text-slate-400">Name
          <input className={inputCls} value={f.name} onChange={set("name")} placeholder="My GPU connection" autoComplete="off" />
        </label>
        <label className="text-2xs text-slate-400">Category
          <select className={inputCls} value={f.category} onChange={(e) => setF((s) => ({ ...s, category: e.target.value as typeof f.category }))}>
            <option value="">Choose…</option>
            <option value="cluster">Cluster</option>
            <option value="instance">Instance</option>
          </select>
        </label>
        <label className="text-2xs text-slate-400">Host
          <input className={inputCls} value={f.host} onChange={set("host")} placeholder="1.2.3.4 or box.example.com" autoComplete="off" />
        </label>
        <label className="text-2xs text-slate-400">Port
          <input className={inputCls} value={f.port} onChange={set("port")} placeholder="22" autoComplete="off" />
        </label>
        <label className="text-2xs text-slate-400">User
          <input className={inputCls} value={f.username} onChange={set("username")} placeholder="username" autoComplete="off" />
        </label>
        <label className="text-2xs text-slate-400">Remote directory
          <input className={inputCls} value={f.remote_parent_dir} onChange={set("remote_parent_dir")} placeholder="/scratch/my-user" autoComplete="off" />
        </label>
      </div>
      <label className="mt-2.5 block text-2xs text-slate-400">Environment
        <input className={`${inputCls} mt-0.5`} value={f.env_setup} onChange={set("env_setup")}
          placeholder="source ~/miniconda3/etc/profile.d/conda.sh && conda activate my-env"
          autoComplete="off" spellCheck={false} />
      </label>
      <p className="mt-1 text-2xs text-slate-500">
        Verification creates <code>zevo</code> inside the remote directory and requires this command to succeed.
      </p>
      <label className="mt-2.5 block text-2xs text-slate-400">
        Container image <span className="text-slate-600">(Cluster only · optional)</span>
        <input
          className={`${inputCls} mt-0.5 disabled:cursor-not-allowed disabled:opacity-40`}
          value={f.container_image}
          onChange={set("container_image")}
          disabled={f.category === "instance"}
          placeholder="/absolute/path/to/runtime.sqsh or registry/image:tag"
          autoComplete="off"
          spellCheck={false}
        />
      </label>
      <div className="mt-2.5 text-2xs text-slate-400">
        Infrastructure SKILL.md <span className="text-slate-600">(optional site rules)</span>
        <SkillFilePicker selected={skillFile} onChange={setSkillFile} />
      </div>
      <p className="mt-1 text-2xs text-slate-500">
        Upload a complete SKILL.md with name and description frontmatter. Zevo installs its instruction body under <code>playbook/skills/infrastructure</code>. Do not include credentials or private-key contents.
      </p>
      <div className="mt-2.5 text-2xs text-slate-400">
        Private key <span className="text-slate-600">(optional)</span>
        <PrivateKeyPicker selected={keyFile} onChange={setKeyFile} />
      </div>
      <label className="mt-2.5 block text-2xs text-slate-400">Password <span className="text-slate-600">(optional)</span>
        <input className={`${inputCls} mt-0.5`} type="password" value={f.password} onChange={set("password")}
          placeholder="SSH password" autoComplete="new-password" spellCheck={false} />
      </label>
      <p className="mt-1 text-2xs text-slate-500">
        Upload the private key file (e.g. <code>id_ed25519</code>) or enter a password, not both. Zevo keeps the key in its own credential store, readable only by Zevo.
      </p>
      {error && <div className="mt-2 rounded-md border border-coral-500/30 bg-coral-500/10 p-2 text-2xs text-coral-300">{error}</div>}
      <div className="mt-4 flex justify-end gap-2">
        <button onClick={close} className="btn justify-center !px-3 !py-1 !text-[12px] uppercase !tracking-[0.1em]">cancel</button>
        <button onClick={submit} disabled={busy || !f.name.trim() || !f.category || !f.host.trim() || !f.username.trim() || !f.remote_parent_dir.trim() || !f.env_setup.trim() || (Boolean(keyFile) === Boolean(f.password))}
          className="btn btn-brass justify-center !px-3 !py-1 !text-[12px] uppercase !tracking-[0.1em] !text-phosphor-300 disabled:opacity-40">
          {busy ? <RotateCw size={11} className="inline animate-spin" /> : "save & verify"}
        </button>
      </div>
      </Modal>
    </>
  );
}

type EditDraft = {
  name: string;
  category: "cluster" | "instance";
  host: string;
  port: string;
  username: string;
  remote_parent_dir: string;
  env_setup: string;
  container_image: string;
  password: string;
};

function editDraft(h: SshHost): EditDraft {
  return {
    name: h.name,
    category: h.category,
    host: h.host,
    port: String(h.port),
    username: h.username,
    remote_parent_dir: h.remote_dir,
    env_setup: h.env_setup,
    container_image: h.container_image,
    password: "",
  };
}

function HostRow({ h, onChanged }: { h: SshHost; onChanged: () => void }) {
  const [busy, setBusy] = useState<"" | "save" | "delete">("");
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<EditDraft>(() => editDraft(h));
  const [skillFile, setSkillFile] = useState<SkillFile | null>(null);
  const [keyFile, setKeyFile] = useState<KeyFile | null>(null);
  const [error, setError] = useState<string | null>(null);

  const set = (key: keyof EditDraft) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>,
  ) => setDraft((current) => ({ ...current, [key]: e.target.value }));

  async function save() {
    setBusy("save"); setError(null);
    try {
      await api(`/hardware/ssh/${h.id}`, {
        method: "PUT",
        body: JSON.stringify({
          name: draft.name.trim(),
          category: draft.category,
          host: draft.host.trim(),
          port: Number(draft.port) || 22,
          username: draft.username.trim(),
          remote_parent_dir: draft.remote_parent_dir.trim(),
          env_setup: draft.env_setup.trim(),
          container_image: draft.category === "cluster" ? draft.container_image.trim() : "",
          ...(skillFile ? {
            skill_filename: skillFile.name,
            skill_markdown: skillFile.markdown,
          } : {}),
          private_key: keyFile?.contents ?? "",
          password: draft.password,
        }),
      });
      setEditing(false);
      setSkillFile(null);
      onChanged();
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(""); }
  }
  async function del() {
    if (!confirm(`Delete SSH connection "${h.name || h.host}"?`)) return;
    setBusy("delete");
    try { await api(`/hardware/ssh/${h.id}`, { method: "DELETE" }); onChanged(); }
    finally { setBusy(""); }
  }

  const inputCls = "w-full rounded-md border border-hair bg-canvas px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-brass-500/50";
  const invalidCredential = Boolean(keyFile) && Boolean(draft.password);
  const saveDisabled = busy === "save" || !draft.name.trim() || !draft.host.trim()
    || !draft.username.trim() || !draft.remote_parent_dir.trim()
    || !draft.env_setup.trim() || invalidCredential;
  return (
    <div className="py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Server size={13} className="text-brass-400" />
            <span className="font-display text-sm font-semibold text-ink">{h.name || h.host}</span>
            <span className="rounded border border-hair bg-raised/40 px-1.5 py-0.5 font-mono text-2xs uppercase tracking-[0.12em] text-slate-400">{h.category}</span>
            {h.status !== "verified" && <StatusBadge status={h.status} />}
          </div>
          <details className="group mt-2 min-w-0">
            <summary className="flex w-fit cursor-pointer list-none items-center gap-1 font-mono text-2xs text-slate-500 transition hover:text-slate-300">
              <ChevronRight size={11} className="transition-transform group-open:rotate-90" />
              Connection details
            </summary>
            <dl className="mt-2 grid min-w-0 gap-x-3 gap-y-1.5 border-l border-hair pl-3 font-mono text-2xs sm:grid-cols-[8.5rem_minmax(0,1fr)]">
              <dt className="text-slate-500">SSH host</dt>
              <dd className="min-w-0 overflow-x-auto whitespace-nowrap pb-3 text-slate-400">{h.username}@{h.host}:{h.port}</dd>
              <dt className="text-slate-500">Remote directory</dt>
              <dd className="min-w-0 overflow-x-auto whitespace-nowrap pb-3 text-slate-400">{h.remote_dir}</dd>
              <dt className="text-slate-500">Environment</dt>
              <dd className="min-w-0 overflow-x-auto whitespace-nowrap pb-3 text-slate-400">{h.env_setup}</dd>
              {h.category === "cluster" && <>
                <dt className="text-slate-500">Container image</dt>
                <dd className="min-w-0 overflow-x-auto whitespace-nowrap pb-3 text-slate-400">{h.container_image || "Not set"}</dd>
              </>}
              <dt className="text-slate-500">Infrastructure Skill</dt>
              <dd className="min-w-0 overflow-x-auto whitespace-nowrap pb-3 text-slate-400">{h.skill_path || "Not set"}</dd>
              {h.authentication === "private_key" ? (
                <>
                  <dt className="text-slate-500">Private key</dt>
                  <dd className="min-w-0 overflow-x-auto whitespace-nowrap pb-3 text-slate-400">{h.private_key_uploaded ? "uploaded · stored by Zevo" : h.private_key_path}</dd>
                </>
              ) : (
                <>
                  <dt className="text-slate-500">Authentication</dt>
                  <dd className="text-slate-400">Password</dd>
                </>
              )}
            </dl>
          </details>
          {h.status !== "verified" && h.last_error && (
            <div className="mt-1.5 text-2xs text-coral-300/80">{h.last_error}</div>
          )}
        </div>
        <div className="flex flex-shrink-0 gap-1.5">
          <button onClick={() => { setDraft(editDraft(h)); setSkillFile(null); setEditing(true); setError(null); }} disabled={!!busy} className="btn !px-2 !py-1 !text-[12px] uppercase !tracking-[0.1em] disabled:opacity-40">
            <Pencil size={11} /> edit
          </button>
          <button onClick={del} disabled={!!busy} className="btn !px-2 !py-1 !text-[12px] uppercase !tracking-[0.1em] !text-coral-300 disabled:opacity-40">
            {busy === "delete" ? <RotateCw size={11} className="inline animate-spin" /> : <><Trash2 size={11} /> delete</>}
          </button>
        </div>
      </div>

      {editing && (
        <div className="mt-3 rounded-md border border-hair bg-canvas/40 p-3">
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            <label className="text-2xs text-slate-400">Name<input className={inputCls} value={draft.name} onChange={set("name")} /></label>
            <label className="text-2xs text-slate-400">Category
              <select className={inputCls} value={draft.category} onChange={set("category")}>
                <option value="cluster">Cluster</option><option value="instance">Instance</option>
              </select>
            </label>
            <label className="text-2xs text-slate-400">Host<input className={inputCls} value={draft.host} onChange={set("host")} /></label>
            <label className="text-2xs text-slate-400">Port<input className={inputCls} value={draft.port} onChange={set("port")} /></label>
            <label className="text-2xs text-slate-400">User<input className={inputCls} value={draft.username} onChange={set("username")} /></label>
            <label className="text-2xs text-slate-400">Remote directory<input className={inputCls} value={draft.remote_parent_dir} onChange={set("remote_parent_dir")} /></label>
          </div>
          <label className="mt-2.5 block text-2xs text-slate-400">Environment<input className={inputCls} value={draft.env_setup} onChange={set("env_setup")} /></label>
          <label className="mt-2.5 block text-2xs text-slate-400">
            Container image <span className="text-slate-600">(Cluster only · optional)</span>
            <input
              className={`${inputCls} disabled:cursor-not-allowed disabled:opacity-40`}
              value={draft.container_image}
              onChange={set("container_image")}
              disabled={draft.category === "instance"}
              placeholder="/absolute/path/to/runtime.sqsh or registry/image:tag"
            />
          </label>
          <div className="mt-2.5 text-2xs text-slate-400">
            Infrastructure SKILL.md <span className="text-slate-600">(optional site rules)</span>
            <SkillFilePicker selected={skillFile} currentPath={h.skill_path} onChange={setSkillFile} />
          </div>
          <div className="mt-2.5 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            <div className="text-2xs text-slate-400">New private key <span className="text-slate-600">(optional)</span><PrivateKeyPicker selected={keyFile} onChange={setKeyFile} replace /></div>
            <label className="text-2xs text-slate-400">New password <span className="text-slate-600">(optional)</span><input className={inputCls} type="password" value={draft.password} onChange={set("password")} placeholder="Keep blank to retain current credential" autoComplete="new-password" /></label>
          </div>
          <p className="mt-1 text-2xs text-slate-500">Leave both credential fields blank to keep the current authentication method.</p>
          <div className="mt-2 flex gap-2">
            <button onClick={save} disabled={saveDisabled} className="btn justify-center !px-3 !py-1 !text-[12px] uppercase !tracking-[0.1em] !text-phosphor-300 disabled:opacity-40">
              {busy === "save" ? <RotateCw size={11} className="inline animate-spin" /> : "save & verify"}
            </button>
            <button onClick={() => { setEditing(false); setError(null); }} className="btn justify-center !px-3 !py-1 !text-[12px] uppercase !tracking-[0.1em]">cancel</button>
          </div>
        </div>
      )}

      {error && <div className="mt-2 rounded-md border border-coral-500/30 bg-coral-500/10 p-2 text-2xs text-coral-300">{error}</div>}
    </div>
  );
}

export function SshConnections({
  embedded = false,
}: {
  embedded?: boolean;
}) {
  const { data, mutate } = useSWR<SshHost[]>("/api/hardware/ssh");
  const hosts = data ?? [];
  const hasConnections = hosts.length > 0;
  return (
    <section className={embedded ? "min-w-0 rounded-md border border-hair bg-panel/40 p-4 sm:p-5" : "mt-9"}>
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className={`font-display font-semibold tracking-tight text-ink ${embedded ? "text-base" : "text-lg"}`}>SSH connections</h2>
        <AddForm onAdded={() => void mutate()} />
      </div>
      <p className="mb-3 max-w-3xl text-2xs leading-relaxed text-slate-500">
        Cluster and Instance use this shared list. Every connection is verified before it can be selected for a Run.
      </p>
      <div className={embedded ? "mt-3" : "rounded-bezel border border-hair bg-panel/40 px-5"}>
        {!hasConnections ? (
          <div className={embedded ? "py-3 text-2xs text-slate-500" : "py-6 text-center text-2xs text-slate-500"}>No SSH connections yet.</div>
        ) : (
          <div className="divide-y divide-hair/70">
            {hosts.map((h) => <HostRow key={h.id} h={h} onChanged={() => void mutate()} />)}
          </div>
        )}
      </div>
    </section>
  );
}
