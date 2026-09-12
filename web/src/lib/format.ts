// Only metrics whose contract is a normalized ratio are percentages. Custom
// metrics default to their raw scale (latency_ms=250 must never become 25000%).
const PERCENTAGE_SCORE_METRICS = new Set([
  "accuracy", "exact_match", "em", "f1", "f1_micro", "f1_macro", "token_f1",
  "precision", "recall", "bleu", "rouge", "rouge_l", "pass_rate", "win_rate",
  "pass@1", "pass_at_1", "mc_loglikelihood", "accuracy_norm", "suite_average",
]);

export function isPercentageMetric(metric: string | null | undefined): boolean {
  const key = (metric ?? "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return PERCENTAGE_SCORE_METRICS.has(key)
    || key.endsWith("_accuracy")
    || key.endsWith("_f1")
    || key.endsWith("_precision")
    || key.endsWith("_recall");
}

export function fmtScore(
  value: number | null | undefined,
  metric: string | null | undefined = "",
): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return isPercentageMetric(metric)
    ? `${(value * 100).toFixed(1)}%`
    : value.toFixed(3);
}

/** Presentation-only casing for metric names; wire values stay untouched. */
export function fmtMetric(value: string | null | undefined): string {
  const metric = value?.trim() ?? "";
  if (!metric) return "—";
  return metric
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part, index) => {
      if (/^f\d+$/i.test(part)) return part.toUpperCase();
      return index === 0 ? part.charAt(0).toUpperCase() + part.slice(1) : part;
    })
    .join(" ");
}

/** Where an Auto run's held-out eval came from, in words. "" while unsettled. */
export function fmtEvalSource(value: string | null | undefined): string {
  switch (value) {
    case "public_benchmark": return "public benchmark";
    case "synthesized": return "synthesized eval";
    default: return "";
  }
}

// Show file paths relative to the run-id segment (a UUID) so absolute host /
// cluster prefixes — usernames, home dirs, group names — aren't exposed on
// screen (e.g. during demos). "/orange/grp/user/zevo/<run-id>/train/model"
// becomes "<run-id>/train/model". Paths without a run-id fall back to trimming
// a leading home/scratch prefix; otherwise returned unchanged.
const _UUID = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;
export function relPath(p: string | null | undefined): string {
  if (!p) return p ?? "";
  const m = p.match(_UUID);
  if (m && m.index !== undefined) return p.slice(m.index);
  // No run-id in the path — strip a common personal prefix if present.
  return p.replace(
    /^\/(?:Users|home)\/[^/]+\/|^\/orange\/[^/]+\/[^/]+\/|^\/tmp\/[^/]+\//,
    "",
  );
}

// Same idea as relPath but for an arbitrary text blob (e.g. a JSON payload dump /
// resolved refs): every embedded absolute path is trimmed down to its run-id
// segment, so usernames + home/cluster prefixes don't leak on screen.
export function scrubPaths(s: string | null | undefined): string {
  if (!s) return s ?? "";
  return s
    // "/anything/<run-id>/rest"  ->  "<run-id>/rest"
    .replace(
      /"\/[^"]*?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/gi,
      '"$1',
    )
    // paths with no run-id: strip a leading personal prefix
    .replace(/"\/(?:Users|home)\/[^/"]+\//g, '"')
    .replace(/"\/orange\/[^/"]+\/[^/"]+\//g, '"');
}

// Scrub FREE-FORM text (e.g. a transcript's bash commands / stdout), where paths
// aren't quoted. Any absolute path with a run-id is trimmed to the run-id; remote
// / home roots (/orange/grp/user, /home/x, /Users/x) collapse to ~. Keeps the
// command readable while hiding usernames + cluster paths on screen.
export function scrubText(s: string | null | undefined): string {
  if (!s) return s ?? "";
  return s
    .replace(
      /\/[^\s"'`]*?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/gi,
      "$1",
    )
    .replace(/\/orange\/[^/\s"'`]+\/[^/\s"'`]+/g, "~")
    .replace(/\/(?:Users|home)\/[^/\s"'`]+/g, "~")
    // bare "/orange" not part of a longer path (e.g. "shared /orange scratch")
    .replace(/ ?\/orange\b(?!\/)/g, "");
}

/** A timestamp as the console shows them: 24-hour, to the minute.
 *
 *  `toLocaleString()` gave a 12-hour clock and seconds — the seconds are noise
 *  in a list of runs hours apart, and AM/PM costs three characters that the
 *  columns beside it need more.
 */
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, {
      year: "numeric", month: "numeric", day: "numeric",
      hour: "2-digit", minute: "2-digit", hour12: false,
    });
  } catch {
    return iso;
  }
}

/** A token count at a glance: 812 · 4.3K · 64.22M · 1.20B.
 *
 *  Lived in two files with the same body; the dashboard needed a third copy,
 *  which is one more than a formatting rule should have.
 */
export function fmtTokens(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0";
  if (n < 1_000) return String(Math.round(n));
  if (n < 1_000_000) return `${(n / 1_000).toFixed(1)}K`;
  if (n < 1_000_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  return `${(n / 1_000_000_000).toFixed(2)}B`;
}

/** Dollars, at a precision scaled to the size of the figure.
 *
 *  The ONE cost formatter — three pages used to carry their own near-copies.
 *  Zero or absent renders as the "—" empty-value glyph: a run that spent
 *  nothing has nothing to report, which is a different fact from $0.00.
 *  Sub-cent figures get 4 places and sub-dollar figures 3, so a cheap
 *  heartbeat reads as its actual cost instead of rounding to $0.00; a dollar
 *  or more reads to the cent like a price. */
export function fmtCost(usd: number | null | undefined): string {
  const n = Number(usd);
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  if (n < 1) return `$${n.toFixed(3)}`;
  return `$${n.toFixed(2)}`;
}

/** A duration in seconds, at the two largest units that matter.
 *
 *  The ONE duration formatter — Runs, run detail, the leaderboard and the
 *  agent page each had their own. Under a minute it is seconds, under an hour
 *  minutes and seconds, above that hours and minutes: two units is what a
 *  column scans at, and the dropped third unit is noise at that magnitude.
 *  Negative or absent renders as the "—" empty-value glyph. */
export function fmtDuration(seconds: number | null | undefined): string {
  const n = Number(seconds);
  if (!Number.isFinite(n) || n < 0) return "—";
  const s = Math.floor(n % 60);
  const m = Math.floor((n / 60) % 60);
  const h = Math.floor(n / 3600);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** A byte count at a glance: 812 B · 4.3 KB · 64.2 MB · 1.20 GB.
 *
 *  The ONE bytes formatter — Files and the artifacts panel had diverging
 *  copies, one of which topped out at MB and printed models as `4300.0 MB`. */
export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

/** A model id without the org that published it: `Qwen/Qwen3-4B` → `Qwen3-4B`.
 *
 *  The org is the same word on every row of a family and never the part being
 *  read. Callers keep the full id in a `title`, so it stays copyable. Ids with
 *  no slash (Bedrock's `moonshotai.kimi-k2.5`) come back unchanged. */
export function shortModel(id: string | null | undefined): string {
  if (!id) return "";
  return id.split("/").pop() || id;
}

/** A dataset's files, grouped by the folder they sit in.
 *
 *  A bundle can split itself into `train/`, `validation/` and `test/`, and the
 *  file names then arrive as `train/train.json`. Grouping on that prefix is
 *  what makes the list readable: three files called "the scorer" are told apart
 *  by which folder they are in, not by their names.
 *
 *  Files sitting loose in the root keep an empty folder name and come last,
 *  unlabelled — a dataset that uses no subfolders then renders exactly as it
 *  did before there were any.
 */
export type FileGroup = {
  folder: string;
  /** What to print above the group, or "" for nothing. */
  label: string;
  /** Whether its files sit under a heading and should be indented. */
  indent: boolean;
  files: { path: string; leaf: string }[];
};

// The pipeline's own order, not the alphabet's: train comes before validation
// comes before test, and sorting by name would put test first.
const FOLDER_ORDER = ["train", "training", "validation", "val", "test", "eval"];

/** The name the root group goes by when it is one group among several. */
export const ROOT_LABEL = "others";

export function groupByFolder(paths: string[]): FileGroup[] {
  const groups = new Map<string, { path: string; leaf: string }[]>();
  for (const p of paths) {
    const cut = p.indexOf("/");
    const folder = cut < 0 ? "" : p.slice(0, cut);
    const leaf = cut < 0 ? p : p.slice(cut + 1);
    if (!groups.has(folder)) groups.set(folder, []);
    groups.get(folder)!.push({ path: p, leaf });
  }
  const rank = (f: string) => {
    if (!f) return 1000;               // the unlabelled group, last
    const i = FOLDER_ORDER.indexOf(f.toLowerCase());
    return i < 0 ? 500 : i;            // unknown folders between the two
  };
  const ordered = [...groups.entries()]
    .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b));
  return ordered.map(([folder, files]) => ({
    folder,
    // The root group is named only when there is something to tell it apart
    // FROM. A dataset that uses no subfolders is not "others", it is just the
    // dataset, and heading its one list would be labelling the obvious.
    label: folder ? `${folder}/` : (ordered.length > 1 ? ROOT_LABEL : ""),
    // And its files are not indented: nothing contains them. Only the files
    // that really sit inside a folder are stepped in under its name.
    indent: !!folder,
    files,
  }));
}

/** The container-side files root — the same place `zevo.paths.files_root()`
 *  serves (compose mounts `./data:/app/data`). The ONE frontend copy. */
export const FILES_ROOT = "/app/data/files";

/** A catalogued file, split into the file set that holds it and its path inside.
 *
 *  `/app/data/files/capybara-if/test/test.csv` is the file set
 *  `capybara-if` and the file `test/test.csv`. Taking the last two segments
 *  instead — which is what this used to do everywhere — reads the SUBFOLDER as
 *  the dataset and shows `test/test.csv`, losing the only part that says which
 *  bundle it belongs to. Worse, that name is what the preview is opened with,
 *  so the link led to a dataset called `test` that does not exist.
 *
 *  Returns null for anything outside the catalogue: a path the user typed has
 *  no dataset to name.
 */
export function splitDatasetPath(
  path: string,
): { dataset: string; file: string; label: string } | null {
  const p = (path || "").replace(/\/+$/, "");
  if (!p.startsWith(`${FILES_ROOT}/`)) return null;
  let rest = p.slice(FILES_ROOT.length + 1);
  // The hosted edition keeps each workspace's file sets under
  // `_t/<workspace id>/`. That namespace is plumbing, not a dataset: skip it
  // so the label and the preview link name the real set.
  const namespaced = rest.match(/^_t\/[^/]+\/(.+)$/);
  if (namespaced) rest = namespaced[1];
  const cut = rest.indexOf("/");
  if (cut < 0) return null;
  const dataset = rest.slice(0, cut);
  const file = rest.slice(cut + 1);
  return { dataset, file, label: `${dataset}/${file}` };
}
