type PredictionArtifact = {
  id: string;
  role: string;
  path: string;
  created_at: string;
  meta: Record<string, unknown>;
};

/** Call separately for each iteration and lane. WorkProducts are history:
 * repeated publication and length trials must not inflate benchmark counts. */
export function currentPredictionArtifacts<T extends PredictionArtifact>(artifacts: T[]): T[] {
  const current = new Map<string, T>();
  const ordered = [...artifacts].sort((a, b) =>
    b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id));
  for (const artifact of ordered) {
    if (artifact.role !== "predictions" || artifact.meta.diagnostic_only) continue;
    const meta = artifact.meta;
    const name = String(meta.suite_member_name || meta.test_set_name || "")
      .replace(/^(validation|test):/, "").trim();
    // Legacy unnamed artifacts can still be matched by their input dataset.
    // With no dataset identity, only collapse references to the same file.
    const key = name ? `member:${name}` : meta.scoring_set
      ? `dataset:${String(meta.scoring_set)}` : `path:${artifact.path || artifact.id}`;
    if (!current.has(key)) current.set(key, artifact);
  }
  return [...current.values()];
}
