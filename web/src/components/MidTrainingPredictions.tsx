import { useState } from "react";
import useSWR from "swr";
import { currentPredictionArtifacts } from "../lib/predictionArtifacts";
import { Pagination, TablePreview } from "./IterationDetailsPanel";
import type { Artifact, ArtifactDetail } from "./IterationDetailsPanel";
import { Kicker } from "./zevo/primitives";

/** Mounted per checkpoint ticket so selection and pagination cannot leak across checks. */
export function MidTrainingPredictions({ runId, ticketId }: { runId: string; ticketId: string }) {
  const { data: artifacts = [], error } = useSWR<Artifact[]>(
    `/api/runs/${encodeURIComponent(runId)}/artifacts`, { refreshInterval: 5000 },
  );
  const predictions = currentPredictionArtifacts(artifacts.filter((item) => item.ticket_id === ticketId));
  const [selectedId, setSelectedId] = useState<string>();
  const [page, setPage] = useState(1);
  const selected = predictions.find((item) => item.id === selectedId) || predictions[0];
  return <section className="space-y-3">
    <Kicker strong>Validation predictions</Kicker>
    {error ? <p className="text-sm text-coral-300">Unable to load predictions.</p>
      : !selected ? <p className="text-sm text-dim">Predictions will appear after checkpoint inference publishes its outputs.</p>
      : <>
        <div className="flex flex-wrap gap-2" aria-label="Prediction benchmarks">
          {predictions.map((item, index) => <button key={item.id} type="button"
            aria-pressed={item.id === selected.id}
            className={`btn ${item.id === selected.id ? "btn-brass" : ""}`}
            onClick={() => { setSelectedId(item.id); setPage(1); }}>
            {String(item.meta.suite_member_name || item.meta.test_set_name || `Validation ${index + 1}`)}
            {item.meta.n_rows != null && ` · ${Number(item.meta.n_rows).toLocaleString()}`}
          </button>)}
        </div>
        <PredictionPreview key={selected.id} runId={runId} artifact={selected} page={page} onPage={setPage} />
      </>}
  </section>;
}

function PredictionPreview({ runId, artifact, page, onPage }: {
  runId: string; artifact: Artifact; page: number; onPage: (page: number) => void;
}) {
  const { data, error } = useSWR<ArtifactDetail>(
    `/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifact.id)}?preview_page=${page}&preview_page_size=10`,
  );
  const rows = Number(artifact.meta.n_rows);
  const pageCount = Number.isFinite(rows) && rows > 0 ? Math.min(10, Math.ceil(rows / 10)) : 1;
  if (error) return <p className="text-sm text-coral-300">Unable to load prediction preview.</p>;
  if (!data) return <p className="text-sm text-dim">Loading preview…</p>;
  if (!data.preview_rows?.length) return <p className="text-sm text-dim">No tabular preview is available.</p>;
  return <>
    <TablePreview columns={data.preview_columns} rows={data.preview_rows} page={page} pageSize={10} />
    <Pagination page={page} pageCount={pageCount} onPage={onPage} />
  </>;
}
