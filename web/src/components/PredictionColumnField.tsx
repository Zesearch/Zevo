import { useEffect, useId } from "react";
import useSWR from "swr";
import { splitDatasetPath } from "../lib/format";

function previewUrl(path: string): string | null {
  const ref = splitDatasetPath(path);
  return ref ? `/api/files/${encodeURIComponent(ref.dataset)}/preview?n=1&file=${encodeURIComponent(ref.file)}` : null;
}

export function PredictionColumnField({ sample, dataset, answerFields, value, onChange }: {
  sample: string;
  dataset: string;
  answerFields: string[];
  value: string;
  onChange: (value: string) => void;
}) {
  const id = useId();
  const { data: submission } = useSWR<{ columns: string[] }>(previewUrl(sample));
  const { data: source } = useSWR<{ columns: string[] }>(previewUrl(dataset));
  const answers = answerFields.map((field) => field.trim());
  const columns = (submission?.columns || []).filter((column) =>
    !source?.columns.includes(column) || answers.includes(column));
  // When both schemas are available, use the same identity-column rule as the scorer.
  const onlyColumn = submission && source && columns.length === 1 ? columns[0] : "";
  useEffect(() => {
    if (!value && onlyColumn) onChange(onlyColumn);
  }, [value, onlyColumn, onChange]);

  return (
    <div>
      <label htmlFor={id} className="field-label mb-1 block">Prediction column</label>
      <input
        id={id} list={`${id}-columns`} value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Auto when there is one output column"
        className="h-10 w-full rounded-md border border-hair bg-canvas px-2.5 font-mono text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none"
      />
      <datalist id={`${id}-columns`}>
        {columns.map((column) => <option key={column} value={column} />)}
      </datalist>
      <p className="mt-1 text-xs text-slate-500">Choose the sample-submission column to score against Answer fields.</p>
    </div>
  );
}
