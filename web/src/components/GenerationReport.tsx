export function GenerationReport({ report }: { report?: Record<string, unknown> }) {
  if (!report || !Object.keys(report).length) return null;
  const benchmarks = report.benchmarks as Record<string, { fraction?: number; max_new_tokens?: number;
    repetition?: { repetitive_fraction_of_truncated?: number }; examples?: { beginning: string; ending: string }[] }> || {};
  const retained = report.retained_original as Record<string, string> || {};
  return <details className="my-3 rounded-bezel border border-sky-500/20 p-3 text-xs text-slate-300">
    <summary className="cursor-pointer text-sky-300">Generation diagnostics</summary>
    <p className="mt-2 text-dim">{report.lane === "held_out_test"
      ? "Held-out report · excluded from training decisions"
      : "Validation report · available to Orchestrator"}</p>
    {Object.entries(benchmarks).map(([name, item]) => <div key={name} className="mt-3">
      <div className="font-mono">{name} · {item.max_new_tokens ?? "?"} tokens · truncated {((item.fraction || 0) * 100).toFixed(1)}%</div>
      <div className="text-dim">Repetitive among truncated responses: {((item.repetition?.repetitive_fraction_of_truncated || 0) * 100).toFixed(1)}%</div>
      {retained[name] && <div className="text-sky-300">Original measurement retained · {retained[name].replaceAll("_", " ")}</div>}
      {!!item.examples?.length && <details className="mt-1"><summary className="cursor-pointer">Output examples</summary>
        {item.examples.map((sample, i) => <pre key={i} className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words">{sample.beginning}{"\n…\n"}{sample.ending}</pre>)}
      </details>}
    </div>)}
  </details>;
}
