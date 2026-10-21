import type { TaskTestSet } from "./api";
import type { RunInputValues, RunSummaryValue } from "../components/RunInputs";

export function gpuAllocationLabel(value: string): string {
  return ({ per_run: "Entire run", per_stage: "Per stage", per_submission: "Per submission (Slurm)" } as Record<string, string>)[value] || value || "Not recorded";
}
export function resourceSummaryValues(inputs: Pick<RunInputValues, "adaptiveMidTrainingEvaluation" | "numGpus" | "gpuAllocationMode" | "generation_backend" | "iterations" | "budget" | "timeLimitHours" | "queueWaitHours" | "stopThreshold">): RunSummaryValue[] {
  return [
    { name: "Maximum GPUs", value: inputs.numGpus.trim() || "Unlimited", overridden: !!inputs.numGpus.trim() },
    { name: "GPU allocation", value: gpuAllocationLabel(inputs.gpuAllocationMode), overridden: true },
    { name: "Mid-training evaluation", value: inputs.adaptiveMidTrainingEvaluation ? "Agent decides" : "Off", overridden: inputs.adaptiveMidTrainingEvaluation },
    { name: "Generation backend", value: (inputs.generation_backend || "vllm").toUpperCase(), overridden: !!inputs.generation_backend },
    { name: "Iterations", value: inputs.iterations.trim() || "Unlimited", overridden: !!inputs.iterations.trim() },
    { name: "Budget", value: inputs.budget.trim() ? `$${inputs.budget.trim()}` : "Unlimited", overridden: !!inputs.budget.trim() },
    { name: "Time limit", value: inputs.timeLimitHours.trim() ? `${inputs.timeLimitHours.trim()} h` : "Unlimited", overridden: !!inputs.timeLimitHours.trim() },
    { name: "Queue limit", value: Number(inputs.queueWaitHours) > 0 ? `${inputs.queueWaitHours.trim()} h` : "Unlimited", overridden: !!inputs.queueWaitHours.trim() },
    { name: "Stop threshold", value: inputs.stopThreshold.trim() || "Not set", overridden: !!inputs.stopThreshold.trim() },
  ];
}
export function scoringSummaryValues(suite: TaskTestSet[], prefix: string): RunSummaryValue[] {
  return suite.flatMap((item, index) => {
    const values: Array<[string, string]> = [
      ["Name", item.name], ["Dataset", item.test_set], ["Split", item.split || "Default"],
      ["Config", item.config || "Default"], ["Maximum rows", item.max_rows ? String(item.max_rows) : "All rows"],
      ["Inference query", item.inference_query], ["Sample submission", item.sample_submission],
      ["Metric type", item.metric_type], ["Metric", item.metric],
      ["Direction", item.metric_direction], ["Answer fields", (item.answer_fields || []).join(", ")],
      ["Prediction column", item.prediction_column || "prediction"], ["Evaluation script", item.evaluation_script || "Built-in"],
    ];
    return values.map(([name, value]) => ({ name: `${prefix} ${index + 1} · ${name}`, value: value || "Not set", overridden: Boolean(value) }));
  });
}
export function agentExecutionLabel(driver: string, model: string): string {
  if (driver === "holdout_data_runner") return "Test data preparation · no LLM";
  if (driver === "evaluation_runner") return "Evaluation runner · no LLM";
  return model === "no-llm" ? "System execution · no LLM" : model;
}
