import type { RunInputValues } from "../components/RunInputs";

type Training = Pick<RunInputValues, "dataset" | "dataQuery" | "baseModel" | "modelQuery" | "trainingMethod" | "methodQuery" | "teacherModel" | "rewardModel" | "usePeft">;
export function trainingStepComplete(value: Training): boolean {
  if (value.trainingMethod === "gkd" && !value.teacherModel.trim()) return false;
  if (value.trainingMethod === "online_dpo" && !value.rewardModel.trim()) return false;
  return [value.dataset, value.dataQuery, value.baseModel, value.modelQuery, value.trainingMethod, value.methodQuery, value.usePeft].some(item => Boolean(item.trim()));
}

type Runtime = Pick<RunInputValues, "numGpus" | "iterations" | "budget" | "timeLimitHours" | "queueWaitHours" | "stopThreshold" | "gpuAllocationMode" | "generation_backend" | "adaptiveMidTrainingEvaluation">;
export function runtimeStepComplete(value: Runtime): boolean {
  const limits = [value.numGpus, value.iterations, value.budget, value.timeLimitHours, value.queueWaitHours];
  if (limits.some(item => item.trim() && (!Number.isFinite(Number(item)) || Number(item) < 0))) return false;
  if (value.stopThreshold.trim() && !Number.isFinite(Number(value.stopThreshold))) return false;
  return limits.some(item => Boolean(item.trim())) || Boolean(value.stopThreshold.trim())
    || value.gpuAllocationMode !== "per_run" || Boolean(value.generation_backend) || value.adaptiveMidTrainingEvaluation;
}

export function trainingStepReady(value: Training): boolean {
  return !(value.trainingMethod === "gkd" && !value.teacherModel.trim()) && !(value.trainingMethod === "online_dpo" && !value.rewardModel.trim());
}
export function runtimeStepReady(value: Runtime): boolean {
  const values = [value.numGpus, value.iterations, value.budget, value.timeLimitHours, value.queueWaitHours];
  return values.every(item => !item.trim() || (Number.isFinite(Number(item)) && Number(item) >= 0))
    && (!value.queueWaitHours.trim() || Number(value.queueWaitHours) <= 168)
    && (!value.stopThreshold.trim() || Number.isFinite(Number(value.stopThreshold)));
}
