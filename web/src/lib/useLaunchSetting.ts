import { useState } from "react";
import type { RunInputValues } from "../components/RunInputs";

const RUN_ONLY_FIELDS = new Set<keyof RunInputValues>([
  "gpuProvider", "cloudBackend", "sshHostId", "numGpus", "gpuAllocationMode",
  "generation_backend", "timeLimitHours", "queueWaitHours",
  // Customized execution preferences are not part of saved Setting identity.
  "promptFraming", "systemPrompt", "lossObjectiveConfig", "inferenceConfig",
  "decodingStrategy", "maxNewTokens", "temperature", "topP", "topK",
  "repetitionPenalty", "seed",
]);

/** Keep the source visible after edits without submitting its stale identity. */
export function useLaunchSetting() {
  const [sourceSetting, setSourceSetting] = useState("");
  const [pickedSetting, setReusableSetting] = useState("");

  function setPickedSetting(id: string) {
    setSourceSetting(id);
    setReusableSetting(id);
  }

  function editSetting(patch: Partial<RunInputValues>, current: RunInputValues) {
    const changed = (Object.keys(patch) as Array<keyof RunInputValues>).some(
      (key) => !RUN_ONLY_FIELDS.has(key) && !Object.is(patch[key], current[key]),
    );
    if (changed) setReusableSetting("");
    return changed;
  }

  return { sourceSetting, pickedSetting, setPickedSetting, editSetting };
}
