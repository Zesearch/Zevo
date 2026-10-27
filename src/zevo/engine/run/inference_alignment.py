"""Align reusable held-out configs to the answer-free Validation contract."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from zevo.contracts.configuration import InferenceRunConfig, load_inference_config
from zevo.contracts.inference_core import inference_core, validate_inference_core


def align_reused_config(path: str, expected: dict, work_dir: str, *, allow_create: bool) -> str:
    """Create a ticket-local config; never rewrite historical evidence.

    Prompt/template changes require regeneration of the exact rendered example,
    which cannot be inferred from a config dictionary. Reject those explicitly.
    Sampling and token settings can be copied from the authoritative core while
    preserving this benchmark's mapping, parsing, budget and resource settings.
    """
    config = load_inference_config(path)
    try:
        if inference_core(config) == expected:
            return path
    except ValueError:
        # Conflicting duplicated sampling controls are replaced below.
        pass
    data = config.model_dump(mode="json")
    fixed = ("base_model", "prompt", "generation_backend", "tokenizer_source",
             "chat_template_source", "chat_template_hash", "template_kwargs")
    changed = [key for key in fixed if data[key] != expected.get(key)]
    impl = data["implementation_config"]
    rendering = impl.get("template_rendering") or {}
    generation_prompt = rendering.get("add_generation_prompt", impl.get("add_generation_prompt", True))
    if generation_prompt != expected.get("add_generation_prompt"):
        changed.append("add_generation_prompt")
    if changed:
        raise ValueError("held-out shared config cannot be safely aligned without regenerating its prompt example: "
                         + ", ".join(changed))
    data["special_token_ids"] = deepcopy(expected["special_token_ids"])
    data["stop_token_ids"] = deepcopy(expected["stop_token_ids"])
    data["measurement"].update(deepcopy(expected["decoding"]))
    data["measurement"]["inference_config"]["stop"] = deepcopy(expected["stop"])
    authoritative = {**expected["decoding"], "stop": expected["stop"],
                     "max_tokens": config.measurement.max_new_tokens}
    for namespace in ("sampling_params", "sampling_extra"):
        previous = impl.get(namespace) or {}
        # Retain explicitly duplicated backend arguments with corrected values;
        # stop_token_ids must remain at its single top-level authority.
        impl[namespace] = {**deepcopy(expected["sampling"]), **{
            key: deepcopy(value) for key, value in authoritative.items() if key in previous
        }}
    impl["add_generation_prompt"] = expected["add_generation_prompt"]
    if "template_rendering" in impl:
        impl["template_rendering"] = {**rendering, "add_generation_prompt": expected["add_generation_prompt"]}
    aligned = InferenceRunConfig.model_validate(data)
    validate_inference_core(aligned, expected)
    body = json.dumps(aligned.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    sha = hashlib.sha256(body.encode()).hexdigest()
    target = Path(work_dir) / "shared-core-configs" / f"{sha}.json"
    if target.exists():
        if target.read_text() != body:
            raise ValueError("aligned held-out inference config was modified after creation")
        return str(target.resolve())
    if not allow_create:
        raise ValueError("cannot align held-out configuration after a GPU execution has been submitted; "
                         "keep its original evidence and request a new measurement")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    target.with_suffix(".provenance.json").write_text(json.dumps({
        "source_path": str(Path(path).resolve()),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "shared_core_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
        "reason": "Align reusable held-out sampling and token settings to shared Validation core",
    }, indent=2), encoding="utf-8")
    return str(target.resolve())
