"""System-owned per-option log-likelihood scoring for multiple-choice eval.

Copy this file beside the generated ``predict.py`` and import it when the
inference configuration sets ``measurement.inference_config.scoring_mode =
"option_loglikelihood"``.  It is the inference-engine half of the opt-in
MMLU-style option-scoring path: instead of generating a letter and string-
matching it, ``predict.py`` scores each candidate answer option under the
frozen prompt and writes the per-option log-likelihoods into the single
prediction column.  The deterministic ``mc_loglikelihood`` (alias
``accuracy_norm``) evaluation metric then length-normalizes and argmaxes those
numbers with no model call, so the "Evaluation is LLM-free" invariant holds.

The emitted prediction cell is a JSON object mapping each option label to the
raw summed completion log-probability plus its token count::

    {"A": {"logprob": -12.34, "num_tokens": 6},
     "B": {"logprob": -9.01,  "num_tokens": 5}, ...}

which is exactly the shape ``eval_metrics._mc_option_scores`` consumes; that
scorer normalizes as ``logprob / num_tokens`` (the ``acc_norm`` convention),
so keeping options long or short does not distort the comparison.

Two standard backends produce the per-token continuation log-probs that feed
:func:`build_option_scores`:

  * **Hugging Face teacher forcing** (:func:`hf_option_scores`) — one forward
    pass over ``prompt + option`` per option; the log-prob of each option token
    is read off the shifted logits and summed.
  * **vLLM ``prompt_logprobs``** (:func:`vllm_option_scores`) — send
    ``prompt + option`` with ``prompt_logprobs`` requested and sum the returned
    log-probs over the option's token span.

The numeric alignment/normalization/emission core (everything except the model
forward pass) is pure Python so it can be unit-tested against mocked token
log-probs with no GPU.  ``torch``/``transformers``/``vllm`` are imported lazily
inside the driver functions only, so this module imports cleanly in contract and
unit checks that never touch a model.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

# ── pure numeric core (unit-tested with mocked token log-probs) ──────────────


def _log_softmax_row(row: Sequence[float]) -> list[float]:
    """Numerically stable log-softmax of one logits row (pure Python).

    Used by the Hugging Face path so the logits→log-prob step can be exercised
    with plain nested lists standing in for a model's output tensor.
    """
    values = [float(x) for x in row]
    if not values:
        raise ValueError("cannot log-softmax an empty logits row")
    peak = max(values)
    sum_exp = math.fsum(math.exp(v - peak) for v in values)
    log_sum_exp = peak + math.log(sum_exp)
    return [v - log_sum_exp for v in values]


def option_score_entry(token_logprobs: Sequence[float]) -> dict[str, Any]:
    """Fold one option's per-token continuation log-probs into an emit entry.

    Returns ``{"logprob": <summed raw log-prob>, "num_tokens": <count>}`` — the
    raw (un-normalized) sum and its token count, which is exactly what the
    ``mc_loglikelihood`` scorer expects and length-normalizes.  Summation uses
    :func:`math.fsum` for order-independent stability.
    """
    logprobs = [float(x) for x in token_logprobs]
    if not logprobs:
        raise ValueError("an option must have at least one continuation token")
    for value in logprobs:
        if not math.isfinite(value):
            raise ValueError("token log-probs must be finite")
    return {"logprob": math.fsum(logprobs), "num_tokens": len(logprobs)}


def build_option_scores(
    option_token_logprobs: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, Any]]:
    """Map ``{label: [per-token log-probs]}`` to the per-option emit dict.

    The result is ``{label: {"logprob": ..., "num_tokens": ...}}`` — the
    object form ``mc_loglikelihood`` consumes and length-normalizes.
    """
    if not option_token_logprobs:
        raise ValueError("no options to score")
    return {
        str(label): option_score_entry(logprobs)
        for label, logprobs in option_token_logprobs.items()
    }


def emit_option_scores(
    option_token_logprobs: Mapping[str, Sequence[float]],
) -> str:
    """Serialize per-option scores to the exact prediction-column JSON string.

    This string is what ``predict.py`` writes into the single prediction cell
    for one row; ``eval_metrics._mc_option_scores`` parses it back.  Keys are
    sorted and separators are compact so the cell is stable across runs.
    """
    scores = build_option_scores(option_token_logprobs)
    return json.dumps(scores, sort_keys=True, separators=(",", ":"))


# ── backend adapters: extract per-token continuation log-probs ───────────────


def teacher_forced_continuation_logprobs(
    logits: Any, token_ids: Sequence[int], continuation_start: int,
) -> list[float]:
    """Per-token log-probs of the continuation tokens under teacher forcing.

    ``logits[i]`` are the next-token logits after consuming ``token_ids[i]``, so
    the log-prob of the token at position ``pos`` is read from row ``pos - 1``.
    Returns the log-probs for ``token_ids[continuation_start:]`` (the option's
    tokens), in order.

    ``logits`` may be a ``torch`` tensor (any object exposing ``.tolist()``) of
    shape ``[seq_len, vocab]`` or a plain nested sequence, so the alignment and
    log-softmax are testable with mocked logits and no GPU.
    """
    rows = logits.tolist() if hasattr(logits, "tolist") else [list(r) for r in logits]
    seq_len = len(token_ids)
    if continuation_start <= 0 or continuation_start >= seq_len:
        raise ValueError(
            "continuation_start must fall inside the sequence after the prompt"
        )
    if len(rows) < seq_len:
        raise ValueError("logits rows fewer than tokens; cannot align")
    out: list[float] = []
    for pos in range(continuation_start, seq_len):
        row_logprobs = _log_softmax_row(rows[pos - 1])
        out.append(row_logprobs[token_ids[pos]])
    return out


def _logprob_value(entry: Any, token_id: int) -> float:
    """Read the log-prob a vLLM ``prompt_logprobs`` entry assigns ``token_id``.

    A real entry is ``{token_id: Logprob(logprob=...)}``; tests pass
    ``{token_id: float}``.  Both a ``Logprob``-like object (``.logprob``) and a
    bare float are accepted.
    """
    if entry is None or token_id not in entry:
        raise ValueError(f"token {token_id} missing from prompt_logprobs entry")
    value = entry[token_id]
    logprob = getattr(value, "logprob", value)
    return float(logprob)


def vllm_continuation_logprobs(
    token_ids: Sequence[int],
    prompt_logprobs: Sequence[Any],
    continuation_start: int,
) -> list[float]:
    """Per-token log-probs of the continuation span from vLLM ``prompt_logprobs``.

    vLLM returns one ``prompt_logprobs`` entry per prompt token (the first is
    ``None`` — the initial token has no context).  Entry ``i`` maps token ids to
    their log-prob at position ``i``; the actual token there is ``token_ids[i]``.
    Sums over ``[continuation_start, len(token_ids))``.
    """
    seq_len = len(token_ids)
    if continuation_start <= 0 or continuation_start >= seq_len:
        raise ValueError(
            "continuation_start must fall inside the sequence after the prompt"
        )
    if len(prompt_logprobs) < seq_len:
        raise ValueError("prompt_logprobs shorter than the token sequence")
    return [
        _logprob_value(prompt_logprobs[pos], token_ids[pos])
        for pos in range(continuation_start, seq_len)
    ]


# ── full drivers (import the model backend lazily; need a GPU to run) ─────────


def hf_option_scores(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    option_texts: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Teacher-forced per-option scores for one row with a HF causal LM.

    For each option, tokenizes ``prompt_text`` then ``prompt_text + option``,
    runs one no-grad forward pass, and sums the option tokens' log-probs via
    :func:`teacher_forced_continuation_logprobs`.  The continuation span starts
    at the prompt's token length, so only the option's tokens are scored.

    Returns the object form ready for :func:`build_option_scores` /
    :func:`emit_option_scores`.  Requires ``torch``; running it needs a model
    and (in practice) a GPU.
    """
    import torch  # noqa: PLC0415 -- lazy: keep this module GPU-free at import

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    continuation_start = len(prompt_ids)
    if continuation_start == 0:
        raise ValueError("prompt tokenized to zero tokens")
    device = getattr(model, "device", None)
    per_option: dict[str, Sequence[float]] = {}
    for label, option_text in option_texts.items():
        full_ids = tokenizer(
            prompt_text + option_text, add_special_tokens=False,
        )["input_ids"]
        if len(full_ids) <= continuation_start:
            raise ValueError(f"option {label!r} added no continuation tokens")
        input_tensor = torch.tensor([full_ids], dtype=torch.long)
        if device is not None:
            input_tensor = input_tensor.to(device)
        with torch.no_grad():
            logits = model(input_tensor).logits[0]
        per_option[str(label)] = teacher_forced_continuation_logprobs(
            logits, full_ids, continuation_start,
        )
    return build_option_scores(per_option)


def vllm_option_scores(
    llm: Any,
    prompt_text: str,
    option_texts: Mapping[str, str],
    tokenizer: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """vLLM ``prompt_logprobs`` per-option scores for one row.

    Sends ``prompt + option`` for every option in one batched ``generate`` call
    with ``prompt_logprobs`` requested and ``max_tokens=1`` (we score the prompt,
    not a generation), then sums each option's continuation log-probs with
    :func:`vllm_continuation_logprobs`.  ``tokenizer`` defaults to
    ``llm.get_tokenizer()`` and is used only to locate the continuation span.

    Requires ``vllm``; running it needs a model and GPU.
    """
    from vllm import SamplingParams  # noqa: PLC0415 -- lazy import

    tok = tokenizer if tokenizer is not None else llm.get_tokenizer()
    prompt_len = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
    if prompt_len == 0:
        raise ValueError("prompt tokenized to zero tokens")
    labels = list(option_texts.keys())
    prompts = [prompt_text + option_texts[label] for label in labels]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    per_option: dict[str, Sequence[float]] = {}
    for label, output in zip(labels, outputs):
        token_ids = list(output.prompt_token_ids)
        per_option[str(label)] = vllm_continuation_logprobs(
            token_ids, output.prompt_logprobs, prompt_len,
        )
    return build_option_scores(per_option)
