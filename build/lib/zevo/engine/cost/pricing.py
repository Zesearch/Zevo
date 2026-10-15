"""Cost estimation for Bedrock / Anthropic model usage.

Drivers emit `turn_completed` events with per-turn `usage` dicts in the
canonical harness shape:

    {"input_tokens": N, "output_tokens": M,
     "cached_input_tokens": K, "reasoning_output_tokens": R}

The runner accumulates these per-heartbeat. At heartbeat-close, we
multiply by a per-model `$/M token` table to record the estimated
spend on the row. Cost is FROZEN at write-time -- historical heartbeats
keep showing what they cost at the time, not what they would cost at
today's prices.

Every rate below comes from OpenRouter's public catalogue
(`GET https://openrouter.ai/api/v1/models`, `pricing.prompt` /
`pricing.input_cache_read` / `pricing.completion`, which are $/token — multiply
by 1e6). One source for all four drivers: whatever the driver, the same model
is billed at the same published rate, and a single fetch re-checks them all.
Anything the catalogue does not carry keeps a hand-entered rate and says so.

Prices move. Re-fetch before trusting a cost, and remember that a run on a
Claude Max or ChatGPT subscription is billed by the plan, not per token — the
meter overstates those.

Prices can be overridden via env without a code change (model id uppercased,
with `-`/`.` -> `_`):

    ZEVO_PRICE_MOONSHOTAI_KIMI_K2_5_INPUT_USD_PER_MTOK=0.60
    ZEVO_PRICE_MOONSHOTAI_KIMI_K2_5_OUTPUT_USD_PER_MTOK=3.00

Unknown models fall through to all-zero; the cost meter just shows
tokens with no $ figure.
"""
from __future__ import annotations

import os
from typing import Any


# Default per-model prices ($ per million tokens). Update as needed --
# or override via env (see module docstring). reasoning_output_tokens
# are billed as output_tokens by OpenAI today, so we double-count them
# only if a model needs them separately (none currently do).
PRICES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # ─── Bedrock open models ────────────────────────────────────────────────
    # $ per 1M tokens. Bedrock publishes its own rates; these are the model
    # publishers' rates as listed on OpenRouter, which is what you would pay
    # for the same model elsewhere. Availability + tool-use verified live
    # against this account's bedrock-runtime.
    "moonshotai.kimi-k2.5":                {"input": 0.57, "cached_input": 0.095, "output": 2.85},
    "deepseek.v3.2":                       {"input": 0.269, "cached_input": 0.1345, "output": 0.4},
    "minimax.minimax-m2.5":                {"input": 0.22, "cached_input": 0.05, "output": 0.9},
    "zai.glm-5":                           {"input": 0.95, "cached_input": 0.2, "output": 2.55},
    "qwen.qwen3-coder-next":               {"input": 0.12, "cached_input": 0.07, "output": 0.8},

    # ─── Claude on Bedrock (if you switch the driver's model) ──────────────
    # Same rates as the Anthropic direct list; ids use the `us.` profile prefix.
    "us.anthropic.claude-opus-5":          {"input": 5.0, "cached_input": 0.5, "output": 25.0},
    "us.anthropic.claude-sonnet-5":        {"input": 2.0, "cached_input": 0.2, "output": 10.0},
    "us.anthropic.claude-opus-4-8":        {"input": 5.0, "cached_input": 0.5, "output": 25.0},
    "us.anthropic.claude-sonnet-4-6":      {"input": 3.0, "cached_input": 0.3, "output": 15.0},
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.0, "cached_input": 0.1, "output": 5.0},

    # ─── Anthropic direct (claude_cli driver) ───────────────────────────────
    # The CLI takes the bare alias; ids carry no date suffix. On a Max/Pro
    # plan the run is billed by the subscription, not per token, and the
    # meter overstates it — same caveat as codex on a ChatGPT plan.
    "claude-opus-5":                       {"input": 5.0, "cached_input": 0.5, "output": 25.0},
    "claude-sonnet-5":                     {"input": 2.0, "cached_input": 0.2, "output": 10.0},
    "claude-fable-5-1":                    {"input": 10.0, "cached_input": 1.0, "output": 50.0},
    "claude-fable-5":                      {"input": 10.0, "cached_input": 1.0, "output": 50.0},
    "claude-opus-4-8":                     {"input": 5.0, "cached_input": 0.5, "output": 25.0},
    "claude-opus-4-7":                     {"input": 5.0, "cached_input": 0.5, "output": 25.0},
    "claude-sonnet-4-6":                   {"input": 3.0, "cached_input": 0.3, "output": 15.0},
    "claude-haiku-4-5":                    {"input": 1.0, "cached_input": 0.1, "output": 5.0},

    # ─── OpenAI direct (codex_cli driver) ───────────────────────────────────
    # Only charged when the driver falls back to OPENAI_API_KEY. With a
    # Codex-managed ChatGPT auth.json session the run comes out of the
    # subscription and these rates do not apply.
    # GPT-5.5 uses OpenRouter's standard endpoint rate. OpenRouter applies a
    # higher tier above 272K prompt tokens; this flat table cannot express that
    # breakpoint, so very-large-context runs are underestimated here.
    "gpt-5.5":                             {"input": 5.0, "cached_input": 0.5, "output": 30.0},
    "gpt-5.3-codex":                       {"input": 1.75, "cached_input": 0.175, "output": 14.0},
    "gpt-5.2-codex":                       {"input": 1.75, "cached_input": 0.175, "output": 14.0},
    "gpt-5.1-codex-max":                   {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    "gpt-5.1-codex-mini":                  {"input": 0.25, "cached_input": 0.03, "output": 2.0},
    "gpt-5":                               {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    "gpt-5-mini":                          {"input": 0.25, "cached_input": 0.025, "output": 2.0},

    # ─── OpenRouter ─────────────────────────────────────────────────────────
    # Namespaced ids, priced from the same catalogue that serves them. The
    # menu deliberately excludes Anthropic and OpenAI models — claude_cli and
    # codex_cli reach those directly. Anything else falls through to 0.00,
    # which the cost panel shows as $0.00 rather than guessing.
    "qwen/qwen3.8-max":                    {"input": 2.0, "cached_input": 0.25, "output": 6.0},
    "moonshotai/kimi-k3":                  {"input": 3.0, "cached_input": 0.3, "output": 15.0},
    "x-ai/grok-4.5":                       {"input": 2.0, "cached_input": 0.3, "output": 6.0},
    "google/gemini-3.6-flash":             {"input": 1.5, "cached_input": 0.15, "output": 7.5},
    "z-ai/glm-5.2":                        {"input": 0.546, "cached_input": 0.1014, "output": 1.716},
    "minimax/minimax-m3":                  {"input": 0.3, "cached_input": 0.06, "output": 1.2},
    "deepseek/deepseek-v4-pro":            {"input": 0.435, "cached_input": 0.0036, "output": 0.87},
    "deepseek/deepseek-v4-flash-0731":     {"input": 0.09, "cached_input": 0.018, "output": 0.18},
    "qwen/qwen3.7-flash":                  {"input": 0.03, "cached_input": 0.006, "output": 0.13},
}


# ─── GPU price by type ($ per GPU-hour) ─────────────────────────────────────
# Used to cost GPU time. Your own cluster has no market price, so we IMPUTE one
# per GPU model. gpu_cost = rate × gpu_count × uptime_hours.
#
# REFERENCE: these defaults are APPROXIMATE on-demand cloud-rental rates — the
# ballpark you'd pay per GPU-hour on Vast.ai / RunPod / Lambda as of early 2026.
# They are NOT authoritative — EDIT THEM to your actual numbers (your cloud
# provider's real price, or your cluster's amortized $/GPU-hour). Override any
# single one without a code change:
#   ZEVO_GPU_PRICE_H100_USD_PER_HOUR=2.50
#
# Matched by substring against the instance's `gpu_name` ("NVIDIA H100 80GB" ->
# "H100"), longest key first. Unknown type -> None -> caller falls back to the
# provider's reported dph (e.g. the actual Vast.ai rental price).
GPU_HOURLY_USD: dict[str, float] = {
    # ── NVIDIA data-center (Blackwell / Hopper / Ampere) ──
    "B200":    5.99,
    "H200":    3.99,
    "H100":    2.99,
    "H800":    2.99,   # China-market H100 variant
    "A100":    1.79,   # 40GB/80GB (name rarely distinguishes)
    "A800":    1.79,   # China-market A100 variant
    "A40":     0.44,
    "A30":     0.35,
    "V100":    0.50,
    # ── NVIDIA L-series (Ada) ──
    "L40S":    1.09,
    "L40":     0.99,
    "L4":      0.30,
    "T4":      0.20,
    # ── Workstation / consumer (RTX) ──
    "A6000":   0.79,   # RTX A6000
    "A5000":   0.36,
    "RTX4090": 0.44,
    "RTX3090": 0.22,
    "RTX8000": 0.40,   # Quadro RTX 8000 (Turing, 48GB)
    # ── AMD ──
    "MI300X":  3.99,
}


def _norm_gpu(name: str) -> str:
    """Uppercase + strip non-alphanumerics so 'NVIDIA H100 80GB' matches 'H100'."""
    return "".join(ch for ch in (name or "").upper() if ch.isalnum())


def gpu_hourly(gpu_name: str) -> float | None:
    """$/GPU-hour for this GPU model, or None if the type is unknown.

    Keys are tried longest first so a key that is a prefix of another cannot
    shadow it: an "L40S" must match L40S, not fall to L40 or L4. Substring
    matching still false-matches names that merely CONTAIN a key (an unknown
    "A1000" would hit A100); env override
    `ZEVO_GPU_PRICE_<KEY>_USD_PER_HOUR` wins over the table.
    """
    norm = _norm_gpu(gpu_name)
    if not norm:
        return None
    for key in sorted(GPU_HOURLY_USD, key=len, reverse=True):
        if _norm_gpu(key) in norm:
            env = os.environ.get(f"ZEVO_GPU_PRICE_{key.upper()}_USD_PER_HOUR")
            if env:
                try:
                    return float(env)
                except ValueError:
                    pass
            return GPU_HOURLY_USD[key]
    return None


def _env_override(model: str, kind: str) -> float | None:
    """Look for ZEVO_PRICE_<MODEL_MANGLED>_<KIND>_USD_PER_MTOK.

    EVERY non-alphanumeric character becomes `_`, not just `-`/`.` — model
    ids carry `/` (all OpenRouter ids) and `:` (dated Bedrock ids), and a
    variable name containing those cannot be exported from a POSIX shell,
    which made the documented override impossible for a third of the table.
    """
    mangled = "".join(c if c.isalnum() else "_" for c in model.upper())
    var = f"ZEVO_PRICE_{mangled}_{kind.upper()}_USD_PER_MTOK"
    raw = os.environ.get(var)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def price(model: str, kind: str) -> float:
    """Resolve $/Mtok for (model, kind). 0.0 if unknown."""
    override = _env_override(model, kind)
    if override is not None:
        return override
    return PRICES_USD_PER_MTOK.get(model, {}).get(kind, 0.0)


def estimate_cost(model: str, usage: dict[str, Any]) -> float:
    """Return the USD cost for one usage dict, rounded to 6 decimals.

    Handles cached-input correctly: OpenAI's reported `input_tokens`
    typically *includes* `cached_input_tokens`. We bill cached separately
    at the cached rate and the remainder at the input rate.
    """
    if not isinstance(usage, dict):
        return 0.0

    def _i(key: str) -> int:
        """Coerce to int defensively: malformed driver output (string,
        None, float) must NOT crash cost capture. Treat bad values as 0
        so a buggy turn shows $0 instead of taking the heartbeat down.
        """
        v = usage.get(key)
        if v is None:
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    inp = _i("input_tokens")
    out = _i("output_tokens")
    cached = _i("cached_input_tokens")
    non_cached = max(0, inp - cached)
    cost = (
        non_cached * price(model, "input")
        + cached    * price(model, "cached_input")
        + out       * price(model, "output")
    ) / 1_000_000.0
    return round(cost, 6)


def accumulate_usage(
    running: dict[str, int], new_usage: dict[str, Any]
) -> dict[str, int]:
    """Add `new_usage` numeric fields into `running` in place. Returns running."""
    for k in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens"):
        try:
            running[k] = int(running.get(k, 0)) + int(new_usage.get(k, 0) or 0)
        except (TypeError, ValueError):
            pass
    return running
