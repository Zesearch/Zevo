"""End-to-end-shape test for the token → cost pipeline.

The runner accumulates `usage` dicts from `turn_completed` events and
multiplies by the per-model rate table in `zevo.engine.cost.pricing`. The risky
piece is Claude's split of `cache_read_input_tokens` /
`cache_creation_input_tokens` vs OpenAI's flat `cached_input_tokens` —
a regression in the Claude normaliser would silently double-count
(double the price) or under-count (silently bill at the wrong tier).

These tests pin:
  1. Claude's normaliser folds cache_read + cache_creation into a
     single `cached_input_tokens` bucket AND includes them in the
     reported `input_tokens` total (which `estimate_cost` then subtracts
     before multiplying by the input rate).
  2. `estimate_cost` charges the cached rate for cached tokens and the
     input rate for the rest — never double-bills cached tokens.
  3. `accumulate_usage` is additive (called once per turn; multiple
     turns add).
  4. The full chain (normalise → accumulate → estimate) reproduces a
     hand-checked dollar figure for each of the 4 driver/model combos
     we ship today.
"""
from __future__ import annotations

from zevo.engine.agent.drivers.claude_cli import _normalise_usage as claude_normalise
from zevo.engine.cost.pricing import PRICES_USD_PER_MTOK, accumulate_usage, estimate_cost


# -------------------------- normaliser invariants ---------------------------

def test_claude_normalise_folds_cache_into_cached_and_total() -> None:
    # Raw Claude usage shape (what stream-json emits):
    raw = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 30,
    }
    out = claude_normalise(raw)
    # cached_input_tokens = cache_read + cache_creation
    assert out["cached_input_tokens"] == 230
    # input_tokens INCLUDES the cached portion (so estimate_cost can
    # subtract it cleanly)
    assert out["input_tokens"] == 100 + 200 + 30 == 330
    assert out["output_tokens"] == 50
    assert out["reasoning_output_tokens"] == 0


def test_claude_normalise_handles_missing_keys() -> None:
    # An assistant turn with no cache hits (cold start)
    raw = {"input_tokens": 1234, "output_tokens": 567}
    out = claude_normalise(raw)
    assert out == {
        "input_tokens": 1234,
        "output_tokens": 567,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }


def test_claude_normalise_handles_all_zero() -> None:
    assert claude_normalise({}) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }


# ----------------------------- cost arithmetic ------------------------------

def test_estimate_cost_never_double_bills_cached_tokens() -> None:
    # claude-sonnet-4-6: $3.00 input, $0.30 cached, $15.00 output per Mtok
    # If cached were billed at BOTH input AND cached, total would be
    # too high. Verify the non-cached portion = input - cached.
    usage = {
        "input_tokens": 1_000_000,   # 1M total
        "cached_input_tokens": 800_000,  # 800K cache hits
        "output_tokens": 0,
    }
    cost = estimate_cost("claude-sonnet-4-6", usage)
    # non_cached = 200_000 → 200_000 * 3/M = 0.60
    # cached     = 800_000 → 800_000 * 0.3/M = 0.24
    # total      = 0.84
    assert abs(cost - 0.84) < 1e-6, f"got {cost}"


def test_estimate_cost_unknown_model_returns_zero_not_crash() -> None:
    assert estimate_cost("totally-made-up-model", {"input_tokens": 1_000_000}) == 0.0


def test_estimate_cost_handles_bad_usage_dict() -> None:
    # Defensive: a malformed usage (None, missing fields, string values)
    # must not raise.
    assert estimate_cost("moonshotai.kimi-k2.5", None) == 0.0  # type: ignore[arg-type]
    assert estimate_cost("moonshotai.kimi-k2.5", {}) == 0.0
    assert estimate_cost("moonshotai.kimi-k2.5", {"input_tokens": "lol"}) == 0.0


# -------------------- accumulation across multiple turns --------------------

def test_accumulate_usage_sums_across_turns() -> None:
    running: dict = {}
    accumulate_usage(running, {"input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 30})
    accumulate_usage(running, {"input_tokens": 200, "output_tokens": 20, "cached_input_tokens": 50})
    accumulate_usage(running, {"input_tokens":  50, "output_tokens":  5})
    assert running["input_tokens"] == 350
    assert running["output_tokens"] == 35
    assert running["cached_input_tokens"] == 80


# ----------------- full chain: normalise → accumulate → cost ----------------

def test_full_chain_claude_sonnet_three_turns_matches_hand_math() -> None:
    # 3 turns of Claude stream-json, with cache hits on turn 2+.
    raw_turns = [
        {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 500},
        {"input_tokens":   50, "output_tokens":  20, "cache_read_input_tokens": 1500, "cache_creation_input_tokens": 0},
        {"input_tokens":   50, "output_tokens":  10, "cache_read_input_tokens": 1500, "cache_creation_input_tokens": 0},
    ]
    running: dict = {}
    for raw in raw_turns:
        accumulate_usage(running, claude_normalise(raw))

    # Hand-check totals:
    # input_tokens = (1000+0+500) + (50+1500+0) + (50+1500+0) = 1500+1550+1550 = 4600
    # output_tokens = 100+20+10 = 130
    # cached       = 500 + 1500 + 1500 = 3500
    assert running["input_tokens"]        == 4600
    assert running["output_tokens"]       == 130
    assert running["cached_input_tokens"] == 3500

    cost = estimate_cost("claude-sonnet-4-6", running)
    # non_cached = 4600 - 3500 = 1100  → 1100 * 3.00/M  = 0.0033
    # cached     = 3500              → 3500 * 0.30/M = 0.00105
    # output     = 130               → 130 * 15.00/M = 0.00195
    # total                                           = 0.0063
    assert abs(cost - 0.0063) < 1e-6, f"got {cost}"


def _expected(model: str, inp: int, out: int, cached: int = 0) -> float:
    """The bill the TABLE implies, computed the long way.

    Published rates change — they were refreshed from OpenRouter and Opus alone
    moved 3x — so these tests check the arithmetic against the table rather than
    against numbers typed in at some point in the past. A test that pins a price
    only ever reports "the price changed", which is not a defect.
    """
    r = PRICES_USD_PER_MTOK[model]
    return ((inp - cached) * r["input"] + cached * r["cached_input"]
            + out * r["output"]) / 1_000_000


def test_full_chain_claude_opus_premium_pricing_applies() -> None:
    # Opus costs strictly more than Sonnet for the same tokens.
    running: dict = {}
    accumulate_usage(
        running,
        claude_normalise({
            "input_tokens": 100_000,
            "output_tokens": 10_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }),
    )
    sonnet = estimate_cost("claude-sonnet-4-6", running)
    opus   = estimate_cost("claude-opus-4-8",   running)
    assert opus > sonnet
    assert abs(opus - _expected("claude-opus-4-8", 100_000, 10_000)) < 1e-9
    assert abs(sonnet - _expected("claude-sonnet-4-6", 100_000, 10_000)) < 1e-9


def test_bedrock_kimi_cost_matches_table() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "cached_input_tokens": 0,
        "output_tokens":   100_000,
    }
    cost = estimate_cost("moonshotai.kimi-k2.5", usage)
    assert abs(cost - _expected("moonshotai.kimi-k2.5", 1_000_000, 100_000)) < 1e-6


def test_bedrock_minimax_cheapest_priced_correctly() -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cached_input_tokens": 0}
    cost = estimate_cost("minimax.minimax-m2.5", usage)
    assert abs(cost - _expected("minimax.minimax-m2.5", 1_000_000, 1_000_000)) < 1e-6
    # still the cheapest thing on the Bedrock menu, which is why it is offered
    for other in ("moonshotai.kimi-k2.5", "zai.glm-5"):
        assert cost < estimate_cost(other, usage)
