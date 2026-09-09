"""Tests for playbook/runners/option_scoring.py — the inference-engine side of
the opt-in MC log-likelihood eval.

These pin (a) the numeric core (log-softmax, per-token summation, length-
preserving emission), (b) the two backend adapters that extract per-token
continuation log-probs (HF teacher forcing via mocked logits, vLLM via mocked
``prompt_logprobs``), and (c) that the emitted JSON is EXACTLY the shape #51's
``mc_loglikelihood`` scorer consumes — verified by round-tripping through that
scorer. No model/GPU is used: token log-probs are mocked throughout.
"""
from __future__ import annotations

import importlib.util
import json
import math

import pytest

from zevo.engine.agent.loader import REPO_ROOT
from zevo.engine.method.eval_metrics import _mc_option_scores, mc_loglikelihood


def _load_helper():
    path = REPO_ROOT / "playbook" / "runners" / "option_scoring.py"
    spec = importlib.util.spec_from_file_location("zevo_test_option_scoring", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OS = _load_helper()


# ─────────────────────────── pure numeric core ──────────────────────────────


def test_log_softmax_row_matches_manual() -> None:
    row = [1.0, 2.0, 3.0]
    got = OS._log_softmax_row(row)
    denom = math.log(sum(math.exp(v) for v in row))
    assert got == pytest.approx([v - denom for v in row])
    # log-softmax is a valid log-distribution: exps sum to 1.
    assert sum(math.exp(v) for v in got) == pytest.approx(1.0)


def test_log_softmax_row_empty_raises() -> None:
    with pytest.raises(ValueError):
        OS._log_softmax_row([])


def test_option_score_entry_sums_and_counts() -> None:
    entry = OS.option_score_entry([-1.0, -2.0, -0.5])
    assert entry == {"logprob": pytest.approx(-3.5), "num_tokens": 3}


def test_option_score_entry_rejects_empty_and_non_finite() -> None:
    with pytest.raises(ValueError):
        OS.option_score_entry([])
    with pytest.raises(ValueError):
        OS.option_score_entry([float("-inf")])


def test_build_option_scores_shape() -> None:
    scores = OS.build_option_scores({"A": [-1.0, -1.0], "B": [-0.5]})
    assert scores == {
        "A": {"logprob": pytest.approx(-2.0), "num_tokens": 2},
        "B": {"logprob": pytest.approx(-0.5), "num_tokens": 1},
    }


def test_build_option_scores_empty_raises() -> None:
    with pytest.raises(ValueError):
        OS.build_option_scores({})


# ─────────────── emitted JSON matches #51's scorer expectations ──────────────


def test_emit_shape_is_object_of_logprob_num_tokens() -> None:
    cell = OS.emit_option_scores({"A": [-1.0, -2.0], "B": [-0.3, -0.4, -0.5]})
    obj = json.loads(cell)
    assert obj == {
        "A": {"logprob": pytest.approx(-3.0), "num_tokens": 2},
        "B": {"logprob": pytest.approx(-1.2), "num_tokens": 3},
    }
    # keys sorted + compact separators => stable cell
    assert cell == json.dumps(obj, sort_keys=True, separators=(",", ":"))


def test_emitted_cell_is_parseable_by_scorer_and_normalizes() -> None:
    # B has the better (higher) length-normalized log-prob: -1.2/3 = -0.4 vs
    # A: -3.0/2 = -1.5. The scorer must pick B under acc_norm.
    cell = OS.emit_option_scores({"A": [-1.0, -2.0], "B": [-0.3, -0.4, -0.5]})
    parsed = _mc_option_scores(cell)
    assert parsed == {"A": pytest.approx(-1.5), "B": pytest.approx(-0.4)}


def test_roundtrip_through_mc_loglikelihood_argmax() -> None:
    # Row 0: option A wins on normalized LL; gold A -> correct.
    # Row 1: option B wins; gold B -> correct.
    row0 = OS.emit_option_scores({"A": [-0.1, -0.1], "B": [-5.0, -5.0]})
    row1 = OS.emit_option_scores({"A": [-9.0], "B": [-0.2]})
    result = mc_loglikelihood([row0, row1], ["A", "B"])
    assert result["accuracy_norm"] == 1.0
    assert result["n_correct"] == 2
    assert result["n_unparsed"] == 0


def test_roundtrip_length_normalization_beats_raw_sum() -> None:
    # A's raw summed LL (-2.4) is WORSE than B's (-1.0), but A is shorter so its
    # per-token LL (-0.4) BEATS B's (-0.5). acc_norm must choose A -> matches
    # gold A. This is the whole point of emitting num_tokens.
    cell = OS.emit_option_scores({"A": [-0.4] * 6, "B": [-0.5] * 2})
    result = mc_loglikelihood([cell], ["A"])
    assert result["accuracy_norm"] == 1.0


# ─────────────────── HF teacher-forcing adapter (mocked logits) ──────────────


def test_teacher_forced_alignment_reads_shifted_rows() -> None:
    # vocab=3, 4 tokens; prompt = [t0, t1], continuation = [t2, t3].
    # Row i predicts token i+1, so continuation token at pos 2 uses row 1,
    # pos 3 uses row 2.
    token_ids = [0, 1, 2, 0]
    logits = [
        [0.0, 0.0, 0.0],   # row 0 (predicts pos 1) — unused
        [2.0, 1.0, 3.0],   # row 1 -> log-prob of token_ids[2]=2
        [5.0, 0.0, 0.0],   # row 2 -> log-prob of token_ids[3]=0
        [0.0, 0.0, 0.0],   # row 3 — unused (no pos 4)
    ]
    got = OS.teacher_forced_continuation_logprobs(logits, token_ids, 2)
    expected = [
        OS._log_softmax_row(logits[1])[2],
        OS._log_softmax_row(logits[2])[0],
    ]
    assert got == pytest.approx(expected)
    assert len(got) == 2  # num_tokens for this option


def test_teacher_forced_accepts_tensor_like_tolist() -> None:
    class FakeTensor:
        def __init__(self, rows):
            self._rows = rows

        def tolist(self):
            return self._rows

    token_ids = [1, 0, 2]  # within vocab size 3
    rows = [[0.1, 0.2, 0.3], [1.0, 2.0, 0.5], [0.0, 0.0, 0.0]]
    got = OS.teacher_forced_continuation_logprobs(FakeTensor(rows), token_ids, 2)
    assert got == pytest.approx([OS._log_softmax_row(rows[1])[2]])


def test_teacher_forced_rejects_bad_continuation_start() -> None:
    with pytest.raises(ValueError):
        OS.teacher_forced_continuation_logprobs([[0.0]], [0], 0)
    with pytest.raises(ValueError):
        OS.teacher_forced_continuation_logprobs([[0.0], [0.0]], [0, 0], 2)


def test_hf_option_scores_end_to_end_with_mocks() -> None:
    # Mock tokenizer: prompt -> 2 tokens; each option appends its own tokens.
    vocab = 5

    def fake_tokenizer(text, add_special_tokens=False):
        table = {
            "PROMPT": [0, 1],
            "PROMPTopt_a": [0, 1, 2, 3],
            "PROMPTopt_b": [0, 1, 4],
        }
        return {"input_ids": table[text]}

    class FakeLogits:
        def __init__(self, rows):
            self.logits = [rows]  # shape [1, seq, vocab]

    class FakeModel:
        device = None

        def __call__(self, input_tensor):
            ids = input_tensor[0]
            rows = [[0.0] * vocab for _ in ids]
            # Make each actual next-token clearly high-prob & deterministic.
            for i, row in enumerate(rows):
                row[(i + 1) % vocab] = 5.0
            return FakeLogits(rows)

    # Minimal torch stand-in so the driver's tensor calls work without torch.
    import sys
    import types

    fake_torch = types.SimpleNamespace(
        long="long",
        no_grad=lambda: _NullCtx(),
        tensor=lambda data, dtype=None: _FakeArray(data),
    )
    sys.modules.setdefault("torch", fake_torch)
    try:
        scores = OS.hf_option_scores(
            FakeModel(), fake_tokenizer, "PROMPT",
            {"A": "opt_a", "B": "opt_b"},
        )
    finally:
        if sys.modules.get("torch") is fake_torch:
            del sys.modules["torch"]

    assert set(scores) == {"A", "B"}
    assert scores["A"]["num_tokens"] == 2  # [2,3] appended
    assert scores["B"]["num_tokens"] == 1  # [4] appended
    # Emitting + parsing must succeed (shape contract with #51).
    cell = OS.emit_option_scores(
        {"A": [scores["A"]["logprob"]], "B": [scores["B"]["logprob"]]}
    )
    assert _mc_option_scores(cell) is not None


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeArray:
    """Barely enough of a tensor: indexable rows + a no-op .to(device)."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        return self._data[idx]

    def to(self, _device):
        return self


# ──────────────────── vLLM prompt_logprobs adapter (mocked) ──────────────────


class _Logprob:
    def __init__(self, logprob):
        self.logprob = logprob


def test_vllm_continuation_logprobs_object_form() -> None:
    token_ids = [10, 11, 12, 13]
    # vLLM: element 0 is None; element i maps token id -> Logprob at pos i.
    prompt_logprobs = [
        None,
        {11: _Logprob(-0.5)},
        {12: _Logprob(-1.5)},
        {13: _Logprob(-0.25)},
    ]
    got = OS.vllm_continuation_logprobs(token_ids, prompt_logprobs, 2)
    assert got == pytest.approx([-1.5, -0.25])


def test_vllm_continuation_logprobs_bare_float_form() -> None:
    token_ids = [1, 2, 3]
    prompt_logprobs = [None, {2: -0.7}, {3: -0.9}]
    got = OS.vllm_continuation_logprobs(token_ids, prompt_logprobs, 2)
    assert got == pytest.approx([-0.9])


def test_vllm_continuation_missing_token_raises() -> None:
    token_ids = [1, 2, 3]
    prompt_logprobs = [None, {2: -0.7}, {99: -0.9}]  # 3 absent at its position
    with pytest.raises(ValueError):
        OS.vllm_continuation_logprobs(token_ids, prompt_logprobs, 2)


def test_vllm_option_scores_end_to_end_with_mocks() -> None:
    def fake_tokenizer(text, add_special_tokens=False):
        return {"input_ids": [0, 1]}  # prompt length 2

    class FakeOutput:
        def __init__(self, prompt_token_ids, prompt_logprobs):
            self.prompt_token_ids = prompt_token_ids
            self.prompt_logprobs = prompt_logprobs

    class FakeLLM:
        def get_tokenizer(self):
            return fake_tokenizer

        def generate(self, prompts, params, use_tqdm=False):
            # A: continuation tokens [2,3]; B: continuation token [4].
            return [
                FakeOutput([0, 1, 2, 3], [None, {1: -0.1}, {2: -0.6}, {3: -0.4}]),
                FakeOutput([0, 1, 4], [None, {1: -0.1}, {4: -0.2}]),
            ]

    import sys
    import types

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: kw
    sys.modules.setdefault("vllm", fake_vllm)
    try:
        scores = OS.vllm_option_scores(
            FakeLLM(), "PROMPT", {"A": "opt_a", "B": "opt_b"},
        )
    finally:
        if sys.modules.get("vllm") is fake_vllm:
            del sys.modules["vllm"]

    assert scores["A"] == {"logprob": pytest.approx(-1.0), "num_tokens": 2}
    assert scores["B"] == {"logprob": pytest.approx(-0.2), "num_tokens": 1}
    # B has better normalized LL (-0.2) than A (-0.5); confirm via the scorer.
    cell = OS.emit_option_scores({"A": [-0.6, -0.4], "B": [-0.2]})
    assert mc_loglikelihood([cell], ["B"])["accuracy_norm"] == 1.0
