"""Remote watcher marker parsing remains lossless and Ticket-owned.

A cluster stage prints `__ATTEMPT__` / `__CONFIG__` / `__PHASE__` /
`__PROGRESS__` into its job-id-specific stdout. The live watcher parses that
stream while the job is running; Collect does not replay the completed log.

These test the pieces used by the live stream: scanning, owner filtering, and
idempotence across a watcher reconnect.
"""

import json

import pytest

from zevo.engine.observe.markers import scan_text


TRAIN_CONFIG = {
    "training_method": "full_sft",
    "prompt_framing": "chat:Qwen/Qwen3-0.6B",
    "learning_rate": 1e-05,
    "prompt": "chat:Qwen/Qwen3-0.6B\n<|im_start|>user\n{INPUT}<|im_end|>\n"
              "<|im_start|>assistant\n",
}
ATTEMPT_ID = "550e8400-e29b-41d4-a716-446655440000"


def test_runner_keeps_live_stdout_marker_parser() -> None:
    """Removing terminal replay must not remove live stdout parsing."""
    from zevo.engine.run import runner

    assert runner.scan_text is scan_text


def _log(ticket: str, cfg: dict) -> str:
    """A remote log as it actually comes back: markers buried in job noise."""
    return (
        "srun: job 38952236 queued and waiting for resources\n"
        f"__PHASE__:{ticket}:loading_model\n"
        "Loading checkpoint shards: 100%|##########| 2/2 [00:01<00:00]\n"
        f"__CONFIG__:{ticket}:{json.dumps(cfg)}\n"
        f"__ATTEMPT__:{ticket}:{ATTEMPT_ID}@1723723199.0\n"
        f"__PHASE__:{ticket}:training\n"
        f'__PROGRESS__:{ticket}:{{"attempt_id":"{ATTEMPT_ID}","step":5,"total":492,"loss":1.4274}}\n'
        f'__PROGRESS__:{ticket}:{{"attempt_id":"{ATTEMPT_ID}","step":25,"total":492,"loss":1.3280}}\n'
        f"__PHASE__:{ticket}:saving_model\n"
    )


def _stream_markers(text: str, own: str):
    """Apply the live watcher's ownership filter to parsed marker bytes."""
    out = []
    for kind, payload in scan_text(text):
        payload = dict(payload)
        owner = payload.pop("owner", "")
        if owner and owner != own:
            continue
        out.append((kind, payload))
    return out


class TestLiveStreamParsing:
    def test_a_streamed_config_is_parsed(self):
        got = _stream_markers(_log("train-028", TRAIN_CONFIG), "train-028")
        cfgs = [p for k, p in got if k == "config"]
        assert len(cfgs) == 1
        assert cfgs[0]["prompt_framing"] == "chat:Qwen/Qwen3-0.6B"
        # The prompt survives intact — it carries newlines, and an earlier scanner
        # cut every marker at the first one and truncated the JSON to nothing.
        assert cfgs[0]["prompt"].endswith("<|im_start|>assistant\n")

    def test_phases_and_progress_come_back_too(self):
        got = _stream_markers(_log("train-028", TRAIN_CONFIG), "train-028")
        assert [p["attempt_id"] for k, p in got if k == "attempt"] == [ATTEMPT_ID]
        assert [p["phase"] for k, p in got if k == "phase"] == [
            "loading_model", "training", "saving_model",
        ]
        assert [p["step"] for k, p in got if k == "progress"] == [5, 25]

    def test_another_ticket_s_markers_are_not_adopted(self):
        # A stage routinely reads a sibling's log — its markers are byte-for-byte
        # the same as this stage's own, and without the owner prefix one training
        # run's steps land on an inference ticket.
        mixed = _log("train-028", TRAIN_CONFIG) + _log("infer-025", {"backend": "vllm"})
        got = _stream_markers(mixed, "train-028")
        assert all(p.get("backend") != "vllm" for k, p in got if k == "config")
        assert len([p for k, p in got if k == "config"]) == 1

    def test_job_noise_around_a_marker_does_not_swallow_it(self):
        # Markers arrive glued to whatever the job printed on the same line.
        text = (
            "\rtraining:  1%|          | 5/492 [00:12<19:44]"
            '__PROGRESS__:train-028:{"step": 5, "total": 492, "loss": 1.4274}\n'
        )
        got = _stream_markers(text, "train-028")
        assert [(k, p["step"]) for k, p in got] == [("progress", 5)]


class TestIdempotence:
    """A marker that already arrived by stdout must not be recorded again."""

    def test_progress_replay_is_dropped(self):
        seen: set[tuple] = set()
        kept = []
        for _ in range(2):  # same log swept twice
            for kind, p in _stream_markers(_log("train-028", TRAIN_CONFIG), "train-028"):
                if kind != "progress":
                    continue
                key = (
                    p.get("attempt_id", ""), p.get("phase", ""),
                    p["step"], p["total"], p["loss"],
                )
                if key in seen:
                    continue
                seen.add(key)
                kept.append(key)
        assert len(kept) == 2  # two distinct steps, not four

    def test_config_replay_is_dropped_on_content(self):
        seen: set[str] = set()
        kept = []
        for _ in range(2):
            for kind, p in _stream_markers(_log("train-028", TRAIN_CONFIG), "train-028"):
                if kind != "config":
                    continue
                cfg = {k: v for k, v in p.items() if k != "t"}
                key = json.dumps(cfg, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                kept.append(key)
        assert len(kept) == 1

    def test_a_genuinely_changed_config_is_not_dropped(self):
        # Keyed on content, so a script that reconfigures mid-run still records
        # the change rather than being silenced by the dedupe.
        seen: set[str] = set()
        kept = []
        for cfg in (TRAIN_CONFIG, {**TRAIN_CONFIG, "learning_rate": 2e-05}):
            key = json.dumps(cfg, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            kept.append(key)
        assert len(kept) == 2


@pytest.mark.parametrize("suffix", ["", "\r\n", "   "])
def test_trailing_whitespace_does_not_break_the_last_marker(suffix):
    got = _stream_markers(_log("train-028", TRAIN_CONFIG) + suffix, "train-028")
    assert [k for k, _ in got][-1] == "phase"
