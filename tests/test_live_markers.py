from pathlib import Path

from zevo.engine.observe.live_markers import LiveMarkerReader


def test_reader_backfills_then_only_returns_appended_markers(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "noise\r__PHASE__:train-001:training@100\n"
        '__PROGRESS__:train-001:{"step":1,"total":10,"loss":2.0}\n'
    )
    reader = LiveMarkerReader(tmp_path, "train-001")

    first = reader.poll()
    assert [kind for kind, _ in first] == ["phase", "progress"]
    assert first[1][1]["step"] == 1
    assert reader.poll() == []

    with log.open("a") as fh:
        fh.write('__PROGRESS__:train-001:{"step":2,"total":10,"loss":1.5}\n')
    assert [(kind, payload["step"]) for kind, payload in reader.poll()] == [
        ("progress", 2)
    ]


def test_reader_waits_for_complete_line_and_filters_other_owner(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_bytes(b'__PROGRESS__:train-001:{"step":3')
    reader = LiveMarkerReader(tmp_path, "train-001")
    assert reader.poll() == []

    with log.open("ab") as fh:
        fh.write(b',"total":10,"loss":1.0}\n')
        fh.write(b'__PROGRESS__:infer-002:{"step":9,"total":10,"loss":9.0}\n')
    assert [(kind, payload["step"]) for kind, payload in reader.poll()] == [
        ("progress", 3)
    ]


def test_reader_keeps_json_escaped_newlines_inside_config(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        '__CONFIG__:train-001:{"prompt":"user\\nassistant","epochs":3}\n'
    )
    got = LiveMarkerReader(tmp_path, "train-001").poll()
    assert got == [("config", {"prompt": "user\nassistant", "epochs": 3})]


def test_runner_mode_skips_stale_file_but_reads_new_bytes(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text('__PROGRESS__:train-001:{"step":1,"total":10,"loss":2.0}\n')
    reader = LiveMarkerReader(tmp_path, "train-001", start_at_end=True)
    assert reader.poll() == []

    with log.open("a") as fh:
        fh.write('__PROGRESS__:train-001:{"step":2,"total":10,"loss":1.5}\n')
    assert reader.poll()[0][1]["step"] == 2
