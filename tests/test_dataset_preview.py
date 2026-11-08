"""Remote samples are bounded, tied to verified data, and survive host release."""
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from zevo.engine import dataset_preview
from zevo.engine.remote_training_data import RemoteDatasetReceipt


def receipt_for(tmp_path, dataset, signature="1" * 64):
    path = tmp_path / "remote_data_receipt.json"
    path.write_text(RemoteDatasetReceipt(
        data_signature=signature, dataset_path=str(dataset), dataset_sha256="2" * 64,
        profile_path="/remote/profile.json", n_rows_before_decontamination=300,
        n_rows=250, decontamination_removed_rows=50, record_fields=["messages"],
    ).model_dump_json())
    return str(path)


@pytest.mark.asyncio
async def test_bounded_backfill_preserves_chat_shape_and_survives_host_release(tmp_path, monkeypatch):
    # Shell metacharacters in an engine-owned path must remain literal.
    data = tmp_path / "data ' $(echo unwanted).jsonl"
    row = {"messages": [{"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"}]}
    data.write_text("\n".join(json.dumps({**row, "id": i}) for i in range(250)))
    receipt = receipt_for(tmp_path, data)
    calls = []

    async def ssh(info, command, **kwargs):
        calls.append(command)
        args = shlex.split(command)
        assert args[3] == str(data)
        output = subprocess.check_output([sys.executable, *args[1:]], text=True)
        return {"ok": True, "stdout": output}

    monkeypatch.setattr(dataset_preview, "_ssh", ssh)
    rows, error = await dataset_preview.remote_training_preview(
        info=object(), dataset_path=str(data), receipt_path=receipt,
    )
    assert not error and len(rows) == 100
    assert rows[0]["messages"] == row["messages"] and rows[-1]["id"] == 99
    data.unlink()  # The training host/file can go away after sampling.
    cached, error = await dataset_preview.remote_training_preview(
        info=None, dataset_path=str(data), receipt_path=receipt,
    )
    assert cached == rows and not error and len(calls) == 1
    receipt_for(tmp_path, data, signature="3" * 64)
    stale, error = await dataset_preview.remote_training_preview(
        info=None, dataset_path=str(data), receipt_path=receipt,
    )
    assert stale == [] and "connection" in error


@pytest.mark.parametrize("reply, reason", [
    ({"ok": False, "error": "ssh credentials must never be shown"}, "could not be reached"),
    ({"ok": True, "stdout": '{"error":"missing"}'}, "no longer present"),
    ({"ok": True, "stdout": '{"error":"permission"}'}, "not readable"),
    ({"ok": True, "stdout": '{"error":"unreadable"}'}, "oversized"),
    ({"ok": True, "stdout": 'garbage'}, "invalid sample"),
    ({"ok": True, "stdout": '{"rows":[]}', "stdout_truncated": True}, "size limit"),
])
@pytest.mark.asyncio
async def test_remote_failure_has_a_safe_reason_without_poisoning_cache(tmp_path, monkeypatch, reply, reason):
    receipt = receipt_for(tmp_path, "/remote/data.jsonl")

    async def ssh(*args, **kwargs):
        return reply

    monkeypatch.setattr(dataset_preview, "_ssh", ssh)
    rows, error = await dataset_preview.remote_training_preview(
        info=object(), dataset_path="/remote/data.jsonl", receipt_path=receipt,
    )
    assert rows == [] and reason in error and "credentials" not in error
    assert not list(tmp_path.glob(".dataset-previews/*.json"))


def test_reader_rejects_oversized_records_and_clips_long_cells(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": "a" * 300_000}))
    def read():
        return json.loads(subprocess.check_output(
            [sys.executable, "-c", dataset_preview._REMOTE_READ, str(data)], text=True,
        ))
    assert read() == {"error": "unreadable"}
    data.write_text(json.dumps({"messages": [{"content": "a" * 5000}]}))
    assert len(read()["rows"][0]["messages"][0]["content"]) == 4001


@pytest.mark.asyncio
async def test_remote_artifact_api_paginates_cached_rows_even_without_local_dataset(tmp_path, monkeypatch):
    from zevo.api.routers.shared import runs
    from zevo.engine.run import remote_jobs

    receipt = receipt_for(tmp_path, "/remote/data.jsonl")

    async def ssh(*args, **kwargs):
        return {"ok": True, "stdout": json.dumps({"rows": [{"id": i} for i in range(23)]})}

    monkeypatch.setattr(dataset_preview, "_ssh", ssh)
    await dataset_preview.remote_training_preview(
        info=object(), dataset_path="/remote/data.jsonl", receipt_path=receipt,
    )
    artifact = SimpleNamespace(
        id="artifact", ticket_id="data", role="training_dataset", path="/remote/data.jsonl",
        meta={"location": "remote", "remote_data_receipt_path": receipt}, created_at=None,
    )
    ticket = SimpleNamespace(lane="optimization", iteration=5)

    async def execute(*args, **kwargs):
        return SimpleNamespace(first=lambda: (artifact, ticket))

    async def retained(*args):
        return None, None, False, 0

    async def no_device(*args):
        return None

    monkeypatch.setattr(runs, "_retained_model_for_run", retained)
    monkeypatch.setattr(remote_jobs, "_device_info_for_ticket", no_device)
    result = await runs.get_artifact_detail(
        "run", "artifact", Request({"type": "http", "headers": []}),
        preview_page=3, preview_page_size=10, db=SimpleNamespace(execute=execute),
    )
    assert result.availability == "remote" and not result.exists
    assert result.preview_rows == [{"id": 20}, {"id": 21}, {"id": 22}]
    assert result.preview_columns == ["id"] and result.preview_total_rows == 23


@pytest.mark.asyncio
async def test_receipt_mismatch_never_reads_an_unverified_file(tmp_path, monkeypatch):
    receipt = receipt_for(tmp_path, "/remote/verified.jsonl")

    async def ssh(*args, **kwargs):
        pytest.fail("mismatched artifact must never be read")

    monkeypatch.setattr(dataset_preview, "_ssh", ssh)
    rows, error = await dataset_preview.remote_training_preview(
        info=object(), dataset_path="/remote/prepared.jsonl", receipt_path=receipt,
    )
    assert not rows and "does not match" in error
