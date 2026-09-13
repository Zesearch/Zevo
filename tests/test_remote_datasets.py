from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_dataset_viewer_retries_429_and_honors_retry_after(monkeypatch) -> None:
    import zevo.engine.remote_datasets as remote

    attempts = 0
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.25"},
                request=request,
            )
        return httpx.Response(200, json={"rows": []}, request=request)

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(remote.asyncio, "sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await remote._get_json(
            client,
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": "owner/data"},
        )

    assert result == {"rows": []}
    assert attempts == 2
    assert slept == [0.25]


@pytest.mark.asyncio
async def test_materialize_uses_saved_token_then_reuses_private_cache(
    tmp_path: Path, monkeypatch,
) -> None:
    import zevo.engine.remote_datasets as remote

    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN='hf_saved_after_backend_started'\n", encoding="utf-8")
    monkeypatch.setenv("ZEVO_ENV_FILE", str(env_file))
    monkeypatch.setenv("HF_TOKEN", "hf_stale_process_value")
    monkeypatch.setenv("ZEVO_HF_DATASET_CACHE_TTL_SECONDS", "86400")
    cache_root = tmp_path / "private-cache"
    monkeypatch.setattr(remote, "_cache_root", lambda: cache_root)

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer hf_saved_after_backend_started"
        if request.url.path == "/splits":
            body = {"splits": [{"config": "default", "split": "test"}]}
        elif request.url.path == "/parquet":
            body = {"parquet_files": [], "partial": False}
        elif request.url.params.get("offset") == "0":
            body = {"rows": [
                {"row": {"question": "one", "answer": 1}},
                {"row": {"question": "two", "answer": 2}},
            ]}
        else:
            body = {"rows": []}
        return httpx.Response(200, json=body, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    first = await remote.materialize(
        hub_id="owner/benchmark",
        split="test",
        config="default",
        out_dir=str(tmp_path / "run-one"),
    )
    request_count = len(requests)
    second = await remote.materialize(
        hub_id="owner/benchmark",
        split="test",
        config="default",
        out_dir=str(tmp_path / "run-two"),
    )

    assert request_count == 4  # splits, Parquet probe, rows, then empty page
    assert len(requests) == request_count, "a cache hit makes no Hub request"
    assert first[1:3] == (["question", "answer"], 2)
    assert second[1:3] == (["question", "answer"], 2)
    assert "reused cached" in second[3]
    assert Path(first[0]).read_text() == Path(second[0]).read_text()
    cache_metadata = next(cache_root.glob("*/metadata.json"))
    assert "hf_saved_after_backend_started" not in cache_metadata.read_text()
    assert json.loads(cache_metadata.read_text())["n_rows"] == 2


@pytest.mark.asyncio
async def test_materialize_prefers_one_parquet_shard_over_many_row_requests(
    tmp_path: Path, monkeypatch,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    import zevo.engine.remote_datasets as remote

    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({
        "question": ["one", "two"],
        "answer": [1, 2],
        "choices": [["a", "b"], ["c", "d"]],
    }), sink)
    payload = sink.getvalue().to_pybytes()
    cache_root = tmp_path / "private-cache"
    monkeypatch.setattr(remote, "_cache_root", lambda: cache_root)
    monkeypatch.setenv("ZEVO_HF_DATASET_CACHE_TTL_SECONDS", "86400")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/splits":
            return httpx.Response(200, json={
                "splits": [{"config": "all", "split": "test"}],
            }, request=request)
        if request.url.path == "/parquet":
            return httpx.Response(200, json={
                "parquet_files": [{
                    "config": "all",
                    "split": "test",
                    "url": "https://huggingface.co/example/shard.parquet",
                    "filename": "0000.parquet",
                    "size": len(payload),
                }],
                "partial": False,
                "pending": [],
                "failed": [],
            }, request=request)
        if request.url.path == "/example/shard.parquet":
            return httpx.Response(200, content=payload, request=request)
        raise AssertionError(f"unexpected request: {request.url}")

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    result = await remote.materialize(
        hub_id="cais/mmlu",
        split="test",
        config="all",
        out_dir=str(tmp_path / "run"),
    )

    assert [request.url.path for request in requests] == [
        "/splits", "/parquet", "/example/shard.parquet",
    ]
    assert result[1:3] == (["question", "answer", "choices"], 2)
    assert "via Parquet" in result[3]
    with Path(result[0]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"question": "one", "answer": "1", "choices": '["a", "b"]'},
        {"question": "two", "answer": "2", "choices": '["c", "d"]'},
    ]
    assert not any(request.url.path == "/rows" for request in requests)


@pytest.mark.asyncio
async def test_parquet_shard_download_resumes_retained_partial_file(
    tmp_path: Path,
) -> None:
    import zevo.engine.remote_datasets as remote

    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
    target = tmp_path / "cached.parquet"
    partial = target.with_suffix(".parquet.part")
    partial.write_bytes(payload[:10])
    seen_ranges: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("Range", ""))
        return httpx.Response(206, content=payload[10:], request=request)

    shard = remote._ParquetShard(
        url="https://huggingface.co/example/shard.parquet",
        filename="0000.parquet",
        size=len(payload),
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await remote._download_parquet_shard(client, shard, target)

    assert seen_ranges == ["bytes=10-"]
    assert result.read_bytes() == payload
    assert not partial.exists()


@pytest.mark.asyncio
async def test_partial_parquet_manifest_falls_back_without_truncation(
    tmp_path: Path, monkeypatch,
) -> None:
    import zevo.engine.remote_datasets as remote

    monkeypatch.setattr(remote, "_cache_root", lambda: tmp_path / "cache")
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/splits":
            body = {"splits": [{"config": "default", "split": "test"}]}
        elif request.url.path == "/parquet":
            body = {
                "partial": True,
                "parquet_files": [{
                    "config": "default", "split": "test",
                    "url": "https://huggingface.co/example/only-one.parquet",
                }],
            }
        elif request.url.params.get("offset") == "0":
            body = {"rows": [{"row": {"question": "complete", "answer": 1}}]}
        else:
            body = {"rows": []}
        return httpx.Response(200, json=body, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    result = await remote.materialize(
        hub_id="owner/partial",
        split="test",
        config="default",
        out_dir=str(tmp_path / "run"),
    )

    assert paths == ["/splits", "/parquet", "/rows", "/rows"]
    assert result[2] == 1
    assert "via rows API" in result[3]
