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
async def test_cached_coding_dataset_rehomes_answers_for_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    import zevo.engine.remote_datasets as remote
    from zevo.code_benchmarks import externalize_code_answers, resolve_code_answer

    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    monkeypatch.setenv(
        "ZEVO_VALIDATION_CODE_ANSWERS_ROOT",
        str(tmp_path / "validation-code-answers"),
    )
    monkeypatch.setenv("ZEVO_HF_DATASET_CACHE_TTL_SECONDS", "86400")
    monkeypatch.setattr(remote, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(remote, "_hf_token", lambda: "")

    row = externalize_code_answers(
        "code_contests",
        {"description": "Add two numbers", "private_tests": {
            "input": ["2 3\n"], "output": ["5\n"],
        }},
    )
    source = tmp_path / "source.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    entry, identity = remote._cache_identity(
        hub_id="deepmind/code_contests", split="train", config="default",
        limit=1, token="",
    )
    remote._store_materialization(
        entry=entry, identity=identity, source=source,
        columns=list(row), n_rows=1, config="default", split="train",
    )

    path, _columns, n_rows, note = await remote.materialize(
        hub_id="deepmind/code_contests", split="train", config="default",
        out_dir=str(tmp_path / "run"), limit=1, answer_scope="validation",
    )
    with Path(path).open(newline="", encoding="utf-8") as handle:
        migrated = next(csv.DictReader(handle))
    assert n_rows == 1
    assert "reused cached" in note
    assert migrated["private_tests"].startswith("zevo-code-answer:validation:v1:")
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "unmounted"))
    assert json.loads(resolve_code_answer(migrated["private_tests"])) == {
        "input": ["2 3\n"], "output": ["5\n"],
    }


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


@pytest.mark.parametrize("scope,prefix", [
    ("test", "zevo-code-answer:v1:"),
    ("validation", "zevo-code-answer:validation:v1:"),
])
def test_livecodebench_parquet_streams_private_answers_to_sidecars(
    tmp_path: Path, monkeypatch, scope: str, prefix: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from zevo.code_benchmarks import resolve_code_answer
    import zevo.engine.remote_datasets as remote

    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    monkeypatch.setenv(
        "ZEVO_VALIDATION_CODE_ANSWERS_ROOT",
        str(tmp_path / "validation-code-answers"),
    )
    private = "encoded-private-case" * 150_000
    shard = tmp_path / "livecodebench.parquet"
    pq.write_table(pa.table({
        "question_id": ["lcb/1"],
        "question_content": ["Solve it"],
        "private_test_cases": [private],
    }), shard)

    path, columns, rows = remote._write_parquet_csv(
        shard_paths=[shard],
        out_dir=str(tmp_path / "converted"),
        limit=0,
        hub_id="sam-paech/livecodebench-code_generation_lite",
        answer_scope=scope,
    )

    with Path(path).open(newline="", encoding="utf-8") as handle:
        converted = next(csv.DictReader(handle))
    assert columns == ["question_id", "question_content", "private_test_cases"]
    assert rows == 1
    assert converted["private_test_cases"].startswith(prefix)
    assert resolve_code_answer(converted["private_test_cases"]) == private


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


@pytest.mark.asyncio
async def test_rows_fallback_streams_pages_and_keeps_late_columns(
    tmp_path: Path, monkeypatch,
) -> None:
    import zevo.engine.remote_datasets as remote

    monkeypatch.setattr(remote, "_cache_root", lambda: tmp_path / "cache")
    records = [
        {"row": {
            "question": f"question {index}", "context": "x" * 1024,
            "answer": index,
            **({"late_field": "present"} if index == 100 else {}),
        }}
        for index in range(101)
    ]
    pages: list[tuple[int, int]] = []

    class StreamingOnlyResponse(httpx.Response):
        def json(self, **kwargs):  # type: ignore[override]
            raise AssertionError("rows response must not be decoded as one JSON object")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/splits":
            return httpx.Response(200, json={
                "splits": [{"config": "default", "split": "test"}],
            }, request=request)
        if request.url.path == "/parquet":
            return httpx.Response(200, json={
                "partial": True, "parquet_files": [],
            }, request=request)
        assert request.url.path == "/rows"
        offset = int(request.url.params["offset"])
        length = int(request.url.params["length"])
        pages.append((offset, length))
        return StreamingOnlyResponse(
            200, content=json.dumps({"rows": records[offset:offset + length]}),
            request=request,
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = await remote.materialize(
        hub_id="owner/benchmark", split="test", config="default",
        out_dir=str(tmp_path / "run"), limit=101,
    )

    assert pages == [(0, 100), (100, 1)]
    assert result[1:3] == (["question", "context", "answer", "late_field"], 101)
    with Path(result[0]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["question"] == "question 0"
    assert rows[-1]["late_field"] == "present"
    assert not list((tmp_path / "run").glob("*.tmp"))
    assert not list((tmp_path / "run").glob(".*.tmp"))


@pytest.mark.asyncio
async def test_rows_fallback_retries_interrupted_page_without_duplicates(
    tmp_path: Path, monkeypatch,
) -> None:
    import zevo.engine.remote_datasets as remote

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"rows":[{"row":{"question":"one"}},'
            raise httpx.ReadError("connection dropped mid-page")

        async def aclose(self) -> None:
            pass

    calls = 0
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.params["offset"] == "0"
        assert request.url.params["length"] == "2"
        if calls == 1:
            return httpx.Response(200, stream=BrokenStream(), request=request)
        return httpx.Response(200, json={"rows": [
            {"row": {"question": "one"}},
            {"row": {"question": "two"}},
        ]}, request=request)

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(remote.asyncio, "sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        path, columns, count = await remote._materialize_rows(
            client=client, hub_id="owner/benchmark", config="default",
            split="test", out_dir=str(tmp_path / "run"), limit=2,
        )

    assert (columns, count) == (["question"], 2)
    assert calls == 2
    assert slept == [1.0]
    with Path(path).open(newline="", encoding="utf-8") as handle:
        assert [row["question"] for row in csv.DictReader(handle)] == ["one", "two"]
    assert not list((tmp_path / "run").glob("*.tmp"))
    assert not list((tmp_path / "run").glob(".*.tmp"))


@pytest.mark.asyncio
async def test_rows_fallback_retries_truncated_json_page(
    tmp_path: Path, monkeypatch,
) -> None:
    import zevo.engine.remote_datasets as remote

    calls = 0
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200, content=b'{"rows":[{"row":{"question":"one"}},',
                request=request,
            )
        return httpx.Response(200, json={"rows": [
            {"row": {"question": "one"}},
            {"row": {"question": "two"}},
        ]}, request=request)

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(remote.asyncio, "sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        path, _columns, count = await remote._materialize_rows(
            client=client, hub_id="owner/benchmark", config="default",
            split="test", out_dir=str(tmp_path / "run"), limit=2,
        )

    assert count == 2
    assert calls == 2
    assert slept == [1.0]
    with Path(path).open(newline="", encoding="utf-8") as handle:
        assert [row["question"] for row in csv.DictReader(handle)] == ["one", "two"]


@pytest.mark.asyncio
async def test_rows_fallback_retries_429_with_retry_after(
    tmp_path: Path, monkeypatch,
) -> None:
    import zevo.engine.remote_datasets as remote

    calls = 0
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429, headers={"Retry-After": "0.25"}, request=request,
            )
        return httpx.Response(200, json={"rows": [
            {"row": {"question": "one"}},
        ]}, request=request)

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(remote.asyncio, "sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        _path, _columns, count = await remote._materialize_rows(
            client=client, hub_id="owner/benchmark", config="default",
            split="test", out_dir=str(tmp_path / "run"), limit=1,
        )

    assert count == 1
    assert calls == 2
    assert slept == [0.25]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,prefix", [
    ("test", "zevo-code-answer:v1:"),
    ("validation", "zevo-code-answer:validation:v1:"),
])
async def test_rows_fallback_preserves_code_answer_sidecars(
    tmp_path: Path, monkeypatch, scope: str, prefix: str,
) -> None:
    from zevo.code_benchmarks import resolve_code_answer
    import zevo.engine.remote_datasets as remote

    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "holdout"))
    monkeypatch.setenv(
        "ZEVO_VALIDATION_CODE_ANSWERS_ROOT",
        str(tmp_path / "validation-code-answers"),
    )
    hidden = {"input": ["secret" * 1000], "output": ["answer"]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["length"] == "1"
        return httpx.Response(200, json={"rows": [{"row": {
            "description": "Solve this", "private_tests": hidden,
        }}]}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        path, columns, count = await remote._materialize_rows(
            client=client, hub_id="deepmind/code_contests", config="default",
            split="train", out_dir=str(tmp_path / "run"), limit=1,
            answer_scope=scope,
        )

    assert (columns, count) == (["description", "private_tests"], 1)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["private_tests"].startswith(prefix)
    assert json.loads(resolve_code_answer(row["private_tests"])) == hidden
