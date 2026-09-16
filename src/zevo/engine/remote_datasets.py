"""What the catalogue says about a hub dataset, so the run does not guess.

A dataset in the catalogue can name files it does not store: `ifeval-train`
holds its evaluation side on disk and pulls its training rows from
`trl-lib/Capybara`. Those entries live in the dataset's `source.json`, and until
now nothing read them at run time — the hub id reached the data agent as a bare
string and the agent downloaded `split="train"`, because that was hardcoded in
its skill.

That is fine right up until it is not. A repo whose training rows live under
`train_sft`, a repo with several configs, or — the case this exists for — a run
that wants the repo's *validation* split as its validation set, cannot be
expressed by a string that only carries the repo name. Worse, the failure is
silent: you get the `train` split of something, it loads, it trains, and the
mismatch surfaces as a disappointing score.

So the declaration is made once, in the UI, on the dataset. Ticket creation
looks it up here and stores the resolved split
on the Data Ticket before a worker runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from zevo.paths import files_root as _files_root


log = logging.getLogger(__name__)

# Same location the files router serves from (zevo.paths.files_root, so the
# ZEVO_FILES_DIR override applies here exactly like it does in the API).
_DATASETS_DIR = Path(_files_root())


@dataclass(frozen=True)
class RemoteSpec:
    """How to load one hub dataset, as the catalogue declares it."""

    id: str
    split: str = ""
    config: str = ""
    role: str = ""
    dataset: str = ""   # the catalogue dataset that declares it
    n_rows: int = 0      # known size of the declared source split, if recorded

    def as_payload(self) -> dict[str, str]:
        """The subset worth putting on a ticket — omitting what was not declared,
        so an empty string never reads as a deliberate choice of ''."""
        out: dict[str, str] = {}
        if self.split:
            out["dataset_split"] = self.split
        if self.config:
            out["dataset_config"] = self.config
        return out


def _role_of(split: str) -> str:
    """What a slice IS, read off its name. Mirrors the datasets router."""
    s = (split or "").strip().lower()
    if s.startswith(("validation", "valid", "dev")):
        return "validation"
    if s.startswith("test"):
        return "test"
    return "train"


def looks_like_hub_id(value: str) -> bool:
    """`owner/name`, with no file extension — the shape of a hub id.

    Deliberately narrow. A path that happens to have one slash and no suffix is
    rare; treating a real file as a hub id would send the data agent to the
    network for something already on disk.
    """
    v = (value or "").strip()
    if not v or v.startswith(("http://", "https://", "/", ".", "~")):
        return False
    return v.count("/") == 1 and not Path(v).suffix


class MaterializeError(RuntimeError):
    """A hub dataset could not be fetched — raised at run creation, where it is
    still a fixable input rather than a stage failing an hour in."""


class _ParquetUnavailable(RuntimeError):
    """The converted Parquet route is unavailable; the rows API may be used."""


class _IncompleteDownload(RuntimeError):
    """A shard response ended before its declared byte size."""


@dataclass(frozen=True)
class _ParquetShard:
    url: str
    filename: str
    size: int


# Dataset Viewer resolves the exact config/split and publishes converted
# Parquet files. Reading those files takes one request per shard instead of one
# request per 100 rows. The rows endpoint remains a compatibility fallback for
# datasets whose Parquet conversion is pending or unavailable.
_SERVER = "https://datasets-server.huggingface.co"
_PAGE = 100
# The rows endpoint may return large records for any dataset. Stream its JSON
# body instead of sizing requests according to a particular benchmark name.
_JSON_STREAM_CHUNK = 64 * 1024
# Not a policy on validation-set size — that is the user's call, and a cap here
# used to silently make it for them by taking a prefix. This is a runaway guard:
# the rows API pages 100 at a time, so a million-row split is ten thousand
# sequential HTTP calls inside run creation. Past this the run is refused with
# the count, which is a thing you can act on; a truncated set is not.
_MAX_FETCH = 200_000

# Dataset Viewer enforces request quotas, and a benchmark suite can require
# dozens of 100-row calls. Authenticated requests receive the account's quota;
# bounded retries absorb a short shared-service throttle instead of making the
# whole Run fail on the first 429.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0)
_MAX_RETRY_DELAY = 30.0

# Materialized held-out rows are reusable Run inputs, not ordinary user files.
# Keep them behind the same private boundary as Test data. A one-day default
# prevents every launch from redownloading a 17-member suite while still
# allowing an unpinned Hub dataset to refresh. Set the TTL to 0 to disable.
_CACHE_VERSION = 1
_DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60


def _hf_token() -> str:
    """Return the latest Hub token without requiring a container restart.

    Settings writes ``/app/.env`` after the backend has started, so consulting
    only ``os.environ`` makes a freshly saved token look configured in the UI
    while Dataset Viewer calls remain anonymous. The env file wins so token
    replacement and clearing also take effect immediately.
    """
    env_path = Path(os.environ.get("ZEVO_ENV_FILE", "/app/.env"))
    try:
        if env_path.is_file():
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() != "HF_TOKEN":
                    continue
                value = value.strip()
                if (
                    len(value) >= 2
                    and value[0] == value[-1]
                    and value[0] in {"'", '"'}
                ):
                    value = value[1:-1]
                return value.strip()
    except OSError as exc:
        # A transient bind-mount/read error must not hide the token Compose
        # already injected into the process.
        log.warning("cannot read HF_TOKEN from %s: %s", env_path, exc)
    return os.environ.get("HF_TOKEN", "").strip()


def _hf_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _retry_delay(response, fallback: float) -> float:
    """Honor Retry-After (seconds or HTTP date), with a finite UI wait."""
    raw = (response.headers.get("Retry-After") or "").strip()
    if raw:
        try:
            delay = float(raw)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw)
                delay = retry_at.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                delay = fallback
        return min(_MAX_RETRY_DELAY, max(0.0, delay))
    return min(_MAX_RETRY_DELAY, fallback)


async def _get_json(client, url: str, *, params: dict[str, object]) -> dict:
    """GET one Dataset Viewer document with bounded transient retries."""
    import httpx

    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if (
                exc.response.status_code not in _RETRYABLE_STATUS
                or attempt >= len(_RETRY_DELAYS)
            ):
                raise
            delay = _retry_delay(exc.response, _RETRY_DELAYS[attempt])
            log.warning(
                "Hugging Face Dataset Viewer returned %s; retrying in %.1fs "
                "(%d/%d)",
                exc.response.status_code,
                delay,
                attempt + 1,
                len(_RETRY_DELAYS),
            )
            await asyncio.sleep(delay)
        except httpx.TransportError:
            if attempt >= len(_RETRY_DELAYS):
                raise
            delay = _RETRY_DELAYS[attempt]
            log.warning(
                "Hugging Face Dataset Viewer request failed; retrying in %.1fs "
                "(%d/%d)",
                delay,
                attempt + 1,
                len(_RETRY_DELAYS),
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


class _AsyncChunkReader:
    """Expose an HTTP byte iterator as the bounded async ``read`` ijson needs."""

    def __init__(self, chunks: AsyncIterator[bytes]) -> None:
        self._chunks = chunks.__aiter__()
        self._buffer = bytearray()
        self._done = False

    async def read(self, size: int = _JSON_STREAM_CHUNK) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            size = _JSON_STREAM_CHUNK
        size = min(size, _JSON_STREAM_CHUNK)
        while len(self._buffer) < size and not self._done:
            try:
                chunk = await self._chunks.__anext__()
            except StopAsyncIteration:
                self._done = True
            else:
                self._buffer.extend(chunk)
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result


def _cache_root() -> Path:
    from zevo.paths import holdout_root

    return Path(holdout_root()) / "huggingface-datasets"


def _cache_ttl_seconds() -> int:
    raw = os.environ.get(
        "ZEVO_HF_DATASET_CACHE_TTL_SECONDS",
        str(_DEFAULT_CACHE_TTL_SECONDS),
    )
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_CACHE_TTL_SECONDS


def _cache_identity(
    *, hub_id: str, split: str, config: str, limit: int, token: str,
) -> tuple[Path, dict[str, object]]:
    # The token itself is never persisted. Its digest prevents a private/gated
    # snapshot fetched with one credential from being silently reused after the
    # installation switches credentials.
    identity: dict[str, object] = {
        "version": _CACHE_VERSION,
        "hub_id": hub_id,
        "requested_split": split,
        "requested_config": config,
        "limit": limit,
        "credential": hashlib.sha256(token.encode()).hexdigest() if token else "anonymous",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _cache_root() / digest, identity


def _cached_materialization(
    *, entry: Path, identity: dict[str, object], out_dir: str,
) -> tuple[str, list[str], int, str] | None:
    ttl = _cache_ttl_seconds()
    data_path = entry / "dataset.csv"
    meta_path = entry / "metadata.json"
    if ttl <= 0 or not data_path.is_file() or not meta_path.is_file():
        return None
    try:
        if time.time() - meta_path.stat().st_mtime > ttl:
            # The completed snapshot expired. Its copied Run artifacts remain
            # intact, but neither the CSV nor its converted shards should pin
            # an unversioned Hub dataset beyond the configured TTL.
            data_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            shutil.rmtree(entry / "parquet", ignore_errors=True)
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("identity") != identity:
            return None
        from zevo.code_benchmarks import code_execution_adapter_for

        if (
            code_execution_adapter_for(str(identity.get("hub_id") or ""))
            in {"livecodebench", "code_contests"}
            and meta.get("code_answer_projection") != "content_addressed_v1"
        ):
            # Older caches embed multi-gigabyte private tests in CSV cells and
            # cannot be streamed by the normal tabular pipeline. Re-project
            # from retained Parquet shards without another Hub download.
            return None
        columns = meta.get("columns")
        n_rows = meta.get("n_rows")
        if (
            not isinstance(columns, list)
            or not all(isinstance(c, str) for c in columns)
            or not isinstance(n_rows, int)
            or n_rows <= 0
            or data_path.stat().st_size <= 0
        ):
            return None
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "validation.csv"
        shutil.copy2(data_path, path)
        cfg = str(meta.get("resolved_config") or "")
        spl = str(meta.get("resolved_split") or "")
        scope = "all" if int(identity["limit"]) <= 0 else "the first"
        note = f"reused cached {scope} {n_rows} rows of {identity['hub_id']} ({cfg}/{spl})"
        return str(path), columns, n_rows, note
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        log.warning("ignoring invalid Hugging Face dataset cache %s: %s", entry, exc)
        return None


def _store_materialization(
    *, entry: Path, identity: dict[str, object], source: Path,
    columns: list[str], n_rows: int, config: str, split: str,
) -> None:
    if _cache_ttl_seconds() <= 0:
        return
    try:
        entry.mkdir(parents=True, exist_ok=True)
        nonce = uuid4().hex
        data_tmp = entry / f"dataset.{nonce}.tmp"
        meta_tmp = entry / f"metadata.{nonce}.tmp"
        shutil.copy2(source, data_tmp)
        from zevo.code_benchmarks import code_execution_adapter_for

        adapter = code_execution_adapter_for(str(identity.get("hub_id") or ""))
        meta_tmp.write_text(json.dumps({
            "identity": identity,
            "resolved_config": config,
            "resolved_split": split,
            "columns": columns,
            "n_rows": n_rows,
            "code_answer_projection": (
                "content_addressed_v1"
                if adapter in {"livecodebench", "code_contests"} else ""
            ),
        }, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(data_tmp, entry / "dataset.csv")
        # Metadata is the commit marker and is therefore replaced last.
        os.replace(meta_tmp, entry / "metadata.json")
    except OSError as exc:
        # Caching is an optimization. A successfully fetched Run must remain
        # launchable when its cache volume is temporarily unwritable.
        log.warning("cannot cache Hugging Face dataset %s: %s", identity["hub_id"], exc)


async def _resolve_split(
    client, hub_id: str, split: str, config: str,
) -> tuple[str, str]:
    """Settle (config, split) against what the repo actually publishes.

    Asking for a split the repo does not have is the failure worth catching
    here: the alternative is a validation set that silently ends up being some
    other split, which nothing downstream can detect.
    """
    try:
        available = (await _get_json(
            client,
            f"{_SERVER}/splits",
            params={"dataset": hub_id},
        )).get("splits") or []
    except Exception as e:  # network, 404, gated repo, malformed JSON
        raise MaterializeError(f"cannot read the splits of {hub_id!r}: {e}")
    if not available:
        raise MaterializeError(f"{hub_id!r} publishes no splits")

    pairs = [(str(s.get("config") or ""), str(s.get("split") or "")) for s in available]

    def _catalogue(limit: int = 12) -> str:
        shown = [f"{c}/{s}" for c, s in pairs[:limit]]
        more = len(pairs) - len(shown)
        return ", ".join(shown) + (f", … and {more} more" if more > 0 else "")

    def _one(matches: list[tuple[str, str]], what: str) -> tuple[str, str]:
        """Exactly one candidate, or say why there is a choice to make.

        A repo can hold many datasets — `cais/mmlu` has 59 subject configs and
        `nyu-mll/glue` twelve unrelated tasks, each with its own `validation`.
        Taking the first match would have quietly validated an MMLU run on
        `abstract_algebra`: a real split, a plausible number, and the wrong
        dataset. Nothing downstream could tell.
        """
        if not matches:
            raise MaterializeError(
                f"{hub_id!r} has no {what}. It has: {_catalogue()}"
            )
        configs = sorted({c for c, _ in matches})
        if len(configs) > 1:
            raise MaterializeError(
                f"{hub_id!r} holds {len(configs)} datasets with a {what} — name "
                f"which one in `config`: {', '.join(configs[:12])}"
                + (f", … and {len(configs) - 12} more" if len(configs) > 12 else "")
            )
        return matches[0]

    if config and split:
        if (config, split) not in pairs:
            raise MaterializeError(
                f"{hub_id!r} has no split {split!r} in config {config!r}. "
                f"It has: {_catalogue()}"
            )
        return config, split
    if split:
        return _one(
            [(c, s) for c, s in pairs if s == split and (not config or c == config)],
            f"{split!r} split",
        )
    # No split named. Prefer a validation-like split rather than silently
    # defaulting to training rows.
    for wanted in ("validation", "valid", "dev", "test"):
        matches = [(c, s) for c, s in pairs if s == wanted and (not config or c == config)]
        if matches:
            return _one(matches, f"{wanted!r} split")
    raise MaterializeError(
        f"{hub_id!r} has no validation-like split — name one explicitly. "
        f"It has: {_catalogue()}"
    )


def _cell(v: object) -> str:
    """A hub row's cell as CSV text. Nested values become JSON, the same way the
    carve flattens JSONL, so a list of options survives the round trip."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)


async def _parquet_shards(
    client, hub_id: str, config: str, split: str,
) -> list[_ParquetShard]:
    """Return the complete converted-Parquet manifest for one resolved split."""
    try:
        document = await _get_json(
            client,
            f"{_SERVER}/parquet",
            params={"dataset": hub_id},
        )
    except Exception as exc:
        raise _ParquetUnavailable(f"cannot list converted Parquet files: {exc}") from exc

    # A partial manifest is not a smaller equivalent dataset. Falling back is
    # preferable to silently evaluating on only the shards converted so far.
    if document.get("partial"):
        raise _ParquetUnavailable("converted Parquet manifest is partial")

    shards: list[_ParquetShard] = []
    for raw in document.get("parquet_files") or []:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("config") or "") != config:
            continue
        if str(raw.get("split") or "") != split:
            continue
        url = str(raw.get("url") or "").strip()
        if not url.startswith(("https://", "http://")):
            continue
        try:
            size = max(0, int(raw.get("size") or 0))
        except (TypeError, ValueError):
            size = 0
        shards.append(_ParquetShard(
            url=url,
            filename=str(raw.get("filename") or Path(url).name or "shard.parquet"),
            size=size,
        ))
    if not shards:
        raise _ParquetUnavailable(
            f"no converted Parquet files for {config}/{split}"
        )
    return shards


@asynccontextmanager
async def _exclusive_file_lock(path: Path) -> AsyncIterator[None]:
    """Serialize writers of a persistent shard cache across API requests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Zevo containers are POSIX
            yield
            return
        await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


async def _download_parquet_shard(
    client, shard: _ParquetShard, target: Path,
) -> Path:
    """Download one shard atomically, resuming a retained `.part` file."""
    import httpx

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    lock = target.with_suffix(target.suffix + ".lock")
    async with _exclusive_file_lock(lock):
        if target.is_file():
            size = target.stat().st_size
            if size > 0 and (shard.size <= 0 or size == shard.size):
                return target
            if shard.size > 0 and 0 < size < shard.size:
                os.replace(target, partial)
            else:
                target.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                offset = partial.stat().st_size if partial.is_file() else 0
                if shard.size > 0 and offset > shard.size:
                    partial.unlink(missing_ok=True)
                    offset = 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                async with client.stream("GET", shard.url, headers=headers) as response:
                    # Some object stores answer a completed Range request with
                    # 416. If the retained bytes equal the manifest size, the
                    # file is already complete and only needs committing.
                    if (
                        response.status_code == 416
                        and shard.size > 0
                        and offset == shard.size
                    ):
                        os.replace(partial, target)
                        return target
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    mode = "ab" if append else "wb"
                    with partial.open(mode) as handle:
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                handle.write(chunk)

                downloaded = partial.stat().st_size
                if downloaded <= 0:
                    raise _IncompleteDownload("downloaded shard is empty")
                if shard.size > 0 and downloaded != shard.size:
                    if downloaded > shard.size:
                        partial.unlink(missing_ok=True)
                    raise _IncompleteDownload(
                        f"expected {shard.size} bytes, received {downloaded}"
                    )
                os.replace(partial, target)
                return target
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if (
                    exc.response.status_code not in _RETRYABLE_STATUS
                    or attempt >= len(_RETRY_DELAYS)
                ):
                    break
                delay = _retry_delay(exc.response, _RETRY_DELAYS[attempt])
            except (httpx.TransportError, _IncompleteDownload) as exc:
                last_error = exc
                if attempt >= len(_RETRY_DELAYS):
                    break
                delay = _RETRY_DELAYS[attempt]
            except OSError as exc:
                raise _ParquetUnavailable(
                    f"cannot cache Parquet shard {shard.filename!r}: {exc}"
                ) from exc
            log.warning(
                "Hugging Face Parquet shard download failed; retrying in %.1fs "
                "(%d/%d): %s",
                delay,
                attempt + 1,
                len(_RETRY_DELAYS),
                last_error,
            )
            await asyncio.sleep(delay)

    raise _ParquetUnavailable(
        f"cannot download Parquet shard {shard.filename!r}: {last_error}"
    )


def _write_parquet_csv(
    *, shard_paths: list[Path], out_dir: str, limit: int, hub_id: str,
) -> tuple[str, list[str], int]:
    """Stream downloaded Parquet shards into Zevo's existing CSV contract."""
    import csv as _csv

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise _ParquetUnavailable("pyarrow is unavailable") from exc

    readers = []
    columns: list[str] = []
    total_rows = 0
    try:
        for path in shard_paths:
            reader = pq.ParquetFile(path)
            readers.append(reader)
            total_rows += int(reader.metadata.num_rows)
            for name in reader.schema_arrow.names:
                if name not in columns:
                    columns.append(name)
    except Exception as exc:
        raise _ParquetUnavailable(f"cannot read converted Parquet: {exc}") from exc

    if total_rows <= 0 or not columns:
        raise _ParquetUnavailable("converted Parquet contains no rows")
    if limit <= 0 and total_rows > _MAX_FETCH:
        raise MaterializeError(
            f"converted split has {total_rows:,} rows, which is too many to pull "
            "into a validation set at run creation. Point at a smaller split, "
            "or upload the rows you want as a file."
        )

    requested = min(total_rows, limit) if limit > 0 else total_rows
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation.csv"
    temporary = out / f"validation.{uuid4().hex}.tmp"
    written = 0
    from zevo.code_benchmarks import (
        code_execution_adapter_for,
        externalize_code_answers,
        store_code_answer_buffer,
    )
    adapter = code_execution_adapter_for(hub_id)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(
                handle, fieldnames=columns, extrasaction="ignore",
            )
            writer.writeheader()
            for reader in readers:
                for batch in reader.iter_batches(
                    batch_size=1
                    if adapter in {"livecodebench", "code_contests"} else 1024,
                ):
                    # Avoid StringScalar.as_py() for LiveCodeBench's enormous
                    # private payload. Its Arrow UTF-8 buffer can be hashed and
                    # compressed directly, without a second full-size string.
                    if adapter == "livecodebench" and "private_test_cases" in batch.schema.names:
                        rows = []
                        for row_index in range(batch.num_rows):
                            row = {}
                            for column_index, name in enumerate(batch.schema.names):
                                scalar = batch.column(column_index)[row_index]
                                if name == "private_test_cases" and scalar.is_valid:
                                    buffer = scalar.as_buffer()
                                    row[name] = (
                                        store_code_answer_buffer(buffer)
                                        if buffer.size else ""
                                    )
                                else:
                                    row[name] = scalar.as_py()
                            rows.append(row)
                    else:
                        rows = batch.to_pylist()
                    for row in rows:
                        row = externalize_code_answers(adapter, row)
                        writer.writerow({c: _cell(row.get(c)) for c in columns})
                        written += 1
                        if written >= requested:
                            break
                    if written >= requested:
                        break
                if written >= requested:
                    break
        os.replace(temporary, path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, MaterializeError):
            raise
        raise _ParquetUnavailable(f"cannot convert Parquet to CSV: {exc}") from exc
    return str(path), columns, written


async def _materialize_parquet(
    *, client, hub_id: str, config: str, split: str, out_dir: str,
    cache_entry: Path, limit: int,
) -> tuple[str, list[str], int]:
    shards = await _parquet_shards(client, hub_id, config, split)
    persistent = _cache_ttl_seconds() > 0
    shard_root = (
        cache_entry / "parquet"
        if persistent
        else Path(out_dir) / ".parquet-shards"
    )
    paths: list[Path] = []
    for index, shard in enumerate(shards):
        identity = hashlib.sha256(
            f"{shard.url}\0{shard.size}".encode()
        ).hexdigest()[:20]
        target = shard_root / f"{index:05d}-{identity}.parquet"
        paths.append(await _download_parquet_shard(client, shard, target))
    completed = False
    try:
        # Conversion is CPU-heavy and LiveCodeBench contains exceptionally
        # large scalar values. Keep it off the API event loop so setup
        # heartbeats and progress polling continue while rows are streamed.
        result = await asyncio.to_thread(
            _write_parquet_csv,
            shard_paths=paths,
            out_dir=out_dir,
            limit=limit,
            hub_id=hub_id,
        )
        completed = True
        return result
    finally:
        from zevo.code_benchmarks import code_execution_adapter_for

        if not persistent or (
            completed and code_execution_adapter_for(hub_id)
            in {"livecodebench", "code_contests"}
        ):
            # Large private tests have been moved to compressed, deduplicated
            # answer sidecars. Retaining the converted Parquet would duplicate
            # several gigabytes without helping resume after completion.
            shutil.rmtree(shard_root, ignore_errors=True)


async def _materialize_rows(
    *, client, hub_id: str, config: str, split: str, out_dir: str,
    limit: int,
) -> tuple[str, list[str], int]:
    """Compatibility path for datasets without a complete Parquet export."""
    import csv as _csv
    import httpx
    import ijson

    from zevo.code_benchmarks import (
        code_execution_adapter_for,
        externalize_code_answers,
    )

    adapter = code_execution_adapter_for(hub_id)
    ceiling = limit if limit > 0 else _MAX_FETCH
    offset = 0
    count = 0
    columns: list[str] = []
    seen_columns: set[str] = set()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation.csv"
    spool = out / f".rows.{uuid4().hex}.tmp"
    temporary = out / f"validation.{uuid4().hex}.tmp"
    try:
        # Column names can appear after the first page. Spool already-
        # projected rows so the CSV header includes every field without
        # retaining an entire dataset (or even one HTTP page) in memory.
        with spool.open("w", encoding="utf-8") as handle:
            while count < ceiling:
                n = min(_PAGE, ceiling - count)
                params = {
                    "dataset": hub_id, "config": config, "split": split,
                    "offset": offset, "length": n,
                }
                for attempt in range(len(_RETRY_DELAYS) + 1):
                    start = handle.tell()
                    page_count = 0
                    page_columns: list[str] = []
                    page_seen: set[str] = set()
                    try:
                        async with client.stream(
                            "GET", f"{_SERVER}/rows", params=params,
                        ) as response:
                            response.raise_for_status()
                            reader = _AsyncChunkReader(response.aiter_bytes(
                                chunk_size=_JSON_STREAM_CHUNK,
                            ))
                            async for item in ijson.items_async(
                                reader, "rows.item", use_float=True,
                            ):
                                row = (item or {}).get("row") or {}
                                if not isinstance(row, dict):
                                    raise ValueError("Dataset Viewer row is not an object")
                                for key in row:
                                    if key not in seen_columns and key not in page_seen:
                                        page_seen.add(key)
                                        page_columns.append(key)
                                projected = externalize_code_answers(adapter, row)
                                json.dump(
                                    projected, handle, ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                handle.write("\n")
                                page_count += 1
                                if page_count >= n:
                                    break
                        break
                    except httpx.HTTPStatusError as exc:
                        handle.seek(start)
                        handle.truncate()
                        if (
                            exc.response.status_code not in _RETRYABLE_STATUS
                            or attempt >= len(_RETRY_DELAYS)
                        ):
                            raise MaterializeError(
                                f"cannot read rows of {hub_id!r} ({config}/{split}): {exc}"
                            ) from exc
                        delay = _retry_delay(exc.response, _RETRY_DELAYS[attempt])
                    except (httpx.TransportError, ijson.IncompleteJSONError) as exc:
                        handle.seek(start)
                        handle.truncate()
                        if attempt >= len(_RETRY_DELAYS):
                            raise MaterializeError(
                                f"cannot read rows of {hub_id!r} ({config}/{split}): {exc}"
                            ) from exc
                        delay = _RETRY_DELAYS[attempt]
                    except Exception as exc:
                        handle.seek(start)
                        handle.truncate()
                        raise MaterializeError(
                            f"cannot read rows of {hub_id!r} ({config}/{split}): {exc}"
                        ) from exc
                    log.warning(
                        "Hugging Face rows stream failed; retrying in %.1fs "
                        "(%d/%d)", delay, attempt + 1, len(_RETRY_DELAYS),
                    )
                    await asyncio.sleep(delay)
                if not page_count:
                    break
                for key in page_columns:
                    seen_columns.add(key)
                    columns.append(key)
                count += page_count
                offset += page_count

        if limit <= 0 and count >= _MAX_FETCH:
            raise MaterializeError(
                f"{hub_id!r} ({config}/{split}) has at least {_MAX_FETCH:,} rows, "
                "which is too many to pull into a validation set at run creation. "
                "Point at a smaller split, or upload the rows you want as a file."
            )
        if not count:
            raise MaterializeError(f"{hub_id!r} ({config}/{split}) returned no rows")

        with spool.open("r", encoding="utf-8") as source, temporary.open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            writer = _csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for line in source:
                row = json.loads(line)
                writer.writerow({column: _cell(row.get(column)) for column in columns})
        os.replace(temporary, path)
        return str(path), columns, count
    finally:
        spool.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


async def materialize(
    *, hub_id: str, split: str = "", config: str = "", out_dir: str,
    limit: int = 0,
) -> tuple[str, list[str], int, str]:
    """Fetch a hub split to a local CSV. Returns (path, columns, n_rows, note).

    A validation set has to exist as a file before the run starts: the split is
    settled at run creation, before any agent — and therefore before anything
    has downloaded anything. Rather than refuse hub ids outright, the backend
    pulls the rows itself.

    `limit=0` means the whole split. Naming a split is naming a set of rows, and
    quietly keeping the first N of them would be measuring something the caller
    did not choose: a split ordered by difficulty, category or source has a
    prefix that is not a sample of it.
    """
    import httpx

    ident = (hub_id or "").strip()
    if not ident:
        raise MaterializeError("no hub id given")
    requested_split = split.strip()
    requested_config = config.strip()
    token = _hf_token()
    cache_entry, cache_identity = _cache_identity(
        hub_id=ident,
        split=requested_split,
        config=requested_config,
        limit=limit,
        token=token,
    )
    cached = _cached_materialization(
        entry=cache_entry,
        identity=cache_identity,
        out_dir=out_dir,
    )
    if cached is not None:
        return cached

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers=_hf_headers(token),
    ) as client:
        cfg, spl = await _resolve_split(client, ident, requested_split, requested_config)
        try:
            path, columns, n_rows = await _materialize_parquet(
                client=client,
                hub_id=ident,
                config=cfg,
                split=spl,
                out_dir=out_dir,
                cache_entry=cache_entry,
                limit=limit,
            )
            route = "Parquet"
        except _ParquetUnavailable as exc:
            log.warning(
                "Hugging Face Parquet unavailable for %s (%s/%s); falling "
                "back to rows API: %s",
                ident,
                cfg,
                spl,
                exc,
            )
            path, columns, n_rows = await _materialize_rows(
                client=client,
                hub_id=ident,
                config=cfg,
                split=spl,
                out_dir=out_dir,
                limit=limit,
            )
            route = "rows API"

    _store_materialization(
        entry=cache_entry,
        identity=cache_identity,
        source=Path(path),
        columns=columns,
        n_rows=n_rows,
        config=cfg,
        split=spl,
    )

    note = f"fetched all {n_rows} rows of {ident} ({cfg}/{spl}) via {route}"
    if 0 < limit <= n_rows:
        # Said out loud: the run is validating on a prefix, not on the split.
        note = (
            f"fetched the first {n_rows} rows of {ident} ({cfg}/{spl}) via "
            f"{route}, capped at {limit}"
        )
    return str(path), columns, n_rows, note


def lookup(hub_id: str, *, role: str = "") -> RemoteSpec | None:
    """The catalogue's declaration for this hub id, or None if it names none.

    `role` narrows the match when one repo is listed several times — pulling a
    repo's train and validation splits is two entries under one id, and the
    caller usually knows which one it is asking about.
    """
    ident = (hub_id or "").strip()
    if not ident or not _DATASETS_DIR.is_dir():
        return None
    wanted = (role or "").strip().lower()
    fallback: RemoteSpec | None = None
    for d in sorted(_DATASETS_DIR.iterdir()):
        source = d / "source.json"
        if not source.is_file():
            continue
        try:
            entries = json.loads(source.read_text()).get("remote") or []
        except (json.JSONDecodeError, OSError, AttributeError):
            # A hand-edited source.json should not take a run down; the run
            # simply proceeds without the declaration, as it did before.
            continue
        for e in entries:
            if not isinstance(e, dict) or str(e.get("id") or "") != ident:
                continue
            try:
                n_rows = max(0, int(e.get("n_rows") or 0))
            except (TypeError, ValueError):
                n_rows = 0
            spec = RemoteSpec(
                id=ident,
                split=str(e.get("split") or ""),
                config=str(e.get("config") or ""),
                role=str(e.get("role") or ""),
                dataset=d.name,
                n_rows=n_rows,
            )
            # Matched on what the entry IS, which the split states directly.
            # An older source.json may carry a `role` that disagrees with its
            # split; the split is the one the loader acts on, so it wins.
            if wanted and _role_of(spec.split or spec.role) == wanted:
                return spec
            if fallback is None:
                fallback = spec
    return fallback
