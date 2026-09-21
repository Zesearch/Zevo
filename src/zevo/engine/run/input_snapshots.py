"""Run-local input copies and live-catalogue change detection.

Tasks and Files are mutable catalogue objects.  A Run therefore copies the
local bytes it will consume and records compact content identities for the
catalogue objects those bytes came from.  Editing or deleting a catalogue
object changes future Runs without rewriting an existing Run.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from zevo.contracts.orchestrator import UserRequest, effective_test_suite
from zevo.engine.method.evaluation_identity import digest_json
from zevo.holdout_storage import resolve_asset
from zevo.paths import files_root, holdout_root, work_dir_root


_IGNORED_CATALOGUE_FILES = {".profile.json"}


def _without_derived_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_derived_fields(child)
            for key, child in value.items()
            if key not in {"source_rows"}
        }
    if isinstance(value, list):
        return [_without_derived_fields(child) for child in value]
    return value


def task_definition_snapshot(
    task: Any, *, task_objective: str | None = None,
) -> dict[str, Any]:
    """Return the stable live-Task identity used by Run warnings."""
    suite = list(task.test_sets or [])
    if not suite:
        suite = [{
            "name": task.name,
            "test_set": task.test_set or "",
            "inference_query": task.task_objective or "Answer the input.",
            "sample_submission": task.test_sample_submission or "",
            "metric_type": task.metric_type,
            "metric": task.metric,
            "answer_fields": list(task.test_answer_fields or []),
            "metric_direction": task.metric_direction,
            "evaluation_script": task.evaluation_script or "",
            "evaluator_sha256": task.evaluator_sha256 or "",
        }]
    contract = {
        "task_objective": str(
            task.task_objective if task_objective is None else task_objective
        ).strip(),
        "test_sets": _without_derived_fields(suite),
    }
    return {"name": str(task.name), "sha256": digest_json(contract)}


def _managed_files_member(value: str) -> Path | None:
    """Map host/container/repo spellings to a Files-catalogue relative path."""
    raw = Path(str(value or "").strip())
    if not str(raw):
        return None
    configured = Path(files_root())
    for root in (configured, Path("/app/data/files")):
        try:
            relative = raw.relative_to(root)
            return relative if relative.parts else None
        except ValueError:
            pass
    if not raw.is_absolute() and len(raw.parts) >= 3 and raw.parts[:2] == ("data", "files"):
        return Path(*raw.parts[2:])
    return None


def _catalogue_dirs(name: str) -> tuple[Path, Path]:
    return Path(files_root()) / name, Path(holdout_root()) / "files" / name


def _catalogue_entries(name: str) -> tuple[bool, tuple[tuple[str, str, int, int], ...]]:
    public, private = _catalogue_dirs(name)
    exists = public.is_dir() or private.is_dir()
    rows: list[tuple[str, str, int, int]] = []
    for namespace, root in (("public", public), ("private", private)):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in _IGNORED_CATALOGUE_FILES:
                continue
            stat = path.stat()
            rows.append((namespace, path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    return exists, tuple(rows)


@lru_cache(maxsize=512)
def _digest_catalogue_entries(
    name: str, entries: tuple[tuple[str, str, int, int], ...],
) -> str:
    public, private = _catalogue_dirs(name)
    hashed = []
    for namespace, relative, _size, _mtime_ns in entries:
        root = private if namespace == "private" else public
        with (root / relative).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        hashed.append({"storage": namespace, "path": relative, "sha256": digest})
    return digest_json(hashed)


def file_set_fingerprint(name: str) -> str | None:
    exists, entries = _catalogue_entries(name)
    if not exists:
        return None
    return _digest_catalogue_entries(name, entries)


def _remote_catalogue_names(identifier: str) -> set[str]:
    root = Path(files_root())
    if not identifier or not root.is_dir():
        return set()
    names: set[str] = set()
    for source in root.glob("*/source.json"):
        try:
            payload = json.loads(source.read_text())
        except (OSError, TypeError, ValueError):
            continue
        if any(
            isinstance(item, dict) and str(item.get("id") or "") == identifier
            for item in (payload.get("remote") or [])
        ):
            names.add(source.parent.name)
    return names


def _catalogue_names(values: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for value in values:
        value = str(value or "").strip()
        if not value:
            continue
        relative = _managed_files_member(value)
        if relative is not None and relative.parts:
            names.add(relative.parts[0])
        else:
            names.update(_remote_catalogue_names(value))
    return names


def catalogue_names(values: Iterable[str]) -> list[str]:
    """Public, deterministic view of Files objects named by input paths."""
    return sorted(_catalogue_names(values))


def request_file_snapshots(request: Any) -> list[dict[str, str]]:
    """Identity every Files object explicitly used by a launch request."""
    values = [str(getattr(request, "dataset", "") or "")]
    for item in effective_test_suite(request):
        values.extend((item.test_set, item.sample_submission, item.evaluation_script))
    for item in list(getattr(request, "validation_sets", []) or []):
        values.extend((item.test_set, item.sample_submission, item.evaluation_script))
    names = sorted(_catalogue_names(values))
    return [
        {"name": name, "sha256": digest}
        for name in names
        if (digest := file_set_fingerprint(name)) is not None
    ]


def launch_input_snapshot(task: Any | None, request: Any) -> dict[str, Any]:
    suite = effective_test_suite(request)
    return {
        "version": 1,
        "local_inputs_copied": True,
        "task": task_definition_snapshot(task) if task is not None else None,
        "files": request_file_snapshots(request),
        "sources": {
            "dataset": str(getattr(request, "dataset", "") or ""),
            "test_sets": [item.model_dump(mode="json") for item in suite],
            "validation_sets": [
                item.model_dump(mode="json")
                for item in list(getattr(request, "validation_sets", []) or [])
            ],
        },
        "evaluation_sha256": "",
    }


def launch_auto_input_snapshot(request: Any) -> dict[str, Any]:
    """Auto starts before a Task/Test contract exists."""
    names = sorted(_catalogue_names([str(getattr(request, "dataset", "") or "")]))
    return {
        "version": 1,
        "local_inputs_copied": True,
        "task": None,
        "files": [
            {"name": name, "sha256": digest}
            for name in names
            if (digest := file_set_fingerprint(name)) is not None
        ],
        "sources": {"dataset": str(getattr(request, "dataset", "") or "")},
        "evaluation_sha256": "",
    }


def _existing_local_path(value: str, *, private: bool) -> Path | None:
    value = str(value or "").strip()
    if not value:
        return None
    # A container-absolute Files path may be read by a host-side local API.
    relative = _managed_files_member(value)
    if relative is not None:
        candidate = Path(resolve_asset(value)) if private else Path(files_root()) / relative
        if candidate.exists():
            return candidate.resolve()
    from zevo.engine.remote_datasets import looks_like_hub_id

    if value.startswith(("http://", "https://")) or looks_like_hub_id(value):
        return None
    path = Path(resolve_asset(value) if private else value)
    if path.exists():
        return path.resolve()
    return None


def _snapshot_destination(run_id: str, value: str, source: Path, *, private: bool) -> Path:
    base = (
        Path(holdout_root()) / "runs" / run_id / "_inputs"
        if private else Path(work_dir_root()) / run_id / "_inputs"
    )
    relative = _managed_files_member(value)
    if relative is not None:
        return base / "files" / relative
    digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    return base / "external" / digest / source.name


def snapshot_local_input(run_id: str, value: str, *, private: bool) -> str:
    """Copy one local input and return its Run-owned absolute path.

    Hub ids, URLs and other non-local references are returned unchanged; their
    materializers already write into the Run's own directory.
    """
    source = _existing_local_path(value, private=private)
    if source is None:
        return str(value or "")
    destination = _snapshot_destination(run_id, value, source, private=private)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        temporary = destination.with_name(f".{destination.name}.snapshot")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    return str(destination.resolve())


def snapshot_user_request(run_id: str, request: UserRequest) -> UserRequest:
    """Rewrite local scoring/training paths to immutable Run-owned copies."""
    suite = [
        item.model_copy(update={
            "test_set": snapshot_local_input(run_id, item.test_set, private=True),
            "sample_submission": snapshot_local_input(
                run_id, item.sample_submission, private=True,
            ),
        })
        for item in effective_test_suite(request)
    ]
    validation = [
        item.model_copy(update={
            "test_set": snapshot_local_input(run_id, item.test_set, private=False),
            "sample_submission": snapshot_local_input(
                run_id, item.sample_submission, private=False,
            ),
        })
        for item in list(request.validation_sets or [])
    ]
    primary = suite[0]
    return request.model_copy(update={
        "dataset": snapshot_local_input(run_id, request.dataset, private=False),
        "test_sets": suite,
        "test_set": primary.test_set,
        "test_sample_submission": primary.sample_submission,
        "validation_sets": validation,
    })


def snapshot_auto_request(run_id: str, request: Any) -> Any:
    """Auto has no scoring contract yet, but may already own Training data."""
    return request.model_copy(update={
        "dataset": snapshot_local_input(run_id, request.dataset, private=False),
    })


def record_evaluation_identity(snapshot: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    out = dict(snapshot or {})
    contract = holdout.get("evaluation_contract")
    out["evaluation_sha256"] = digest_json(contract) if isinstance(contract, dict) else ""
    return out


def detect_input_changes(run: Any, task: Any | None) -> list[dict[str, str]]:
    """Compare one Run's launch identities with today's live catalogue."""
    snapshot = dict(getattr(run, "input_snapshot", None) or {})
    if snapshot.get("version") != 1:
        return []
    changes: list[dict[str, str]] = []
    task_snapshot = snapshot.get("task")
    if isinstance(task_snapshot, dict):
        if task is None:
            changes.append({
                "kind": "task", "name": str(task_snapshot.get("name") or run.task_name),
                "status": "deleted",
            })
        elif task_definition_snapshot(task).get("sha256") != task_snapshot.get("sha256"):
            changes.append({
                "kind": "task", "name": str(task_snapshot.get("name") or run.task_name),
                "status": "modified",
            })
    for item in snapshot.get("files") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        current = file_set_fingerprint(name)
        if current is None:
            changes.append({"kind": "file", "name": name, "status": "deleted"})
        elif current != item.get("sha256"):
            changes.append({"kind": "file", "name": name, "status": "modified"})
    return changes
