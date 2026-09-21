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
    return {"name": str(task.name), "sha256": digest_json(contract), "definition": contract}


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


def file_set_snapshot(name: str) -> dict[str, Any] | None:
    """Keep bounded previews and metadata, never an entire dataset in the DB."""
    parts = Path(name).parts
    if not parts or Path(name).is_absolute() or ".." in parts:
        return None
    # Old online snapshots sometimes fingerprinted the shared `_t` root.
    # That identity may still support a warning, never a cross-tenant preview.
    if parts[0] == "_t" and len(parts) < 3:
        fingerprint = file_set_fingerprint(name)
        return {"name": name, "sha256": fingerprint} if fingerprint else None
    exists, entries = _catalogue_entries(name)
    if not exists:
        return None
    return _file_set_snapshot(name, entries)


@lru_cache(maxsize=128)
def _file_set_snapshot(name: str, entries: tuple[tuple[str, str, int, int], ...]) -> dict[str, Any]:
    import csv

    public, private = _catalogue_dirs(name)
    manifest = []
    text_budget = 128 * 1024
    for namespace, relative, size, _ in entries:
        path = (private if namespace == "private" else public) / relative
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        item: dict[str, Any] = {"path": relative, "storage": namespace, "size": size, "sha256": digest}
        if path.suffix.lower() == ".csv":
            try:
                with path.open(encoding="utf-8-sig") as source:
                    header = source.readline(65536)
                if header.endswith("\n") or size < 65536:
                    item["columns"] = next(csv.reader([header]), [])
            except (UnicodeError, csv.Error):
                pass
        elif path.suffix.lower() in {".txt", ".md", ".py", ".json", ".yaml", ".yml"} and size <= min(32768, text_budget):
            try:
                item["text"] = path.read_text(encoding="utf-8")
                text_budget -= size
            except UnicodeError:
                pass
        manifest.append(item)
    fingerprint = digest_json([
        {key: item[key] for key in ("storage", "path", "sha256")} for item in manifest
    ])
    return {"name": name, "sha256": fingerprint, "entries": manifest}


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
            if ".." in relative.parts:
                continue
            if relative.parts[0] == "_t":
                if len(relative.parts) >= 3:
                    names.add(Path(*relative.parts[:3]).as_posix())
            else:
                names.add(relative.parts[0])
        else:
            names.update(_remote_catalogue_names(value))
    return names


def catalogue_names(values: Iterable[str]) -> list[str]:
    """Public, deterministic view of Files objects named by input paths."""
    return sorted(_catalogue_names(values))


def request_file_snapshots(request: Any) -> list[dict[str, Any]]:
    """Identity every Files object explicitly used by a launch request."""
    values = [str(getattr(request, "dataset", "") or "")]
    for item in effective_test_suite(request):
        values.extend((item.test_set, item.sample_submission, item.evaluation_script))
    for item in list(getattr(request, "validation_sets", []) or []):
        values.extend((item.test_set, item.sample_submission, item.evaluation_script))
    names = sorted(_catalogue_names(values))
    return [
        snapshot
        for name in names
        if (snapshot := file_set_snapshot(name)) is not None
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
            snapshot
            for name in names
            if (snapshot := file_set_snapshot(name)) is not None
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


def input_change_details(run: Any, task: Any | None) -> list[dict[str, Any]]:
    """Dashboard-only comparison; historical values must match their saved hash."""
    snapshot = dict(getattr(run, "input_snapshot", None) or {})
    groups = []

    def differences(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
        if before == after:
            return []
        if isinstance(before, dict) and isinstance(after, dict):
            return [row for key in sorted(before.keys() | after.keys())
                    for row in differences(before.get(key), after.get(key), f"{path} / {key}".strip(" /"))]
        if isinstance(before, list) and isinstance(after, list) and before and after and all(
            isinstance(x, dict) and x.get("name") for x in before + after
        ) and len({x["name"] for x in before}) == len(before) and len({x["name"] for x in after}) == len(after):
            return differences({x["name"]: x for x in before}, {x["name"]: x for x in after}, path)
        return [{"field": path, "before": before, "after": after}]

    for change in detect_input_changes(run, task):
        group: dict[str, Any] = {**change, "fields": [], "note": ""}
        if change["kind"] == "task":
            saved = snapshot.get("task") or {}
            before = saved.get("definition")
            if before is None:
                # Older Runs may retain the full original contract separately.
                for suite in [(snapshot.get("sources") or {}).get("test_sets"),
                              (getattr(run, "holdout", None) or {}).get("test_sets")]:
                    candidate = {"task_objective": str(getattr(run, "task_objective", "") or "").strip(),
                                 "test_sets": _without_derived_fields(suite)}
                    if suite and digest_json(candidate) == saved.get("sha256"):
                        before = candidate
                        break
            if before is not None and digest_json(before) == saved.get("sha256"):
                after = task_definition_snapshot(task)["definition"] if task is not None else None
                group["fields"] = differences(before, after) if after is not None else [
                    {"field": key, "before": value, "after": None} for key, value in before.items()
                ]
            else:
                group["note"] = "This older Run saved a fingerprint only. Its original Task definition is unavailable for comparison."
        else:
            saved = next(x for x in snapshot.get("files", []) if x.get("name") == change["name"])
            parts = Path(change["name"]).parts
            if "entries" not in saved or (parts and parts[0] == "_t" and len(parts) < 3):
                group["note"] = "This older Run has no saved file manifest. The File changed, but its original contents cannot be compared."
            else:
                current = file_set_snapshot(change["name"])
                before_files = {x["storage"] + "/" + x["path"]: x for x in saved["entries"]}
                after_files = {x["storage"] + "/" + x["path"]: x for x in (current or {}).get("entries", [])}
                for key in sorted(before_files.keys() | after_files.keys()):
                    old, new = before_files.get(key), after_files.get(key)
                    if old and new and old["sha256"] == new["sha256"]:
                        continue
                    def display(value: dict | None) -> dict:
                        return {k: v for k, v in (value or {}).items() if k not in {"path", "storage"}}
                    group["fields"].extend(differences(display(old), display(new), key))
                group["note"] = "Small text files include saved text. Datasets show size, columns when available, and content fingerprints; rows are not expanded."
        # Bound diff work for long user-authored instructions. Full values are
        # still available even when line highlighting is omitted.
        from difflib import SequenceMatcher
        for field in group["fields"]:
            before, after = field["before"], field["after"]
            if isinstance(before, str) and isinstance(after, str) and len(before) + len(after) <= 65536:
                left, right = before.splitlines(), after.splitlines()
                if len(left) + len(right) > 2000:
                    continue
                field["before_lines"], field["after_lines"] = [], []
                for tag, a, b, c, d in SequenceMatcher(None, left, right).get_opcodes():
                    field["before_lines"].extend(
                        {"text": left[i] + ("\n" if i < len(left) - 1 else ""), "changed": tag != "equal"}
                        for i in range(a, b)
                    )
                    field["after_lines"].extend(
                        {"text": right[i] + ("\n" if i < len(right) - 1 else ""), "changed": tag != "equal"}
                        for i in range(c, d)
                    )
        groups.append(group)
    return groups
