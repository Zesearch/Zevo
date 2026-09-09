"""The user-uploaded FILES catalogue: GET/POST /files, /files/{name}/preview.

One entry is a named set of files the user uploaded — typically a training
file, a validation file, a test file and a submission template. The UI calls
this "Files" in its nav, its page title and its buttons, so the endpoints and
DTOs here do too.

NOT the ML sense of "dataset" that runs through the agent contracts
(`UserRequest.dataset`, the `training_dataset` ArtifactBinding, `dataset.jsonl`). Those name the data a
run TRAINS on and are untouched by this module; the only place the two meet is
`DatasetProfile`, which the engine computes about a file's contents.

On disk these live under the configured files root, normally
`data/files/<name>/` on both host and container.

We catalog by directory name; the preview endpoint uses the format-aware
parsers in this module and returns only a bounded sample.
"""
from __future__ import annotations

import csv
import json
import re
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from zevo.paths import files_root, holdout_root

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from zevo.contracts._base import StrictBody
from zevo.api.ui_access import is_trusted_ui_request

from zevo.engine.dataset_profiler import (
    DatasetProfile,
    is_cache_valid,
    load_cached,
    profile_file,
    save_cached,
)


router = APIRouter()


REPO_ROOT = Path(__file__).resolve().parents[5]
DATASETS_DIR = Path(files_root())
PRIVATE_DATASETS_DIR = Path(holdout_root()) / "files"


def _pick_primary_data_file(d: Path) -> Path | None:
    """For profiling: the TRAINING file if the bundle has one, else the first
    data file by extension preference.

    Alphabetical order alone picked `sample_submission.csv` — an empty template
    — over `train.csv`, so every shipped bundle profiled as "no parseable rows".
    A dataset's profile is context about what a model would learn from, so a
    name starting with `train` wins.

    Being training data beats being a preferred FORMAT, and that order matters:
    suffix-first meant a bundle whose training file is `train/train.json` got
    profiled as whatever `.csv` happened to sit beside it, because `.json` is
    last in the preference list. The card then described a file the run does not
    read. Suffix preference only breaks ties among equally-training files.
    """
    if not d.exists():
        return None
    # Subfolders included: a bundle keeps its training file in train/, and the
    # profile is meant to describe what a model would learn from.
    order = {".jsonl": 0, ".csv": 1, ".json": 2}
    files = sorted(
        (f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in order),
        key=lambda f: (order[f.suffix.lower()], str(f)),
    )
    if not files:
        return None
    train = [f for f in files
             if f.stem.lower().startswith("train")
             or f.parent.name.lower().startswith("train")]
    if not train:
        return files[0]
    # Among training files, the one actually CALLED `train` is the training
    # file; `train_if.csv` beside it is a variant the run may or may not use.
    # Without this the tie fell to suffix order and a `.csv` variant beat
    # `train.json`.
    exact = [f for f in train if f.stem.lower() == "train"]
    return (exact or train)[0]


def _profile_one(dataset_dir: Path, name: str) -> DatasetProfile | None:
    """Run the profiler (using the on-disk cache when possible) for the
    primary data file in the dataset dir."""
    f = _pick_primary_data_file(dataset_dir)
    if f is None:
        return None
    cached = load_cached(dataset_dir)
    if cached is not None and is_cache_valid(dataset_dir, f, cached):
        return cached
    profile = profile_file(name, f)
    save_cached(dataset_dir, profile)
    return profile


class RemoteFile(BaseModel):
    """A file of this dataset that is NOT on disk — it is fetched at run time.

    It still belongs in the file list: "this dataset has no local train set" and
    "this dataset trains on trl-lib/Capybara from HuggingFace" are the same fact,
    and splitting them into a file list plus a provenance footnote made the
    dataset look like it was missing its training data.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["train", "validation", "test"]
    kind: Literal["huggingface", "url"]
    id: str        # the hub id or short name shown in the list
    url: str
    # Which slice of the hub repo to pull. A hub dataset is not one table: it
    # ships `train`, `validation` and `test`, and the acquire step used to
    # hardcode `train` — so a repo whose training data lives under `train_sft`,
    # or one you want the validation half of, could not be expressed at all.
    # HuggingFace slicing syntax works here too (`train[:2000]`, `train[:80%]`).
    # "" = let the data agent pick, which still means `train` in practice.
    split: str = ""
    # The named subset, for repos that ship several (`load_dataset("glue",
    # "mnli")`). "" = the repo's default config. Not the same axis as `split`:
    # one repo, many configs, each with its own splits.
    config: str = ""


class FileSetSource(BaseModel):
    """Where a dataset came from, and what it is for.

    Absent when no provenance was recorded.
    """

    model_config = ConfigDict(extra="forbid")

    note: str = ""
    # There is no training/evaluation `kind` on a dataset any more. A bundle is
    # a folder of files, and what each file IS gets decided where it is USED: a
    # task names its test set, a setting names its training and validation
    # data. Declaring it twice let the two disagree, and only one of them was
    # the one a run actually read.
    remote: list[RemoteFile] = Field(default_factory=list)


class FileSetDTO(BaseModel):
    name: str
    path: str
    size_bytes: int
    files: list[str]
    # When the dataset entered the catalogue. Datasets are directories, not
    # rows, so this is the directory's own timestamp rather than a column.
    created_at: str = ""
    source: FileSetSource | None = None


# Bookkeeping, not data: kept out of `files` so the list shows what the dataset
# actually contains.
_META_FILES = {"source.json", ".profile.json"}

# What an uploaded file can be declared as, and the filename it is stored under.
# The `_public` halves are gone from this list along with the files themselves:
# the questions-only copy is derived from the task's declared answer fields
# now, so a place to upload one by hand is a second source of truth for a file
# that has one.
_FILE_ROLE_NAMES = {
    "train": "train",
    "validation": "validation",
    "test": "test",
    "sample_submission": "sample_submission",
}


def _read_source(d: Path) -> FileSetSource | None:
    f = d / "source.json"
    if not f.is_file():
        return None
    try:
        return FileSetSource(**json.loads(f.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        # A hand-edited source.json should not take the whole page down.
        return None


def _dataset_files(d: Path) -> list[str]:
    """Every file in the dataset, named relative to its root.

    One level down is `train/train.json` rather than `train.json`, because the
    subfolder is part of which file this IS: a bundle that splits itself into
    train/ validation/ test/ has three files called the same thing otherwise.
    The UI groups on the prefix; anything sitting loose in the root has none.
    """
    root = d.resolve()
    out = []
    for f in d.rglob("*"):
        if not f.is_file() or f.name in _META_FILES:
            continue
        rel = f.relative_to(d)
        # Editor droppings and caches are not data.
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return sorted(out)


def _private_set_dir(d: Path) -> Path:
    """Private mirror of a public catalogue directory."""
    return PRIVATE_DATASETS_DIR / d.name


def _all_dataset_files(d: Path) -> list[str]:
    """Logical members across the public bundle and its private Test mirror."""
    return sorted(set(_dataset_files(d)) | set(_dataset_files(_private_set_dir(d))))


def _member(d: Path, rel: str) -> Path | None:
    """The file `rel` names inside `d`, or None if it names something else.

    `..` has to be caught here rather than by a character check: the file names
    now legitimately contain `/`, so the test is where the path LANDS, not what
    it looks like.
    """
    for candidate_root in (d, _private_set_dir(d)):
        root = candidate_root.resolve()
        try:
            target = (candidate_root / rel).resolve()
        except (OSError, ValueError):
            continue
        if target.is_relative_to(root) and target.is_file():
            return target
    return None


def _dto_for(d: Path) -> FileSetDTO:
    files = _all_dataset_files(d)
    return FileSetDTO(
        name=d.name, path=str(d),
        size_bytes=sum(
            target.stat().st_size
            for f in files
            if (target := _member(d, f)) is not None
        ),
        files=files,
        created_at=datetime.fromtimestamp(d.stat().st_ctime, tz=timezone.utc).isoformat(),
        source=_read_source(d),
    )


@router.get("/files", response_model=list[FileSetDTO])
async def list_files() -> list[FileSetDTO]:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    names = {
        p.name for root in (DATASETS_DIR, PRIVATE_DATASETS_DIR)
        for p in root.iterdir() if p.is_dir()
    }
    # A public catalogue directory carries source.json/profile metadata even
    # when every data member is private.  Ensure it exists for old private-only
    # bundles before building their DTO.
    out = []
    for name in sorted(names):
        d = _set_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        out.append(_dto_for(d))
    return out


# `.py` is here because a scoring set is not only data: a task ships the eval.py
# that grades it, and a bundle you cannot put the scorer into is a bundle whose
# files live in two places. It is stored and shown like any other file — the
# preview renders it as source, and nothing here runs it.
_ALLOWED_UPLOAD_SUFFIXES = {
    ".csv", ".jsonl", ".json", ".pdf", ".txt", ".md", ".parquet", ".py",
}


# A subfolder name, not a path: one segment of letters, digits, dash and
# underscore. Anything else — a slash, a dot, an empty string — is refused
# rather than sanitized, because a caller who meant `train` and got
# `traindata` silently would find out much later.
_FOLDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")

# A file-set name is exactly one directory segment. The same rule guards
# every `{name}` route: without it, `DELETE /files/..` (path-as-is) resolves
# to the data root itself and `rmtree` follows.
_SET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,63}$")


def _set_dir(name: str) -> Path:
    """Resolve a file-set name to its directory, refusing path escapes.

    Central chokepoint for every endpoint that takes `{name}`: the name must
    be a single well-formed segment AND the resolved directory must stay
    inside the files root.
    """
    if not _SET_NAME_RE.match(name or "") or ".." in name:
        raise HTTPException(400, f"invalid file set name {name!r}")
    d = (DATASETS_DIR / name).resolve()
    if d.parent != DATASETS_DIR.resolve():
        raise HTTPException(400, f"invalid file set name {name!r}")
    return d


@router.post("/files", response_model=FileSetDTO)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(""),
    role: str = Form(""),
    folder: str = Form(""),
) -> FileSetDTO:
    """Upload a single dataset file. The dataset name is the directory
    name (falls back to the filename stem).

    `role` says what this one FILE is, and renames it to the convention the
    rest of the system reads: `validation` lands as `validation.csv`. That
    naming is not cosmetic: `zevo.engine.method.validation_split.resolve` adopts a
    `validation.csv` sitting beside the test set, so a correctly-shaped
    validation set uploaded as `my_held_out_rows.csv` is invisible to it.
    """
    if not file.filename:
        raise HTTPException(400, "filename required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            400,
            f"unsupported file type {suffix!r}; expected one of "
            f"{sorted(_ALLOWED_UPLOAD_SUFFIXES)}",
        )
    ds_name = (name or Path(file.filename).stem).strip().replace(" ", "_")
    if not ds_name:
        raise HTTPException(400, "file set name resolved to empty string")
    dataset_dir = _set_dir(ds_name)
    # `folder` puts the file in a subfolder of the dataset: the bundle that
    # splits itself into train/ validation/ test/ is assembled one upload at a
    # time, and without this the only way to get that layout was on disk.
    sub = (folder or "").strip().strip("/")
    if sub and not _FOLDER_RE.match(sub):
        raise HTTPException(
            400,
            f"folder {folder!r} is not a name: use letters, digits, dash or "
            "underscore, one level only",
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    wanted_role = (role or "").strip().lower()
    private_upload = (
        sub.lower() == "test"
        or wanted_role in {"test", "sample_submission"}
    )
    storage_dir = _private_set_dir(dataset_dir) if private_upload else dataset_dir
    target_dir = storage_dir / sub if sub else storage_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    # basename only: the client controls `filename`, and a path in it must
    # not steer the write outside target_dir.
    stored_name = os.path.basename(file.filename)
    if not stored_name or stored_name.startswith("."):
        raise HTTPException(400, f"invalid filename {file.filename!r}")
    if wanted_role:
        if wanted_role not in _FILE_ROLE_NAMES:
            raise HTTPException(
                400, f"role must be one of {', '.join(_FILE_ROLE_NAMES)}",
            )
        stored_name = f"{_FILE_ROLE_NAMES[wanted_role]}{suffix}"
    target_path = target_dir / stored_name
    # Same ceiling the attachments endpoint enforces; without one, a single
    # request can fill the disk.
    _MAX_UPLOAD = 200 * 1024 * 1024
    written = 0
    with target_path.open("wb") as fh:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > _MAX_UPLOAD:
                fh.close()
                target_path.unlink(missing_ok=True)
                raise HTTPException(413, "file exceeds the 200MB upload limit")
            fh.write(chunk)
    size = target_path.stat().st_size
    # F.2 — fire-and-forget profiler so the dataset has structured
    # context ready before any agent inspects it. Errors are swallowed
    # (logged inside the profiler); the dataset is usable either way.
    background_tasks.add_task(_profile_in_background, dataset_dir, ds_name)
    return _dto_for(dataset_dir)


class NoteBody(StrictBody):
    """What the dataset card says about itself: the one-line description."""

    note: str = ""


@router.put("/files/{name}/note", response_model=FileSetDTO)
async def set_dataset_note(name: str, body: NoteBody) -> FileSetDTO:
    """Set what a dataset IS. Kept in the dataset's own source.json so it
    travels with the files rather than living in a table beside them."""
    d = _set_dir(name)
    if not d.is_dir():
        raise HTTPException(404, f"file set {name!r} not found")
    src = _read_source(d) or FileSetSource()
    src.note = body.note.strip()
    (d / "source.json").write_text(json.dumps(src.model_dump(), indent=2) + "\n")
    return _dto_for(d)


class RemoteBody(StrictBody):
    """A file this dataset names but does not store."""

    id: str                       # hub id, or a URL
    kind: Literal["huggingface", "url"] = "huggingface"
    split: str = ""               # which slice of the repo. "" = train
    config: str = ""              # named subset, for repos that ship several
    role: Literal["", "train", "validation", "test"] = ""


def _role_of(split: str) -> str:
    """What a remote entry IS, read off the split it names.

    Asked separately once, and it was the same question twice: an entry whose
    split is `validation` is the validation set, and a form that let you say
    `train` beside a `validation` split only made it possible to disagree with
    yourself. `role` stays on the record because source.json files already
    carry it and it reads better than re-deriving downstream.
    """
    s = (split or "").strip().lower()
    if s.startswith(("validation", "valid", "dev")):
        return "validation"
    if s.startswith("test"):
        return "test"
    return "train"


@router.post("/files/{name}/remote", response_model=FileSetDTO)
async def add_dataset_remote(name: str, body: RemoteBody) -> FileSetDTO:
    """Add a HuggingFace id (or URL) as one of the dataset's files.

    A dataset can be partly on disk and partly fetched — ifeval ships its test
    side and pulls its training set from the hub — so the two live in one list
    rather than in a file list plus a footnote. A file set may also store
    NOTHING locally (hub-only), so a missing directory is created here the
    same way an upload would create it, not treated as an error.
    """
    d = _set_dir(name.strip().replace(" ", "_"))
    d.mkdir(parents=True, exist_ok=True)
    ident = body.id.strip()
    if not ident:
        raise HTTPException(400, "id is required")
    is_url = ident.startswith(("http://", "https://"))
    if body.kind == "url" and not is_url:
        raise HTTPException(400, "kind='url' requires an http:// or https:// id")
    if body.kind == "huggingface" and is_url:
        raise HTTPException(400, "kind='huggingface' requires a Hub dataset id, not a URL")
    split = body.split.strip()
    role = (body.role or "").strip().lower() or _role_of(split)
    src = _read_source(d) or FileSetSource()
    # Keyed by (id, split): the same repo listed twice is how a dataset says
    # "train on this repo's train split and validate on its validation split",
    # which is the ordinary case and used to collide as a duplicate.
    if any(r.id == ident and (r.split or "") == split for r in src.remote):
        raise HTTPException(
            409,
            f"{ident!r}{f' ({split})' if split else ''} is already listed",
        )
    url = (ident if ident.startswith("http")
           else f"https://huggingface.co/datasets/{ident}")
    src.remote.append(RemoteFile(
        role=role, kind=body.kind, id=ident, url=url,
        split=split, config=body.config.strip(),
    ))
    (d / "source.json").write_text(json.dumps(src.model_dump(), indent=2) + "\n")
    return _dto_for(d)


class RemotePatchBody(StrictBody):
    """Fields to change on a listed remote. Omitted = left alone."""

    role: Literal["train", "validation", "test"] | None = None
    split: str | None = None
    config: str | None = None


@router.patch("/files/{name}/remote/{ident:path}", response_model=FileSetDTO)
async def patch_dataset_remote(
    name: str, ident: str, body: RemotePatchBody, split: str = "",
) -> FileSetDTO:
    """Fix a remote's role/split/config in place.

    Without this, correcting a mistyped split means deleting the entry and
    re-adding it — and on a dataset that lists the same repo twice, deleting by
    id alone hits whichever one comes first. `?split=` names which entry when
    the id is ambiguous.
    """
    d = _set_dir(name)
    if not d.is_dir():
        raise HTTPException(404, f"file set {name!r} not found")
    src = _read_source(d) or FileSetSource()
    matches = [r for r in src.remote if r.id == ident
               and (not split or (r.split or "") == split)]
    if not matches:
        raise HTTPException(404, f"{ident!r} is not listed on {name!r}")
    if len(matches) > 1:
        raise HTTPException(
            409, f"{ident!r} is listed more than once — pass ?split= to say which",
        )
    entry = matches[0]
    if body.role is not None:
        entry.role = body.role.strip().lower() or _role_of(entry.split)
    if body.split is not None:
        entry.split = body.split.strip()
        if body.role is None:
            entry.role = _role_of(entry.split)
    if body.config is not None:
        entry.config = body.config.strip()
    (d / "source.json").write_text(json.dumps(src.model_dump(), indent=2) + "\n")
    return _dto_for(d)


@router.delete("/files/{name}/remote/{ident:path}", response_model=FileSetDTO)
async def delete_dataset_remote(name: str, ident: str, split: str = "") -> FileSetDTO:
    """Drop a remote entry. Nothing is deleted anywhere — it was never stored.

    `?split=` picks which entry when the dataset lists the same repo more than
    once — pulling a repo's train and validation splits is two entries with one
    id, and removing "the first match" would be a coin toss.
    """
    d = _set_dir(name)
    if not d.is_dir():
        raise HTTPException(404, f"file set {name!r} not found")
    src = _read_source(d) or FileSetSource()
    kept = [r for r in src.remote
            if not (r.id == ident and (not split or (r.split or "") == split))]
    if len(kept) == len(src.remote):
        raise HTTPException(404, f"{ident!r} is not listed on {name!r}")
    src.remote = kept
    (d / "source.json").write_text(json.dumps(src.model_dump(), indent=2) + "\n")
    return _dto_for(d)


@router.delete("/files/{name}/contents/{filename:path}", response_model=FileSetDTO)
async def delete_file_set_member(name: str, filename: str) -> FileSetDTO:
    """Remove one file from a set. The set itself stays, even empty — deleting
    the last file is not the same request as deleting the set.

    `contents`, not a second `files`: this collection is already called `files`,
    so `/files/{name}/files/{filename}` reads as though the two mean the same
    thing when one is the named set and the other is a member of it."""
    d = _set_dir(name)
    if not d.is_dir():
        raise HTTPException(404, f"file set {name!r} not found")
    target = _member(d, filename)
    if target is None:
        raise HTTPException(404, f"{filename!r} is not a file of file set {name!r}")
    if target.name in _META_FILES:
        raise HTTPException(400, f"{filename!r} is bookkeeping, not data")
    target.unlink()
    # An emptied subfolder is not a file the dataset has; leaving it behind
    # makes the next listing show a folder with nothing in it.
    member_root = (
        _private_set_dir(d).resolve()
        if target.is_relative_to(_private_set_dir(d).resolve())
        else d.resolve()
    )
    for parent in target.parents:
        if parent == member_root or not parent.is_relative_to(member_root):
            break
        if any(parent.iterdir()):
            break
        parent.rmdir()
    return _dto_for(d)


@router.delete("/files/{name}", status_code=204)
async def delete_dataset(name: str) -> Response:
    """Drop the whole directory. Settings and Tasks keep file references as plain text, so
    a task pointing here is left naming a file that no longer exists — the run
    will fail preflight rather than silently train on nothing."""
    d = _set_dir(name)
    private = _private_set_dir(d)
    if not d.is_dir() and not private.is_dir():
        raise HTTPException(404, f"file set {name!r} not found")
    if d.is_dir():
        shutil.rmtree(d)
    if private.is_dir():
        shutil.rmtree(private)
    return Response(status_code=204)


def _profile_in_background(dataset_dir: Path, name: str) -> None:
    """Background-task wrapper that never raises."""
    import sys
    try:
        _profile_one(dataset_dir, name)
    except Exception as e:  # noqa: BLE001
        print(
            f"[datasets.profile] background profile failed for {name}: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )


class FileSetPreview(BaseModel):
    """Best-effort preview. Always returns 200 for an existing dataset;
    the `kind` field tells the UI what to render."""
    name: str
    file: str
    # 'csv' | 'jsonl' | 'json' | 'pdf' | 'txt' | 'script' | 'parquet' | 'binary' | 'empty'
    kind: str
    # Tabular fields (csv/jsonl/parquet). Empty for text/binary.
    n_columns: int = 0
    columns: list[str] = Field(default_factory=list)
    n_rows: int = 0
    preview_rows: list[dict] = Field(default_factory=list)
    # Text fields (pdf/txt/markdown). Empty for tabular/binary.
    sample_text: str = ""
    # Human-readable note for the UI to show alongside whatever it can.
    notes: str = ""


def _pick_preview_file(d: Path) -> Path | None:
    """Pick the most informative file in the dataset dir.

    Priority: csv > jsonl > parquet > pdf > txt > md > anything else.
    Returns None when there is nothing to preview.

    Bookkeeping is skipped: `source.json` is the dataset's own record of what it
    is, and a dataset that stores nothing locally (it names a HuggingFace id
    instead) has only that — which the previewer used to parse as data and
    render as a table of parse errors.
    """
    files = [target for f in _all_dataset_files(d)
             if (target := _member(d, f)) is not None]
    if not files:
        return None
    priority = [".csv", ".jsonl", ".json", ".parquet", ".pdf", ".txt", ".md"]
    for ext in priority:
        for f in files:
            if f.suffix.lower() == ext:
                return f
    return files[0]


@router.get("/files/{name}/preview", response_model=FileSetPreview)
async def preview_dataset(
    request: Request, name: str, n: int = 5, file: str = "",
) -> FileSetPreview:
    """Run bounded, synchronous parsers outside the API event loop."""
    return await run_in_threadpool(
        _preview_dataset_sync, name, n, file, is_trusted_ui_request(request),
    )


def _preview_dataset_sync(
    name: str, n: int = 5, file: str = "", trusted_ui: bool = False,
) -> FileSetPreview:
    """Preview one file. `file` names it; without it the most informative one
    is picked, which is what the card links to."""
    d = _set_dir(name)
    if not d.exists() or not d.is_dir():
        raise HTTPException(404, f"file set {name} not found")

    if file:
        # Resolve inside the dataset dir — a name with .. must not escape it.
        candidate = _member(d, file)
        if candidate is None:
            raise HTTPException(404, f"{file!r} is not a file of file set {name!r}")
        target: Path | None = candidate
    else:
        target = _pick_preview_file(d)
    if target is None:
        # Nothing stored here. A dataset that names a hub id keeps no local
        # copy, and saying "no files" about it would be wrong — it has one, it
        # is just fetched when a run starts.
        src = _read_source(d)
        remote = (src.remote if src else []) or []
        if remote:
            where = "HuggingFace" if all(r.kind == "huggingface" for r in remote) else "the web"
            notes = (
                f"An online dataset on {where} ({', '.join(r.id for r in remote)}), "
                f"so there is nothing here to preview. Click the link above to open "
                f"it on {where}."
            )
        else:
            notes = "No files in this dataset directory."
        return FileSetPreview(name=name, file="", kind="empty", notes=notes)

    # The backend can preview private Test assets for the dashboard, but an
    # optimization Agent can also reach the backend network.  Do not turn the
    # Files API into a proxy around the filesystem boundary.
    try:
        is_private = target.resolve().is_relative_to(PRIVATE_DATASETS_DIR.resolve())
    except (OSError, ValueError):
        is_private = False
    if is_private and not trusted_ui:
        raise HTTPException(403, "held-out Test files are available only to the trusted UI")

    suffix = target.suffix.lower()
    n_preview = max(1, min(n, 50))

    try:
        if suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(target, nrows=n_preview)
            with target.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                n_rows = max(0, sum(1 for _ in csv.reader(fh)) - 1)
            return FileSetPreview(
                name=name, file=target.name, kind="csv",
                n_columns=len(df.columns),
                columns=list(df.columns),
                n_rows=n_rows,
                preview_rows=df.fillna("").to_dict(orient="records"),
            )

        if suffix in (".jsonl", ".json"):
            # `.json` is one document, `.jsonl` is one per line, and reading the
            # first as the second gives a "table" whose every row is a parse
            # error and whose row count is the file's line count. The suffix is
            # not enough on its own — plenty of `.json` files in the wild are
            # really JSONL — so the shape decides: a leading `[` is an array.
            rows: list[dict] = []
            with target.open("r", encoding="utf-8", errors="replace") as head_fh:
                head = head_fh.read(64).lstrip()
            as_array = head.startswith("[")
            if as_array:
                import ijson

                n_rows = 0
                with target.open("rb") as fh:
                    for item in ijson.items(fh, "item"):
                        n_rows += 1
                        if len(rows) < n_preview:
                            rows.append(item if isinstance(item, dict) else {"value": item})
            else:
                with target.open("r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i >= n_preview:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parsed = json.loads(line)
                            rows.append(parsed if isinstance(parsed, dict) else {"value": parsed})
                        except json.JSONDecodeError:
                            rows.append({"_parse_error": line[:200]})
                n_rows = sum(1 for ln in target.open("r", encoding="utf-8") if ln.strip())
            cols: list[str] = []
            for r in rows:
                for k in r.keys():
                    if k not in cols:
                        cols.append(k)
            # A nested value has no cell to sit in; show it as the JSON it is
            # rather than as a Python dict repr.
            flat = [{k: (json.dumps(v, ensure_ascii=False)
                         if isinstance(v, (dict, list)) else v)
                     for k, v in r.items()} for r in rows]
            return FileSetPreview(
                name=name, file=target.name, kind="json" if as_array else "jsonl",
                n_columns=len(cols), columns=cols,
                n_rows=n_rows, preview_rows=flat,
            )

        if suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(target)
            table = (
                parquet.read_row_group(0).slice(0, n_preview)
                if parquet.num_row_groups
                else None
            )
            df = table.to_pandas() if table is not None else None
            columns = list(parquet.schema_arrow.names)
            return FileSetPreview(
                name=name, file=target.name, kind="parquet",
                n_columns=len(columns),
                columns=columns,
                n_rows=int(parquet.metadata.num_rows),
                preview_rows=([] if df is None else df.fillna("").to_dict(orient="records")),
            )

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(target))
                pages_to_show = min(2, len(reader.pages))
                buf: list[str] = []
                for i in range(pages_to_show):
                    try:
                        buf.append(reader.pages[i].extract_text() or "")
                    except Exception as exc:  # pragma: no cover
                        buf.append(f"[page {i+1} extraction failed: {exc}]")
                text = "\n\n--- page break ---\n\n".join(buf).strip()
                return FileSetPreview(
                    name=name, file=target.name, kind="pdf",
                    sample_text=text[:8000],
                    notes=f"{len(reader.pages)} page(s); showing first {pages_to_show}.",
                )
            except ImportError:
                return FileSetPreview(
                    name=name, file=target.name, kind="pdf",
                    notes="pypdf not installed in the backend container; install it to enable PDF previews.",
                )

        # Scripts are read whole, not sampled: an eval script's point is what
        # it scores, and the first five lines are its imports.
        if suffix in (".py", ".sh"):
            src = target.read_text(encoding="utf-8", errors="replace")
            return FileSetPreview(
                name=name, file=target.name, kind="script",
                sample_text=src[:20000],
                notes=f"{len(src.splitlines())} lines"
                      + (" (truncated at 20k chars)" if len(src) > 20000 else ""),
            )

        if suffix in (".txt", ".md"):
            lines: list[str] = []
            with target.open("r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= n_preview:
                        break
                    lines.append(line.rstrip("\n"))
            return FileSetPreview(
                name=name, file=target.name, kind="txt",
                sample_text="\n".join(lines)[:8000],
                notes=f"First {len(lines)} line(s).",
            )

        # Unknown / binary -- still return something useful.
        size = target.stat().st_size
        return FileSetPreview(
            name=name, file=target.name, kind="binary",
            notes=f"{suffix or 'no extension'} file ({size} B). No preview available.",
        )
    except Exception as exc:
        # Never let a broken parser turn into 500 -- the UI just renders
        # the message and the user moves on.
        return FileSetPreview(
            name=name, file=target.name if target else "",
            kind="binary",
            notes=f"preview failed: {type(exc).__name__}: {exc}",
        )


# ──────────────────────────── profile endpoints ──────────────────────────────


@router.get("/files/{name}/profile", response_model=DatasetProfile)
async def get_dataset_profile(name: str, file: str = "") -> DatasetProfile:
    """Return the structured DatasetProfile for the given dataset.

    Hits the on-disk cache first; if the underlying file changed (sha
    mismatch) or the profiler version bumped, re-runs the profiler.
    Returns 404 if the dataset directory has no profileable file.
    """
    d = _set_dir(name)
    if not d.is_dir():
        raise HTTPException(404, f"file set {name!r} not found")
    if file:
        # A named file is profiled directly. The on-disk cache holds one profile
        # per dataset dir, so it can only serve the primary file; profiling here
        # is cheap enough that a fresh run beats a wrong cache hit.
        target = _member(d, file)
        if target is None:
            raise HTTPException(404, f"{file!r} is not a file of file set {name!r}")
        if target.suffix.lower() not in (".csv", ".jsonl", ".json", ".parquet"):
            raise HTTPException(404, f"{file!r} is not tabular — nothing to profile")
        return profile_file(f"{name}/{target.name}", target)
    profile = _profile_one(d, name)
    if profile is None:
        raise HTTPException(
            404,
            f"file set {name!r} has no .jsonl/.csv/.json file to profile "
            f"(PDF/TXT files are inspected by the data agent at run-time)",
        )
    return profile


@router.post("/files/{name}/profile/refresh", response_model=DatasetProfile)
async def refresh_dataset_profile(name: str) -> DatasetProfile:
    """Force re-profile, ignoring the cache. Use when the user has
    edited the file outside of the upload flow."""
    d = _set_dir(name)
    if not d.is_dir():
        raise HTTPException(404, f"file set {name!r} not found")
    # Bust cache by deleting it.
    from zevo.engine.dataset_profiler import cache_path_for
    cp = cache_path_for(d)
    if cp.exists():
        cp.unlink()
    profile = _profile_one(d, name)
    if profile is None:
        raise HTTPException(404, f"file set {name!r} has no profileable file")
    return profile
