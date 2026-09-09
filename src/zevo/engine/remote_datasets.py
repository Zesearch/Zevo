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

import json
from dataclasses import dataclass
from pathlib import Path

from zevo.paths import files_root as _files_root

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


# The datasets-server REST API rather than the `datasets` library: the backend
# needs a few hundred rows of a public split, not a dependency that pulls in
# torch.
_SERVER = "https://datasets-server.huggingface.co"
_PAGE = 100
# Not a policy on validation-set size — that is the user's call, and a cap here
# used to silently make it for them by taking a prefix. This is a runaway guard:
# the rows API pages 100 at a time, so a million-row split is ten thousand
# sequential HTTP calls inside run creation. Past this the run is refused with
# the count, which is a thing you can act on; a truncated set is not.
_MAX_FETCH = 200_000


async def _resolve_split(
    client, hub_id: str, split: str, config: str,
) -> tuple[str, str]:
    """Settle (config, split) against what the repo actually publishes.

    Asking for a split the repo does not have is the failure worth catching
    here: the alternative is a validation set that silently ends up being some
    other split, which nothing downstream can detect.
    """
    try:
        r = await client.get(f"{_SERVER}/splits", params={"dataset": hub_id})
        r.raise_for_status()
        available = r.json().get("splits") or []
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
    import csv as _csv

    import httpx

    ident = (hub_id or "").strip()
    if not ident:
        raise MaterializeError("no hub id given")
    rows: list[dict] = []
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        cfg, spl = await _resolve_split(client, ident, split.strip(), config.strip())
        ceiling = limit if limit > 0 else _MAX_FETCH
        offset = 0
        while len(rows) < ceiling:
            n = min(_PAGE, ceiling - len(rows))
            try:
                r = await client.get(f"{_SERVER}/rows", params={
                    "dataset": ident, "config": cfg, "split": spl,
                    "offset": offset, "length": n,
                })
                r.raise_for_status()
                batch = [x.get("row") or {} for x in (r.json().get("rows") or [])]
            except Exception as e:
                raise MaterializeError(f"cannot read rows of {ident!r} ({cfg}/{spl}): {e}")
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
        if limit <= 0 and len(rows) >= _MAX_FETCH:
            raise MaterializeError(
                f"{ident!r} ({cfg}/{spl}) has at least {_MAX_FETCH:,} rows, which is "
                "too many to pull into a validation set at run creation. Point at a "
                "smaller split, or upload the rows you want as a file."
            )
    if not rows:
        raise MaterializeError(f"{ident!r} ({cfg}/{spl}) returned no rows")

    columns: list[str] = []
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: _cell(row.get(c)) for c in columns})

    note = f"fetched all {len(rows)} rows of {ident} ({cfg}/{spl})"
    if 0 < limit <= len(rows):
        # Said out loud: the run is validating on a prefix, not on the split.
        note = (f"fetched the first {len(rows)} rows of {ident} ({cfg}/{spl}), "
                f"capped at {limit}")
    return str(path), columns, len(rows), note


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
            spec = RemoteSpec(
                id=ident,
                split=str(e.get("split") or ""),
                config=str(e.get("config") or ""),
                role=str(e.get("role") or ""),
                dataset=d.name,
            )
            # Matched on what the entry IS, which the split states directly.
            # An older source.json may carry a `role` that disagrees with its
            # split; the split is the one the loader acts on, so it wins.
            if wanted and _role_of(spec.split or spec.role) == wanted:
                return spec
            if fallback is None:
                fallback = spec
    return fallback
