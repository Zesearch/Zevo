"""Two runs must never land on the same card.

`instance` mode attaches to an allocation the user started by hand, with
`srun --overlap` — the flag that tells Slurm explicitly NOT to schedule. So
nothing below this layer arbitrates: if the lease allocator hands card 3 to
two runs, both launch on it and the failure surfaces as an OOM in whichever
one loses, three stages later, with nothing pointing back here.

The cases are the ones that describe the intent:

  * a slice comes from ONE allocation. 3 cards wanted, two allocations with 2
    free each -> refused. Free cards on different jobs do not add up.
  * 2 + 2 across two allocations -> both placed, on different jobs.
  * 1 + 1 + 1 + 1 into two 2-GPU allocations -> all four fit.
  * concurrent requests for the same last card -> exactly one wins.

The last one is the only one that cannot be satisfied by careful reading of
free cards, and it is the reason the partial unique index exists.
"""
from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.api.database import get_db
from zevo.api.main import app
from zevo.db.models import Base, GpuLease, Run
from zevo.contracts.infrastructure import GpuLeaseRequest


# ───────────────────────────── fixtures ──────────────────────────────────────


@pytest_asyncio.fixture
async def sessionmaker_(tmp_path):
    # On DISK, not `:memory:`. Every in-memory connection gets its own private
    # database, so the concurrency case would pass vacuously: eight requests
    # would each insert card 0 into eight different databases and never meet.
    # A file is the only way the unique index is actually under test.
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/leases.db", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def client(sessionmaker_):
    async def _override():
        async with sessionmaker_() as s:
            yield s

    app.dependency_overrides[get_db] = _override
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


def _allocs(*pairs: tuple[str, int]) -> list[dict]:
    """(jobid, gpu_count) -> the inventory shape squeue reporting produces."""
    return [
        {
            "jobid": j,
            "node": f"node{j}",
            "gpu_count": n,
            "idle_gpu_indices": list(range(n)),
            "gpu_name": "B200",
            "vram_gb": 180,
        }
        for j, n in pairs
    ]


def test_lease_contract_has_one_closed_allocation_key() -> None:
    body = GpuLeaseRequest.model_validate({
        "run_id": "run-a", "ticket_id": "infra-a-001", "num_gpus": 1,
        "min_vram_gb": 24, "allocations": _allocs(("101", 1)),
    })
    assert body.allocations[0].jobid == "101"
    with pytest.raises(ValidationError, match="candidates"):
        GpuLeaseRequest.model_validate({
            "run_id": "run-a", "num_gpus": 1,
            "candidates": _allocs(("101", 1)),
        })


async def _acquire(client, run_id: str, want: int, allocs: list[dict]):
    return await client.post(
        "/api/gpu/leases",
        json={"run_id": run_id, "num_gpus": want, "allocations": allocs},
    )


# ───────────────────────────── placement ─────────────────────────────────────


@pytest.mark.asyncio
async def test_slice_never_spans_allocations(client):
    """3 cards, two allocations with 2 each: refused, not split 2+1."""
    r = await _acquire(client, "run-a", 3, _allocs(("101", 2), ("102", 2)))
    assert r.status_code == 409
    detail = r.json()["detail"]
    # The refusal has to say what is free where, or the user cannot act on it.
    assert "101" in detail and "102" in detail
    assert "2/2 idle and unleased" in detail


@pytest.mark.asyncio
async def test_two_pairs_land_on_different_allocations(client):
    allocs = _allocs(("101", 2), ("102", 2))
    a = (await _acquire(client, "run-a", 2, allocs)).json()
    b = (await _acquire(client, "run-b", 2, allocs)).json()

    assert a["gpu_count"] == 2 and b["gpu_count"] == 2
    assert a["jobid"] != b["jobid"], "the second run took cards on the first's job"
    assert a["visible_devices"] == "0,1" and b["visible_devices"] == "0,1"

    # And now there is genuinely nothing left.
    assert (await _acquire(client, "run-c", 1, allocs)).status_code == 409


@pytest.mark.asyncio
async def test_four_singles_fill_two_dual_allocations(client):
    allocs = _allocs(("101", 2), ("102", 2))
    grants = []
    for i in range(4):
        r = await _acquire(client, f"run-{i}", 1, allocs)
        assert r.status_code == 200, r.text
        grants.append(r.json())

    cards = {(g["jobid"], tuple(g["gpu_indices"])) for g in grants}
    assert len(cards) == 4, f"a card was handed out twice: {grants}"
    assert (await _acquire(client, "run-x", 1, allocs)).status_code == 409


@pytest.mark.asyncio
async def test_best_fit_keeps_the_empty_allocation_whole(client):
    """A single must not be scattered onto the box a later 4-GPU job needs."""
    allocs = _allocs(("101", 4), ("102", 4))
    await _acquire(client, "run-a", 3, allocs)   # 101 -> 3 used, 1 free
    single = (await _acquire(client, "run-b", 1, allocs)).json()
    assert single["jobid"] == "101", "the single fragmented the empty allocation"

    big = (await _acquire(client, "run-c", 4, allocs)).json()
    assert big["jobid"] == "102" and big["gpu_count"] == 4


@pytest.mark.asyncio
async def test_live_probe_excludes_busy_allocation_even_without_zevo_lease(client):
    """Foreign GPU work is absent from the ledger but present in the idle probe."""
    allocs = _allocs(("101", 1), ("102", 1))
    allocs[0]["idle_gpu_indices"] = []

    grant = (await _acquire(client, "run-a", 1, allocs)).json()
    assert grant["jobid"] == "102"
    assert grant["gpu_indices"] == [0]


@pytest.mark.asyncio
async def test_allocator_uses_only_explicitly_idle_indices(client):
    allocs = _allocs(("101", 4))
    allocs[0]["idle_gpu_indices"] = [1, 3]

    grant = (await _acquire(client, "run-a", 2, allocs)).json()
    assert grant["gpu_indices"] == [1, 3]


@pytest.mark.asyncio
async def test_no_idle_gpu_fails_only_after_all_candidates_are_considered(client):
    allocs = _allocs(("101", 2), ("102", 2))
    for allocation in allocs:
        allocation["idle_gpu_indices"] = []

    response = await _acquire(client, "run-a", 1, allocs)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "101" in detail and "102" in detail
    assert "idle and unleased" in detail


@pytest.mark.asyncio
async def test_idle_indices_are_required_and_must_be_physical_indices(client):
    missing = _allocs(("101", 2))[0]
    missing.pop("idle_gpu_indices")
    response = await _acquire(client, "run-a", 1, [missing])
    assert response.status_code == 422

    invalid = _allocs(("101", 2))[0]
    invalid["idle_gpu_indices"] = [2]
    response = await _acquire(client, "run-a", 1, [invalid])
    assert response.status_code == 422

    duplicate_jobs = _allocs(("101", 1), ("101", 1))
    response = await _acquire(client, "run-a", 1, duplicate_jobs)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_zero_means_take_one_allocation_whole(client):
    """num_gpus unset keeps the pre-lease behaviour: a lone run gets the box."""
    g = (await _acquire(client, "run-a", 0, _allocs(("101", 8)))).json()
    assert g["gpu_count"] == 8
    assert g["visible_devices"] == "0,1,2,3,4,5,6,7"


@pytest.mark.asyncio
async def test_reacquire_returns_the_same_cards(client):
    """A re-woken infra ticket must not bid twice for the same run."""
    allocs = _allocs(("101", 4))
    first = (await _acquire(client, "run-a", 2, allocs)).json()
    again = (await _acquire(client, "run-a", 2, allocs)).json()
    assert again["reused"] is True
    assert again["gpu_indices"] == first["gpu_indices"]
    # ...and it did not consume four cards in the process.
    assert (await _acquire(client, "run-b", 2, allocs)).status_code == 200


# ───────────────────────────── the race ──────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_requests_never_share_a_card(client, sessionmaker_):
    """Eight runs, one 4-GPU allocation, all asking at once: four win."""
    allocs = _allocs(("101", 4))
    results = await asyncio.gather(
        *(_acquire(client, f"run-{i}", 1, allocs) for i in range(8)),
        return_exceptions=True,
    )
    granted = [r.json() for r in results if not isinstance(r, Exception) and r.status_code == 200]
    held = [tuple(g["gpu_indices"]) for g in granted]
    assert len(held) == len(set(held)), f"two runs got the same card: {granted}"

    # Exactly four, not "at most four": a `<=` here would also pass if the
    # allocator refused everyone, which is the failure this test exists to
    # catch in reverse. If this ever trips, the cause is in `others` — an
    # unhandled OperationalError under SQLite write contention would drop a
    # request to a 500 and take a real grant with it, and without printing it
    # the failure looks like an allocator bug it is not.
    # Seen to fail ONCE, inside the full suite, and not reproduced in ~100
    # subsequent runs (60 solo, 25 full-suite, random order). So the diagnostic
    # is the fix here rather than a guess at the cause: the next failure has to
    # arrive with the reason attached, because re-running will not produce one.
    #
    # `.text` on a 500 carries the traceback, which is what would distinguish
    # the two live hypotheses — SQLite write contention surfacing as an
    # unhandled OperationalError, versus a genuine allocator fault. The DB row
    # count separates "the grant never happened" from "it happened and the
    # response was lost".
    others = [
        repr(r) if isinstance(r, Exception) else f"{r.status_code} {r.text[:400]}"
        for r in results
        if isinstance(r, Exception) or r.status_code != 200
    ]
    async with sessionmaker_() as s:
        live = (await s.execute(
            select(GpuLease).where(GpuLease.released_at.is_(None))
        )).scalars().all()
    ledger = sorted((x.jobid, x.gpu_index, x.run_id) for x in live)
    assert sorted(i for (i,) in held) == [0, 1, 2, 3], (
        f"granted {held}; ledger holds {ledger}; non-grants were: {others}"
    )


# ───────────────────────────── reaping ───────────────────────────────────────


@pytest.mark.asyncio
async def test_candidate_omission_does_not_release_an_active_runs_lease(client):
    """One request's candidate list is not a global liveness snapshot."""
    await _acquire(client, "run-a", 2, _allocs(("101", 2)))
    g = (await _acquire(client, "run-b", 2, _allocs(("102", 2)))).json()
    assert g["jobid"] == "102"

    live = (await client.get("/api/gpu/leases")).json()
    assert {r["jobid"] for r in live} == {"101", "102"}, live


@pytest.mark.asyncio
async def test_vram_floor_rejects_unknown_and_undersized_allocations(client):
    allocs = _allocs(("101", 1), ("102", 1))
    allocs[0]["vram_gb"] = 0
    allocs[1]["vram_gb"] = 24

    response = await client.post(
        "/api/gpu/leases",
        json={
            "run_id": "run-a",
            "num_gpus": 1,
            "min_vram_gb": 40,
            "allocations": allocs,
        },
    )
    assert response.status_code == 409
    assert "verified per-device VRAM" in response.json()["detail"]


@pytest.mark.asyncio
async def test_vram_floor_selects_a_verified_eligible_allocation(client):
    allocs = _allocs(("101", 1), ("102", 1), ("103", 1))
    allocs[0]["vram_gb"] = 0
    allocs[1]["vram_gb"] = 24
    allocs[2]["vram_gb"] = 80

    response = await client.post(
        "/api/gpu/leases",
        json={
            "run_id": "run-a",
            "num_gpus": 1,
            "min_vram_gb": 40,
            "allocations": allocs,
        },
    )
    assert response.status_code == 200
    assert response.json()["jobid"] == "103"


@pytest.mark.asyncio
async def test_terminal_run_frees_its_cards(client, sessionmaker_):
    allocs = _allocs(("101", 2))
    async with sessionmaker_() as s:
        s.add(Run(metric="accuracy", id="run-a", status="running"))
        await s.commit()
    await _acquire(client, "run-a", 2, allocs)
    assert (await _acquire(client, "run-b", 1, allocs)).status_code == 409

    async with sessionmaker_() as s:
        run = await s.get(Run, "run-a")
        run.status = "success"
        await s.commit()

    assert (await _acquire(client, "run-b", 2, allocs)).status_code == 200


@pytest.mark.asyncio
async def test_release_hands_the_cards_back(client):
    allocs = _allocs(("101", 2))
    await _acquire(client, "run-a", 2, allocs)
    r = await client.delete("/api/gpu/leases", params={"run_id": "run-a"})
    assert r.json()["released"] == 2
    assert (await _acquire(client, "run-b", 2, allocs)).status_code == 200


@pytest.mark.asyncio
async def test_no_verified_fixed_host_says_so(client):
    r = await _acquire(client, "run-a", 1, [])
    assert r.status_code == 409
    assert "no verified fixed instance host" in r.json()["detail"]


@pytest.mark.asyncio
async def test_released_rows_do_not_block_the_next_holder(client, sessionmaker_):
    """The unique index is partial — history must not fill the allocation up."""
    allocs = _allocs(("101", 1))
    for i in range(3):
        assert (await _acquire(client, f"run-{i}", 1, allocs)).status_code == 200
        await client.delete("/api/gpu/leases", params={"run_id": f"run-{i}"})

    async with sessionmaker_() as s:
        rows = (await s.execute(select(GpuLease))).scalars().all()
    assert len(rows) == 3
    assert sum(1 for r in rows if r.released_at is None) == 0


@pytest.mark.asyncio
async def test_unknown_run_is_not_reported_as_a_race(client, sessionmaker_):
    """A bad run_id is a caller error, not "someone took the card".

    Both arrive as IntegrityError. Retrying a foreign-key violation six times
    and then blaming the allocations sends whoever reads it looking at Slurm
    for a problem that is in the request.
    """
    from sqlalchemy import event

    # SQLite leaves foreign keys off by default, which is the only reason the
    # rest of this file can use invented run ids. Turn them on for this case.
    engine = sessionmaker_.kw["bind"]

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    await engine.dispose()  # force new connections so the pragma applies
    r = await _acquire(client, "no-such-run", 1, _allocs(("101", 2)))
    assert r.status_code == 400, r.text
    assert "kept changing" not in r.json()["detail"]


@pytest.mark.asyncio
async def test_refusal_distinguishes_occupied_from_too_small(client):
    """Both are refusals; only one is worth relaunching for.

    There is no queue: a run that cannot be placed fails. So the message is
    the entire remedy, and "wait for a run to finish" is wrong advice when no
    allocation was ever big enough.
    """
    allocs = _allocs(("101", 2), ("102", 2))

    # Too big for any allocation, empty or not.
    too_big = await _acquire(client, "run-a", 4, allocs)
    assert too_big.status_code == 409
    assert "no allocation is big enough" in too_big.json()["detail"]
    assert "largest is 2" in too_big.json()["detail"]

    # Would fit, but the cards are taken.
    await _acquire(client, "run-b", 2, allocs)
    await _acquire(client, "run-c", 2, allocs)
    busy = await _acquire(client, "run-d", 2, allocs)
    assert busy.status_code == 409
    assert "leased by another run" in busy.json()["detail"]
    assert "Relaunch once one of them finishes" in busy.json()["detail"]


@pytest.mark.asyncio
async def test_cancelling_a_run_hands_its_cards_back(client, sessionmaker_):
    """Cancel does not go through the reconciler, so it releases its own.

    `cancel_run` writes the terminal status directly; the reconciler's
    closed-run sweep never sees it. Without a release on this path the cards
    stay held until something else happens to call the acquire endpoint.
    """
    from zevo.api.routers.shared.runs import _mark_run_instances_released

    allocs = _allocs(("101", 2))
    async with sessionmaker_() as s:
        s.add(Run(metric="accuracy", id="run-a", status="running"))
        await s.commit()
    assert (await _acquire(client, "run-a", 2, allocs)).status_code == 200

    async with sessionmaker_() as s:
        await _mark_run_instances_released(s, "run-a")

    live = (await client.get("/api/gpu/leases")).json()
    assert live == [], live
