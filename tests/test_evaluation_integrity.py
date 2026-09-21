from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from zevo.api.file_integrity import publish_upload
from zevo.engine.artifact_validation import _semantic_fingerprint, semantic_record_fingerprints
from zevo.engine.method.evaluation_identity import (
    snapshot_test_contract, inference_identity, complete_identity, compatible_identities,
)
from zevo.db.models import Base, Run, Task, ScoreEvent, Ticket, WorkProduct


def test_semantic_inputs_have_stable_typed_boundaries():
    a = {"question": "What?", "context": {"left": 2, "right": False}, "answer": "secret"}
    b = {"answer": "changed", "context": {"right": False, "left": 2}, "question": " What? "}
    assert _semantic_fingerprint(a) == _semantic_fingerprint(b)
    variants = [{"x": value} for value in [0, 1, False, None, "0", "false", [0], {"v": 0}]]
    identities = [_semantic_fingerprint(item) for item in variants]
    assert all(identities) and len(set(identities)) == len(variants)
    assert _semantic_fingerprint({"input": ["a", "b"]}) != _semantic_fingerprint({"input": "a\nb"})
    assert _semantic_fingerprint({"input": [1, 2]}) != _semantic_fingerprint({"input": [2, 1]})
    assert _semantic_fingerprint({"instruction": {"2": "b", "1": "a"}}) == _semantic_fingerprint({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "private"}, {"role": "user", "content": "b"}]})


def test_numeric_and_reordered_populations_overlap(tmp_path):
    left = tmp_path / "test.json"
    right = tmp_path / "validation.json"
    left.write_text(json.dumps([{"x": 0, "y": True, "answer": "one"}]))
    right.write_text(json.dumps([{"y": True, "answer": "two", "x": 0}]))
    assert semantic_record_fingerprints(str(left), excluded_fields=["answer"]) & semantic_record_fingerprints(str(right), excluded_fields=["answer"])


def test_upload_replaces_live_file_atomically_and_is_idempotent(tmp_path):
    target = tmp_path / "train.csv"
    assert publish_upload(BytesIO(b"original"), target, max_bytes=20) == 8
    assert publish_upload(BytesIO(b"original"), target, max_bytes=20) == 8
    with pytest.raises(HTTPException) as error:
        publish_upload(BytesIO(b"replacement"), target, max_bytes=3)
    assert error.value.status_code == 413
    assert target.read_bytes() == b"original"
    assert publish_upload(BytesIO(b"replacement"), target, max_bytes=20) == 11
    assert target.read_bytes() == b"replacement"
    class FailingSource:
        def read(self, _size):
            raise OSError("upload interrupted")
    with pytest.raises(OSError):
        publish_upload(FailingSource(), target, max_bytes=20)
    assert target.read_bytes() == b"replacement"
    assert list(tmp_path.iterdir()) == [target]


def _suite(tmp_path):
    data = tmp_path / "test.csv"
    data.write_text("question,answer\nfirst,yes\n")
    sample = tmp_path / "sample.csv"
    sample.write_text("prediction\n\n")
    return [{"name": "test", "test_set": str(data), "sample_submission": str(sample), "inference_query": "Answer {question}", "metric": "accuracy", "metric_direction": "max", "answer_fields": ["answer"], "metric_type": "builtin"}]


def _identity(contract, config=None):
    config = config or {"prompt": {"system_prompt": "Be exact"}, "measurement": {"decoding_config": {"temperature": 0}}, "generation_backend": "hf"}
    return complete_identity(contract, {member["name"]: {"test_contract": contract, "inference_identity": inference_identity(config)} for member in contract["members"]})


def test_comparison_requires_population_scorer_suite_and_inference_evidence(tmp_path):
    suite = _suite(tmp_path)
    contract = snapshot_test_contract(suite)
    assert contract
    original = _identity(contract)
    assert compatible_identities(original, deepcopy(original))
    assert not compatible_identities(original, None)
    for key, value in [("inference_query", "Different prompt"), ("metric", "exact_match"), ("metric_direction", "min")]:
        changed = deepcopy(suite)
        changed[0][key] = value
        assert not compatible_identities(original, _identity(snapshot_test_contract(changed)))
    changed = deepcopy(suite)
    changed.append({**changed[0], "name": "second"})
    assert not compatible_identities(original, _identity(snapshot_test_contract(changed)))
    altered_config = {"prompt": {"system_prompt": "Be exact"}, "measurement": {"decoding_config": {"temperature": 1}}, "generation_backend": "hf"}
    assert not compatible_identities(original, _identity(contract, altered_config))
    Path(suite[0]["test_set"]).write_text("question,answer\nsecond,no\n")
    assert not compatible_identities(original, _identity(snapshot_test_contract(suite)))
    # Same metric label but different custom evaluator bytes are incompatible.
    script = tmp_path / "evaluate.py"
    script.write_text("score = 1\n")
    from zevo.engine.method.evaluation_identity import file_digest
    suite[0].update(metric_type="custom", evaluation_script=str(script), evaluator_sha256=file_digest(script))
    custom = _identity(snapshot_test_contract(suite))
    script.write_text("score = 0\n")
    assert snapshot_test_contract(suite) is None
    suite[0]["evaluator_sha256"] = file_digest(script)
    assert not compatible_identities(custom, _identity(snapshot_test_contract(suite)))


@pytest.mark.asyncio
async def test_validation_publication_rolls_back_marker_and_retries_once(tmp_path, monkeypatch):
    from zevo.engine.run import runner
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        run = Run(id="r", task_name="suite", status="running", metric="suite_average", metric_direction="max", validation_metric="suite_average", validation_metric_direction="max", holdout={"validation_sets": [{"name": name, "validation_set": f"/data/{name}.csv", "metric": "accuracy", "metric_direction": "max"} for name in ("a", "b")]})
        inference = Ticket(id="infer", run_id="r", agent_id="inference", iteration=1, status="succeeded", lane="optimization", payload={"model_source": "checkpoint", "base_model": "owner/model"})
        evaluations = [Ticket(id=f"eval-{name}", run_id="r", agent_id="evaluation", iteration=1, status="succeeded", lane="optimization", payload={"test_set_name": f"validation:{name}", "metric": "accuracy"}, inputs={"predictions": {"source_ticket_id": "infer"}}) for name in ("a", "b")]
        path = tmp_path / "metrics.json"
        path.write_text('{}')
        product = WorkProduct(ticket_id="eval-b", role="metrics", path=str(path), meta={})
        db.add_all([run, inference, *evaluations, product])
        await db.commit()
        assert (await runner._record_validation_component(db, run, evaluations[0], .4))[0] is False
        original = Path.write_text
        def fail_metrics(self, *args, **kwargs):
            if self == path:
                raise OSError("simulated disk full")
            return original(self, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "write_text", fail_metrics)
            with pytest.raises(ValueError, match="persist aggregate"):
                await runner._record_validation_component(db, run, evaluations[1], .8)
        run = await db.get(Run, "r")
        evaluation = await db.get(Ticket, "eval-b")
        assert not run.holdout.get("validation_suite_recorded")
        assert (await db.execute(select(func.count()).select_from(ScoreEvent))).scalar_one() == 0
        assert (await runner._record_validation_component(db, run, evaluation, .8))[0] is True
        # A delivery retry (even with a different claimed value) cannot rewrite
        # the published components or append a second event.
        assert (await runner._record_validation_component(db, run, evaluation, .1))[0] is True
        event = (await db.execute(select(ScoreEvent))).scalar_one()
        assert event.score == pytest.approx(.6)
        assert run.history[0]["score"] == pytest.approx(.6)
        assert run.iterations_completed == 1
        assert run.holdout["validation_suite_results"]["trained|1|owner/model"]["b"]["score"] == .8
    await engine.dispose()


@pytest.mark.asyncio
async def test_settlement_rejects_reordered_numeric_overlap(tmp_path, monkeypatch):
    from zevo.contracts.orchestrator import TaskTestSet, UserRequest
    from zevo.engine.run.split_settlement import settle_splits, SplitSettlementError
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))
    test = tmp_path / "test.json"
    validation = tmp_path / "validation.json"
    test.write_text(json.dumps([{"x": i, "y": True, "answer": "yes"} for i in range(250)]))
    validation.write_text(json.dumps([{"answer": "no", "y": True, "x": i} for i in range(200)]))
    sample = tmp_path / "sample.csv"
    sample.write_text("prediction\nexample\n")
    def member(name, path):
        return TaskTestSet(name=name, test_set=str(path), sample_submission=str(sample), inference_query="Classify {x}", metric="accuracy", metric_direction="max", answer_fields=["answer"])
    request = UserRequest(task_objective="Classify values", test_sets=[member("test", test)], validation_sets=[member("validation", validation)], metric="accuracy", metric_direction="max", test_set=str(test), test_answer_fields=["answer"], test_sample_submission=str(sample), training_method="", dataset="", base_model="owner/model", constraints=[])
    run = Run(id="numeric-split", task_name="numeric", metric="accuracy", metric_direction="max")
    with pytest.raises(SplitSettlementError, match="share 200 question"):
        await settle_splits(run, request, work_dir_root=str(tmp_path / "runs"))


@pytest.mark.asyncio
async def test_measurement_captures_identity_and_comparison_fails_closed(tmp_path):
    from zevo.engine.run.runner import _record_holdout_score
    from zevo.api.routers.ui.models import compare
    from zevo.db.models import RegistryModel
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    suite = _suite(tmp_path)
    contract = snapshot_test_contract(suite)
    configuration = {"prompt": {"system_prompt": "Be exact"}, "measurement": {"decoding_config": {"temperature": 0}}, "generation_backend": "hf"}
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        for name, score in [("a", .4), ("b", .8)]:
            run = Run(id=name, task_name="same-name", metric="accuracy", metric_direction="max", status="running", holdout={"test_sets": suite, "evaluation_contract": contract}, history=[{"iteration": 1, "source": "trained", "score": score}])
            infer = Ticket(id=f"infer-{name}", run_id=name, agent_id="inference", iteration=1, status="succeeded", lane="held_out_test", payload={"model_source": "checkpoint", "base_model": "owner/model"})
            evaluation = Ticket(id=f"eval-{name}", run_id=name, agent_id="evaluation", iteration=1, status="succeeded", lane="held_out_test", payload={"test_set_name": "test", "metric": "accuracy"}, inputs={"predictions": {"source_ticket_id": infer.id}})
            predictions = WorkProduct(ticket_id=infer.id, role="predictions", path=f"/unused/{name}.csv", meta={"suite_primary": True, "configuration": configuration})
            model = RegistryModel(version_tag=f"M-{name}", run_id=name, iteration=1, base_model="owner/model", training_method="sft", metric="accuracy", metric_direction="max")
            db.add_all([run, infer, evaluation, predictions, model])
            await db.commit()
            await _record_holdout_score(db, run, evaluation, score)
        result = await compare("M-a", "M-b", db)
        assert result.champion_test_score_delta == pytest.approx(.4)
        events = (await db.execute(select(ScoreEvent).order_by(ScoreEvent.run_id))).scalars().all()
        assert events[0].extras["evaluation_identity"]["sha256"]
        # A post-run source edit cannot change the immutable measured evidence.
        Path(suite[0]["test_set"]).write_text("question,answer\nchanged,no\n")
        assert (await compare("M-a", "M-b", db)).champion_test_score_delta == pytest.approx(.4)
        # Legacy data with only matching metric strings is never comparable.
        events[1].extras = {}
        await db.commit()
        result = await compare("M-a", "M-b", db)
        assert result.champion_test_score_delta is None
        assert "unverified evaluation contracts" in result.headline_summary
    await engine.dispose()


def test_remote_standalone_helper_matches_scheduler_identity():
    import runpy
    from zevo.engine import remote_training_data
    remote = runpy.run_path(remote_training_data.__file__, run_name="standalone_test")
    samples = [
        {"question": "hello", "context": {"z": 2, "a": False}},
        {"x": 0, "y": None},
        {"messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "private"}]},
        {"instruction": {"2": "two", "1": "one"}},
    ]
    for sample in samples:
        assert remote["_semantic_fingerprint"](sample) == _semantic_fingerprint(sample)


def test_concurrent_uploads_publish_only_complete_versions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    target = tmp_path / "dataset.csv"
    def upload(content):
        try:
            publish_upload(BytesIO(content), target, max_bytes=1024)
            return (200, content)
        except HTTPException as error:
            return (error.status_code, content)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(upload, [b"version-one", b"version-two"]))
    assert sorted(code for code, _ in results) == [200, 200]
    assert target.read_bytes() in {b"version-one", b"version-two"}
    assert list(tmp_path.iterdir()) == [target]
