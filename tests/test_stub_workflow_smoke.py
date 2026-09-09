"""One cheap end-to-end contract smoke test: no LLM, SSH, GPU, or scorer."""
from __future__ import annotations

from pathlib import Path
import json

from zevo.contracts.configuration import load_inference_config, load_train_config
from zevo.contracts.data import load_data_recipe
from zevo.engine.agent.drivers.stub import (
    _stub_data,
    _stub_eval,
    _stub_infer,
    _stub_infra,
    _stub_registry,
    _stub_train,
)
from zevo.engine.artifact_validation import materialize_system_scoring_artifacts


def test_stub_baseline_and_one_training_iteration(tmp_path: Path) -> None:
    infra = _stub_infra(
        {"ticket_id": "infra-smoke", "provider": "instance", "num_gpus": 1},
        tmp_path / "infra",
    )
    data = _stub_data(
        {
            "ticket_id": "data-smoke",
            "operation": "prepare_run_data",
            "dataset": "/provided/train.jsonl",
            "dataset_source": "/provided/train.jsonl",
            "expected_source_identity": "/provided/train.jsonl",
            "training_method": "full_sft",
            "answer_fields": ["answer"],
        },
        tmp_path / "data",
    )
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps([
        {"id": "v1", "instruction": "say hello", "answer": "hello"},
    ]), encoding="utf-8")
    sample = tmp_path / "sample.csv"
    sample.write_text("id,prediction\nexample,\n", encoding="utf-8")
    scoring = materialize_system_scoring_artifacts(
        scoring_source=str(validation),
        answer_fields=["answer"],
        sample_submission=str(sample),
        out_dir=str(tmp_path / "scoring"),
    )

    baseline = _stub_infer(
        {
            "ticket_id": "infer-smoke-0",
            "base_model": "Qwen/Qwen3-0.6B-Base",
            "generation_backend": "vllm",
            "verified_model_reasoning_type": "thinking",
            "inference_data_profile_path": scoring.profile_path,
        },
        tmp_path / "infer0",
    )
    baseline_eval = _stub_eval(
        {"ticket_id": "eval-smoke-0", "metric": "token_f1"},
        tmp_path / "eval0",
    )

    trained = _stub_train(
        {
            "ticket_id": "train-smoke-1",
            "iteration": 1,
            "base_model": "Qwen/Qwen3-0.6B-Base",
            "training_method_pin": "full_sft",
            "inference_config_path": baseline.inference_config_path,
            "generation_backend": "vllm",
        },
        tmp_path / "train1",
    )
    candidate = _stub_infer(
        {
            "ticket_id": "infer-smoke-1",
            "base_model": "Qwen/Qwen3-0.6B-Base",
            "generation_backend": "vllm",
            "inference_config_path": baseline.inference_config_path,
            "reusable_predict_script_path": baseline.predict_script_path,
        },
        tmp_path / "infer1",
    )
    candidate_eval = _stub_eval(
        {"ticket_id": "eval-smoke-1", "metric": "token_f1"},
        tmp_path / "eval1",
    )
    registered = _stub_registry(
        {
            "ticket_id": "registry-smoke-1",
            "run_id": "12345678-smoke",
            "iteration": 1,
            "checkpoint_path": trained.checkpoint_path,
            "metrics_path": candidate_eval.metrics_path,
            "base_model": load_inference_config(
                baseline.inference_config_path
            ).base_model,
            "training_method": load_train_config(trained.train_config_path).training_method,
            "dataset_source": load_data_recipe(data.data_recipe_path).source_identity,
            "task_objective": "smoke",
            "metric": "token_f1",
            "metric_direction": "max",
        },
        tmp_path / "registry",
    )

    assert all(result.status == "succeeded" for result in (
        infra, data, baseline, baseline_eval, trained, candidate,
        candidate_eval, registered,
    ))
    assert candidate.inference_config_path == baseline.inference_config_path
    assert candidate.predict_script_path == baseline.predict_script_path
    assert Path(trained.train_config_path).is_file()
    inference_config = load_inference_config(baseline.inference_config_path)
    train_config = load_train_config(trained.train_config_path)
    assert inference_config.prompt.model_reasoning_type == "thinking"
    assert inference_config.template_kwargs == {"enable_thinking": True}
    assert any(
        "<think>\n<THINKING_TRACE>\n</think>" in sequence
        for sequence in train_config.training_data_example.rendered_sequences.values()
    )
    assert inference_config.implementation_config == {}
    assert train_config.training.logging_steps == 20
    assert "ZevoTrainerTelemetryCallback" in Path(
        trained.train_script_path
    ).read_text(encoding="utf-8")
    assert Path(trained.log_path).is_file()
    assert Path(registered.registry_path).is_file()
    assert Path(registered.registry_path) == tmp_path / "registry" / "registry.yaml"
