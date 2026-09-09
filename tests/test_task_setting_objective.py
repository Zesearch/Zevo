"""A task states the problem; a setting states what a run pins down.

The shipped catalogue used to bake the setting into each task's objective PROSE,
which is why one problem needed four task rows. The c9e1a4b7d2f3 migration takes
those sentences back out and `compose_objective` puts the right ones back per
run. These tests hold that pair honest: the four levels of a domain must strip
to the SAME problem statement (or they were not one task), and composing it with
a setting must say what that setting decided.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from zevo.api.routers.ui.tasks import autonomy_level, compose_objective


def _migration():
    import zevo

    path = (Path(zevo.__file__).resolve().parents[2]
            / "alembic" / "versions" / "c9e1a4b7d2f3_merge_tasks_by_domain.py")
    spec = importlib.util.spec_from_file_location("merge_tasks_by_domain", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_MODEL = "Qwen/Qwen3-4B"
_PROBLEM = ("Fine-tune {model} to answer USMLE-style medical multiple-choice questions. "
            "reason, then give the final answer as a boxed letter, e.g. \\boxed{{C}}. ")
_TAIL = ("There is no budget cap. Iterate freely (no iteration cap) to maximize test "
         "accuracy; keep the best and STOP when accuracy plateaus.")

# The four med rows exactly as the catalogue shipped them.
SHIPPED = {
    "L1": (_PROBLEM.format(model=_MODEL)
           + "Train on the PROVIDED training set. Use the rft training method. " + _TAIL),
    "L2": (_PROBLEM.format(model=_MODEL)
           + "Train on the PROVIDED training set. Choose the training method(s) yourself. " + _TAIL),
    "L3": (_PROBLEM.format(model=_MODEL)
           + "NO training set is provided — acquire and curate the training data yourself. "
           + "Choose the training method(s) yourself. " + _TAIL),
    "L4": (_PROBLEM.format(model="a base model you choose")
           + "NO training set is provided — acquire and curate the training data yourself. "
           + "Choose the training method(s) yourself. Pick the base model yourself. " + _TAIL),
}
# What the historical catalogue rows pinned down. This remains migration
# evidence; it is not the current autonomy ladder.
SHIPPED_SETTINGS = {
    "L1": ("/data/medqa", _MODEL, "rft"),
    "L2": ("/data/medqa", _MODEL, ""),
    "L3": ("", _MODEL, ""),
    "L4": ("", "", ""),
}

# The current ladder releases data first, then method, then model.
LADDER_SETTINGS = {
    "L1": ("/data/medqa", _MODEL, "rft"),
    "L2": ("", _MODEL, "rft"),
    "L3": ("", _MODEL, ""),
    "L4": ("", "", ""),
}


def test_every_level_strips_to_the_same_problem() -> None:
    """If the four rows are one task, they must reduce to one objective."""
    strip = _migration().strip_setting
    stripped = {lvl: strip(obj, SHIPPED_SETTINGS[lvl][1]) for lvl, obj in SHIPPED.items()}
    assert len(set(stripped.values())) == 1, stripped
    problem = next(iter(stripped.values()))
    # The problem must carry no trace of the setting — that is the whole point.
    for word in ("PROVIDED", "yourself", "training method", "Qwen"):
        assert word not in problem, f"{word!r} survived in {problem!r}"
    # …and must still carry the problem and the run policy.
    assert problem.startswith("Fine-tune a model to answer USMLE-style")
    assert problem.endswith(_TAIL)


def test_composed_objective_states_the_setting() -> None:
    strip = _migration().strip_setting
    problem = strip(SHIPPED["L1"], _MODEL)

    ds, model, method = SHIPPED_SETTINGS["L1"]
    l1 = compose_objective(problem, dataset=ds, base_model=model, training_method=method)
    assert "Train on the PROVIDED training set." in l1
    assert "Use the rft training method." in l1
    assert f"Fine-tune {_MODEL}." in l1
    assert l1.startswith(problem)

    l4 = compose_objective(problem, dataset="", base_model="", training_method="")
    assert "NO training set is provided" in l4
    assert "Choose the training method(s) yourself." in l4
    assert "Pick the base model yourself." in l4
    assert "rft" not in l4 and _MODEL not in l4


def test_level_follows_the_setting_not_the_task() -> None:
    for lvl, (ds, model, method) in LADDER_SETTINGS.items():
        assert autonomy_level(ds, model, method) == lvl


def test_non_ladder_setting_is_not_given_a_misleading_level() -> None:
    assert autonomy_level("/data/medqa", _MODEL, "") == "Custom"


def test_downgrade_restores_the_shipped_prose_sentences() -> None:
    """The inverse has to put back what it took out, sentence for sentence."""
    mod = _migration()
    problem = mod.strip_setting(SHIPPED["L1"], _MODEL)
    back = mod.restore_setting(problem, *[SHIPPED_SETTINGS["L1"][i] for i in (0, 1, 2)])
    assert "Train on the PROVIDED training set." in back
    assert "Use the rft training method." in back
    # The model was pinned, so no "pick one yourself" sentence comes back.
    assert "Pick the base model yourself." not in back


def test_user_written_objective_is_left_alone() -> None:
    """A task somebody typed contains none of the catalogue's sentences."""
    mine = "Make the model summarise support tickets in two sentences."
    assert _migration().strip_setting(mine, "") == mine


def test_where_a_run_executes_is_not_part_of_the_setting(monkeypatch) -> None:
    """The GPU provider and the generation_backend are not in the key, and not stored.

    They answer a different question from the rest of a setting: which model,
    which method, which data is a line of attack worth comparing on a board;
    which box you rented today is a fact about a machine. Renting one and using
    your own cluster does not make two experiments out of one.

    Keeping them there also broke launches. A setting recorded before `cloud`
    had a value of its own stored the provider as "", meaning "whatever this
    deployment defaults to". Picking that setting wrote the blank into a
    dropdown that has no blank option, so the browser showed the first one —
    `instance` — while the state held "". The launch went out with no provider,
    the server resolved it from a Vast.ai key being present, and a run that
    looked like it was attaching to the user's own allocation rented a GPU.
    """
    from zevo.api.routers.ui.tasks import setting_identity
    from zevo.db import TaskSetting

    monkeypatch.setenv("ZEVO_DEFAULT_GPU_PROVIDER", "cloud")
    base = {"base_model": "Qwen3-4B", "training_method": "full_sft",
            "validation_set": "trl-lib/Capybara"}
    same = setting_identity(base)
    assert setting_identity({**base, "gpu_provider": "cloud"}) == same
    assert setting_identity({**base, "generation_backend": "vllm"}) == same

    # The fields that DO make an experiment still tell settings apart.
    assert setting_identity({**base, "base_model": "Qwen3-8B"}) != same
    assert setting_identity({**base, "training_method": "lora_sft"}) != same
    assert setting_identity({**base, "metric_direction": "min"}) == same
    assert setting_identity({**base, "dataset": "data.json"}) != same
    assert setting_identity({**base, "validation_set": ""}) != same
    assert setting_identity({**base, "data_query": "prefer curated data"}) != same
    assert setting_identity({**base, "model_query": "prefer a compact model"}) != same
    assert setting_identity({**base, "method_query": "prefer supervised methods"}) != same

    # And a setting cannot carry them at all, so nothing can write a blank one
    # into a form that has no way to show it.
    columns = {c.name for c in TaskSetting.__table__.columns}
    assert "gpu_provider" not in columns
    assert "generation_backend" not in columns
    assert "metric_direction" not in columns
    assert {"data_query", "model_query", "method_query"} <= columns


def test_a_different_budget_is_a_different_setting() -> None:
    """Reported from the launch dialog, and it cost the record.

    Picking the saved `l1-nolimit` (stored cap 0) and typing 60 into Budget
    left the four compared fields untouched, so the form answered "same as the
    saved setting, so nothing new is saved". The run did honour the $60 — but
    the configuration was never written down, and the row kept claiming a cap
    of 0 that no run of it had used.

    A setting STORES its budget. A field the row keeps is a field that tells
    two rows apart, or it is a number nobody maintains.
    """
    from zevo.api.routers.ui.tasks import setting_identity

    nolimit = {"base_model": "Qwen3-0.6B-Base", "training_method": "full_sft"}
    assert setting_identity({**nolimit, "max_cost_usd": 60}) != setting_identity(nolimit)
    assert setting_identity({**nolimit, "iteration_budget": 5}) != setting_identity(nolimit)


def test_absent_and_zero_are_the_same_cap() -> None:
    """Both say "no cap". A form that omits the box and one that sends 0
    describe one setting, and hashing them apart would offer to save a
    duplicate of the row the user just picked."""
    from zevo.api.routers.ui.tasks import setting_identity

    base = {"base_model": "Q", "training_method": "full_sft"}
    assert setting_identity(base) == setting_identity({**base, "max_cost_usd": 0})
    assert setting_identity(base) == setting_identity({**base, "max_cost_usd": ""})
    assert setting_identity(base) == setting_identity({**base, "max_cost_usd": None})


def test_a_row_and_a_form_describing_it_agree() -> None:
    """The row stores a list; a query string sends a comma-joined string.

    They reach the identity as different Python types, and the run-creation
    check compares a `UserRequest` dump against stored rows — so a shape
    mismatch would make every launch look like a new setting.
    """
    from zevo.api.routers.ui.tasks import setting_identity
    from zevo.db import TaskSetting

    row = TaskSetting(
        task_name="capybara", name="s1", base_model="Q", training_method="full_sft",
        dataset="d.json", validation_set="val.json",
        validation_answer_fields=["answer", "turn"],
        iteration_budget=0, max_cost_usd=60.0,
    )
    form = {"base_model": "Q", "training_method": "full_sft", "dataset": "d.json",
            "validation_set": "val.json",
            # As the form sends it: what the user typed, not a parsed list.
            "validation_answer_fields": "answer, turn", "max_cost_usd": "60"}
    assert setting_identity(row) == setting_identity(form)


def test_test_derived_validation_details_do_not_create_duplicate_settings() -> None:
    """Inherited Test values and the newer blank sentinel are one Setting.

    Rows created before the blank request sentinel was introduced persisted
    the inherited answer field and metric. New rows leave those
    request fields blank and resolve them only when the Run starts. With no
    independent Validation set, both representations mean exactly the same
    experiment.
    """
    from zevo.api.routers.ui.tasks import setting_identity

    common = {
        "dataset": "",
        "base_model": "allenai/OLMo-2-0425-1B",
        "training_method": "full_sft",
        "validation_set": "",
    }
    legacy = {
        **common,
        "validation_split": "legacy-split",
        "validation_config": "legacy-config",
        "validation_answer_fields": ["response"],
        "validation_sample_submission": "/old/validation-sample.csv",
        "validation_metric_type": "builtin",
        "validation_metric": "token_f1",
        "validation_metric_direction": "max",
        "validation_evaluation_script": "/old/evaluator.py",
    }
    derived = {
        **common,
        "validation_answer_fields": [],
        "validation_metric_type": "",
        "validation_metric": "",
        "validation_metric_direction": "",
    }

    assert setting_identity(legacy) == setting_identity(derived)

    independent = {**legacy, "validation_set": "/data/validation.json"}
    changed_metric = {**independent, "validation_metric": "accuracy"}
    assert setting_identity(independent) != setting_identity(changed_metric)
