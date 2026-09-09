"""Tests for the per-agent skill pool (zevo.engine.agent.loader).

A skill pool is a directory of method cards at
`playbook/skills/<agent-id>/<name>/SKILL.md`, beside `playbook/agents/`. The loader surfaces a
compact INDEX into the prompt rather than the card bodies: an agent reads the one
card it needs when it needs it.

Pinned here:
  - discovery from the top-level pool, sorted, keyed by METHOD id
  - the `skills:` frontmatter as an allowlist AND an ordering
  - a typo in the allowlist fails loud rather than silently dropping a method
  - the rendered section is an index, not the card bodies
  - the real agents load their real pools
  - the orchestrator gets a catalog of every other agent's pool
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zevo.engine.agent import loader as agent_loader
from zevo.engine.agent import skills as agent_skills
from zevo.engine.agent.loader import (
    REPO_ROOT,
    SkillCard,
    _load_skill_pool,
    _render_skill_catalog,
    _render_skill_pool,
    load_agent,
    list_agent_ids,
)


def _write_card(root: Path, agent: str, name: str, *, description: str = "does a thing",
                body: str = "BODY-SENTINEL") -> None:
    """One card on disk, in the layout the loader reads."""
    d = root / agent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f'---\nname: {name}\ndescription: "{description}"\n---\n\n{body}\n',
        encoding="utf-8",
    )


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """A throwaway SKILLS_ROOT. The loader resolves the pool from the repo root,
    not from the agent directory handed to it, so redirecting the module global
    is what isolates a test from the shipped skills."""
    root = tmp_path / "skills"
    monkeypatch.setattr(agent_loader, "SKILLS_ROOT", root)
    return root


# ─────────────────────────── discovery / allowlist ──────────────────────────


def test_auto_discovers_all_cards_sorted_when_no_allowlist(pool, tmp_path):
    for n in ("charlie", "alpha", "bravo"):
        _write_card(pool, "agentx", n)
    cards = _load_skill_pool(tmp_path / "agents" / "agentx", allow=[])
    assert [c.name for c in cards] == ["alpha", "bravo", "charlie"]
    assert all(isinstance(c, SkillCard) for c in cards)


def test_method_id_is_the_hyphenated_name_with_underscores(pool, tmp_path):
    # The canonical method id is `lora_sft`; the directory is `lora-sft`.
    _write_card(pool, "agentx", "lora-sft")
    (card,) = _load_skill_pool(tmp_path / "agents" / "agentx", allow=[])
    assert card.name == "lora-sft"
    assert card.method == "lora_sft"


def test_allowlist_filters_and_preserves_order(pool, tmp_path):
    for n in ("alpha", "bravo", "charlie"):
        _write_card(pool, "agentx", n)
    cards = _load_skill_pool(tmp_path / "agents" / "agentx", allow=["charlie", "alpha"])
    assert [c.name for c in cards] == ["charlie", "alpha"]  # subset + given order


def test_all_token_loads_everything(pool, tmp_path):
    _write_card(pool, "agentx", "alpha")
    _write_card(pool, "agentx", "bravo")
    cards = _load_skill_pool(tmp_path / "agents" / "agentx", allow=["all"])
    assert {c.name for c in cards} == {"alpha", "bravo"}


def test_unknown_skill_in_allowlist_raises(pool, tmp_path):
    # Loud, because the alternative is an agent quietly missing a method it was
    # told it had.
    _write_card(pool, "agentx", "alpha")
    with pytest.raises(KeyError):
        _load_skill_pool(tmp_path / "agents" / "agentx", allow=["alpha", "ghost"])


def test_no_pool_and_no_allowlist_returns_empty(pool, tmp_path):
    assert _load_skill_pool(tmp_path / "agents" / "agentx", allow=[]) == []


def test_no_pool_but_an_allowlist_raises(pool, tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_skill_pool(tmp_path / "agents" / "agentx", allow=["alpha"])


def test_card_fields_parsed_from_frontmatter(pool, tmp_path):
    _write_card(pool, "agentx", "alpha", description="my description")
    (card,) = _load_skill_pool(tmp_path / "agents" / "agentx", allow=[])
    assert card.description == "my description"
    assert card.path.is_absolute() and card.path.name == "SKILL.md"


# ─────────────────────────── index rendering ────────────────────────────────


def test_render_points_at_the_installed_skills_rather_than_listing_them(pool, tmp_path):
    """The prompt announces that a pool exists and how to reach it; it does not
    enumerate. Cards are STAGED into the workspace as real Skills, so the live
    list is whatever is installed — repeating it here would be a second copy to
    keep in sync, and the stale one would win whenever they disagreed."""
    _write_card(pool, "agentx", "alpha", description="do alpha", body="BODY-SENTINEL")
    cards = _load_skill_pool(tmp_path / "agents" / "agentx", allow=[])
    rendered = _render_skill_pool(cards)

    assert "# Skills" in rendered
    assert "look at your OWN skills" in rendered
    assert "`training_method`" in rendered   # how the intended one is named
    assert "`.agents/skills`" in rendered    # Codex's native repo-skill root
    assert "BODY-SENTINEL" not in rendered   # never the card body


def test_render_empty_pool_is_empty_string():
    assert _render_skill_pool([]) == ""


def test_codex_stages_and_cleans_skills_in_agents_layout(pool, tmp_path, monkeypatch):
    _write_card(pool, "infrastructure", "sample-cluster", body="SITE-SENTINEL")
    monkeypatch.setattr(agent_skills, "SKILLS_ROOT", pool)
    workspace = tmp_path / "ticket"

    staged = agent_skills.stage_skills(
        "infrastructure", str(workspace), layout="codex",
    )

    assert staged == workspace / ".agents" / "skills"
    assert (staged / "sample-cluster" / "SKILL.md").is_file()
    assert not (workspace / ".claude").exists()

    agent_skills.unstage_skills(str(workspace), layout="codex")
    assert not (workspace / ".agents").exists()


def test_default_skill_layout_remains_claude_for_native_and_tool_drivers(
    pool, tmp_path, monkeypatch,
):
    _write_card(pool, "train", "full-sft")
    monkeypatch.setattr(agent_skills, "SKILLS_ROOT", pool)
    workspace = tmp_path / "ticket"

    staged = agent_skills.stage_skills("train", str(workspace))

    assert staged == workspace / ".claude" / "skills"
    assert (staged / "full-sft" / "SKILL.md").is_file()
    assert not (workspace / ".agents").exists()


def test_cluster_site_skills_are_shared_with_gpu_stage_agents(
    pool, tmp_path, monkeypatch,
):
    _write_card(pool, "train", "full-sft")
    _write_card(pool, "infrastructure", "cluster-alpha")
    _write_card(pool, "infrastructure", "cluster-beta")
    monkeypatch.setattr(agent_skills, "SKILLS_ROOT", pool)

    train_dir = agent_skills.stage_skills("train", str(tmp_path / "train"))
    infer_dir = agent_skills.stage_skills("inference", str(tmp_path / "infer"))

    assert (train_dir / "full-sft" / "SKILL.md").is_file()
    assert (train_dir / "cluster-alpha" / "SKILL.md").is_file()
    assert (train_dir / "cluster-beta" / "SKILL.md").is_file()
    assert (infer_dir / "cluster-alpha" / "SKILL.md").is_file()
    assert (infer_dir / "cluster-beta" / "SKILL.md").is_file()


# ─────────────────────────── real agents ────────────────────────────────────


def test_train_loads_its_pool():
    bp = load_agent("train")
    methods = {s.method for s in bp.skills}
    # The SFT pair and the preference family are the pool's backbone; asserting
    # the whole list would break every time a method is added, which is a thing
    # that is supposed to be easy.
    assert {"lora_sft", "full_sft", "dpo", "grpo"} <= methods
    assert "# Skills" in bp.instructions
    assert bp.output_schema.__name__ == "TrainResult"


def test_data_loads_its_pool():
    bp = load_agent("data")
    methods = {s.method for s in bp.skills}
    assert {"reformat_csv", "passthrough_jsonl", "reformat_jsonl",
            "pdf_to_qa", "acquire_hf", "synthesize_llm",
            "distill_augment"} <= methods
    assert "# Skills" in bp.instructions
    assert "required `training_method`" in bp.instructions
    assert "Do not expect or invent prompt rendering" in bp.instructions
    assert "report every canonical Skill id" in bp.instructions
    assert "`method_ids`" in bp.instructions
    assert "`audit_steps`" in bp.instructions
    assert bp.output_schema.__name__ == "DataResult"


def test_infrastructure_loads_without_bundled_site_skill(pool):
    bp = load_agent("infrastructure")
    assert bp.skills == []
    assert "# Skills" not in bp.instructions
    assert bp.output_schema.__name__ == "InfraResult"


def test_every_real_card_carries_what_the_catalog_prints():
    for agent_id in ("data", "train", "infrastructure"):
        for card in load_agent(agent_id).skills:
            assert card.description, f"{agent_id}/{card.name} missing description"
            assert card.path.exists()


def test_agents_without_a_pool_still_load():
    """Pools are opt-in: an agent with none gets an empty list and no section."""
    bp = load_agent("inference")
    assert bp.skills == []
    assert "# Skills" not in bp.instructions


def test_evaluation_is_a_system_runner_not_an_llm_agent() -> None:
    assert "evaluation" not in list_agent_ids()
    manifest = REPO_ROOT / "playbook" / "runners" / "evaluation.md"
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert "driver: evaluation_runner" in text
    assert "model: no-llm" in text


# ─────────────────────── orchestrator skill catalog ─────────────────────────


def test_specialists_own_skill_selection_not_orchestrator():
    """The supervisor governs provenance; specialists receive their own menu."""
    orchestrator = load_agent("orchestrator")
    assert "# Downstream Skill Catalog" not in orchestrator.instructions
    assert orchestrator.skills == []
    assert {skill.name for skill in load_agent("train").skills} >= {
        "lora-sft", "dpo", "grpo",
    }
    assert {skill.name for skill in load_agent("data").skills} >= {
        "reformat-csv", "reformat-jsonl",
    }


def test_render_skill_catalog_excludes_self_and_poolless_agents(pool):
    _write_card(pool, "data", "sample-data")
    _write_card(pool, "train", "sample-train")
    _write_card(pool, "infrastructure", "sample-cluster")
    cat = _render_skill_catalog(exclude_id="orchestrator")
    assert "**data**" in cat and "**train**" in cat and "**infrastructure**" in cat
    assert "site matched by typed `ssh_host`" in cat
    assert "**orchestrator**" not in cat   # excluded
    assert "**inference**" not in cat      # no pool → omitted
