"""Load an agent's identity.md (frontmatter + body) and resolve metadata.

Used by both drivers. Returns an AgentBlueprint with everything a driver
needs to spawn a heartbeat: instructions text, model, tool callables,
output schema class.

The output-schema class lives in `zevo.contracts` (the string-import path in
frontmatter resolves at load time). In-process function tools were removed —
agents work through their driver's own bash/file tools.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import frontmatter
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTS_DIR = REPO_ROOT / "playbook" / "agents"
SHARED_DIR = AGENTS_DIR / "_shared"
# Skills live beside the agent docs under `playbook/skills/<agent>/<skill>/
# SKILL.md`. They are staged into the ticket workspace at run time (see
# `zevo.engine.agent.skills`) and invoked NATIVELY — Claude Code/Codex skill
# discovery, or an in-process driver's `Skill` tool — not `read` from a path
# baked into the prompt.
SKILLS_ROOT = REPO_ROOT / "playbook" / "skills"

# Ticket input formats. This is deliberately separate from Run.mode.
INPUT_FORMAT_TYPED = "typed"
INPUT_FORMAT_FREEFORM = "freeform"

# identity.md carries the agent's frontmatter (metadata) AND its identity prose
# body — it is the per-agent definition file (there is no AGENTS.md anymore).
_IDENTITY_FILE = "identity.md"
# Per-agent doc files folded into the prompt after identity, in this order,
# then the mode-selected invocation doc.
_GOAL_FILE = "goal.md"
_PLATFORM_FILE = "platform.md"
_TYPED_FILE = "typed.md"
_FREEFORM_FILE = "freeform.md"
# Shared across every agent.
_COMMONS_FILE = "commons.md"
# Per-agent skill pool at `playbook/skills/<agent>/<skill>/SKILL.md`. The loader surfaces
# only a compact index into the prompt (name + description); the FULL skill is
# staged into the workspace and invoked natively (Claude Code/Codex discovery
# or an in-process `Skill` tool), never `read` from a baked-in path.


def _read_section(path: Path, title: str) -> str:
    """Return the file's text under a '# <title>' header.

    Missing or empty files RAISE. Every section here is part of the agent's
    contract — a checkout that lacks one would otherwise assemble a prompt
    silently missing, say, the whole Typed Ticket contract, and the agent
    would fail in ways that look like a model problem rather than a deploy
    problem.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError) as e:
        raise FileNotFoundError(
            f"agent doc missing: {path} — the playbook checkout is incomplete"
        ) from e
    if not text:
        raise ValueError(f"agent doc is empty: {path}")
    return f"# {title}\n\n{text}"


@dataclass
class SkillCard:
    """One selectable method in an agent's skill pool.

    A skill is `playbook/skills/<agent>/<skill>/SKILL.md` with `name` + `description`
    frontmatter (the Agent Skills format). `method` is the Skill catalog id
    the orchestrator sets — the skill folder name with hyphens turned back into
    underscores (`lora-sft` ⇄ `lora_sft`). We never ship the implementation; the
    skill body is the contract and the agent writes the code.
    """

    name: str          # skill/folder name, hyphenated (Claude-legal)
    method: str        # Skill catalog id, underscored
    description: str
    path: Path         # the SKILL.md


def _load_skill_pool(agent_dir: Path, allow: list[str]) -> list[SkillCard]:
    """Discover the agent's skill pool from `playbook/skills/<agent>/*/SKILL.md`.

    Every card under `playbook/skills/<agent>/` loads, sorted by name — dropping a new
    SKILL.md in is all it takes. `allow` is the optional `skills:` frontmatter
    list of METHOD ids (underscore, e.g. `lora_sft`); no agent declares one now,
    but when present it narrows and orders the pool and an unknown name raises,
    which is how you pin an agent to a subset. `agent_dir` is
    `playbook/agents/<id>`; skills come from `playbook/skills/<id>`.
    """
    skills_dir = SKILLS_ROOT / agent_dir.name
    if not skills_dir.is_dir():
        if allow and allow != ["all"]:
            raise FileNotFoundError(
                f"agent declares skills {allow} but no skills dir at {skills_dir}"
            )
        return []

    cards: dict[str, SkillCard] = {}   # keyed by method id (underscore)
    for sk in sorted(skills_dir.glob("*/SKILL.md")):
        doc = frontmatter.load(sk)
        fm = dict(doc.metadata)
        name = str(fm.get("name") or sk.parent.name).strip()
        method = name.replace("-", "_")
        cards[method] = SkillCard(
            name=name,
            method=method,
            description=str(fm.get("description") or "").strip(),
            path=sk.resolve(),
        )

    if allow and allow != ["all"]:
        selected: list[SkillCard] = []
        for want in allow:
            key = str(want).replace("-", "_")
            if key not in cards:
                raise KeyError(
                    f"skill {want!r} declared in frontmatter but no "
                    f"{key.replace('_', '-')}/SKILL.md in {skills_dir} "
                    f"(have: {sorted(cards)})"
                )
            selected.append(cards[key])
        return selected

    return list(cards.values())


def _render_skill_pool(cards: list[SkillCard], agent_id: str = "") -> str:
    """Point the agent at its OWN skills — do NOT enumerate them.

    Skills are STAGED for this stage; the Agent discovers them itself (Claude
    Code and Codex surface their native workspace layouts; in-process drivers
    list them via the `Skill` tool) and invokes what fits. We deliberately do
    not list them here so the prompt stays lean and the Agent looks at the real
    installed skills.
    """
    if not cards:
        return ""
    if agent_id == "data":
        return (
            "# Skills\n\n"
            "You have Data method **Skills** installed for this stage. Inspect "
            "your OWN skills and load every one that materially governs the "
            "work:\n"
            "- **Claude Code:** available Skills are surfaced automatically; "
            "invoke the applicable Skills.\n"
            "- **Codex CLI:** available Skills are surfaced automatically from "
            "the Ticket workspace `.agents/skills`; invoke the applicable Skills.\n"
            "- **Other models (bedrock etc.):** call `Skill(action=\"list\")`, "
            "then `Skill(action=\"load\", name=\"<skill>\")`.\n\n"
            "A Data ticket may compose "
            "collection, reformatting, and curation techniques. Inspect the source "
            "and select only necessary conversion or validation-safe curation. "
            "The required `training_method` selects the semantic "
            "record family. Do not expect or invent prompt rendering, loss, or "
            "training/inference hyperparameters. Source and target-size contracts "
            "remain fixed. Write and run the implementation yourself; "
            "report every canonical Skill id actually used in `method_ids`; "
            "put concise human-readable execution details in `audit_steps`."
        )
    if agent_id == "infrastructure":
        return (
            "# Skills\n\n"
            "You have site-operation **Skills** installed for this stage. "
            "Inspect your OWN skills and load the one whose site trigger "
            "matches the typed `ssh_host` or explicitly named cluster:\n"
            "- **Claude Code:** invoke the matching surfaced Skill.\n"
            "- **Codex CLI:** invoke the matching Skill surfaced from the Ticket "
            "workspace `.agents/skills`.\n"
            "- **Other models (bedrock etc.):** call `Skill(action=\"list\")`, "
            "then `Skill(action=\"load\", name=\"<skill>\")`.\n\n"
            "Infrastructure has no `training_method` selector. A site Skill specializes "
            "routing, scheduler, storage, and policy; it must not change the "
            "requested provider or weaken the shared ownership contract. Load "
            "one matching Skill at most. If none matches, use the generic "
            "platform contract; if the site is ambiguous, fail visibly."
        )
    return (
        "# Skills\n\n"
        "You have method **Skills** installed for this stage. Don't rely on this "
        "prompt to enumerate them — look at your OWN skills and pick what fits:\n"
        "- **Claude Code:** your available Skills are surfaced to you "
        "automatically — invoke the one that fits (it loads its full instructions).\n"
        "- **Codex CLI:** your available Skills are surfaced automatically from "
        "the Ticket workspace `.agents/skills` — invoke the one that fits.\n"
        "- **Other models (bedrock etc.):** call `Skill(action=\"list\")` to see "
        "your skills, then `Skill(action=\"load\", name=\"<skill>\")`.\n\n"
        "For Train, select every unpinned `training_method` yourself from the "
        "available compatible Skills; a non-empty `training_method_pin` is binding "
        "and `configuration_suggestions.training_method` is advisory. "
        "Invoke the selected Skill (underscores→hyphens, e.g. `lora_sft` → "
        "`lora-sft`). An unknown selected method is a contract failure. Orchestrator "
        "guidance is advisory unless it carries a user pin. We give the contract; "
        "**you write the code**. Echo the method you used back in "
        "`training_method`. Run ONE only."
    )


def _render_skill_catalog(exclude_id: str) -> str:
    """Render every OTHER agent's skill pool — for the orchestrator's prompt.

    The orchestrator needs to see the full menu. It declares Data's semantic
    method family and may suggest Train's method, while each Specialist chooses
    its own concrete Skills. We list each agent's available method names and
    summaries so the menu is whatever is actually installed.
    """
    blocks: list[str] = []
    for aid in list_agent_ids():
        if aid == exclude_id:
            continue
        idp = AGENTS_DIR / aid / _IDENTITY_FILE
        try:
            fm = dict(frontmatter.load(idp).metadata)
        except (FileNotFoundError, OSError):
            continue
        allow = [str(s) for s in (fm.get("skills") or [])]
        try:
            cards = _load_skill_pool(idp.parent, allow)
        except (KeyError, FileNotFoundError):
            continue
        if not cards:
            continue
        rows = "\n".join(f"  - `{c.method}` — {c.description}" for c in cards)
        if aid == "data":
            selector = (
                "declare one compatible payload `training_method`; Data chooses "
                "and reports its own canonical `method_ids`"
            )
        elif aid == "infrastructure":
            selector = "site matched by typed `ssh_host`; no method selector"
        else:
            selector = (
                "set `training_method_pin` only from a user pin; otherwise give "
                "optional high-level `configuration_suggestions.training_method`"
            )
        blocks.append(f"**{aid}** ({selector}):\n{rows}")
    if not blocks:
        return ""
    intro = (
        "Use the ownership named on each worker's catalog block. Data receives one "
        "method solely to choose required record semantics, then chooses its own "
        "preparation Skills. Train chooses one executable method unless the user "
        "pinned it; Orchestrator advice remains high-level. Infrastructure matches "
        "a site-operation Skill from its typed SSH route."
    )
    return "# Downstream Skill Catalog\n\n" + intro + "\n\n" + "\n\n".join(blocks)


def _assemble_instructions(
    agent_dir: Path, input_format: str, identity_body: str, skills: list[SkillCard]
) -> str:
    """Build the system prompt from the split doc files.

    Order: identity -> goal -> shared commons -> platform -> skill pool ->
    the matching typed/freeform invocation contract. `identity_body` is
    identity.md's prose (its
    frontmatter is stripped by the caller). The skill-pool section is omitted
    for agents with no pool. Typed is the default.
    """
    identity_text = identity_body.strip()
    sections: list[str] = [
        f"# Identity\n\n{identity_text}" if identity_text else "",
        _read_section(agent_dir / _GOAL_FILE, "Goal"),
        _read_section(SHARED_DIR / _COMMONS_FILE, "Platform Runtime (shared)"),
        _read_section(agent_dir / _PLATFORM_FILE, "Platform"),
        _render_skill_pool(skills, agent_dir.name),
    ]
    if input_format == INPUT_FORMAT_FREEFORM:
        sections.append(_read_section(agent_dir / _FREEFORM_FILE, "Freeform Ticket"))
    else:
        sections.append(_read_section(agent_dir / _TYPED_FILE, "Typed Ticket"))

    return "\n\n---\n\n".join(s for s in sections if s)


@dataclass
class AgentBlueprint:
    """Everything a driver needs to invoke an agent."""

    id: str
    name: str
    title: str
    reports_to: str
    default_driver: str
    default_model: str
    output_schema: type[BaseModel] | None
    tools: list[Callable[..., Any]]
    instructions: str
    identity_path: Path
    skills: list[SkillCard] = field(default_factory=list)
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)


def _import_attr(dotted: str) -> Any:
    """Resolve 'pkg.mod:Name' (preferred) or 'pkg.mod.Name'."""
    if ":" in dotted:
        mod_path, _, attr = dotted.partition(":")
    else:
        mod_path, _, attr = dotted.rpartition(".")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def load_agent(
    agent_id: str, input_format: str = INPUT_FORMAT_TYPED
) -> AgentBlueprint:
    """Read playbook/agents/<id>/identity.md (frontmatter + body) and resolve import-paths.

    identity.md is the per-agent definition file: its YAML frontmatter holds the
    metadata (id, name, driver, model, tools, output_schema) and its body is the
    Identity section of the prompt. `input_format` selects which Ticket invocation
    contract is folded into `instructions`. The body is assembled
    from the split docs: identity -> goal -> shared commons -> platform ->
    skill pool -> (typed | freeform). The skill pool is the agent's
    `skills:` frontmatter (method ids) resolved against
    `playbook/skills/<id>/*/SKILL.md`.
    """
    path = AGENTS_DIR / agent_id / _IDENTITY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"identity.md not found for agent '{agent_id}' at {path}"
        )

    doc = frontmatter.load(path)
    fm = dict(doc.metadata)

    output_schema_path: str = fm.get("output_schema", "") or ""
    output_schema: type[BaseModel] | None = None
    if output_schema_path:
        candidate = _import_attr(output_schema_path)
        if not (isinstance(candidate, type) and issubclass(candidate, BaseModel)):
            raise TypeError(
                f"output_schema {output_schema_path!r} for agent {agent_id} "
                "is not a Pydantic BaseModel subclass"
            )
        output_schema = candidate

    # In-process Python tools were removed — every agent runs via a driver
    # (claude_cli / codex_cli / bedrock / openrouter / stub) and does its work with the driver's own bash/file
    # tools, so no agent declares `tools:`. If one ever does, fail loud.
    tool_paths: list[str] = list(fm.get("tools") or [])
    if tool_paths:
        raise ImportError(
            f"agent {agent_id!r} declares tools {tool_paths} but in-process tools "
            f"were removed; use a driver (claude_cli / codex_cli / bedrock / openrouter / stub) instead."
        )
    tools: list[Callable[..., Any]] = []

    # Skill pool: auto-discovered from playbook/skills/<id>/. The optional `skills:`
    # frontmatter narrows and orders it; omitted (the normal case) = take all.
    skill_names: list[str] = [str(s) for s in (fm.get("skills") or [])]
    skills = _load_skill_pool(path.parent, skill_names)

    if input_format not in (INPUT_FORMAT_TYPED, INPUT_FORMAT_FREEFORM):
        raise ValueError(f"unsupported ticket input_format {input_format!r}")
    instructions = _assemble_instructions(
        path.parent, input_format, doc.content, skills
    )
    # An optional supervisor catalogue is retained for deployments that need
    # capability awareness. It is never authority to author specialist values.
    if fm.get("show_skill_catalog"):
        catalog = _render_skill_catalog(exclude_id=fm.get("id", agent_id))
        if catalog:
            instructions = f"{instructions}\n\n---\n\n{catalog}"

    return AgentBlueprint(
        id=fm.get("id", agent_id),
        name=fm.get("name", agent_id),
        title=fm.get("title", ""),
        reports_to=fm.get("reports_to", "") or "",
        # identity.md supplies the shipped default; an explicit CLI argument or
        # non-empty DB row overrides it at runtime.
        default_driver=fm.get("default_driver", "") or "",
        default_model=fm.get("default_model", "") or "",
        output_schema=output_schema,
        tools=tools,
        instructions=instructions,
        identity_path=path,  # now identity.md (the per-agent definition file)
        skills=skills,
        raw_frontmatter=fm,
    )


def list_agent_ids() -> list[str]:
    """Return all agent ids found under playbook/agents/."""
    if not AGENTS_DIR.exists():
        return []
    return sorted(
        p.parent.name
        for p in AGENTS_DIR.glob(f"*/{_IDENTITY_FILE}")
        if (p.parent / _IDENTITY_FILE).exists()
    )
