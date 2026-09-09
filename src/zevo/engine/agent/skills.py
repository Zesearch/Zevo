"""Per-stage skill staging.

Skills live in the repo at `playbook/skills/<agent>/<skill>/SKILL.md` (one folder per
skill, `SKILL.md` with `name` + `description` frontmatter — the Agent Skills /
Claude Code format). They are NOT baked into an agent's prompt; instead, right
before a driver runs an agent, that agent's ZEVO method skills are STAGED into
the ticket's workspace and removed afterwards. This scopes the Playbook-owned
method catalogue to the current stage. A CLI may also expose its own globally
installed built-ins; those are driver capabilities, not ZEVO method skills.

Native CLIs require different repository layouts:

  - claude_cli: Claude Code discovers `<cwd>/.claude/skills/` natively when its
    subprocess cwd is the workspace (`--add-dir` does NOT surface skills, so the
    driver must set cwd). The agent then invokes them via the native Skill tool.
  - codex_cli: Codex discovers repository skills from
    `<cwd>/.agents/skills/`; `.claude/skills/` is not a Codex discovery root.
  - bedrock/openrouter: our in-process loops can't auto-discover, so they expose
    a `Skill` tool that lists + loads from the Claude-layout staged dir.

Staging is per-driver-run (one heartbeat). It's idempotent and concurrency-safe:
each heartbeat has its own workspace, so staged dirs never collide.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import frontmatter

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILLS_ROOT = REPO_ROOT / "playbook" / "skills"

SkillLayout = Literal["claude", "codex"]

# Driver-native discovery roots inside one Ticket workspace. Bedrock and
# OpenRouter use the Claude layout as storage for Zevo's explicit Skill tool.
_STAGE_SUBPATHS: dict[SkillLayout, tuple[str, str]] = {
    "claude": (".claude", "skills"),
    "codex": (".agents", "skills"),
}


def _stage_path(workspace_dir: str, layout: SkillLayout) -> Path:
    try:
        subpath = _STAGE_SUBPATHS[layout]
    except KeyError as exc:  # defensive for untyped external callers
        raise ValueError(f"unknown skill layout: {layout!r}") from exc
    return Path(workspace_dir).joinpath(*subpath)


def agent_skill_dirs(agent_id: str) -> list[Path]:
    """Every Skill staged for an agent, including shared cluster site rules."""
    owners = [agent_id]
    # Cluster jobs now belong to the consuming stage. Those agents therefore
    # need the same site-specific Slurm constraints Infrastructure used while
    # resolving the access/resource plan.
    if agent_id in {"train", "inference"}:
        owners.append("infrastructure")
    dirs: list[Path] = []
    for owner in owners:
        base = SKILLS_ROOT / owner
        if base.is_dir():
            dirs.extend(
                d for d in sorted(base.iterdir())
                if (d / "SKILL.md").is_file()
            )
    return dirs


def stage_skills(
    agent_id: str,
    workspace_dir: str,
    *,
    layout: SkillLayout = "claude",
) -> Path | None:
    """Copy an Agent's skills into its driver's native workspace layout.

    Returns the staged skills directory, or None if the Agent has no skills.
    Overwrites any prior staged copy (idempotent across re-wakes)."""
    dirs = agent_skill_dirs(agent_id)
    if not dirs:
        return None
    dest = _stage_path(workspace_dir, layout)
    dest.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        target = dest / d.name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(d, target)
    return dest


def unstage_skills(
    workspace_dir: str,
    *,
    layout: SkillLayout = "claude",
) -> None:
    """Remove staged skills and their empty driver-specific parent directory."""
    dest = _stage_path(workspace_dir, layout)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    parent = dest.parent
    try:
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


# ---- helpers for the bedrock Skill tool (reads the staged dir) ----


def _parse_skill(md_path: Path) -> tuple[str, str, str]:
    """(name, description, body) parsed from a SKILL.md."""
    doc = frontmatter.load(md_path)
    fm = dict(doc.metadata)
    name = str(fm.get("name") or md_path.parent.name).strip()
    return name, str(fm.get("description") or "").strip(), doc.content.strip()


def list_staged_skills(staged_dir: Path | None) -> list[dict[str, str]]:
    """[{name, description}] for each staged skill — for the Skill tool's list."""
    if not staged_dir or not Path(staged_dir).is_dir():
        return []
    out: list[dict[str, str]] = []
    for d in sorted(Path(staged_dir).iterdir()):
        sk = d / "SKILL.md"
        if not sk.is_file():
            continue
        name, desc, _ = _parse_skill(sk)
        out.append({"name": name, "description": desc})
    return out


def load_staged_skill(staged_dir: Path | None, name: str) -> str:
    """Full instructions body of one staged skill, or '' if not found.

    Accepts the skill name in either hyphen (`lora-sft`) or underscore
    (`lora_sft`, the canonical method id) form."""
    if not staged_dir or not name:
        return ""
    want = {name, name.replace("_", "-"), name.replace("-", "_")}
    for d in Path(staged_dir).iterdir():
        if d.name in want and (d / "SKILL.md").is_file():
            _, _, body = _parse_skill(d / "SKILL.md")
            return body
    return ""
