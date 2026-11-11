"""Conservative preflight for generated Python, not proof of runtime behavior.

Resolve ordinary imports, aliases and local modules instead of requiring one
source spelling. Constant-dead branches do not supply capability evidence.
Completion still depends on the runtime's receipts and artifact validators.
"""
from __future__ import annotations

import ast
from pathlib import Path


def calls_capability(path: str | Path, capabilities: set[str], *, _seen=None) -> bool:
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        return False
    seen.add(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: dict[str, str] = {}
    modules: set[str] = set()
    found = False

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = qualified(node.value)
            return f"{parent}.{node.attr}" if parent else ""
        return ""

    def visit(node):
        nonlocal found
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
            for child in (node.body if node.test.value else node.orelse):
                visit(child)
            return
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and not node.test.value:
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name] = item.name
                modules.add(item.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = qualified(node.value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = value
        elif isinstance(node, ast.Call):
            name = qualified(node.func)
            if name in capabilities:
                found = True
            elif any(qualified(child) in capabilities and qualified(child).endswith('Callback')
                     for arg in [*node.args, *(k.value for k in node.keywords)]
                     for child in ast.walk(arg)):
                # Trainer accepts either a callback instance or its class.
                found = True
            else:
                # Follow a called helper in a sibling module. Mere imports do
                # not establish that a capability was used.
                for module in modules:
                    if name.startswith(module + "."):
                        local = path.parent.joinpath(*module.split('.')).with_suffix('.py')
                        if local.is_file() and local.parent == path.parent:
                            found |= calls_capability(local, capabilities, _seen=seen)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def capability_problem(path: str, capabilities: set[str], description: str) -> str:
    try:
        return "" if calls_capability(path, capabilities) else description
    except (OSError, SyntaxError) as exc:
        return f"cannot inspect generated implementation: {exc}"


def gpu_launch_problem(body: str, executable: Path) -> str:
    # A GPU allocation is not evidence of a GPU workload: Data curation may
    # be entirely CPU-only. Its durable batch receipts still apply. GPU Data
    # programs must use the runtime, but allocation size cannot classify them.
    if executable.name == "prepare_data.py":
        return ""
    # Shell references are a submission preflight only; registered implementation
    # hashes, process supervision and completion receipts remain authoritative.
    import shlex
    try:
        tokens = shlex.split(body, comments=True)
    except ValueError:
        # A valid shell heredoc can contain Python prose with unmatched shell
        # quotes. Never reject that shell program as malformed Python evidence.
        tokens = []
        for line in body.splitlines():
            try:
                tokens.extend(shlex.split(line, comments=True))
            except ValueError:
                continue
    launchers = {"zevo_gpu_runtime.py", "zevo_parallel_inference.py", "zevo_adaptive_training.py"}
    if any(Path(token).name in launchers for token in tokens):
        return ""
    return capability_problem(str(executable),
        {"zevo_gpu_runtime.run", "zevo_gpu_runtime.launch"},
        "GPU workload must launch through the system GPU runtime CLI or Python run/launch API")
