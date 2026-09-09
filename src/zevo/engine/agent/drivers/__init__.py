"""Driver registry. Pick a driver by name."""
import os
import shutil

from zevo.engine.agent.drivers.base import Driver, DriverRunResult


_KNOWN_DRIVER_NAMES = {
    "claude_cli", "claude-cli", "claude",
    "codex_cli", "codex-cli", "codex",
    "bedrock", "aws", "aws_bedrock",
    "openrouter", "open_router",
    "evaluation_runner", "evaluation-runner",
    "stub",
}


def is_known_driver_name(name: str) -> bool:
    return (name or "").lower().strip() in _KNOWN_DRIVER_NAMES


def get_driver(name: str) -> Driver:
    """Resolve a driver by string name.

    `ZEVO_DEFAULT_DRIVER` env var overrides the empty-string default.

    Available drivers:
      - `claude_cli` -- `claude -p ... --output-format stream-json`
                        subprocess (default). Has bash / edit / read /
                        search built into the subprocess. Auth: Claude
                        Max/Pro OAuth via `claude auth login` (creds at
                        $HOME/.claude/) or CLAUDE_CODE_OAUTH_TOKEN, then
                        ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY fallbacks.
                        The child environment enforces that order.
      - `codex_cli`  -- `codex exec --json` subprocess. Same shape as
                        claude_cli, and the only other driver that can
                        run on a SUBSCRIPTION rather than per-token
                        billing: it uses Codex's complete auth.json login
                        cache. Falls back to OPENAI_API_KEY when absent.
      - `bedrock`    -- AWS Bedrock Converse API via boto3. Covers
                        Claude, Llama, Mistral, Nova, etc.; tools
                        are built into the driver itself.
      - `openrouter` -- OpenRouter chat-completions over HTTP. One key
                        fronts several hundred models, so switching model
                        is a string change. Auth: OPENROUTER_API_KEY.
      - `stub`       -- test-only; returns canned artifacts, no LLM call.
      - `evaluation_runner` -- deterministic system evaluator; Bash invokes
                               the task scorer directly and no LLM is called.

    The CLI driver requires the `claude` binary on PATH; if missing, the
    call raises with a hint to install it or pick another driver explicitly.
    """
    n = (name or "").lower().strip()
    if not n:
        n = os.environ.get("ZEVO_DEFAULT_DRIVER", "claude_cli").lower().strip()

    if n == "stub":
        from zevo.engine.agent.drivers.stub import StubDriver
        return StubDriver()

    if n in ("evaluation_runner", "evaluation-runner"):
        from zevo.engine.agent.drivers.evaluation_runner import EvaluationRunnerDriver
        return EvaluationRunnerDriver()

    if n in ("claude_cli", "claude-cli", "claude"):
        claude_bin = os.environ.get("ZEVO_CLAUDE_BIN", "").strip()
        if not claude_bin and not shutil.which("claude"):
            raise RuntimeError(
                "claude_cli driver requested but the `claude` binary is not "
                "on PATH. Install it (`npm i -g @anthropic-ai/claude-code`), "
                "set ZEVO_CLAUDE_BIN, or pick another driver (--driver bedrock)."
            )
        from zevo.engine.agent.drivers.claude_cli import ClaudeCliDriver
        return ClaudeCliDriver()

    if n in ("codex_cli", "codex-cli", "codex"):
        codex_bin = os.environ.get("ZEVO_CODEX_BIN", "").strip()
        if not codex_bin and not shutil.which("codex"):
            raise RuntimeError(
                "codex_cli driver requested but the `codex` binary is not "
                "on PATH. Install it (`npm i -g @openai/codex`), set "
                "ZEVO_CODEX_BIN, or pick another driver."
            )
        from zevo.engine.agent.drivers.codex_cli import CodexCliDriver
        return CodexCliDriver()

    if n in ("openrouter", "open_router"):
        from zevo.engine.agent.drivers.openrouter_driver import OpenRouterDriver
        return OpenRouterDriver()

    if n in ("bedrock", "aws", "aws_bedrock"):
        from zevo.engine.agent.drivers.bedrock_driver import BedrockDriver
        return BedrockDriver()

    raise ValueError(f"Unknown driver: {name!r}")


__all__ = ["Driver", "DriverRunResult", "get_driver", "is_known_driver_name"]
