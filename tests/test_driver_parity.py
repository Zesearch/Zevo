"""Driver parity tests — pin the contract every driver MUST satisfy.

Four LLM drivers exist (claude_cli, codex_cli, bedrock, openrouter), plus the
deterministic `evaluation_runner` and test-only `stub`. They share the runner
interface, but
each has its own prompt rendering, usage normalisation, and event
normalisation. These tests pin:

  1. Each class imports + exposes a .name and an async run_agent().
  2. Each driver's _render_* function (now via _prompt.render_*) is a
     pure function that returns a non-empty string containing the
     payload + AGENTS body (for the CLI driver) and the schema name.
  3. Each driver's usage-mapping function converts its provider-specific
     keys (cache_read_input_tokens vs cacheReadInputTokens, etc.) into
     the harness canonical dict shape that pricing.estimate_cost expects.
  4. The shared _prompt.render_agent_prompt produces deterministic
     output for the same inputs (no UUIDs, no timestamps).

No live API calls. No subprocess invocations. Pure-Python structural
verification so the suite stays in CI and fails the SECOND any driver
drifts.
"""
from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import BaseModel

from zevo.engine.agent.loader import AgentBlueprint
from zevo.engine.agent.drivers import _prompt
from zevo.engine.agent.drivers.base import Driver
from zevo.engine.agent.drivers.bedrock_driver import (
    BedrockDriver,
    _map_usage as bedrock_map_usage,
    _render_user_text as bedrock_render,
)
from zevo.engine.agent.drivers.claude_cli import (
    ClaudeCliDriver,
    _normalise_usage as claude_normalise,
    _render_prompt as claude_render,
    _resolve_auth as claude_resolve_auth,
)
from zevo.engine.agent.drivers.codex_cli import (
    CodexCliDriver,
    _CodexTranscriptGate as CodexTranscriptGate,
    _codex_output_schema as codex_output_schema,
    _final_message as codex_final_message,
    _line_to_events as codex_line_to_events,
    _resolve_auth as codex_resolve_auth,
)
from zevo.engine.agent.drivers.openrouter_driver import (
    OpenRouterDriver,
    _map_usage as openrouter_map_usage,
    _to_openai_tool,
)
from zevo.engine.agent.drivers.evaluation_runner import EvaluationRunnerDriver
from zevo.engine.agent.drivers._agent_loop import BASH_TOOL, SKILL_TOOL
from zevo.engine.agent.drivers._json_utils import extract_trailing_json
from zevo.contracts.data import (
    DataRecipe,
    DataTaskInput,
    DataResult,
)
from zevo.contracts.inference import InferenceResult
from zevo.contracts.infrastructure import InfraResult
from zevo.contracts.model_registry import RegisterResult
from zevo.contracts.orchestrator import SupervisorAction
from zevo.contracts.train import TrainResult


def test_structured_output_extractor_keeps_nested_data_result_outermost() -> None:
    """Nested memory details must never shadow their containing DataResult."""
    result = {
        "status": "succeeded",
        "ticket_id": "data-001",
        "operation": "prepare_run_data",
        "error_message": "",
        "notes": "prepared and verified",
        "training_dataset_path": "/tmp/dataset.jsonl",
        "data_recipe_path": "/tmp/data_recipe.json",
        "memory_updates": [{
            "kind": "verified_fact",
            "key": "source.shape",
            "summary": "Source records use messages.",
            "details": [
                {"key": "record_family", "value": "messages"},
                {"key": "columns", "value": "messages"},
            ],
            "visibility": "agent_local",
        }],
        "n_rows_in": 10,
        "n_rows_out": 10,
    }
    import json

    text = "Work complete.\n\n```json\n" + json.dumps(result) + "\n```"
    extracted = extract_trailing_json(
        text, min_keys=2, output_schema=DataResult,
    )

    assert extracted == result
    assert DataResult.model_validate(extracted).operation == "prepare_run_data"


# ─────────────────────────── fixtures ────────────────────────────────────────


@pytest.fixture
def blueprint() -> AgentBlueprint:
    return AgentBlueprint(
        id="test-agent", name="Test Agent", title="Test", reports_to="",
        default_driver="claude_cli", default_model="claude-opus-4-7",
        instructions=(
            "# Test Agent\n\n"
            "You are a test agent. Do the thing."
        ),
        output_schema=DataResult,
        tools=[], identity_path="",
    )


@pytest.fixture
def payload() -> DataTaskInput:
    # Field names here drifted from the schema at some point and every test
    # that takes this fixture has been erroring at setup since, which is
    # invisible in a suite that already reports errors for missing deps.
    return DataTaskInput(
        ticket_id="test-001",
        operation="prepare_run_data",
        run_id="run-001",
        dataset_source="/tmp/x.jsonl",
        dataset="/tmp/x.jsonl",
        data_query="",
        expected_source_identity="/tmp/x.jsonl",
        training_method="full_sft",
        data_intent_signature="a" * 64,
        data_recipe_schema=DataRecipe.model_json_schema(),
        data_recipe_validation_command=(
            "python -m zevo.contracts.data validate-recipe <recipe> <dataset>"
        ),
        artifacts_validation_command=(
            "python -m zevo.engine.artifact_validation validate-training-data <outputs>"
        ),
        work_dir="/tmp/work",
    )


# ─────────────────────── 1. Protocol surface  ────────────────────────────────


@pytest.mark.parametrize("driver_cls,expected_name", [
    (ClaudeCliDriver, "claude_cli"),
    (CodexCliDriver, "codex_cli"),
    (BedrockDriver, "bedrock"),
    (OpenRouterDriver, "openrouter"),
    (EvaluationRunnerDriver, "evaluation_runner"),
])
def test_driver_exposes_name_and_async_run_agent(driver_cls, expected_name) -> None:
    """Every driver MUST have a .name str class attr AND an async
    run_agent method matching the Protocol signature. If you add a new
    driver and forget either, this fails."""
    assert hasattr(driver_cls, "name")
    assert driver_cls.name == expected_name

    assert hasattr(driver_cls, "run_agent")
    sig = inspect.signature(driver_cls.run_agent)
    params = set(sig.parameters)
    # Every driver's run_agent must accept the canonical kwargs the
    # runner passes.
    for required in ("blueprint", "input_payload", "workspace_dir",
                     "stdout_sink", "event_sink", "conversation",
                     "model_override"):
        assert required in params, (
            f"{driver_cls.__name__}.run_agent is missing kwarg {required!r}"
        )

    # Must be async (the runner awaits it).
    assert inspect.iscoroutinefunction(driver_cls.run_agent), (
        f"{driver_cls.__name__}.run_agent must be async"
    )


def test_driver_protocol_is_satisfied_structurally() -> None:
    """Both classes have the attributes Driver Protocol requires. Driver
    is a plain Protocol (not @runtime_checkable), so we check by attribute
    name rather than isinstance — same guarantee for the runner's purposes
    (it just calls .run_agent + reads .name)."""
    for cls in (ClaudeCliDriver, BedrockDriver, EvaluationRunnerDriver):
        inst = cls()  # no network/IO side effects at construction
        assert hasattr(inst, "name"), f"{cls.__name__} missing .name"
        assert hasattr(inst, "run_agent"), f"{cls.__name__} missing .run_agent"
        assert callable(inst.run_agent), f"{cls.__name__}.run_agent not callable"


# ───────────────────── 2. Prompt rendering parity ────────────────────────────


def test_claude_cli_uses_shared_render_agent_prompt(blueprint, payload) -> None:
    """The CLI driver renders via the shared _prompt module: AGENTS body
    + payload block + do-the-work block + inlined schema."""
    claude_out = claude_render(blueprint, payload, conversation=None)

    assert claude_out.startswith("# Test Agent"), "claude prompt missing AGENTS body"
    assert "Ticket payload (DataTaskInput)" in claude_out
    assert '"dataset": "/tmp/x.jsonl"' in claude_out
    assert "DO THE WHOLE TASK" in claude_out
    assert "DataResult" in claude_out
    assert "HARD RULES:" in claude_out          # strict_json=True
    assert '"properties"' in claude_out         # claude inlines the schema body


def test_bedrock_sdk_uses_shared_render_user_message(blueprint, payload) -> None:
    """Bedrock (SDK driver) renders the user message via _prompt — it does
    NOT include the AGENTS body (that's the system prompt for SDK drivers),
    but still carries the payload + schema instruction."""
    bed = bedrock_render(payload, blueprint.output_schema, conversation=None)
    assert not bed.startswith("# Test Agent")
    assert "Ticket payload (DataTaskInput)" in bed
    assert "DataResult" in bed


def test_conversation_block_appears_when_provided(blueprint, payload) -> None:
    conv = [
        {"author": "user", "body": "actually change train_size to 50"},
        {"author": "data", "body": "ack — re-doing"},
    ]
    out = claude_render(blueprint, payload, conversation=conv)
    assert "CONVERSATION SO FAR" in out
    assert "actually change train_size to 50" in out
    assert "**[user]**" in out


def test_render_agent_prompt_is_deterministic(blueprint, payload) -> None:
    """Same inputs → byte-identical output, twice."""
    a = _prompt.render_agent_prompt(blueprint, payload, conversation=None,
                                    strict_json=True, schema_inline=True)
    b = _prompt.render_agent_prompt(blueprint, payload, conversation=None,
                                    strict_json=True, schema_inline=True)
    assert a == b


def test_no_schema_omits_required_output_block(payload) -> None:
    """When the agent has no output_schema (e.g. orchestrator's freeform
    branch), the REQUIRED FINAL OUTPUT block is omitted."""
    bp_noschema = AgentBlueprint(
        id="x", name="x", title="x", reports_to="", default_driver="claude_cli",
        default_model="claude-opus-4-7", instructions="hi", output_schema=None,
        tools=[], identity_path="",
    )
    out = _prompt.render_agent_prompt(bp_noschema, payload)
    assert "REQUIRED FINAL OUTPUT" not in out
    assert "HARD RULES:" not in out


# ───────────────────── 3. Usage normalisation parity ─────────────────────────


def test_usage_mappers_produce_canonical_keys() -> None:
    """Every usage mapper must produce a dict with EXACTLY these keys:
    input_tokens, output_tokens, cached_input_tokens, reasoning_output_tokens.
    The pricing module then accumulates + estimates from this shape."""
    CANON = {"input_tokens", "output_tokens", "cached_input_tokens",
             "reasoning_output_tokens"}

    # Claude CLI (stream-json: cache_read_input_tokens + cache_creation_input_tokens)
    claude_raw = {
        "input_tokens": 100, "output_tokens": 50,
        "cache_read_input_tokens": 30, "cache_creation_input_tokens": 5,
    }
    claude_out = claude_normalise(claude_raw)
    assert set(claude_out.keys()) == CANON
    # Claude CLI sums cache_read + cache_creation into cached_input_tokens
    # AND adds them into input_tokens (so cost subtraction works).
    assert claude_out["cached_input_tokens"] == 35
    assert claude_out["input_tokens"] == 135

    # Bedrock (camelCase: inputTokens / cacheReadInputTokens)
    bed_raw = {"inputTokens": 300, "outputTokens": 90, "cacheReadInputTokens": 80}
    bed_out = bedrock_map_usage(bed_raw)
    assert set(bed_out.keys()) == CANON
    assert bed_out["cached_input_tokens"] == 80
    assert bed_out["input_tokens"] == 300


def test_usage_mappers_handle_missing_or_none() -> None:
    """All mappers must defend against None / missing keys without
    raising — the runner can't recover from a normaliser crash."""
    assert claude_normalise({}) == {
        "input_tokens": 0, "output_tokens": 0,
        "cached_input_tokens": 0, "reasoning_output_tokens": 0,
    }
    assert bedrock_map_usage(None) == {}
    assert bedrock_map_usage({}) == {
        "input_tokens": 0, "output_tokens": 0,
        "cached_input_tokens": 0, "reasoning_output_tokens": 0,
    }


# ───────────────────── 4. Registry resolves the drivers ──────────────────────


def test_get_driver_resolves_known_names() -> None:
    """The registry MUST know LLM, deterministic, and test drivers."""
    from zevo.engine.agent.drivers import get_driver
    for name in ("claude_cli", "bedrock", "evaluation_runner", "stub"):
        try:
            d = get_driver(name)
            assert d.name in (name, name.replace("_", "-"))
        except (FileNotFoundError, RuntimeError) as e:
            # Acceptable: binary / lib missing in test env. The point is
            # the registry didn't raise "Unknown driver".
            assert "Unknown driver" not in str(e)


def test_get_driver_rejects_unknown_name() -> None:
    from zevo.engine.agent.drivers import get_driver
    with pytest.raises(Exception, match=r"[Uu]nknown driver"):
        get_driver("totally-made-up-driver")


# ─────────────────── 5. codex_cli event + auth mapping ───────────────────────


@pytest.mark.parametrize("raw,expected", [
    ({"msg": {"type": "agent_message", "message": "hi"}}, "agent_message"),
    ({"type": "item.completed", "item": {
        "id": "item_0", "type": "agent_message", "text": "hi",
    }}, "agent_message"),
    ({"msg": {"type": "exec_command_begin", "command": ["bash", "-lc", "ls"]}}, "tool_call"),
    ({"type": "item.completed", "item": {
        "id": "item_1", "type": "command_execution", "command": "ls",
        "aggregated_output": "file.txt\n", "exit_code": 0,
    }}, "tool_call"),
    ({"msg": {"type": "agent_reasoning", "text": "thinking"}}, "reasoning"),
    ({"type": "turn.completed", "usage": {
        "input_tokens": 10, "output_tokens": 2,
    }}, "turn_completed"),
    ({"msg": {"type": "error", "message": "boom"}}, "error"),
    # An event Codex adds later must still reach the transcript.
    ({"msg": {"type": "invented_next_release"}}, "raw"),
])
def test_codex_events_map_onto_the_harness_contract(raw, expected) -> None:
    import json as _json
    events = codex_line_to_events(_json.dumps(raw))
    assert [e["type"] for e in events] == [expected]


@pytest.mark.parametrize("event_factory", [
    lambda text: {"msg": {"type": "agent_message", "message": text}},
    lambda text: {"type": "item.completed", "item": {
        "id": "item_0", "type": "agent_message", "text": text,
    }},
])
def test_codex_result_shaped_intermediate_message_is_omitted(event_factory) -> None:
    """A provisional Result is neither progress nor a Ticket result."""
    import json as _json

    provisional = _json.dumps({
        "status": "failed",
        "ticket_id": "infra-001",
        "error_message": "Checking idle GPUs before acquiring the lease.",
        "notes": "",
    })
    gate = CodexTranscriptGate()
    events = []
    for event in codex_line_to_events(_json.dumps(event_factory(provisional))):
        events.extend(gate.feed(event))
    # Seeing later execution proves that the pending assistant message was an
    # intermediate pseudo-Result rather than the final structured Result.
    for event in codex_line_to_events(_json.dumps({
        "type": "item.started",
        "item": {"type": "command_execution", "command": "nvidia-smi"},
    })):
        events.extend(gate.feed(event))

    assert [event["type"] for event in events] == ["tool_call"]


def test_codex_preserves_plain_intermediate_progress() -> None:
    gate = CodexTranscriptGate()
    assert gate.feed({
        "type": "agent_message",
        "payload": {"message": "Checking idle GPUs."},
    }) == []
    visible = gate.feed({
        "type": "tool_call",
        "payload": {"name": "run_bash", "phase": "begin"},
    })
    assert [event["type"] for event in visible] == [
        "agent_message", "tool_call",
    ]
    assert visible[0]["payload"]["message"] == "Checking idle GPUs."


def test_codex_empty_final_result_narration_is_not_duplicated_in_transcript() -> None:
    import json as _json

    result = _json.dumps({
        "status": "succeeded",
        "ticket_id": "infra-001",
        "error_message": "",
        "notes": "",
    })
    raw = {"type": "item.completed", "item": {
        "id": "item_0", "type": "agent_message", "text": result,
    }}
    gate = CodexTranscriptGate()
    visible = []
    for event in codex_line_to_events(_json.dumps(raw)):
        visible.extend(gate.feed(event))
    assert visible == []
    for event in codex_line_to_events(_json.dumps({
        "type": "turn.completed", "usage": {"input_tokens": 10},
    })):
        visible.extend(gate.feed(event))
    assert [event["type"] for event in visible] == ["turn_completed"]


@pytest.mark.parametrize("status,message", [
    ("failed", "Checking idle GPUs before acquiring the lease."),
    ("failed", "Cannot execute shell commands because the response format was forced."),
    ("succeeded", "Provisioning complete, but performing one more verification."),
])
def test_codex_omits_every_intermediate_result_envelope(status, message) -> None:
    import json as _json

    provisional = _json.dumps({
        "status": status,
        "ticket_id": "infra-001",
        "error_message": message,
        "notes": "",
    })
    gate = CodexTranscriptGate()
    assert gate.feed({
        "type": "agent_message", "payload": {"message": provisional},
    }) == []
    visible = gate.feed({
        "type": "tool_call",
        "payload": {"name": "run_bash", "phase": "begin"},
    })
    assert [event["type"] for event in visible] == ["tool_call"]


def test_codex_preserves_real_command_failure_event() -> None:
    gate = CodexTranscriptGate()
    failure = {
        "type": "tool_call",
        "payload": {
            "name": "run_bash", "phase": "end",
            "command": "false", "exit_code": 1,
        },
    }
    assert gate.feed(failure) == [failure]


def test_codex_final_failure_is_parsed_but_not_shown_as_progress() -> None:
    import json as _json

    result = _json.dumps({
        "status": "failed",
        "ticket_id": "infra-001",
        "error_message": "No idle GPU is available.",
        "notes": "",
    })
    raw_message = _json.dumps({
        "type": "item.completed",
        "item": {"type": "agent_message", "text": result},
    })
    raw_done = _json.dumps({"type": "turn.completed", "usage": {}})
    stream = "\n".join([raw_message, raw_done])

    gate = CodexTranscriptGate()
    visible = []
    for line in stream.splitlines():
        for event in codex_line_to_events(line):
            visible.extend(gate.feed(event))

    assert [event["type"] for event in visible] == ["turn_completed"]
    assert _json.loads(codex_final_message(stream))["status"] == "failed"


def test_codex_keeps_stdout_markers() -> None:
    """Execution markers drive the live progress bar, and they
    arrive as plain stdout lines rather than as Codex events."""
    assert codex_line_to_events("__PHASE__:train-001:training@100")[0]["type"] == "phase"
    assert codex_line_to_events(
        "__ATTEMPT__:train-001:550e8400-e29b-41d4-a716-446655440000@100"
    )[0]["type"] == "attempt"
    assert codex_line_to_events("")==[]


def test_codex_final_message_is_the_last_agent_message() -> None:
    """The typed Result is parsed out of the final assistant message, so
    picking the wrong one silently fails every typed ticket."""
    import json as _json
    stream = "\n".join([
        _json.dumps({"msg": {"type": "agent_message", "message": "first"}}),
        _json.dumps({"msg": {"type": "exec_command_end", "exit_code": 0}}),
        _json.dumps({"msg": {"type": "agent_message", "message": "last"}}),
    ])
    assert codex_final_message(stream) == "last"


def test_codex_final_message_supports_item_event_stream() -> None:
    """Codex 0.147 emits the final assistant text as item.completed."""
    import json as _json
    stream = "\n".join([
        _json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
        _json.dumps({"type": "item.completed", "item": {
            "id": "item_0", "type": "agent_message", "text": "final-json",
        }}),
        _json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 10, "output_tokens": 2,
        }}),
    ])
    assert codex_final_message(stream) == "final-json"


@pytest.mark.parametrize("output_model", [
    SupervisorAction,
    DataResult,
    InferenceResult,
    InfraResult,
    RegisterResult,
    TrainResult,
])
def test_codex_output_schema_requires_every_property_recursively(
    output_model: type[BaseModel],
) -> None:
    """Responses strict schemas require defaults in `required` too."""
    schema = codex_output_schema(output_model)

    def assert_strict(node) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                assert node.get("required") == list(properties)
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            for value in node.values():
                assert_strict(value)
        elif isinstance(node, list):
            for value in node:
                assert_strict(value)

    assert_strict(schema)
    assert set(schema.get("properties", {})) == set(schema.get("required", []))


def test_codex_output_schema_rejects_open_mappings_locally() -> None:
    class OpenMappingResult(BaseModel):
        details: dict[str, Any]

    with pytest.raises(ValueError, match="open-ended object"):
        codex_output_schema(OpenMappingResult)


def test_codex_prefers_the_login_cache_over_the_api_key(tmp_path) -> None:
    """A complete ChatGPT login cache must win over the usage-billed key."""
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"auth_mode":"chatgpt","tokens":{"access_token":"test"}}'
    )

    assert codex_resolve_auth({
        "CODEX_HOME": str(codex_home), "OPENAI_API_KEY": "k",
    })[0] == "chatgpt_plan"
    assert codex_resolve_auth({
        "CODEX_HOME": str(tmp_path / "missing"), "OPENAI_API_KEY": "k",
    })[0] == "api_key"
    assert codex_resolve_auth({"HOME": "/nonexistent"})[0] == "none"


def test_claude_subscription_auth_removes_billing_credentials(tmp_path) -> None:
    env = {
        "HOME": str(tmp_path),
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth",
        "ANTHROPIC_AUTH_TOKEN": "bearer",
        "ANTHROPIC_API_KEY": "key",
    }
    assert claude_resolve_auth(env)[0] == "oauth_token"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_claude_oauth_session_wins_and_sanitizes_child_env(tmp_path) -> None:
    credentials = tmp_path / ".claude"
    credentials.mkdir()
    (credentials / ".credentials.json").write_text("{}")
    env = {
        "HOME": str(tmp_path),
        "ANTHROPIC_AUTH_TOKEN": "bearer",
        "ANTHROPIC_API_KEY": "key",
    }
    assert claude_resolve_auth(env)[0] == "oauth_session"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_claude_bearer_token_wins_over_api_key(tmp_path) -> None:
    env = {
        "HOME": str(tmp_path),
        "ANTHROPIC_AUTH_TOKEN": "bearer",
        "ANTHROPIC_API_KEY": "key",
    }
    assert claude_resolve_auth(env)[0] == "auth_token"
    assert "ANTHROPIC_AUTH_TOKEN" in env
    assert "ANTHROPIC_API_KEY" not in env


def test_codex_api_key_cache_is_not_reported_as_a_chatgpt_plan(tmp_path) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"auth_mode":"apikey","OPENAI_API_KEY":"test"}'
    )
    assert codex_resolve_auth({"CODEX_HOME": str(codex_home)})[0] == "api_key"


def test_codex_invalid_cache_blocks_silent_billing_fallback(tmp_path) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text("{}")
    assert codex_resolve_auth({
        "CODEX_HOME": str(codex_home), "OPENAI_API_KEY": "test",
    })[0] == "invalid"


# ─────────────────── 6. openrouter tool + usage mapping ──────────────────────


@pytest.mark.parametrize("tool", [BASH_TOOL, SKILL_TOOL])
def test_openrouter_wraps_shared_tools_as_openai_functions(tool) -> None:
    """The shared tool specs are the harness's own shape; OpenRouter
    speaks OpenAI's. A dropped schema here means the model calls a tool
    with no arguments."""
    t = _to_openai_tool(tool)
    assert t["type"] == "function"
    assert t["function"]["name"] == tool["name"]
    assert t["function"]["parameters"] == tool["input_schema"]


def test_openrouter_usage_maps_to_the_canonical_shape() -> None:
    """pricing.estimate_cost reads these four keys; anything OpenRouter
    nests has to be flattened into them or the run costs $0."""
    assert openrouter_map_usage(None) == {}
    got = openrouter_map_usage({
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 30},
        "completion_tokens_details": {"reasoning_tokens": 5},
    })
    assert got == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_input_tokens": 30,
        "reasoning_output_tokens": 5,
    }


def test_openrouter_missing_key_names_the_fix() -> None:
    """A driver that cannot run should say which variable to set and
    where to get one, not raise a KeyError from inside a request."""
    import os as _os
    saved = _os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            OpenRouterDriver._api_key()
    finally:
        if saved is not None:
            _os.environ["OPENROUTER_API_KEY"] = saved
