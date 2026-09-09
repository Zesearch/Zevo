"""Classify a terminal Ticket for an explicit operator rerun.

Automatic repair ownership lives in :mod:`failure_policy`. This module runs
only after that bounded lifecycle has ended: it tells the operator whether an
unchanged manual rerun is sensible and supplies the explanation/recovery shown
by the API, CLI, and web UI.

The catalog deliberately lives beside the executable regexes. A classifier
rule and its operator guidance therefore cannot drift apart or depend on a
Markdown parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


Verdict = Literal["transient", "structural", "cancelled", "unknown"]


@dataclass(frozen=True)
class FailureMode:
    code: str
    verdict: Verdict
    retryable: bool
    pattern: re.Pattern[str]
    reason: str
    description: str
    recovery: str


@dataclass(frozen=True)
class RetryClassification:
    verdict: Verdict
    retryable: bool
    reason: str
    code: str = "unknown"
    matched_pattern: str = ""
    description: str = ""
    recovery: str = ""


FAILURE_MODES: tuple[FailureMode, ...] = (
    FailureMode(
        code="cancelled_by_user",
        verdict="cancelled",
        retryable=True,
        pattern=re.compile(r"cancelled by user|SIGTERM|received SIGTERM", re.I),
        reason="ticket was cancelled by the user",
        description="The operator stopped this Ticket or its process received SIGTERM.",
        recovery="Rerun only when the cancellation was unintended or the operator now wants the work to continue.",
    ),
    FailureMode(
        code="result_schema_validation",
        verdict="structural",
        retryable=False,
        pattern=re.compile(r"ValidationError:.*for \w+Result", re.I | re.S),
        reason="the Agent did not satisfy its typed final Result after automatic repair",
        description="The final structured result is missing required fields or contains invalid keys or values.",
        recovery="Inspect the last transcript and the exact Result schema. Correct the producing prompt or implementation before forcing a rerun.",
    ),
    FailureMode(
        code="specialist_contract_validation",
        verdict="structural",
        retryable=False,
        pattern=re.compile(
            r"INVALID (?:Inference|Train)RunConfig|"
            r"runtime experiment contract violated: (?!inference_data_profile)|"
            r"reported success but .* differs|"
            r"(?:configuration|registry entry|artifact) schema is invalid",
            re.I | re.S,
        ),
        reason="a deterministic Specialist configuration or artifact check failed after automatic repair",
        description="A generated YAML, declared result, or committed artifact disagrees with its machine-readable authority.",
        recovery="Use the exact validator error and supplied schema to correct the producing Specialist or its upstream work order; do not infer keys from older Runs.",
    ),
    FailureMode(
        code="inference_data_profile",
        verdict="structural",
        retryable=False,
        pattern=re.compile(
            r"invalid inference_data_profile|inference_data_profile(?:\.[A-Za-z0-9_]+|\b)",
            re.I,
        ),
        reason="the answer-free Inference data profile does not match its Data authority",
        description="Baseline configuration selection cannot safely use a profile with the wrong schema, mapping, row shape, or answer-removal evidence.",
        recovery="Repair and validate the setup Data artifact, then rerun Baseline Inference from the corrected registered profile.",
    ),
    FailureMode(
        code="execution_preflight",
        verdict="structural",
        retryable=False,
        pattern=re.compile(
            r"(?:execution|runtime) preflight.*(?:reject|fail)|"
            r"(?:model|template|backend|device|method).*(?:incompatible|cannot be honored)",
            re.I | re.S,
        ),
        reason="the selected configuration cannot execute on the bound runtime or inputs",
        description="The configuration may be schema-valid while still incompatible with the actual model, template, backend, device, method, or data.",
        recovery="Preserve the evidence and change the owning configuration or upstream work order before rerunning.",
    ),
    FailureMode(
        code="unresolved_input_binding",
        verdict="structural",
        retryable=False,
        pattern=re.compile(r"unresolved input binding", re.I),
        reason="an upstream Ticket did not emit the required WorkProduct",
        description="A child input binding points to an absent or failed upstream artifact.",
        recovery="Repair and rerun the named upstream Ticket first; rerun this child only after the required WorkProduct exists.",
    ),
    FailureMode(
        code="inputs_unresolvable",
        verdict="structural",
        retryable=False,
        pattern=re.compile(r"inputs unresolvable", re.I),
        reason="the runner could not resolve an upstream artifact binding",
        description="Input resolution failed before the worker could start.",
        recovery="Repair the upstream Ticket or binding, confirm its WorkProduct exists, and then rerun this Ticket.",
    ),
    FailureMode(
        code="payload_missing_field",
        verdict="structural",
        retryable=False,
        pattern=re.compile(r"(?:KeyError:.*)?payload missing required field", re.I | re.S),
        reason="the Ticket payload is missing a required field",
        description="The work order does not satisfy the closed stored-payload schema.",
        recovery="Correct the upstream Ticket creator or payload against its typed schema before rerunning.",
    ),
    FailureMode(
        code="no_trailing_json",
        verdict="structural",
        retryable=False,
        pattern=re.compile(r"no parseable trailing JSON line", re.I),
        reason="automatic repair could not recover a typed final Result",
        description="The Agent activation ended without the required terminal JSON object.",
        recovery="Inspect the transcript and correct the Agent prompt, driver, or output implementation before forcing a rerun.",
    ),
    FailureMode(
        code="unsupported_input",
        verdict="structural",
        retryable=False,
        pattern=re.compile(r"unsupported (?:file type|target_format|task_type)", re.I),
        reason="the selected Agent or Skill does not support this input shape",
        description="The bound file, target format, or task type is outside the selected implementation contract.",
        recovery="Choose a compatible upstream representation, Agent operation, or Skill before rerunning.",
    ),
    FailureMode(
        code="file_not_found",
        verdict="structural",
        retryable=False,
        pattern=re.compile(r"file does not exist|No such file or directory", re.I),
        reason="a required file is absent from the executing filesystem",
        description="The path is wrong, the file was removed, or the required workspace/volume was not mounted.",
        recovery="Restore or re-upload the exact file, or correct its authoritative binding before rerunning.",
    ),
    FailureMode(
        code="oom",
        verdict="transient",
        retryable=True,
        pattern=re.compile(r"\b(?:OOM|out of memory|CUDA out of memory)\b", re.I),
        reason="GPU OOM; a new Agent activation may choose a smaller in-scope configuration",
        description="The realized model, sequence, batch, or execution strategy exceeded available device memory.",
        recovery="Rerun once so the owning Specialist can reduce memory pressure within the same work order; use a different resource plan only if no feasible configuration fits.",
    ),
    FailureMode(
        code="ssh_unreachable",
        verdict="transient",
        retryable=True,
        pattern=re.compile(
            r"\bssh: connect to host .* (?:Operation timed out|Connection refused)\b",
            re.I,
        ),
        reason="the assigned remote SSH endpoint was not reachable",
        description="The allocation may still be starting or the remote SSH service may be unavailable.",
        recovery="Check the assigned lease/allocation state and SSH probe, then retry; replace the resource only when it is no longer usable.",
    ),
    FailureMode(
        code="network",
        verdict="transient",
        retryable=True,
        pattern=re.compile(r"\bConnection (?:timed out|refused|reset)\b", re.I),
        reason="a network connection failed transiently",
        description="A provider, remote host, or service endpoint was temporarily unreachable.",
        recovery="Retry once. Repeated failures require checking the endpoint, provider status, and network path.",
    ),
    FailureMode(
        code="rate_limit",
        verdict="transient",
        retryable=True,
        pattern=re.compile(r"\b(?:rate limit|429|RateLimitError)\b", re.I),
        reason="the model provider rate-limited the request",
        description="The provider refused the activation because request or token throughput was temporarily too high.",
        recovery="Wait for the provider window to recover and rerun. Repeated limits require lowering request concurrency or changing provider capacity.",
    ),
    FailureMode(
        code="provider_5xx",
        verdict="transient",
        retryable=True,
        pattern=re.compile(r"\b(?:503|502|504|Bad Gateway|Service Unavailable)\b", re.I),
        reason="an upstream provider returned a temporary server failure",
        description="The remote service returned a gateway or availability error.",
        recovery="Check provider status and retry once. Repeated identical failures indicate an endpoint or configuration problem.",
    ),
    FailureMode(
        code="request_timeout",
        verdict="transient",
        retryable=True,
        pattern=re.compile(r"\b(?:Read timed out|ReadTimeout|Timeout)\b", re.I),
        reason="a provider or remote operation exceeded its timeout",
        description="The operation did not complete within its configured deadline.",
        recovery="Inspect the heartbeat to distinguish slow healthy work from a dead endpoint, then retry once or correct the relevant timeout.",
    ),
    FailureMode(
        code="dns_resolve",
        verdict="transient",
        retryable=True,
        pattern=re.compile(r"\b(?:Temporary failure in name resolution|DNS)\b", re.I),
        reason="hostname resolution failed transiently",
        description="The container or host could not resolve a required service name.",
        recovery="Retry once; if it persists, inspect the Docker/host DNS configuration.",
    ),
)


UNKNOWN_MODE = RetryClassification(
    verdict="unknown",
    retryable=True,
    reason="failure did not match a known classifier rule; inspect the heartbeat before rerunning",
    code="unknown",
    description="No deterministic manual-rerun rule currently matches this terminal error.",
    recovery="Inspect the final heartbeat and retry once only when the failure appears transient. Add a code rule and test if the pattern recurs.",
)


def failure_catalog() -> tuple[FailureMode, ...]:
    """Return the executable manual-rerun catalog in first-match order."""
    return FAILURE_MODES


def failure_mode_by_code(code: str) -> FailureMode | None:
    return next((mode for mode in FAILURE_MODES if mode.code == code), None)


def classify(error_message: str, exit_code: int = 0) -> RetryClassification:
    """Map the latest terminal error to one manual-rerun decision."""
    message = (error_message or "").strip()
    if not message and exit_code == 0:
        return RetryClassification(
            verdict="unknown",
            retryable=False,
            reason="ticket already succeeded — nothing to retry",
            code="not_applicable",
        )
    if exit_code in (143, -15) and ("cancel" in message.lower() or not message):
        mode = failure_mode_by_code("cancelled_by_user")
        assert mode is not None
        return _classification(mode)
    for mode in FAILURE_MODES:
        if mode.pattern.search(message):
            return _classification(mode)
    return UNKNOWN_MODE


def _classification(mode: FailureMode) -> RetryClassification:
    return RetryClassification(
        verdict=mode.verdict,
        retryable=mode.retryable,
        reason=mode.reason,
        code=mode.code,
        matched_pattern=mode.pattern.pattern,
        description=mode.description,
        recovery=mode.recovery,
    )
