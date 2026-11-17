"""Conservative response adaptation for generated training pipelines (stdlib only).

No answer inference: spans and gold must come from documented source metadata
or an unambiguous source-specific parser. Generic SFT uses preserve mode.
"""
from __future__ import annotations

import re


def _answer_line(template: str | None, gold: str | None) -> str:
    if (not template or template.count("{answer}") != 1
            or "{" in template.replace("{answer}", "")
            or "}" in template.replace("{answer}", "")
            or "\n" in template or "\r" in template
            or template != template.strip()):
        raise ValueError("answer-line adaptation requires a single-line template with one {answer}")
    if not gold or gold != gold.strip() or "\n" in gold or "\r" in gold:
        raise ValueError("answer-line adaptation requires reliable single-line gold")
    return template.replace("{answer}", gold)


def prepare_response(
    response: str, *, scope: str = "preserve", output_format: str = "preserve",
    requirement: str = "", gold: str | None = None,
    answer_span: tuple[int, int] | None = None,
    answer_template: str | None = None,
) -> str:
    """Adapt the identified answer or terminal answer line, preserving reasoning."""
    if scope not in {"preserve", "task", "matching_records"}:
        raise ValueError("unknown response requirement scope")
    if output_format not in {"preserve", "boxed", "answer_line"}:
        raise ValueError("unsupported response adaptation; preserve or request Data repair")
    if output_format == "preserve":
        if answer_template is not None:
            raise ValueError("an applicable answer-line requirement cannot use preserve mode")
        return response
    if scope == "preserve" or not requirement.strip():
        raise ValueError("answer adaptation requires an explicit applicable requirement")
    if not gold or answer_span is None:
        raise ValueError("return to Data: answer adaptation requires reliable gold and span")
    start, end = answer_span
    if not 0 <= start < end <= len(response):
        raise ValueError("invalid source answer span")
    original = response[start:end]
    boxed = r"\boxed{" + gold + "}"
    if original not in {gold, boxed}:
        raise ValueError("return to Data: answer span disagrees with gold; do not guess")
    if output_format == "answer_line":
        target = _answer_line(answer_template, gold)
        line_start = response.rfind("\n", 0, start) + 1
        prefix, suffix = response[line_start:start], response[end:]
        # Replace only a dedicated terminal answer line, never explanation prose.
        if response[line_start:].rstrip() == target:
            return response
        if (not re.fullmatch(r"[ \t]*(?:(?:Final Answer|Answer)[ \t]*:[ \t]*)?", prefix)
                or suffix.strip() not in {"", "."}):
            raise ValueError("answer-line adaptation needs an unambiguous terminal answer line; preserve reasoning")
        return response[:line_start] + target
    return response[:start] + boxed + response[end:]


def validate_training_response(
    response: str, *, expected_response: str,
    input_ids: list[int], labels: list[int],
    required_format: str = "preserve", gold: str | None = None,
    answer_token_span: tuple[int, int] | None = None,
    answer_template: str | None = None,
) -> None:
    """Check actual constructed response and post-truncation labels before training.

    answer_token_span is measured in the UNTRUNCATED rendered sequence using
    the actual tokenizer. For answer tasks it must cover the final gold answer.
    Call for each supervised turn/branch. This does not replace prefix-mask checks.
    """
    if response != expected_response:
        raise ValueError("training response changed source content outside approved adaptation")
    if required_format not in {"preserve", "boxed", "answer_line"}:
        raise ValueError("unsupported output requirement")
    if answer_template is not None and required_format != "answer_line":
        raise ValueError("an applicable answer-line requirement cannot use preserve or boxed mode")
    if required_format == "answer_line":
        target = _answer_line(answer_template, gold)
        if response.rstrip().split("\n")[-1] != target:
            raise ValueError("training target conflicts with required final answer line")
        if answer_token_span is None:
            raise ValueError("answer-line target requires final-answer token span")
    if required_format == "boxed":
        if not gold or r"\boxed{" + gold + "}" not in response:
            raise ValueError("training target conflicts with required boxed answer")
        if answer_token_span is None:
            raise ValueError("boxed target requires final-answer token span")
    if len(input_ids) != len(labels) or not any(x != -100 for x in labels):
        raise ValueError("training sequence has invalid or empty supervision")
    if any(label != -100 and label != token for token, label in zip(input_ids, labels)):
        raise ValueError("training labels disagree with rendered tokens")
    if answer_token_span is not None:
        start, end = answer_token_span
        if not 0 <= start < end <= len(labels):
            raise ValueError("final answer was truncated")
        if any(label == -100 for label in labels[start:end]):
            raise ValueError("final answer is masked out of training loss")
