"""Task-owned semantic instructions for measured inference.

The model-level prompt contract (chat template, tokenizer, special tokens and
system role) must stay aligned between Train and Inference.  That contract does
not, however, say what one evaluation row asks the model to do.  A sample
submission says even less: it is only the shape of ``predictions.csv``.

``TaskInferenceProtocol`` fills that gap.  It is deliberately small enough to
derive from an objective for the common case, while remaining explicit and
serializable once a Task is created.  The resolved values are copied into the
Run's inference mapping, so Baseline and every trained checkpoint execute the
same semantic task protocol.
"""
from __future__ import annotations

import re
import string
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskType = Literal[
    "generation", "classification", "multiple_choice", "math_reasoning", "code",
]
ResponseFormat = Literal["plain_text", "label", "choice", "boxed_answer", "code"]
AnswerParser = Literal["raw", "choice", "boxed", "regex", "code"]


class TaskInferenceProtocol(BaseModel):
    """The task semantics applied to every row of one dataset evaluation.

    ``{input}`` in ``user_prompt_template`` is a Zevo placeholder for the
    deterministic rendering of the selected input fields.  Advanced callers
    may instead name fields directly (for example ``{question}``); those are
    checked against the materialized data profile before generation.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source: Literal["auto", "user", "backend"] = "auto"
    task_type: TaskType = "generation"
    instruction: str = Field(min_length=1)
    user_prompt_template: str = Field(default="{input}", min_length=1)
    output_instruction: str = Field(min_length=1)
    response_format: ResponseFormat = "plain_text"
    answer_parser: AnswerParser = "raw"
    answer_regex: str = ""

    @model_validator(mode="after")
    def validate_protocol(self) -> "TaskInferenceProtocol":
        self.instruction = self.instruction.strip()
        self.user_prompt_template = self.user_prompt_template.strip()
        self.output_instruction = self.output_instruction.strip()
        self.answer_regex = self.answer_regex.strip()
        if not self.instruction:
            raise ValueError("inference protocol instruction must not be blank")
        if not self.user_prompt_template:
            raise ValueError("inference protocol user_prompt_template must not be blank")
        if not self.output_instruction:
            raise ValueError("inference protocol output_instruction must not be blank")
        if self.answer_parser == "regex":
            if not self.answer_regex:
                raise ValueError("answer_parser='regex' requires answer_regex")
            try:
                re.compile(self.answer_regex)
            except re.error as exc:
                raise ValueError(f"inference protocol answer_regex is invalid: {exc}") from exc
        elif self.answer_regex:
            raise ValueError("answer_regex is valid only with answer_parser='regex'")
        expected = {
            "boxed": "boxed_answer",
            "choice": "choice",
            "code": "code",
        }.get(self.answer_parser)
        if expected and self.response_format != expected:
            raise ValueError(
                f"answer_parser={self.answer_parser!r} requires "
                f"response_format={expected!r}"
            )
        return self

    def inference_mapping(self) -> dict[str, str]:
        """Return the immutable portion of ``measurement.inference_config``."""
        parser_regex = {
            "boxed": r"\\boxed\{([^{}]+)\}",
            "choice": r"(?i)(?:final\s+answer\s*[:：]?\s*)?\(?([A-Z])\)?\s*$",
        }.get(self.answer_parser, "")
        return {
            "task_instruction": self.instruction,
            "user_prompt_template": self.user_prompt_template,
            "output_instruction": self.output_instruction,
            "response_format": self.response_format,
            "answer_parser": self.answer_parser,
            **(
                {"answer_regex": self.answer_regex or parser_regex}
                if self.answer_regex or parser_regex else {}
            ),
        }

    def render_user_content(self, input_values: dict[str, object]) -> str:
        """Render the user turn exactly as measured Inference must render it."""
        return render_task_user_content(self.inference_mapping(), input_values)


def render_task_user_content(
    mapping: dict[str, object], input_values: dict[str, object],
) -> str:
    """Compose instruction, row input and output requirement deterministically."""
    template = str(mapping.get("user_prompt_template") or "{input}")
    if len(input_values) == 1:
        joined = str(next(iter(input_values.values())))
    else:
        joined = "\n".join(f"{key}: {value}" for key, value in input_values.items())
    values = {key: str(value) for key, value in input_values.items()}
    values["input"] = joined
    fields = [name for _, name, _, _ in string.Formatter().parse(template) if name]
    unknown = sorted(set(fields) - set(values))
    if unknown:
        raise ValueError(
            "user_prompt_template references fields absent from the scoring data: "
            + ", ".join(unknown)
        )
    rendered_input = template.format_map(values).strip()
    return "\n\n".join(
        part.strip()
        for part in (
            str(mapping.get("task_instruction") or ""),
            rendered_input,
            str(mapping.get("output_instruction") or ""),
        )
        if part.strip()
    )


_MATH_WORDS = re.compile(
    r"\b(math|mathematics|mathematical|arithmetic|algebra|geometry|calculus|equation|aime|omega)\b|"
    r"数学|算术|代数|几何|方程",
    re.IGNORECASE,
)
_CODE_WORDS = re.compile(
    r"\b(code|coding|program|programming|function|implementation|unit tests?|humaneval|mbpp)\b|"
    r"代码|编程|函数|程序",
    re.IGNORECASE,
)
_CHOICE_WORDS = re.compile(
    r"\b(multiple[ -]?choice|mcq|choose (?:the )?(?:correct|best)|options?)\b|"
    r"选择题|选项",
    re.IGNORECASE,
)
_CLASSIFICATION_WORDS = re.compile(
    r"\b(classif(?:y|ication)|categor(?:y|ize|isation|ization)|label|sentiment)\b|"
    r"分类|标签|情感",
    re.IGNORECASE,
)


def infer_task_type(objective: str, metric: str = "") -> TaskType:
    """Deterministically choose a conservative protocol preset."""
    text = f"{objective} {metric}".strip()
    if _CODE_WORDS.search(text):
        return "code"
    if _MATH_WORDS.search(text):
        return "math_reasoning"
    if metric.strip().lower() in {"mc_loglikelihood", "accuracy_norm"} or _CHOICE_WORDS.search(text):
        return "multiple_choice"
    if _CLASSIFICATION_WORDS.search(text):
        return "classification"
    return "generation"


def default_task_inference_protocol(
    objective: str, metric: str = "",
) -> TaskInferenceProtocol:
    """Resolve the zero-form-fields path into an explicit frozen protocol."""
    instruction = objective.strip()
    if not instruction:
        raise ValueError("task objective is required to derive an inference protocol")
    task_type = infer_task_type(instruction, metric)
    presets: dict[str, dict[str, str]] = {
        "math_reasoning": {
            "output_instruction": (
                "Show the reasoning needed to solve the problem, then end with the final "
                "answer in the form \\boxed{answer}."
            ),
            "response_format": "boxed_answer",
            "answer_parser": "boxed",
        },
        "multiple_choice": {
            "output_instruction": "Return only the label of the selected option.",
            "response_format": "choice",
            "answer_parser": "choice",
        },
        "classification": {
            "output_instruction": "Return only the predicted label.",
            "response_format": "label",
            "answer_parser": "raw",
        },
        "code": {
            "output_instruction": "Return only the implementation code, without commentary.",
            "response_format": "code",
            "answer_parser": "code",
        },
        "generation": {
            "output_instruction": "Return the requested response directly.",
            "response_format": "plain_text",
            "answer_parser": "raw",
        },
    }
    return TaskInferenceProtocol(
        source="auto",
        task_type=task_type,
        instruction=instruction,
        **presets[task_type],
    )
