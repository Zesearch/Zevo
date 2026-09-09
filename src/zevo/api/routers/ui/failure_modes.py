"""Machine-readable manual-rerun failure catalog and classifier."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from zevo.contracts._base import StrictBody
from zevo.engine.run.retry_policy import classify, failure_catalog


router = APIRouter()


class FailureMode(BaseModel):
    code: str
    title: str
    verdict: Literal["transient", "structural", "cancelled", "unknown"]
    pattern: str
    description: str
    recovery: str


class CatalogResponse(BaseModel):
    catalog: list[FailureMode]


class ClassifyBody(StrictBody):
    error_message: str
    exit_code: int = 0


class ClassifyResponse(BaseModel):
    verdict: str
    retryable: bool
    reason: str
    code: str
    matched_pattern: str
    catalog_entry: FailureMode | None = None


def _public_mode(mode) -> FailureMode:
    return FailureMode(
        code=mode.code,
        title=mode.code,
        verdict=mode.verdict,
        pattern=mode.pattern.pattern,
        description=mode.description,
        recovery=mode.recovery,
    )


@router.get("/failure-modes", response_model=CatalogResponse)
async def list_failure_modes() -> CatalogResponse:
    return CatalogResponse(catalog=[_public_mode(mode) for mode in failure_catalog()])


@router.post("/failure-modes/classify", response_model=ClassifyResponse)
async def classify_error(body: ClassifyBody) -> ClassifyResponse:
    result = classify(body.error_message, body.exit_code)
    entry = next(
        (mode for mode in failure_catalog() if mode.code == result.code),
        None,
    )
    return ClassifyResponse(
        verdict=result.verdict,
        retryable=result.retryable,
        reason=result.reason,
        code=result.code,
        matched_pattern=result.matched_pattern,
        catalog_entry=_public_mode(entry) if entry is not None else None,
    )
