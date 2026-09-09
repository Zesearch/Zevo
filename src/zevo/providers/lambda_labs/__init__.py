"""Lambda Cloud (lambda.ai) REST API client. Pure Python (httpx); no framework deps.

A second `cloud`-mode GPU backend alongside `zevo.providers.vastai`, with a mirrored surface
so the infrastructure agent can use either behind the same calls.
"""

from .provider import (
    LambdaCloudProvider,
    load_api_key,
    LAMBDA_API_BASE,
    LAMBDA_SSH_USER,
    LAMBDA_SSH_PORT,
)

__all__ = [
    "LambdaCloudProvider",
    "load_api_key",
    "LAMBDA_API_BASE",
    "LAMBDA_SSH_USER",
    "LAMBDA_SSH_PORT",
]
