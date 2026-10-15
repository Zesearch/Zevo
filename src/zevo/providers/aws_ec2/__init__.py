"""AWS EC2 GPU instances as a `cloud`-mode backend. Pure boto3; no framework deps.

Third `cloud` backend alongside `zevo.providers.vastai` and
`zevo.providers.lambda_labs`, with the same mirrored surface so the
infrastructure agent and the shipped acquire helper drive it with the same
calls.
"""

from .provider import (
    AwsEc2Provider,
    aws_credentials_present,
    AWS_EC2_SSH_USER,
    AWS_EC2_SSH_PORT,
    AWS_EC2_CREDENTIAL_ENVS,
    DEFAULT_CREDENTIALS_OPT_IN_ENV,
    MARKET_ENV,
)

__all__ = [
    "AwsEc2Provider",
    "aws_credentials_present",
    "AWS_EC2_SSH_USER",
    "AWS_EC2_SSH_PORT",
    "AWS_EC2_CREDENTIAL_ENVS",
    "DEFAULT_CREDENTIALS_OPT_IN_ENV",
    "MARKET_ENV",
]
