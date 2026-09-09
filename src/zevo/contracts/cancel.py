"""What to do with the trained weights when a Run is cancelled.

`POST /runs/{id}/cancel` takes one of these. `download` and `hf` keep the
champion checkpoint (copied off the GPU box before it is destroyed); `discard`
is the old behaviour: kill everything and tear down at once.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CancelWeights = Literal["download", "hf", "discard"]

_HF_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


class CancelWeightsPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: CancelWeights = "download"
    # Absolute folder visible to the Zevo server. Empty means the run's own
    # models folder under the work dir, which the registry stage also uses.
    local_dir: str = ""
    # `hf` only: `owner/name` on the Hugging Face Hub. The HF_TOKEN from
    # Settings must be able to write it.
    hf_repo_id: str = ""
    hf_private: bool = True
    # `hf` only: keep the local copy the upload was made from.
    keep_local_copy: bool = Field(default=True)

    @model_validator(mode="after")
    def _check(self) -> "CancelWeightsPolicy":
        if self.local_dir and not Path(self.local_dir).is_absolute():
            raise ValueError("local_dir must be an absolute path")
        if self.weights == "hf" and not _HF_REPO.match(self.hf_repo_id):
            raise ValueError("hf_repo_id must look like owner/name")
        return self
