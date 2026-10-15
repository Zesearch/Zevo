"""User-owned destination for a completed champion (also used by cancellation)."""
from pathlib import PurePosixPath
from typing import Literal
import re
from pydantic import BaseModel, ConfigDict, model_validator

class ModelSavePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weights: Literal["hf", "remote"] = "hf"
    hf_repo_id: str = ""
    hf_private: bool = True
    remote_dir: str = ""

    @model_validator(mode="after")
    def validate_destination(self):
        self.hf_repo_id = self.hf_repo_id.strip()
        self.remote_dir = self.remote_dir.strip()
        if self.weights == "hf" and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+", self.hf_repo_id):
            raise ValueError("hf_repo_id must look like owner/model-name")
        if self.weights == "remote":
            path = PurePosixPath(self.remote_dir)
            if not path.is_absolute() or ".." in path.parts or not str(path).strip("/") or any(c in self.remote_dir for c in "\x00\n\r"):
                raise ValueError("remote_dir must be an absolute directory on the GPU machine, other than /")
        return self
