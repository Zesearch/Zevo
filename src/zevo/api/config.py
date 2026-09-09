"""Runtime config for the FastAPI backend."""
from __future__ import annotations

import os

from zevo.paths import work_dir_root


class Settings:
    app_name: str = "Zevo"
    app_version: str = "0.2.0"
    api_prefix: str = "/api"

    @property
    def cors_origins(self) -> list[str]:
        """Whitelist for CORS (set to localhost dev + any explicit origin)."""
        extra = os.environ.get("ZEVO_CORS_ORIGINS", "").strip()
        defaults = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ]
        if extra:
            return defaults + [o.strip() for o in extra.split(",") if o.strip()]
        return defaults

    @property
    def work_dir_root(self) -> str:
        return work_dir_root()

settings = Settings()
