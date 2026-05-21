"""
backend/config.py — Backend configuration from env vars with sensible defaults.

Pattern: @lru_cache singleton, fallback defaults — same as src/config.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class BackendConfig:
    database_url: str = (
        "postgresql+asyncpg://legaldiff:legaldiff_secret@localhost:5432/legaldiff"
    )
    database_url_sync: str = (
        "postgresql://legaldiff:legaldiff_secret@localhost:5432/legaldiff"
    )
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    storage_root: str = "./backend/storage"
    max_upload_size_mb: int = 100
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx")

    log_level: str = "INFO"


@lru_cache
def get_backend_config() -> BackendConfig:
    """Return a frozen BackendConfig singleton populated from env vars."""
    return BackendConfig(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://legaldiff:legaldiff_secret@localhost:5432/legaldiff",
        ),
        database_url_sync=os.getenv(
            "DATABASE_URL_SYNC",
            "postgresql://legaldiff:legaldiff_secret@localhost:5432/legaldiff",
        ),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        jwt_secret_key=os.getenv("JWT_SECRET_KEY", "change-me-in-production"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        access_token_expire_minutes=int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
        ),
        refresh_token_expire_days=int(
            os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")
        ),
        llm_base_url=os.getenv(
            "LLM_BASE_URL", "http://localhost:8000/v1"
        ),
        llm_model_name=os.getenv(
            "LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"
        ),
        storage_root=os.getenv("STORAGE_ROOT", "./backend/storage"),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "100")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
