"""
backend/services/settings_service.py — Per-user LLM settings business logic.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.user_settings import UserSettings
from backend.schemas.settings import LLMConfigSchema

logger = logging.getLogger(__name__)

# Ensure project root is importable so `src.config` (the same source the worker
# uses) is available for resolving the current default config.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Static model presets for hosted providers (Ollama is fetched live).
_DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]


def _system_defaults() -> LLMConfigSchema:
    """Resolve the current system defaults from src.config, with a safe fallback."""
    try:
        from src.config import get_config, get_llm_config

        llm = get_llm_config(provider="local")
        pipe = get_config()
        return LLMConfigSchema(
            llm_provider="local",
            llm_model_name=llm.get("model_name", "qwen2.5:14b"),
            llm_base_url=llm.get("base_url", "http://localhost:11434/v1"),
            llm_api_key=llm.get("api_key", "not-needed"),
            llm_temperature_acu=float(llm.get("temperature_acu", 0.05)),
            llm_temperature_summary=float(llm.get("temperature_summary", 0.3)),
            max_tokens_acu=int(llm.get("max_tokens_acu", 8192)),
            max_tokens_summary=int(llm.get("max_tokens_summary", 1024)),
            max_concurrency=int(pipe.get("comparison", {}).get("max_concurrency", 1)),
            llm_timeout_seconds=float(llm.get("timeout_seconds", 300.0)),
            llm_max_retries=int(llm.get("max_retries", 3)),
            match_threshold=float(pipe.get("alignment", {}).get("match_threshold", 0.65)),
            skip_phase3=False,
            # Phase 3 speed knobs — defaults match the comparison engine.
            enable_second_pass=True,
            enable_number_enumeration=True,
            min_confidence_to_include=float(
                pipe.get("comparison", {}).get("min_confidence_to_include", 0.2)
            ),
            max_comparison_pairs=None,  # None = unlimited
        )
    except Exception:
        logger.warning("Could not load src.config defaults; using schema defaults.", exc_info=True)
        return LLMConfigSchema()


class SettingsService:
    """CRUD + helpers for per-user LLM settings."""

    async def get(self, db: AsyncSession, user_id: uuid.UUID) -> LLMConfigSchema:
        row = await self._row(db, user_id)
        if row is None:
            return _system_defaults()
        # Merge stored values over the system defaults so new fields are filled in
        # even if the row predates them.
        base = _system_defaults().model_dump()
        base.update(row.llm_config or {})
        return LLMConfigSchema(**base)

    async def upsert(
        self, db: AsyncSession, user_id: uuid.UUID, config: LLMConfigSchema
    ) -> LLMConfigSchema:
        row = await self._row(db, user_id)
        data = config.model_dump()
        if row is None:
            db.add(UserSettings(user_id=user_id, llm_config=data))
        else:
            row.llm_config = data
        await db.flush()
        return config

    async def get_overrides(self, db: AsyncSession, user_id: uuid.UUID) -> dict:
        """Return the stored config as a plain dict, or {} if none saved.

        {} means jobs use the YAML defaults unchanged.
        """
        row = await self._row(db, user_id)
        if row is None:
            return {}
        return dict(row.llm_config or {})

    async def list_models(self, provider: str, base_url: str | None = None) -> list[str]:
        """Available models for a provider. Local → live Ollama; deepseek → presets."""
        if provider == "deepseek":
            return list(_DEEPSEEK_MODELS)
        return await self._list_ollama_models(base_url or "http://localhost:11434/v1")

    @staticmethod
    async def _row(db: AsyncSession, user_id: uuid.UUID) -> UserSettings | None:
        result = await db.execute(
            select(UserSettings).where(UserSettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def _list_ollama_models(base_url: str) -> list[str]:
        tags_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(tags_url)
                resp.raise_for_status()
                payload = resp.json()
            return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
        except Exception:
            logger.debug("Ollama model list unavailable at %s", tags_url, exc_info=True)
            return []
