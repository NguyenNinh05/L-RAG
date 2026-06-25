"""
backend/api/settings.py — Per-user LLM settings endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.models.user import User
from backend.schemas.settings import (
    LLMConfigSchema,
    LLMSettingsResponse,
    AvailableModelsResponse,
)
from backend.services.settings_service import SettingsService

router = APIRouter()

PROVIDERS = ["local", "deepseek"]


def _svc() -> SettingsService:
    return SettingsService()


@router.get("/llm", response_model=LLMSettingsResponse)
async def get_llm_settings(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: SettingsService = Depends(_svc),
):
    config = await svc.get(db, user.id)
    models = await svc.list_models(config.llm_provider, config.llm_base_url)
    return LLMSettingsResponse(config=config, providers=PROVIDERS, models=models)


@router.put("/llm", response_model=LLMSettingsResponse)
async def save_llm_settings(
    body: LLMConfigSchema,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: SettingsService = Depends(_svc),
):
    config = await svc.upsert(db, user.id, body)
    models = await svc.list_models(config.llm_provider, config.llm_base_url)
    return LLMSettingsResponse(config=config, providers=PROVIDERS, models=models)


@router.get("/models", response_model=AvailableModelsResponse)
async def list_models(
    provider: str = Query("local"),
    base_url: str | None = Query(None),
    user: User = Depends(get_current_user),
    svc: SettingsService = Depends(_svc),
):
    models = await svc.list_models(provider, base_url)
    base = svc.provider_base_url(provider)
    return AvailableModelsResponse(provider=provider, models=models, base_url=base)
