"""
backend/api/health.py — Health check endpoints.
"""

from __future__ import annotations

import httpx
import redis as sync_redis
from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_backend_config
from backend.database import _async_engine
from backend.schemas.common import HealthResponse
from backend.workers.gpu_lock import GPULock

router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def ready():
    cfg = get_backend_config()
    resp = HealthResponse()

    # Database check
    try:
        engine = _async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        resp.database = "connected"
    except Exception:
        resp.database = "disconnected"
        resp.status = "degraded"

    # Redis check
    try:
        r = sync_redis.from_url(cfg.redis_url)
        r.ping()
        r.close()
        resp.redis = "connected"
    except Exception:
        resp.redis = "disconnected"
        resp.status = "degraded"

    # Worker check (Celery + GPU lock)
    try:
        lock = GPULock()
        is_busy = lock.is_locked()
        if is_busy:
            resp.worker = "busy"
        else:
            resp.worker = "available"
    except Exception:
        resp.worker = "unreachable"
        resp.status = "degraded"

    # LLM server check
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{cfg.llm_base_url}/models")
            if r.status_code == 200:
                resp.llm_server = "connected"
            else:
                resp.llm_server = "error"
    except Exception:
        resp.llm_server = "disconnected"
        resp.status = "degraded"

    if resp.database == "connected" and resp.redis == "connected" and resp.worker != "unreachable":
        resp.status = "ok"

    return resp
