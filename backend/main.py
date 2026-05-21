"""
backend/main.py — FastAPI application factory.

Start with: uvicorn backend.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router import api_router
from backend.config import get_backend_config
from backend.ws.handlers import ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_backend_config()
    logging.basicConfig(level=cfg.log_level)
    logger = logging.getLogger(__name__)

    logger.info("LegalDiff Backend starting...")
    from backend.services.storage import FileStorageManager
    storage = FileStorageManager(cfg.storage_root)
    storage.ensure_directories()
    logger.info("Storage directories ensured: %s", cfg.storage_root)

    yield

    logger.info("LegalDiff Backend shutting down...")


def create_app() -> FastAPI:
    cfg = get_backend_config()

    app = FastAPI(
        title="LegalDiff API",
        description="Backend API for LegalDiff — Vietnamese Legal Document Comparison",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Local deployment; tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.include_router(ws_router)

    return app


app = create_app()
