"""
backend/database.py — SQLAlchemy async engine + session factory.
"""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.config import get_backend_config


@lru_cache
def _async_engine():
    cfg = get_backend_config()
    return create_async_engine(
        cfg.database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


def _async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        _async_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_db():
    """FastAPI dependency: yield an async database session."""
    factory = _async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Synchronous engine for Celery worker (non-async context)
@lru_cache
def _sync_engine():
    from sqlalchemy import create_engine

    cfg = get_backend_config()
    return create_engine(
        cfg.database_url_sync,
        echo=False,
        poolclass=NullPool,  # Celery worker: one connection per task
    )


def get_sync_session():
    """Yield a synchronous DB session (for Celery tasks)."""
    from sqlalchemy.orm import Session

    with Session(_sync_engine()) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
