"""
backend/schemas/common.py — Shared API schemas (pagination, errors, health).
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class HealthResponse(BaseModel):
    status: str = "ok"
    database: str = "unknown"
    redis: str = "unknown"
    worker: str = "unknown"
    llm_server: str = "unknown"
