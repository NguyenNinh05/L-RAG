"""
backend/api/router.py — Top-level API router aggregation.
"""

from fastapi import APIRouter

from backend.api.auth import router as auth_router
from backend.api.documents import router as documents_router
from backend.api.jobs import router as jobs_router
from backend.api.reports import router as reports_router
from backend.api.health import router as health_router

api_router = APIRouter(prefix="/api")
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
api_router.include_router(reports_router, prefix="/reports", tags=["reports"])
api_router.include_router(health_router, prefix="/health", tags=["health"])
