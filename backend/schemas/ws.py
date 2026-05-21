"""
backend/schemas/ws.py — WebSocket message schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WSProgressMessage(BaseModel):
    event: str = Field(..., description="progress | phase_change | completed | error")
    job_id: str
    progress_pct: int
    current_phase: str
    message: str = ""
    error: str | None = None
