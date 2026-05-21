"""
backend/schemas/report.py — Report response schemas.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict


class ReportSummaryResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    pair_id: str
    match_type: str
    location_context: str
    total_acus: int
    passed_acus: int
    rejected_acus: int
    hallucination_rate: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportDetailResponse(ReportSummaryResponse):
    executive_summary: dict | None
    markdown_report: str | None
    verified_acus: list | None
    rejected_acus_detail: list | None
