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


# ---------------------------------------------------------------------------
# Aggregated diff report (matches frontend DiffReport / AcuItem / Evidence).
# One job → many ComparisonReportModel rows; we flatten their verified ACUs
# into a single {summary, acus[]} payload for the report viewer.
# ---------------------------------------------------------------------------


class EvidenceResponse(BaseModel):
    text: str = ""
    article: str | None = None
    citation: str | None = None


class AcuItemResponse(BaseModel):
    id: str
    type: str
    severity: str
    title: str
    description: str
    v1_evidence: EvidenceResponse
    v2_evidence: EvidenceResponse


class DiffReportSummaryResponse(BaseModel):
    total_acus: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    v1_label: str
    v2_label: str
    generated_at: str


class DiffReportResponse(BaseModel):
    summary: DiffReportSummaryResponse
    acus: list[AcuItemResponse]
