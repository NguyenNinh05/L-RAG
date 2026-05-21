"""
backend/api/reports.py — Report endpoints.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.models.user import User
from backend.models.comparison_report import ComparisonReportModel
from backend.models.comparison_job import ComparisonJob
from backend.schemas.report import ReportDetailResponse, ReportSummaryResponse

router = APIRouter()


async def _get_report(
    db: AsyncSession, report_id: uuid.UUID, user_id: uuid.UUID
) -> ComparisonReportModel | None:
    result = await db.execute(
        select(ComparisonReportModel)
        .join(ComparisonJob, ComparisonReportModel.job_id == ComparisonJob.id)
        .where(
            ComparisonReportModel.id == report_id,
            ComparisonJob.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


@router.get("/{report_id}", response_model=ReportDetailResponse)
async def get_report(
    report_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    report = await _get_report(db, report_id, user.id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@router.get("/{report_id}/markdown")
async def get_report_markdown(
    report_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    report = await _get_report(db, report_id, user.id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return Response(
        content=report.markdown_report or "",
        media_type="text/markdown; charset=utf-8",
    )
