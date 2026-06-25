"""
backend/api/jobs.py — Comparison job endpoints.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db, get_storage
from backend.models.user import User
from backend.models.comparison_job import ComparisonJob
from backend.models.comparison_report import ComparisonReportModel
from backend.models.document import Document
from backend.schemas.common import PaginatedResponse
from backend.schemas.job import CreateJobRequest, JobResponse, JobStatusResponse
from backend.schemas.report import (
    ReportSummaryResponse,
    DiffReportResponse,
    DiffReportSummaryResponse,
    AcuItemResponse,
    EvidenceResponse,
)
from backend.services.job_service import JobService
from backend.services.storage import FileStorageManager

router = APIRouter()


# Severity heuristic by change_type — ACUs have no native severity field,
# so we derive one for the frontend's filter/sort UI.
_SEVERITY_BY_TYPE = {
    "addition": "high",
    "deletion": "high",
    "structural": "medium",
    "numerical": "medium",
    "terminology": "medium",
    "reorder": "low",
}


def _acu_to_item(acu: dict) -> AcuItemResponse:
    """Map a stored ACU dict (ACUOutput.to_dict()) to the frontend AcuItem shape."""
    change_type = str(acu.get("change_type", "structural"))
    original = str(acu.get("original_value", "") or "")
    new = str(acu.get("new_value", "") or "")
    title = (new or original or change_type).strip()

    return AcuItemResponse(
        id=str(acu.get("acu_id", "")),
        type=change_type,
        severity=_SEVERITY_BY_TYPE.get(change_type, "medium"),
        title=title[:200],
        description=str(acu.get("reasoning", "") or title),
        v1_evidence=EvidenceResponse(
            text=str(acu.get("verbatim_evidence_v1", "") or ""),
            article=str(acu.get("location_v1", "") or "") or None,
        ),
        v2_evidence=EvidenceResponse(
            text=str(acu.get("verbatim_evidence_v2", "") or ""),
            article=str(acu.get("location_v2", "") or "") or None,
        ),
    )


def _svc(storage: FileStorageManager = Depends(get_storage)) -> JobService:
    return JobService(storage)


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(
    body: CreateJobRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    try:
        job = await svc.create(
            db,
            user.id,
            body.document_v1_id,
            body.document_v2_id,
            body.skip_phase3,
            body.config_overrides,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return _to_job_response(job)


@router.get("", response_model=PaginatedResponse[JobResponse])
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    jobs, total = await svc.list_by_user(db, user.id, page, page_size)
    total_pages = (total + page_size - 1) // page_size
    return PaginatedResponse(
        items=[_to_job_response(j) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    job = await svc.get_by_id(db, job_id, user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _to_job_response(job)


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    job = await svc.get_by_id(db, job_id, user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    try:
        cancelled = await svc.cancel(db, job_id, user.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    if not cancelled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return {"id": str(job_id), "status": "cancelled"}


@router.delete("/{job_id}", status_code=204)
async def delete_job(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    try:
        deleted = await svc.delete(db, job_id, user.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")


@router.get("/{job_id}/reports", response_model=PaginatedResponse[ReportSummaryResponse])
async def list_job_reports(
    job_id: uuid.UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    job = await svc.get_by_id(db, job_id, user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    base = select(ComparisonReportModel).where(
        ComparisonReportModel.job_id == job_id
    )

    from sqlalchemy import func
    count_result = await db.execute(
        select(func.count()).select_from(base.subquery())
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        base.order_by(ComparisonReportModel.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    reports = list(result.scalars().all())
    total_pages = (total + page_size - 1) // page_size
    return PaginatedResponse(
        items=reports,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{job_id}/report", response_model=DiffReportResponse)
async def get_job_report(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    """
    Aggregated report for a job — flattens every pair's verified ACUs into a
    single {summary, acus[]} payload matching the frontend DiffReport shape.
    """
    job = await svc.get_by_id(db, job_id, user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    # Labels from the source documents
    doc_rows = (
        await db.execute(
            select(Document).where(
                Document.id.in_([job.document_v1_id, job.document_v2_id])
            )
        )
    ).scalars().all()
    doc_names = {str(d.id): d.original_filename for d in doc_rows}
    v1_label = doc_names.get(str(job.document_v1_id), "V1")
    v2_label = doc_names.get(str(job.document_v2_id), "V2")

    # All report rows for this job, newest first
    report_rows = (
        await db.execute(
            select(ComparisonReportModel)
            .where(ComparisonReportModel.job_id == job_id)
            .order_by(ComparisonReportModel.created_at.desc())
        )
    ).scalars().all()

    acus: list[AcuItemResponse] = []
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    generated_at = job.completed_at or job.created_at

    for row in report_rows:
        for raw_acu in row.verified_acus or []:
            if not isinstance(raw_acu, dict):
                continue
            item = _acu_to_item(raw_acu)
            acus.append(item)
            by_type[item.type] = by_type.get(item.type, 0) + 1
            by_severity[item.severity] = by_severity.get(item.severity, 0) + 1
        if row.created_at and (generated_at is None or row.created_at > generated_at):
            generated_at = row.created_at

    return DiffReportResponse(
        summary=DiffReportSummaryResponse(
            total_acus=len(acus),
            by_type=by_type,
            by_severity=by_severity,
            v1_label=v1_label,
            v2_label=v2_label,
            generated_at=(generated_at.isoformat() if generated_at else ""),
        ),
        acus=acus,
    )


@router.get("/{job_id}/catalog")
async def get_job_catalog(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: JobService = Depends(_svc),
):
    job = await svc.get_by_id(db, job_id, user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job.catalog or {}


def _to_job_response(job: ComparisonJob) -> JobResponse:
    document_v1 = job.__dict__.get("document_v1")
    document_v2 = job.__dict__.get("document_v2")

    return JobResponse(
        id=job.id,
        document_v1_id=job.document_v1_id,
        document_v2_id=job.document_v2_id,
        v1_filename=getattr(document_v1, "original_filename", ""),
        v2_filename=getattr(document_v2, "original_filename", ""),
        status=job.status,
        current_phase=job.current_phase,
        progress_pct=job.progress_pct,
        error_message=job.error_message,
        total_pairs=job.total_pairs,
        matched_count=job.matched_count,
        added_count=job.added_count,
        deleted_count=job.deleted_count,
        split_count=job.split_count,
        merge_count=job.merge_count,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
