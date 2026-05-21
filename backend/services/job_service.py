"""
backend/services/job_service.py — ComparisonJob CRUD business logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.celery_app import celery_app
from backend.config import get_backend_config
from backend.models.comparison_job import ComparisonJob, JobStatus, JobPhase
from backend.models.document import Document
from backend.services.storage import FileStorageManager


class JobService:
    def __init__(self, storage: FileStorageManager) -> None:
        self._storage = storage

    async def create(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        document_v1_id: uuid.UUID,
        document_v2_id: uuid.UUID,
        skip_phase3: bool = False,
        config_overrides: dict | None = None,
    ) -> ComparisonJob:
        # Validate documents exist and belong to user
        v1 = await db.execute(
            select(Document).where(
                Document.id == document_v1_id,
                Document.user_id == user_id,
            )
        )
        if not v1.scalar_one_or_none():
            raise ValueError("Document V1 not found")

        v2 = await db.execute(
            select(Document).where(
                Document.id == document_v2_id,
                Document.user_id == user_id,
            )
        )
        if not v2.scalar_one_or_none():
            raise ValueError("Document V2 not found")

        cfg = get_backend_config()

        job = ComparisonJob(
            user_id=user_id,
            document_v1_id=document_v1_id,
            document_v2_id=document_v2_id,
            status=JobStatus.PENDING.value,
            current_phase=JobPhase.QUEUED.value,
            config_snapshot=config_overrides or {},
        )
        db.add(job)
        await db.flush()
        await db.refresh(job)

        # Resolve absolute file paths
        v1_doc = v1.scalar_one_or_none()
        v2_doc = v2.scalar_one_or_none()
        v1_path = str(self._storage.get_absolute_path(v1_doc.storage_path))
        v2_path = str(self._storage.get_absolute_path(v2_doc.storage_path))

        # Dispatch to Celery worker
        task = celery_app.send_task(
            "run_pipeline",
            args=[str(job.id), v1_path, v2_path],
            kwargs={"config_overrides": config_overrides},
        )

        job.celery_task_id = str(task.id)
        await db.flush()

        return job

    async def get_by_id(
        self, db: AsyncSession, job_id: uuid.UUID, user_id: uuid.UUID
    ) -> ComparisonJob | None:
        result = await db.execute(
            select(ComparisonJob).where(
                ComparisonJob.id == job_id,
                ComparisonJob.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ComparisonJob], int]:
        base = select(ComparisonJob).where(ComparisonJob.user_id == user_id)

        count_result = await db.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar() or 0

        result = await db.execute(
            base.order_by(ComparisonJob.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    async def cancel(
        self, db: AsyncSession, job_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        job = await self.get_by_id(db, job_id, user_id)
        if not job:
            return False

        if job.status not in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
            raise ValueError("Job is not in a cancellable state")

        if job.celery_task_id:
            celery_app.control.revoke(job.celery_task_id, terminate=True)

        job.status = JobStatus.CANCELLED.value
        job.completed_at = datetime.now(timezone.utc)
        await db.flush()
        return True

    async def delete(
        self, db: AsyncSession, job_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        job = await self.get_by_id(db, job_id, user_id)
        if not job:
            return False

        if job.status == JobStatus.PROCESSING.value:
            raise ValueError("Cannot delete a running job")

        await db.delete(job)
        await db.flush()
        return True
