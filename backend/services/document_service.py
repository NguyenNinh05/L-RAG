"""
backend/services/document_service.py — Document CRUD business logic.
"""

from __future__ import annotations

import uuid

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_backend_config
from backend.models.document import Document
from backend.models.comparison_job import ComparisonJob, JobStatus
from backend.services.storage import FileStorageManager


class DocumentService:
    def __init__(self, storage: FileStorageManager) -> None:
        self._storage = storage

    async def upload(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        file: UploadFile,
    ) -> Document:
        cfg = get_backend_config()
        filename = file.filename or "unknown"

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if f".{ext}" not in cfg.allowed_extensions:
            raise ValueError(
                f"Unsupported file type '.{ext}'. Allowed: {cfg.allowed_extensions}"
            )

        content = await file.read()
        if len(content) > cfg.max_upload_size_mb * 1024 * 1024:
            raise ValueError(
                f"File too large. Max: {cfg.max_upload_size_mb}MB"
            )

        storage_path, sha256, file_size = self._storage.store_upload(content, filename)

        # Dedup: same hash, same user → return existing
        result = await db.execute(
            select(Document).where(
                Document.content_hash_sha256 == sha256,
                Document.user_id == user_id,
            )
        )
        if existing := result.scalar_one_or_none():
            return existing

        doc = Document(
            user_id=user_id,
            original_filename=filename,
            storage_path=storage_path,
            file_size_bytes=file_size,
            mime_type=file.content_type or "application/octet-stream",
            content_hash_sha256=sha256,
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        return doc

    async def get_by_id(
        self, db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
    ) -> Document | None:
        result = await db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Document], int]:
        base = select(Document).where(Document.user_id == user_id)

        count_result = await db.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar() or 0

        result = await db.execute(
            base.order_by(Document.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    async def delete(
        self, db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        doc = await self.get_by_id(db, document_id, user_id)
        if not doc:
            return False

        # Check for active jobs referencing this document
        active_result = await db.execute(
            select(ComparisonJob).where(
                (
                    (ComparisonJob.document_v1_id == document_id)
                    | (ComparisonJob.document_v2_id == document_id)
                ),
                ComparisonJob.status.in_(
                    [JobStatus.PENDING.value, JobStatus.PROCESSING.value]
                ),
            )
        )
        if active_result.scalar_one_or_none():
            raise ValueError("Cannot delete document referenced by an active job")

        self._storage.delete_file(doc.storage_path)
        await db.delete(doc)
        await db.flush()
        return True
