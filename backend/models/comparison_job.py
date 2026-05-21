"""
backend/models/comparison_job.py — ComparisonJob ORM model.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    from backend.models.user import User
    from backend.models.document import Document
    from backend.models.comparison_report import ComparisonReportModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPhase(str, Enum):
    QUEUED = "queued"
    INGESTION = "ingestion"
    ALIGNMENT = "alignment"
    COMPARISON = "comparison"
    DONE = "done"


class ComparisonJob(Base, TimestampMixin):
    __tablename__ = "comparison_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid_pk)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_v1_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False
    )
    document_v2_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False
    )

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(20), default=JobStatus.PENDING.value, index=True
    )
    current_phase: Mapped[str] = mapped_column(
        String(20), default=JobPhase.QUEUED.value
    )
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Pipeline output — catalog stored as JSONB
    catalog: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    v1_doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    v2_doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Denormalized catalog summary
    total_pairs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    matched_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    added_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deleted_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    split_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    merge_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Audit
    config_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True
    )

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="jobs")
    document_v1: Mapped["Document"] = relationship(
        "Document", foreign_keys=[document_v1_id], back_populates="jobs_as_v1"
    )
    document_v2: Mapped["Document"] = relationship(
        "Document", foreign_keys=[document_v2_id], back_populates="jobs_as_v2"
    )
    reports: Mapped[list["ComparisonReportModel"]] = relationship(
        "ComparisonReportModel", back_populates="job", cascade="all, delete-orphan"
    )
