"""
backend/models/document.py — Document ORM model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    from backend.models.user import User
    from backend.models.comparison_job import ComparisonJob


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid_pk)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Upload metadata
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    content_hash_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )

    # Populated by Phase 1 ingestion
    doc_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    doc_number: Mapped[str | None] = mapped_column(String(128), nullable=True)
    signing_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parties: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    article_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="documents")

    jobs_as_v1: Mapped[list["ComparisonJob"]] = relationship(
        "ComparisonJob",
        foreign_keys="ComparisonJob.document_v1_id",
        back_populates="document_v1",
    )
    jobs_as_v2: Mapped[list["ComparisonJob"]] = relationship(
        "ComparisonJob",
        foreign_keys="ComparisonJob.document_v2_id",
        back_populates="document_v2",
    )
