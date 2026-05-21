"""
backend/models/comparison_report.py — ComparisonReport ORM model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    from backend.models.comparison_job import ComparisonJob


class ComparisonReportModel(Base, TimestampMixin):
    """One row per matched DiffPair processed by Phase 3."""

    __tablename__ = "comparison_reports"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid_pk)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("comparison_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    pair_id: Mapped[str] = mapped_column(String(64), nullable=False)
    match_type: Mapped[str] = mapped_column(String(20), nullable=False)
    location_context: Mapped[str] = mapped_column(String(512), default="")

    # Denormalized stats
    total_acus: Mapped[int] = mapped_column(Integer, default=0)
    passed_acus: Mapped[int] = mapped_column(Integer, default=0)
    rejected_acus: Mapped[int] = mapped_column(Integer, default=0)
    hallucination_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # Full output
    executive_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    markdown_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    verified_acus: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    rejected_acus_detail: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Relationship
    job: Mapped["ComparisonJob"] = relationship(
        "ComparisonJob", back_populates="reports"
    )
