"""
backend/models/eval.py — Benchmark evaluation run & per-pair result ORM models.

These store the output of the offline evaluation harness
(`evaluation/outputs/<run>/`) so the frontend can render dashboards and
detailed per-pair reports. Data is global (not user-scoped) benchmark data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    pass


class EvalRun(Base, TimestampMixin):
    """One benchmark run (e.g. 'v2', 'v3', 'deepseek')."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid_pk)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    source_path: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    num_pairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    aggregate: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    markdown_report: Mapped[str | None] = mapped_column(Text, nullable=True)


class EvalPair(Base, TimestampMixin):
    """One pair result within an EvalRun (e.g. '01-tand_signed')."""

    __tablename__ = "eval_pairs"
    __table_args__ = (UniqueConstraint("run_id", "pair_name", name="uq_eval_pairs_run_pair"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid_pk)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pair_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    v1_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    v2_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    gt_file: Mapped[str | None] = mapped_column(String(512), nullable=True)

    phase1: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    phase2: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    phase3: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    ground_truth: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    change_detection: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    per_category: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    per_category_semantic: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    acus: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    errors: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    total_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
