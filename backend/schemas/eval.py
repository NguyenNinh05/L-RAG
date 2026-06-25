"""
backend/schemas/eval.py — Benchmark evaluation response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel

from backend.schemas.report import AcuItemResponse


class EvalRunSummary(BaseModel):
    """Headline metrics for one run — the dashboard payload."""

    slug: str
    label: str
    num_pairs: int
    change_f1: float
    change_f1_semantic: float
    change_recall: float
    change_precision: float
    hallucination_rate: float
    avg_time_per_pair_seconds: float
    total_acus_generated: int
    total_acus_passed: int
    total_acus_rejected: int
    per_category: dict
    per_category_semantic: dict


class EvalRunDetail(BaseModel):
    slug: str
    label: str
    num_pairs: int
    aggregate: dict
    markdown_report: str | None


class EvalPairSummary(BaseModel):
    pair_name: str
    recall: float
    precision: float
    f1: float
    recall_semantic: float
    recall_article: float
    hallucination_rate: float
    total_time: float
    num_acus: int


class EvalPairDetail(BaseModel):
    pair_name: str
    v1_file: str | None
    v2_file: str | None
    gt_file: str | None
    phase1: dict
    phase2: dict
    phase3: dict
    ground_truth: dict
    change_detection: dict
    per_category: dict
    per_category_semantic: dict
    acus: list[AcuItemResponse]
    total_time: float
    errors: list
