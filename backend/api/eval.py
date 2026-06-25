"""
backend/api/eval.py — Benchmark evaluation dashboards & detailed reports.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.api.jobs import _acu_to_item
from backend.models.eval import EvalPair, EvalRun
from backend.models.user import User
from backend.schemas.eval import (
    EvalPairDetail,
    EvalPairSummary,
    EvalRunDetail,
    EvalRunSummary,
)

router = APIRouter()


def _run_summary(run: EvalRun) -> EvalRunSummary:
    a = run.aggregate or {}
    g = lambda key, default=0.0: a.get(key, default)  # noqa: E731

    return EvalRunSummary(
        slug=run.slug,
        label=run.label,
        num_pairs=run.num_pairs,
        change_f1=float(g("change_f1")),
        change_f1_semantic=float(g("change_f1_semantic")),
        change_recall=float(g("change_recall")),
        change_precision=float(g("change_precision")),
        hallucination_rate=float(g("hallucination_rate")),
        avg_time_per_pair_seconds=float(g("avg_time_per_pair_seconds")),
        total_acus_generated=int(g("total_acus_generated")),
        total_acus_passed=int(g("total_acus_passed")),
        total_acus_rejected=int(g("total_acus_rejected")),
        per_category=a.get("per_category", {}) or {},
        per_category_semantic=a.get("per_category_semantic", {}) or {},
    )


async def _get_run(db: AsyncSession, slug: str) -> EvalRun:
    row = await db.execute(select(EvalRun).where(EvalRun.slug == slug))
    run = row.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Eval run not found")
    return run


@router.get("/runs", response_model=list[EvalRunSummary])
async def list_runs(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.execute(select(EvalRun).order_by(EvalRun.slug))
    return [_run_summary(r) for r in rows.scalars().all()]


@router.get("/runs/{slug}", response_model=EvalRunDetail)
async def get_run(
    slug: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_run(db, slug)
    return EvalRunDetail(
        slug=run.slug,
        label=run.label,
        num_pairs=run.num_pairs,
        aggregate=run.aggregate or {},
        markdown_report=run.markdown_report,
    )


@router.get("/runs/{slug}/pairs", response_model=list[EvalPairSummary])
async def list_pairs(
    slug: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_run(db, slug)
    rows = await db.execute(
        select(EvalPair).where(EvalPair.run_id == run.id).order_by(EvalPair.pair_name)
    )
    out: list[EvalPairSummary] = []
    for p in rows.scalars().all():
        cd = p.change_detection or {}
        ph3 = p.phase3 or {}
        recall = float(cd.get("recall", 0.0))
        precision = float(cd.get("precision", 0.0))
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        out.append(
            EvalPairSummary(
                pair_name=p.pair_name,
                recall=recall,
                precision=precision,
                f1=f1,
                recall_semantic=float(cd.get("recall_semantic", 0.0)),
                recall_article=float(cd.get("recall_article", 0.0)),
                hallucination_rate=float(ph3.get("hallucination_rate", 0.0)),
                total_time=float(p.total_time or 0.0),
                num_acus=len(p.acus or []),
            )
        )
    return out


@router.get("/runs/{slug}/pairs/{pair_name}", response_model=EvalPairDetail)
async def get_pair(
    slug: str,
    pair_name: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_run(db, slug)
    row = await db.execute(
        select(EvalPair).where(EvalPair.run_id == run.id, EvalPair.pair_name == pair_name)
    )
    p = row.scalar_one_or_none()
    if p is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Eval pair not found")

    # Map stored ACU dicts → frontend AcuItem shape (eval ACUs have no acu_id).
    acus = []
    for i, raw in enumerate(p.acus or []):
        if not isinstance(raw, dict):
            continue
        a = dict(raw)
        if not a.get("acu_id"):
            a["acu_id"] = f"acu_{i}"
        acus.append(_acu_to_item(a))

    return EvalPairDetail(
        pair_name=p.pair_name,
        v1_file=p.v1_file,
        v2_file=p.v2_file,
        gt_file=p.gt_file,
        phase1=p.phase1 or {},
        phase2=p.phase2 or {},
        phase3=p.phase3 or {},
        ground_truth=p.ground_truth or {},
        change_detection=p.change_detection or {},
        per_category=p.per_category or {},
        per_category_semantic=p.per_category_semantic or {},
        acus=acus,
        total_time=float(p.total_time or 0.0),
        errors=p.errors or [],
    )
