"""
backend/workers/pipeline_task.py — Celery task to execute the 3-phase AI pipeline.

Orchestrates: GPU lock → Phase 1 → Phase 2 → Phase 3 → store results → release lock.
Progress is reported via Redis pub/sub for WebSocket broadcast.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import redis as sync_redis
from celery import Task
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.celery_app import celery_app
from backend.config import get_backend_config
from backend.database import _sync_engine
from backend.models.comparison_job import ComparisonJob, JobStatus, JobPhase
from backend.models.comparison_report import ComparisonReportModel
from backend.models.document import Document
from backend.workers.gpu_lock import GPULock, GPUBusyError

logger = logging.getLogger(__name__)

# Ensure the project root is on the path so src/* imports work
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _publish_progress(job_id: str, pct: int, phase: str, message: str) -> None:
    cfg = get_backend_config()
    try:
        r = sync_redis.from_url(cfg.redis_url)
        r.publish(
            f"job:{job_id}:progress",
            json.dumps({
                "event": "progress",
                "job_id": job_id,
                "progress_pct": pct,
                "current_phase": phase,
                "message": message,
            }),
        )
        r.close()
    except Exception:
        logger.warning("[Job %s] Failed to publish progress", job_id)


def _update_job_db(job_id: str, **kwargs) -> None:
    """Synchronously update a comparison_job row."""
    engine = _sync_engine()
    with Session(engine) as session:
        stmt = (
            update(ComparisonJob)
            .where(ComparisonJob.id == job_id)
            .values(**kwargs, updated_at=datetime.now(timezone.utc))
        )
        session.execute(stmt)
        session.commit()


class PipelineTask(Task):
    autoretry_for = (GPUBusyError, ConnectionError, TimeoutError)
    max_retries = 3
    default_retry_delay = 30

    def __init__(self) -> None:
        self._gpu_lock = GPULock()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0] if args else kwargs.get("job_id", "unknown")
        _update_job_db(
            job_id,
            status=JobStatus.FAILED.value,
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc),
        )
        _publish_progress(
            job_id, 0, JobPhase.DONE.value,
            f"Thất bại: {exc}",
        )
        try:
            self._gpu_lock.release(str(job_id))
        except Exception:
            pass


@celery_app.task(base=PipelineTask, bind=True, name="run_pipeline")
def run_pipeline(
    self,
    job_id: str,
    file_v1_path: str,
    file_v2_path: str,
    config_overrides: dict | None = None,
) -> dict:
    """
    Execute the full 3-phase LegalDiff pipeline.

    Args:
        job_id: UUID string of the ComparisonJob.
        file_v1_path: Absolute path to V1 file in storage.
        file_v2_path: Absolute path to V2 file in storage.
        config_overrides: Optional pipeline config overrides.

    Returns:
        Summary dict with status, total_reports, catalog_summary.
    """
    cfg = get_backend_config()
    job_id_str = str(job_id)

    # ---------- GPU Lock ----------
    if not self._gpu_lock.acquire(job_id_str):
        raise GPUBusyError("GPU is busy with another job")

    try:
        _update_job_db(
            job_id_str,
            status=JobStatus.PROCESSING.value,
            started_at=datetime.now(timezone.utc),
        )

        # ---------- Phase 1: Ingestion (0-30%) ----------
        _publish_progress(
            job_id_str, 5, JobPhase.INGESTION.value, "Đang phân tích tài liệu..."
        )
        _update_job_db(job_id_str, current_phase=JobPhase.INGESTION.value, progress_pct=5)

        from src.pipeline import LegalDiffPipeline, PipelineRunConfig

        def progress_cb(pct: int, phase: str, message: str) -> None:
            _publish_progress(job_id_str, pct, phase, message)
            _update_job_db(job_id_str, current_phase=phase, progress_pct=pct)

        pipeline_cfg = PipelineRunConfig(
            file_v1=file_v1_path,
            file_v2=file_v2_path,
            progress_callback=progress_cb,
        )

        # Apply config overrides
        if config_overrides:
            for key, val in config_overrides.items():
                if hasattr(pipeline_cfg, key):
                    setattr(pipeline_cfg, key, val)

        pipeline = LegalDiffPipeline(run_config=pipeline_cfg)

        # Run the pipeline (synchronous — wraps asyncio internally)
        _publish_progress(job_id_str, 10, JobPhase.INGESTION.value, "Đang parse tài liệu...")
        _update_job_db(job_id_str, progress_pct=10)

        result = pipeline.run()

        # ---------- Store Results ----------
        _publish_progress(job_id_str, 95, JobPhase.COMPARISON.value, "Đang lưu kết quả...")
        _update_job_db(job_id_str, progress_pct=95)

        _store_results(job_id_str, result)

        # ---------- Done ----------
        _update_job_db(
            job_id_str,
            status=JobStatus.COMPLETED.value,
            current_phase=JobPhase.DONE.value,
            progress_pct=100,
            completed_at=datetime.now(timezone.utc),
        )
        _publish_progress(
            job_id_str, 100, JobPhase.DONE.value, "Hoàn thành!"
        )

        return {
            "status": "completed",
            "job_id": job_id_str,
            "total_reports": len(result.get("reports", [])),
        }

    except Exception:
        logger.exception("[Job %s] Pipeline failed", job_id_str)
        raise

    finally:
        self._gpu_lock.release(job_id_str)


def _store_results(job_id_str: str, result: dict) -> None:
    """Store pipeline output in PostgreSQL."""
    engine = _sync_engine()

    doc_v1 = result.get("doc_v1")
    doc_v2 = result.get("doc_v2")
    catalog = result.get("catalog")
    reports = result.get("reports", [])

    with Session(engine) as session:
        # Update catalog on the job
        catalog_dict = catalog.to_report_dict() if catalog else {}
        summary = catalog.summary() if catalog else {}

        stmt = (
            update(ComparisonJob)
            .where(ComparisonJob.id == job_id_str)
            .values(
                catalog=catalog_dict,
                v1_doc_id=str(doc_v1.doc_id) if doc_v1 else None,
                v2_doc_id=str(doc_v2.doc_id) if doc_v2 else None,
                total_pairs=summary.get("total_pairs"),
                matched_count=summary.get("matched"),
                added_count=summary.get("added"),
                deleted_count=summary.get("deleted"),
                split_count=summary.get("split"),
                merge_count=summary.get("merged"),
                updated_at=datetime.now(timezone.utc),
            )
        )
        session.execute(stmt)

        # Insert reports
        for report in reports:
            try:
                report_dict = report.to_machine_readable()
            except AttributeError:
                report_dict = {}

            row = ComparisonReportModel(
                job_id=job_id_str,
                pair_id=getattr(report, "pair_id", ""),
                match_type=getattr(report, "match_type", "matched")
                if hasattr(report, "match_type")
                else "matched",
                location_context=getattr(report, "location_context", ""),
                total_acus=getattr(report, "total_acus_passed", 0)
                + getattr(report, "total_acus_rejected", 0),
                passed_acus=getattr(report, "total_acus_passed", 0),
                rejected_acus=getattr(report, "total_acus_rejected", 0),
                hallucination_rate=getattr(report, "hallucination_rate", 0.0),
                executive_summary=(
                    report.executive_summary.model_dump()
                    if getattr(report, "executive_summary", None)
                    else None
                ),
                markdown_report=getattr(report, "markdown_report", ""),
                verified_acus=[
                    acu.to_dict() if hasattr(acu, "to_dict") else acu
                    for acu in getattr(report, "verified_acus", [])
                ],
                rejected_acus_detail=[
                    {
                        "acu": v.acu.to_dict() if hasattr(v, "acu") and hasattr(v.acu, "to_dict") else {},
                        "status": getattr(v, "status", "failed"),
                        "reason": getattr(v, "rejection_reason", ""),
                    }
                    for v in getattr(report, "rejected_acus", [])
                ],
            )
            session.add(row)

        session.commit()
