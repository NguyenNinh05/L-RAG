"""
backend/celery_app.py — Celery application configuration.

Critical: worker_concurrency=1 to ensure only one GPU-bound pipeline runs at a time.
"""

from __future__ import annotations

from celery import Celery

from backend.config import get_backend_config

cfg = get_backend_config()

celery_app = Celery(
    "legaldiff",
    broker=cfg.redis_url,
    backend=cfg.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=86400 * 7,
    task_track_started=True,
    task_store_errors_even_if_ignored=True,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Autodiscover tasks
celery_app.autodiscover_tasks(["backend.workers"])
