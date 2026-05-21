"""
backend/workers/gpu_lock.py — Redis-based distributed mutex for GPU access.

Ensures only one pipeline uses GPU-bound operations (Phase 2: BGE-M3)
at any time across all Celery workers.
"""

from __future__ import annotations

import logging

import redis

from backend.config import get_backend_config

logger = logging.getLogger(__name__)

LOCK_KEY = "gpu:lock"
LOCK_TIMEOUT = 900  # 15 minutes max pipeline time


class GPUBusyError(Exception):
    """Raised when GPU is locked by another job."""


class GPULock:
    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or get_backend_config().redis_url
        self._client = redis.from_url(self._redis_url)

    def acquire(self, job_id: str) -> bool:
        """
        Try to acquire the GPU lock. Non-blocking.

        Returns True if lock was acquired, False if GPU is busy.
        """
        return bool(
            self._client.set(LOCK_KEY, job_id, nx=True, ex=LOCK_TIMEOUT)
        )

    def release(self, job_id: str) -> None:
        """Release the GPU lock only if held by this job_id."""
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        result = self._client.eval(script, 1, LOCK_KEY, job_id)
        if result:
            logger.info("[GPU Lock] Released for job %s", job_id)

    def is_locked(self) -> bool:
        return bool(self._client.exists(LOCK_KEY))

    def current_holder(self) -> str | None:
        holder = self._client.get(LOCK_KEY)
        return holder.decode() if holder else None
