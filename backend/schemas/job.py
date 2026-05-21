"""
backend/schemas/job.py — Job request/response schemas.
"""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, model_validator
import uuid


class CreateJobRequest(BaseModel):
    document_v1_id: uuid.UUID
    document_v2_id: uuid.UUID
    skip_phase3: bool = False
    config_overrides: dict | None = None

    @model_validator(mode="after")
    def _documents_must_differ(self):
        if self.document_v1_id == self.document_v2_id:
            raise ValueError("document_v1_id and document_v2_id must be different")
        return self


class JobResponse(BaseModel):
    id: uuid.UUID
    document_v1_id: uuid.UUID
    document_v2_id: uuid.UUID
    v1_filename: str = ""
    v2_filename: str = ""
    status: str
    current_phase: str
    progress_pct: int
    error_message: str | None = None
    total_pairs: int | None = None
    matched_count: int | None = None
    added_count: int | None = None
    deleted_count: int | None = None
    split_count: int | None = None
    merge_count: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class JobStatusResponse(BaseModel):
    id: uuid.UUID
    status: str
    current_phase: str
    progress_pct: int
    error_message: str | None = None
    celery_task_id: str | None = None

    model_config = ConfigDict(from_attributes=True)
