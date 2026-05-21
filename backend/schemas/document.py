"""
backend/schemas/document.py — Document request/response schemas.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict


class DocumentUploadResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    file_size_bytes: int | None
    mime_type: str | None
    is_processed: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    file_size_bytes: int | None
    mime_type: str | None
    doc_title: str | None
    doc_number: str | None
    signing_date: str | None
    parties: list[str] | None
    page_count: int | None
    article_count: int | None
    is_processed: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
