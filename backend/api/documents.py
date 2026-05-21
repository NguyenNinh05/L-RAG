"""
backend/api/documents.py — Document upload and management endpoints.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db, get_storage
from backend.database import get_db as _get_db
from backend.models.user import User
from backend.schemas.document import DocumentResponse, DocumentUploadResponse
from backend.schemas.common import PaginatedResponse
from backend.services.document_service import DocumentService
from backend.services.storage import FileStorageManager

router = APIRouter()


def _svc(storage: FileStorageManager = Depends(get_storage)) -> DocumentService:
    return DocumentService(storage)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: DocumentService = Depends(_svc),
):
    try:
        doc = await svc.upload(db, user.id, file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return doc


@router.get("", response_model=PaginatedResponse[DocumentResponse])
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: DocumentService = Depends(_svc),
):
    docs, total = await svc.list_by_user(db, user.id, page, page_size)
    total_pages = (total + page_size - 1) // page_size
    return PaginatedResponse(
        items=docs,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: DocumentService = Depends(_svc),
):
    doc = await svc.get_by_id(db, document_id, user.id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    svc: DocumentService = Depends(_svc),
):
    try:
        deleted = await svc.delete(db, document_id, user.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
