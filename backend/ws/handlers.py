"""
backend/ws/handlers.py — WebSocket endpoint for job progress.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.models.comparison_job import ComparisonJob
from backend.services.auth_service import verify_token
from backend.ws.manager import ConnectionManager

ws_router = APIRouter()
manager = ConnectionManager()


@ws_router.websocket("/ws/jobs/{job_id}")
async def job_progress_websocket(
    websocket: WebSocket,
    job_id: str,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    # Authenticate via query-param token (WebSocket can't use Authorization header)
    try:
        payload = verify_token(token)
        user_id = payload.get("sub")
    except Exception:
        await websocket.close(code=4001)
        return

    # Authorize: user must own the job
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        await websocket.close(code=4004)
        return

    result = await db.execute(
        select(ComparisonJob).where(ComparisonJob.id == job_uuid)
    )
    job = result.scalar_one_or_none()

    if not job or str(job.user_id) != user_id:
        await websocket.close(code=4003)
        return

    await manager.connect(job_id, websocket)

    redis_task = asyncio.create_task(manager.listen_redis(job_id))

    try:
        while True:
            try:
                data = await websocket.receive_json()
                if data.get("event") == "ping":
                    await websocket.send_json({"event": "pong"})
            except WebSocketDisconnect:
                break
    finally:
        redis_task.cancel()
        try:
            await redis_task
        except asyncio.CancelledError:
            pass
        await manager.disconnect(job_id, websocket)
