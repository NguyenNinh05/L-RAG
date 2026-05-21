"""
backend/ws/manager.py — WebSocket connection manager with Redis pub/sub bridge.

Enables multiple uvicorn workers to broadcast progress to clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

import redis.asyncio as aredis
from fastapi import WebSocket

from backend.config import get_backend_config

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._redis_url = get_backend_config().redis_url
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[job_id].add(websocket)
        logger.info("[WS] Connected to job %s (%d total)", job_id, len(self._connections[job_id]))

    async def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        self._connections[job_id].discard(websocket)
        if not self._connections[job_id]:
            self._connections.pop(job_id, None)
        logger.info("[WS] Disconnected from job %s", job_id)

    async def _broadcast(self, job_id: str, message: dict) -> None:
        dead: set[WebSocket] = set()
        for ws in self._connections.get(job_id, set()):
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections[job_id].discard(ws)

    async def listen_redis(self, job_id: str) -> None:
        """Subscribe to Redis pub/sub and broadcast messages to local clients."""
        try:
            r = aredis.from_url(self._redis_url)
            pubsub = r.pubsub()
            await pubsub.subscribe(f"job:{job_id}:progress")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await self._broadcast(job_id, data)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[WS] Redis listener error for job %s", job_id)
        finally:
            try:
                await pubsub.unsubscribe()
                await r.close()
            except Exception:
                pass
