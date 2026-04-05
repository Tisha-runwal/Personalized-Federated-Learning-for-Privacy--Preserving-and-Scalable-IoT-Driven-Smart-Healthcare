"""WebSocket endpoint that streams live training metrics to connected dashboards."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# List of all currently-connected WebSocket clients
_connected: list[WebSocket] = []


@router.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Accept a new WebSocket connection and keep it open until the client disconnects."""
    await websocket.accept()
    _connected.append(websocket)
    logger.info("WebSocket client connected. Total: %d", len(_connected))
    try:
        # Keep connection alive; we only push, but we still need to listen for
        # close frames or ping/pong.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _connected:
            _connected.remove(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(_connected))


async def broadcast_metric(data: dict) -> None:
    """Send *data* as JSON to every connected dashboard client.

    Disconnected sockets are silently removed from the connection list.
    """
    if not _connected:
        return

    payload = json.dumps(data)
    dead: list[WebSocket] = []

    for ws in list(_connected):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)

    for ws in dead:
        if ws in _connected:
            _connected.remove(ws)
