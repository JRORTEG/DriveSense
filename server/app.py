"""FastAPI relay: Windows PC (producer) -> Mac (this app) -> browser HUD (consumer).

Task 10 of Plan.md. This app has two WebSocket endpoints so the producer and
consumers never share a socket:

- ``/ws/ingest``  Windows PC connects here as a client and pushes JSON text
  frames shaped like one of:
    {"type": "frame", "data": "<base64 JPEG>", "timestamp": float}
    {"type": "event", "alert": bool, "reason": str | null, "timestamp": float, ...}
    {"type": "audio", "reason": str, "data": "<base64>"}   # added by Task 15

- ``/ws/stream``  Browser clients connect here. Every message received on
  ``/ws/ingest`` is broadcast to every connected ``/ws/stream`` client.

Frames use latest-frame-wins backpressure: a slow browser tab drops stale
frames instead of building a queue, so the HUD stays real-time. Events are
never dropped this way -- they drive DB logging (Task 13) and the alert
banner (Task 16), so they queue instead.

Must bind to host="0.0.0.0" (see main.py) so the Windows PC can reach this
over LAN, and CORS must allow all origins for the same reason. Both are
hackathon LAN-only choices, not production-safe.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("drivesense.server")

EVENT_QUEUE_MAXSIZE = 256


class StreamClient:
    """Per-browser-client fan-out buffer: latest-frame-wins + queued events."""

    def __init__(self) -> None:
        self.latest_frame: str | None = None
        self.events: asyncio.Queue[str] = asyncio.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
        self.wake = asyncio.Event()

    def offer_frame(self, raw: str) -> None:
        self.latest_frame = raw
        self.wake.set()

    def offer_event(self, raw: str) -> None:
        try:
            self.events.put_nowait(raw)
        except asyncio.QueueFull:
            # Client is stuck badly enough to back up the event queue too.
            # Drop the oldest event rather than the newest -- newest carries
            # the freshest alert state.
            try:
                self.events.get_nowait()
                self.events.put_nowait(raw)
            except asyncio.QueueEmpty:
                pass
            logger.warning("stream client event queue full, dropped oldest event")
        self.wake.set()

    async def writer(self, websocket: WebSocket) -> None:
        """Drain events (never dropped) before the latest frame (may be stale)."""
        try:
            while True:
                await self.wake.wait()
                self.wake.clear()

                while not self.events.empty():
                    event_raw = await self.events.get()
                    await websocket.send_text(event_raw)

                if self.latest_frame is not None:
                    frame_raw = self.latest_frame
                    self.latest_frame = None
                    await websocket.send_text(frame_raw)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("stream client writer failed")


class ConnectionHub:
    """Tracks connected /ws/stream clients and relays /ws/ingest traffic to them."""

    def __init__(self) -> None:
        self.clients: set[StreamClient] = set()
        self.frames_relayed = 0
        self.events_relayed = 0

    def register(self) -> StreamClient:
        client = StreamClient()
        self.clients.add(client)
        return client

    def unregister(self, client: StreamClient) -> None:
        self.clients.discard(client)

    def broadcast(self, raw: str, msg_type: str) -> None:
        if msg_type == "frame":
            self.frames_relayed += 1
        else:
            self.events_relayed += 1
        for client in self.clients:
            if msg_type == "frame":
                client.offer_frame(raw)
            else:
                client.offer_event(raw)

    def summary(self) -> dict:
        return {
            "status": "ok",
            "stream_clients": len(self.clients),
            "frames_relayed": self.frames_relayed,
            "events_relayed": self.events_relayed,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.hub = ConnectionHub()
    # Task 12 will attach app.state.db_pool here too.
    yield


app = FastAPI(title="DriveSense Relay", lifespan=lifespan)

# Hackathon LAN scope only -- allow every origin so the Windows PC and any
# browser on the local network can talk to this server without a CORS
# preflight failure. Not appropriate outside a demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return app.state.hub.summary()


@app.websocket("/ws/ingest")
async def ws_ingest(websocket: WebSocket) -> None:
    """Windows PC connects here and pushes frame/event/audio JSON messages."""
    await websocket.accept()
    logger.info("producer connected on /ws/ingest")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type")
            except (json.JSONDecodeError, AttributeError):
                logger.warning("dropping malformed /ws/ingest message: %.200s", raw)
                continue

            if msg_type not in ("frame", "event", "audio"):
                logger.warning("dropping unknown message type: %r", msg_type)
                continue

            try:
                app.state.hub.broadcast(raw, msg_type)
            except Exception:
                logger.exception("broadcast failed for type=%s", msg_type)
    except WebSocketDisconnect:
        logger.info("producer disconnected from /ws/ingest")
    except Exception:
        logger.exception("unexpected error on /ws/ingest")


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """Browser clients connect here and receive everything broadcast above."""
    await websocket.accept()
    hub: ConnectionHub = app.state.hub
    client = hub.register()
    writer_task = asyncio.create_task(client.writer(websocket))
    logger.info("browser client connected on /ws/stream (%d total)", len(hub.clients))
    try:
        while True:
            # We don't expect browser->server traffic on this endpoint; this
            # recv just lets us notice the disconnect promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("browser client disconnected from /ws/stream")
    except Exception:
        logger.exception("unexpected error on /ws/stream")
    finally:
        writer_task.cancel()
        hub.unregister(client)
