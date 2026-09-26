"""Telemetry logging service (Plan.md Task 13).

Persists frame-level detection events and alerts from /ws/ingest into
Tiger Data without blocking the WebSocket relay. Writes never happen inline
in the ingest loop -- messages are enqueued synchronously and drained by a
background batch writer, so a slow or momentarily unreachable Tiger Cloud
never lags the video feed.

Two things every caller must get right:
- Frames are NOT logged wholesale (~15/sec of base64 JPEG would be
  multi-MB/sec of JSONB). Only every Nth frame is sampled, via
  maybe_log_frame().
- The base64 payload (`data`) is always stripped before storage -- a ~200
  byte telemetry row, not a ~50KB one.
"""

import asyncio
import logging
from datetime import datetime, timezone

import asyncpg

import config

logger = logging.getLogger("drivesense.telemetry")

QUEUE_MAXSIZE = 2000
BATCH_MAX = 100
BATCH_INTERVAL = 1.0  # seconds

# Keys never persisted: base64 payloads (frame JPEG, Task 15 audio) and the
# envelope's own "type" (that becomes event_type, not part of the payload).
_STRIP_KEYS = ("data", "type")

# Shutdown sentinel pushed through the telemetry queue -- see stop().
_STOP = object()


def to_timestamptz(raw) -> datetime:
    """Producer sends time.time() floats; the column is TIMESTAMPTZ. asyncpg
    does not coerce floats, so this must happen explicitly. Anything
    unparseable falls back to "now" rather than failing the whole row."""
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(tz=timezone.utc)


def _sanitize_payload(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k not in _STRIP_KEYS}


class TelemetryLogger:
    """Owns the write queue + background batch-insert worker for
    detection_events, plus a direct (low-frequency) writer for
    reaction_times used by Task 14."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._queue: asyncio.Queue[tuple[datetime, str, dict]] = asyncio.Queue(
            maxsize=QUEUE_MAXSIZE
        )
        self._worker_task: asyncio.Task | None = None
        self._frame_counter = 0
        self.written = 0
        self.dropped = 0
        self.failed = 0

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Shut the worker down without losing anything. Pushes a sentinel
        through the same queue rather than task.cancel()-ing the worker --
        cancelling mid-collection would discard whatever the worker had
        already dequeued into its in-progress batch but not yet written
        (items are removed from the queue the moment _collect_batch() calls
        .get(), well before the batch is actually flushed). The sentinel is
        FIFO-ordered after every real item already queued, so the worker
        collects them all, writes that final batch, then exits on its own."""
        if self._worker_task is None:
            return
        await self._queue.put(_STOP)  # blocking put: waits for room if full
        await self._worker_task

    def log_event(self, msg: dict) -> None:
        """Fire-and-forget enqueue for event/alert/audio messages. Never
        awaits, never raises -- a full queue just drops and counts."""
        event_type = "alert" if msg.get("alert") else msg.get("type", "event")
        self._enqueue(event_type, msg)

    def maybe_log_frame(self, msg: dict) -> None:
        """Samples every TELEMETRY_FRAME_SAMPLE_N-th frame instead of every
        frame -- see module docstring for why."""
        self._frame_counter += 1
        if self._frame_counter % config.TELEMETRY_FRAME_SAMPLE_N != 0:
            return
        self._enqueue("frame_sample", msg)

    def _enqueue(self, event_type: str, msg: dict) -> None:
        timestamp = to_timestamptz(msg.get("timestamp"))
        payload = _sanitize_payload(msg)
        try:
            self._queue.put_nowait((timestamp, event_type, payload))
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning("telemetry queue full, dropped %s event", event_type)

    async def log_reaction_time(
        self,
        alert_timestamp: datetime,
        driver_reaction_timestamp: datetime,
        delta_ms: int,
        reason: str | None,
    ) -> None:
        """Low-frequency (one row per alert) -- written directly rather than
        through the batch queue. Used by Task 14."""
        try:
            await self._pool.execute(
                """
                INSERT INTO reaction_times
                    (alert_timestamp, driver_reaction_timestamp, delta_ms, reason)
                VALUES ($1, $2, $3, $4)
                """,
                alert_timestamp,
                driver_reaction_timestamp,
                delta_ms,
                reason,
            )
        except Exception:
            self.failed += 1
            logger.exception("failed to write reaction_times row")

    def stats(self) -> dict:
        return {
            "queued": self._queue.qsize(),
            "written": self.written,
            "dropped": self.dropped,
            "failed": self.failed,
        }

    async def _run(self) -> None:
        while True:
            try:
                batch, stopping = await self._collect_batch()
                if batch:
                    await self._write_batch(batch)
                if stopping:
                    return
            except Exception:
                logger.exception("telemetry worker iteration failed")

    async def _collect_batch(self) -> tuple[list[tuple[datetime, str, dict]], bool]:
        """Blocks for the first item, then drains up to BATCH_MAX more
        without blocking -- batches under load, low-latency when quiet.
        Returns (batch, stopping): stopping is True once _STOP is dequeued,
        which stop() relies on to flush its final batch before the worker
        exits (see stop()'s docstring)."""
        batch: list[tuple[datetime, str, dict]] = []
        item = await self._queue.get()
        if item is _STOP:
            return batch, True
        batch.append(item)

        try:
            while len(batch) < BATCH_MAX:
                item = await asyncio.wait_for(self._queue.get(), timeout=BATCH_INTERVAL)
                if item is _STOP:
                    return batch, True
                batch.append(item)
        except asyncio.TimeoutError:
            pass
        return batch, False

    async def _write_batch(self, batch: list[tuple[datetime, str, dict]]) -> None:
        try:
            await self._pool.executemany(
                """
                INSERT INTO detection_events (timestamp, event_type, payload)
                VALUES ($1, $2, $3)
                """,
                batch,
            )
            self.written += len(batch)
        except Exception:
            self.failed += len(batch)
            logger.exception("failed to write telemetry batch of %d row(s)", len(batch))
