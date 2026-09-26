"""Reaction-time capture (Plan.md Task 14).

Measures the gap between an alert firing (Task 7) and the driver reacting --
defined by Plan.md as Task 6's ego_stationary flipping True -> False.

Wire contract this depends on (documented in server/app.py's module
docstring too): the "event" message on /ws/ingest is sent every frame, not
only on alert, and always carries ego_stationary:
    {"type": "event", "alert": bool, "reason": str | null,
     "ego_stationary": bool, "timestamp": float}
Task 7's alert engine already needs ego_stationary as an input every frame,
so this is a near-free extension for the (still unbuilt) Task 11 to satisfy.

Single-pending-alert model: a new alert always replaces whatever was
pending. If the driver never reacted before the next alert fires, the most
recent alert is the one that matters for the demo -- there is no multi-alert
queue.
"""

import asyncio
import logging
from datetime import datetime

from db.telemetry import TelemetryLogger, to_timestamptz

logger = logging.getLogger("drivesense.reaction")


class ReactionTracker:
    def __init__(self, telemetry: TelemetryLogger) -> None:
        self._telemetry = telemetry
        self._pending: dict | None = None  # {"alert_timestamp": datetime, "reason": str | None}

    @property
    def pending_since(self) -> datetime | None:
        return self._pending["alert_timestamp"] if self._pending is not None else None

    def on_event(self, msg: dict) -> None:
        """Sync, called from ws_ingest for every "event" message. The only
        actual I/O (the DB write) is fired via create_task so this never
        blocks the ingest loop -- same rule as Task 13's telemetry calls."""
        ts = to_timestamptz(msg.get("timestamp"))

        if msg.get("alert"):
            self._pending = {"alert_timestamp": ts, "reason": msg.get("reason")}
            return

        if self._pending is not None and msg.get("ego_stationary") is False:
            alert_timestamp = self._pending["alert_timestamp"]
            reason = self._pending["reason"]
            self._pending = None

            delta_ms = round((ts - alert_timestamp).total_seconds() * 1000)
            if delta_ms < 0:
                # Producer clock jitter guard -- never store a negative
                # reaction time.
                delta_ms = 0

            asyncio.create_task(
                self._telemetry.log_reaction_time(alert_timestamp, ts, delta_ms, reason)
            )
