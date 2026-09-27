"""Read-path queries for Tiger Data telemetry (Plan.md Task 17).

Kept separate from db/telemetry.py on purpose: that module owns the batched
write queue (TelemetryLogger) and must never be touched by a read path. This
module only ever issues SELECTs against the tables telemetry.py eventually
flushes to, for GET /api/metrics/summary.
"""

import asyncio

import asyncpg

_EMPTY_SUMMARY = {
    "db": "unavailable",
    "total_alerts": 0,
    "avg_reaction_ms": None,
    "reason_counts": {},
    "events_timeline": [],
    "reaction_timeline": [],
}

_REASON_COUNTS_SQL = """
    SELECT payload->>'reason' AS reason, count(*) AS n
    FROM detection_events
    WHERE event_type = 'alert'
    GROUP BY reason
"""

# date_bin(), not Timescale's time_bucket() -- plain Postgres 14+, so this
# still works if try_create_hypertable() degraded to a plain table.
_EVENTS_TIMELINE_SQL = """
    SELECT date_bin('30 seconds', timestamp, TIMESTAMPTZ '2000-01-01 00:00:00+00') AS bucket,
           payload->>'reason' AS reason,
           count(*) AS n
    FROM detection_events
    WHERE event_type = 'alert'
    GROUP BY bucket, reason
    ORDER BY bucket
"""

_AVG_REACTION_SQL = "SELECT avg(delta_ms)::float AS avg_ms FROM reaction_times"

_REACTION_TIMELINE_SQL = """
    SELECT alert_timestamp, delta_ms, reason
    FROM reaction_times
    ORDER BY alert_timestamp
"""


async def get_summary(pool: asyncpg.Pool | None) -> dict:
    """One-shot aggregate read for GET /api/metrics/summary. A missing pool
    (Tiger Cloud unconfigured/unreachable) returns the same zeroed shape the
    dashboard renders as "no data" -- same db-may-be-None convention as the
    rest of the repo (see db/pool.py's module docstring)."""
    if pool is None:
        return dict(_EMPTY_SUMMARY)

    reason_rows, timeline_rows, avg_row, reaction_rows = await asyncio.gather(
        pool.fetch(_REASON_COUNTS_SQL),
        pool.fetch(_EVENTS_TIMELINE_SQL),
        pool.fetchrow(_AVG_REACTION_SQL),
        pool.fetch(_REACTION_TIMELINE_SQL),
    )

    reason_counts = {(r["reason"] or "unknown"): r["n"] for r in reason_rows}

    events_timeline = [
        {
            "bucket": r["bucket"].isoformat(),
            "reason": r["reason"] or "unknown",
            "count": r["n"],
        }
        for r in timeline_rows
    ]

    reaction_timeline = [
        {
            "alert_timestamp": r["alert_timestamp"].isoformat(),
            "delta_ms": r["delta_ms"],
            "reason": r["reason"],
        }
        for r in reaction_rows
    ]

    return {
        "db": "connected",
        "total_alerts": sum(reason_counts.values()),
        "avg_reaction_ms": avg_row["avg_ms"] if avg_row is not None else None,
        "reason_counts": reason_counts,
        "events_timeline": events_timeline,
        "reaction_timeline": reaction_timeline,
    }
