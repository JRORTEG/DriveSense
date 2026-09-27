"""Tiger Data (Postgres/TimescaleDB) connection pool + schema management.

Task 12 of Plan.md. Target is Tiger Cloud exclusively (no local fallback,
no Docker -- see temp/TIGER_CLOUD_SETUP.md for why). The pool is designed to
degrade gracefully: a missing or unreachable TIGER_DATA_DSN must never crash
the server, since the WebSocket relay (Task 10) is the demo's core and has
to keep running with or without a database. Every caller in Task 13+ must
treat create_pool() returning None as "skip DB logging", not as an error.
"""

import json
import logging
from pathlib import Path

import asyncpg

import config

logger = logging.getLogger("drivesense.db")

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Anything that looks like the unfilled .env.example placeholder should be
# treated the same as "no DSN configured" -- don't attempt to connect to it.
_PLACEHOLDER_MARKERS = ("user:password", "<password>", "<host>")


def _dsn_looks_configured(dsn: str | None) -> bool:
    if not dsn:
        return False
    return not any(marker in dsn for marker in _PLACEHOLDER_MARKERS)


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register a JSONB codec so dicts round-trip natively. asyncpg has no
    built-in dict<->JSONB mapping -- without this, every insert would need
    an explicit $n::jsonb cast and every read would come back as a string
    needing manual json.loads. Task 13's telemetry writer and Task 17's
    dashboard reads both depend on this being registered once, here."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def create_pool() -> asyncpg.Pool | None:
    """Connect to Tiger Cloud. Returns None (never raises) if unconfigured
    or unreachable, so the caller can boot without a database."""
    dsn = config.TIGER_DATA_DSN
    if not _dsn_looks_configured(dsn):
        logger.warning(
            "TIGER_DATA_DSN not configured (still a placeholder or unset) -- "
            "starting without a database. See DEMO.md and .env.example."
        )
        return None

    try:
        pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=5,
            timeout=10,
            command_timeout=10,
            init=_init_connection,
        )
        logger.info("connected to Tiger Data")
        return pool
    except Exception:
        logger.exception("failed to connect to TIGER_DATA_DSN -- starting without a database")
        return None


async def apply_schema(pool: asyncpg.Pool) -> None:
    """Run db/schema.sql. Idempotent -- every statement is IF NOT EXISTS."""
    try:
        ddl = SCHEMA_PATH.read_text()
        async with pool.acquire() as conn:
            await conn.execute(ddl)
        logger.info("schema applied (detection_events, reaction_times)")
    except Exception:
        logger.exception("failed to apply schema.sql")


async def try_create_hypertable(pool: asyncpg.Pool) -> bool:
    """Attempt to convert detection_events into a TimescaleDB hypertable.

    Kept separate from schema.sql on purpose: if the Tiger Cloud service
    was created as plain Postgres (wrong service type) or the extension
    can't be installed, this logs a warning and the caller keeps running
    against a plain table instead of aborting startup.
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            await conn.execute(
                "SELECT create_hypertable('detection_events', 'timestamp', "
                "if_not_exists => TRUE);"
            )
        logger.info("detection_events is a TimescaleDB hypertable")
        return True
    except Exception:
        logger.warning(
            "timescaledb unavailable, using plain table for detection_events",
            exc_info=True,
        )
        return False


async def close_pool(pool: asyncpg.Pool | None) -> None:
    if pool is not None:
        await pool.close()
        logger.info("Tiger Data pool closed")
