-- DriveSense telemetry schema (Plan.md Task 12).
-- Applied idempotently by db/pool.py:apply_schema() on FastAPI startup.

-- Frame-level detection events + alerts (Task 7, 13). Composite primary key
-- (id, timestamp) is required by TimescaleDB: create_hypertable() needs the
-- partitioning column in every unique/primary key constraint. Do not shrink
-- this to PRIMARY KEY (id) -- that breaks the hypertable conversion in
-- db/pool.py:try_create_hypertable().
CREATE TABLE IF NOT EXISTS detection_events (
    id          BIGSERIAL,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type  TEXT        NOT NULL,
    payload     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (id, timestamp)
);

-- One row per alert (Task 14), not per frame -- no hypertable needed here.
-- reason is an addition beyond Plan.md's stated interface: carries Task 7's
-- "light_green" / "lead_accelerating" so Task 17 can break reaction time
-- down by trigger type.
CREATE TABLE IF NOT EXISTS reaction_times (
    id                         BIGSERIAL PRIMARY KEY,
    alert_timestamp            TIMESTAMPTZ NOT NULL,
    driver_reaction_timestamp  TIMESTAMPTZ NOT NULL,
    delta_ms                   INTEGER     NOT NULL,
    reason                     TEXT
);

CREATE INDEX IF NOT EXISTS idx_detection_events_type_time
    ON detection_events (event_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_reaction_times_alert_time
    ON reaction_times (alert_timestamp DESC);
