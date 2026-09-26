# DriveSense — Hackathon Execution Plan

AI-powered Smart Dashcam: contextual navigation + distraction alerts at intersections. Deadline: Sunday 11:00 AM.

Order to build (risk/value first, then satisfy sponsor tracks): 0→1→2→3→4→5→6→7 (core CV + alert logic) → 10→11 (streaming demo safety net) → 8→9 (HUD polish) → 12→13→14 (DB, MLH track) → 15 (ElevenLabs, MLH track) → 16→17 (frontend) → 18 (hardening).

Each task block below is standalone — copy-paste it back as its own prompt to generate the code.

---

## Hardware Infrastructure

Split dual-machine architecture:
1. **Windows PC (NVIDIA):** Runs the heavy CV pipeline (Tasks 1-9). Producer — streams annotated frames + event JSON payloads over WebSocket/HTTP to the Mac.
2. **Mac (M2 Pro):** Runs the central FastAPI server, Tiger Data database, ElevenLabs cached audio player, and hosts the HTML5 HUD / Chart.js frontend UI (Tasks 10-18).

FastAPI endpoints must bind to `0.0.0.0` (not `127.0.0.1`) with CORS enabled, so the Windows PC can stream frames and event payloads to the Mac over the local network.

---

## TASK 0: Project Scaffold & Environment (COMPLETE)

**Objective:** Set up repo skeleton, dependencies, config loading.

**Technical Approach:**
- Python 3.11 venv. `requirements.txt`: `opencv-python`, `ultralytics` (YOLOv8), `fastapi`, `uvicorn[standard]`, `websockets`, `psycopg2-binary` or `asyncpg`, `python-dotenv`, `elevenlabs`, `numpy`.
- `.env` for `TIGER_DATA_DSN`, `ELEVENLABS_API_KEY`. **Split by machine:** Windows PC's `.env` needs `SERVER_URL` (Mac's LAN address, e.g. `ws://<mac-ip>:8000/ws/ingest`); Mac's `.env` needs `TIGER_DATA_DSN`, `ELEVENLABS_API_KEY`.
- Folder layout: `cv/` (Windows), `server/`, `db/`, `frontend/`, `audio/` (Mac), `main.py`.

**Interface:** Input: none. Output: runnable skeleton, `pip install -r requirements.txt` succeeds, `.env.example` committed on both machines.

---

## TASK 1: Video Ingestion Module (Windows) (COMPLETE — merged to main, `cv/frame_source.py`)

**Objective:** Unified frame source — webcam or pre-recorded file, looped.

**Technical Approach:** `cv2.VideoCapture(source)` wrapped in a class `FrameSource` with `.read() -> np.ndarray | None`, auto-loop on file EOF, configurable target FPS via `cv2.CAP_PROP_FPS` or manual sleep throttle.

**Interface:** Input: `source: int|str` (webcam index or file path), `loop: bool`. Output: generator/iterator yielding `(frame: np.ndarray, timestamp: float)`.

**Status:** Implemented in `cv/frame_source.py`, merged to `main`.

---

## TASK 2: Traffic Light State Detector (HSV) (Windows) (COMPLETE — `cv/traffic_light.py`, branch `task-2-traffic-light`, not yet merged to main)

**Objective:** Detect traffic light bounding region and classify state (red/yellow/green) via HSV masking; emit transition events (red→green).

**Technical Approach:** Fixed or YOLO-detected ROI for light housing → convert ROI to HSV → `cv2.inRange` masks for red (two hue ranges wrapping 0/180), yellow, green → largest contour + pixel count threshold decides active color → maintain last-state to detect red→green edge.

**Interface:** Input: `frame: np.ndarray`, optional `roi: (x,y,w,h)`. Output: JSON `{"state": "red"|"yellow"|"green"|"unknown", "transitioned_to_green": bool, "timestamp": float}`.

**Status:** Implemented in `cv/traffic_light.py` (`TrafficLightDetector`). Tuned on real footage (`ampel_red_to_green.ogv`): lit lamps overexpose to a near-white core, so the HSV floors are S≥80, V≥110 (hue: red 0–10 & 165–179, yellow 15–35, green 40–100); min blob 8 px / 0.2% of ROI; morphological opening only for ROIs ≥100 px per side. State changes need 3 consecutive frames (~0.1 s at 30 FPS); the red→green edge compares against the last *known* state so brief `unknown` gaps don't hide it. No ROI → `unknown` (never scans the whole frame). Preview: `python -m cv.traffic_light <video> --select` (or `--roi x,y,w,h`, `--yolo yolov8n.pt`). Retune against dashcam clips, especially night footage.

---

## TASK 3: Vehicle Detector (YOLO) (Windows) (COMPLETE — `cv/vehicle_detector.py`, branch `task-3-vehicle-detector`, not yet merged to main)

**Objective:** Detect vehicles per frame with bounding boxes + class.

**Technical Approach:** `ultralytics.YOLO("yolov8n.pt")`, filter classes `car, truck, bus, motorcycle`, confidence threshold ~0.4.

**Interface:** Input: `frame: np.ndarray`. Output: JSON list `[{"id": null, "bbox": [x1,y1,x2,y2], "class": "car", "conf": 0.87}, ...]`.

**Status:** Implemented in `cv/vehicle_detector.py` (`VehicleDetector`). Auto-picks `cuda:0` when available (FP16 via `quantize=16`, warm-up inference at construction) and falls back to CPU. Verified on `intersection_montreal_720p.webm`: ~85 FPS on an RTX 2000 Ada Laptop GPU, correct boxes on cars/trucks, cyclist correctly excluded. Preview/benchmark: `python -m cv.vehicle_detector <video> [--out annotated.mp4]`. Needs a Python 3.11 venv (`py install 3.11`; the system default was 3.14, which `torch`/`ultralytics` don't support yet) with CUDA `torch` (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`, matched to the installed driver's CUDA version) installed before `pip install -r requirements.txt`. On Windows, `torch`'s DLLs also need the Microsoft VC++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe) — without it, `import torch` fails with `WinError 126`.

---

## TASK 4: Simple Multi-Object Tracker (Windows) (COMPLETE — `cv/tracker.py`, branch `task-4-vehicle-tracker`, not yet merged to main)

**Objective:** Assign persistent IDs to detected vehicles across frames (needed for lead-vehicle acceleration + reference-vehicle highlight).

**Technical Approach:** Centroid-distance tracker (IoU or Euclidean centroid matching, no external lib) — dict of `track_id -> {centroid, bbox, last_seen, velocity_history}`. Prune tracks unseen > N frames.

**Interface:** Input: current-frame detection list (Task 3 output) + previous track state. Output: JSON `[{"track_id": int, "bbox": [...], "class": str, "velocity": [vx,vy]}, ...]` + updated internal state.

**Status:** Implemented in `cv/tracker.py` (`VehicleTracker`, stateful — call `.update(detections)` once per frame). Greedy nearest-match: IoU when boxes overlap ≥ `min_iou` (0.15 default), else centroid distance within a `max_center_dist` gate (120px default); IoU matches always win over distance-only ones. No scipy/Hungarian algorithm needed. Class is majority-voted over a track's history so single-frame car/truck misclassification doesn't flicker the reported label. Velocity is the average frame-to-frame centroid displacement over the last `velocity_window` frames (5 default), in px/frame. Tracks unseen for `max_age` frames (15 default, ~0.5s at 30 FPS) are pruned and a later reappearance gets a new ID (no re-identification). `update()` returns only tracks matched in the current frame, not stale coasted boxes. Verified on `intersection_montreal_720p.webm`: ~95 FPS end-to-end with Task 3's detector on an RTX 2000 Ada; a parked car held the same ID across all 300 test frames, and two crossing vehicles didn't swap IDs. Unit-tested for occlusion recovery, pruning, and new-ID-on-reappearance. Preview: `python -m cv.tracker <video>`.

---

## TASK 5: Lead-Vehicle Acceleration Detector (Windows)

**Objective:** Identify the "lead vehicle" (closest tracked vehicle roughly centered ahead) and detect when it starts accelerating from rest.

**Technical Approach:** Pick track with largest bbox area + smallest horizontal offset from frame center as lead vehicle. Maintain rolling centroid-position buffer (~10 frames); compute speed via frame-to-frame displacement; flag `accelerating=True` when speed crosses threshold after being ~0.

**Interface:** Input: tracked vehicle list (Task 4 output). Output: JSON `{"lead_vehicle_id": int|null, "accelerating": bool, "speed_px_per_frame": float}`.

---

## TASK 6: Ego-Vehicle Stationary Detector (Windows)

**Objective:** Determine if our own (camera) vehicle is stationary, using only visual input (no speed sensor).

**Technical Approach:** Sparse optical flow (`cv2.calcOpticalFlowPyrLK`) on static background features (road/lane markings, static corners via `cv2.goodFeaturesToTrack` in lower frame region excluding moving vehicle bboxes). Low mean flow magnitude over N frames = stationary.

**Interface:** Input: current + previous grayscale frame, list of vehicle bboxes to exclude. Output: JSON `{"ego_stationary": bool, "mean_flow_magnitude": float}`.

---

## TASK 7: Distraction Alert Event Engine (Windows)

**Objective:** Fuse Tasks 2, 5, 6 into the core alert trigger: fire when (light turned green OR lead vehicle accelerates) AND ego is stationary, with debounce so it fires once per event.

**Technical Approach:** Simple state machine class holding last-alert timestamp; debounce window (e.g. 3s); on trigger, package event dict and hand off to audio module (Task 12) and DB logger (Task 10).

**Interface:** Input: `light_state` dict (Task 2), `lead_vehicle` dict (Task 5), `ego_stationary` dict (Task 6), `frame_timestamp`. Output: JSON `{"alert": bool, "reason": "light_green"|"lead_accelerating"|null, "timestamp": float}`.

---

## TASK 8: Reference-Vehicle Selection & HUD Highlight Logic (Windows)

**Objective:** Pick a "reference vehicle" (e.g., a car turning, matching a heuristic) and produce highlight overlay data for contextual navigation.

**Technical Approach:** Heuristic on Task 4 tracks: select vehicle whose bbox centroid trajectory shows consistent lateral drift beyond threshold (turning) within an ROI relevant to the upcoming maneuver (configurable "expected turn direction" input, e.g. from a stub GPS instruction). Output styling metadata (color, label) for that track ID only.

**Interface:** Input: tracked vehicle list (Task 4), `expected_direction: "left"|"right"|"straight"`. Output: JSON `{"reference_track_id": int|null, "highlight_color": "#RRGGBB", "label": "Follow this car"}`.

---

## TASK 9: Frame Annotation Renderer (Windows)

**Objective:** Draw all overlays (vehicle boxes, reference highlight, traffic-light state badge, alert banner) onto the frame for HUD output.

**Technical Approach:** `cv2` drawing primitives (`rectangle`, `putText`, semi-transparent overlay via `cv2.addWeighted` for alert banner flash). Pure function, no state. Output frame is not displayed locally — it is handed to Task 11's network client, which streams it to the Mac.

**Interface:** Input: `frame`, tracked vehicles (Task 4), reference-vehicle data (Task 8), light state (Task 2), alert event (Task 7). Output: annotated `np.ndarray` frame, passed to Task 11 for network transmission.

---

## TASK 10: FastAPI App + WebSocket Streaming Endpoint (Mac) (COMPLETE — merged to main, `server/app.py`)

**Objective:** Receive annotated frames and event JSON pushed from the Windows PC, and serve them to the browser frontend over WebSocket.

**Technical Approach:** FastAPI app, bound to `host="0.0.0.0"` in `uvicorn.run(...)` (not `127.0.0.1`) so the Windows PC can reach it over LAN. Add `CORSMiddleware` (`allow_origins=["*"]` for hackathon LAN scope — demo-only) so cross-origin requests from the Windows PC succeed. Two separate WebSocket endpoints:
- `@app.websocket("/ws/ingest")` — Windows PC connects here as a client and pushes `{"type":"frame", "data":"<base64 JPEG>"}` and `{"type":"event", ...}` messages.
- `@app.websocket("/ws/stream")` — browser frontend connects here; server broadcasts whatever it receives on `/ws/ingest` to all connected `/ws/stream` clients via a shared `asyncio.Queue`.

Splitting ingest from stream keeps the producer (Windows) and consumers (browsers) from colliding on the same socket.

**Interface:** Input: WebSocket messages pushed from Windows PC on `/ws/ingest` (JSON schema as above). Output: same messages broadcast over `/ws/stream` to any connected frontend client.

---

## TASK 11: Main Processing Loop (Pipeline Orchestration) (Windows)

**Objective:** Wire Tasks 1-9 into one loop: read frame → detect → track → decide → annotate → send to the Mac over the network.

**Technical Approach:** Single async or threaded loop function `run_pipeline()`, run standalone on the Windows PC (no local FastAPI server needed — this machine is a client, not a host). Loop opens an outbound WebSocket connection to the Mac's `/ws/ingest` endpoint (`SERVER_URL` from `.env`) and pushes `{frame, events}` each iteration instead of an in-process queue. Reconnect with backoff if the connection to the Mac drops. DB logging (Task 13) and reaction-time/audio triggers (Task 14/15) run on the Mac side once it receives the event JSON — Windows only ships the payload.

**Interface:** Input: `FrameSource` (Task 1) instance. Output: pushes `{frame, events}` over WebSocket to the Mac's `/ws/ingest` (Task 10) each iteration; no return value (long-running loop).

---

## TASK 12: Tiger Data (Postgres) Schema + Connection (Mac) (COMPLETE — merged to main, `db/schema.sql` + `db/pool.py`; verified against Tiger Cloud, `/health` reports `"db":"connected","hypertable":true`)

**Objective:** Define telemetry schema and connection pool for Tiger Data.

**Technical Approach:** Tables: `detection_events(id, timestamp, event_type, payload JSONB)`, `reaction_times(id, alert_timestamp, driver_reaction_timestamp, delta_ms)`. Use `asyncpg` pool created on FastAPI startup, DSN from `.env`. Consider Tiger Data hypertable (`SELECT create_hypertable('detection_events','timestamp')`) for high-frequency telemetry.

**Interface:** Input: DDL script `schema.sql`. Output: connection pool object available to app state (`app.state.db_pool`); tables created idempotently.

---

## TASK 13: Telemetry Logging Service (Mac) (COMPLETE — merged to main, `db/telemetry.py`; verified against Tiger Cloud)

**Objective:** Persist frame-level detection events and alert/reaction-time data asynchronously without blocking the video loop.

**Technical Approach:** Async writer function using `app.state.db_pool.execute(...)`; called from Task 11 pipeline on each alert event and periodically (e.g. every N frames) for general telemetry. Batch inserts if frequency is high to avoid connection saturation.

**Interface:** Input: event dict (from Task 7/8), `db_pool`. Output: row inserted in `detection_events`; returns nothing (fire-and-forget with error logging).

**Status:** Implemented in `db/telemetry.py` (`TelemetryLogger`), hooked into `server/app.py`'s `/ws/ingest` handler (the actual Mac-side integration point — see Task 11's note on why). Non-blocking `put_nowait` enqueue, background worker batches up to 100 rows or flushes every 1s via `executemany`. Frames sampled every `TELEMETRY_FRAME_SAMPLE_N` (30) instead of logged wholesale; base64 `data` stripped from every payload before storage. Shutdown uses a `_STOP` sentinel pushed through the same queue (not `task.cancel()`) so the final in-flight batch always flushes — a cancel-based version was tried first and silently dropped queued-but-uncollected rows on shutdown; caught and fixed during verification. Verified against live Tiger Cloud: exact event/frame-sample counts, sanitized payloads (max row 85 bytes, zero leaked `data` keys), `payload->>'reason'` resolves, relay throughput unaffected by live DB writes, graceful degradation intact with no DSN.

---

## TASK 14: Reaction-Time Capture (Mac) (COMPLETE — branch `feat/task-14-reaction-time`, not yet merged to main; `server/reaction_tracker.py`)

**Objective:** Measure and log time between alert firing and driver "reacting" (ego vehicle starts moving again, from Task 6 flipping to `ego_stationary=False`).

**Technical Approach:** On alert trigger, store `alert_timestamp` in memory; watch subsequent frames' `ego_stationary` output; on first `False` after an active alert, compute `delta_ms` and write to `reaction_times` table via Task 13's logger.

**Interface:** Input: alert event (Task 7) stream, ego-stationary stream (Task 6). Output: DB row `{"alert_timestamp", "driver_reaction_timestamp", "delta_ms"}`.

**Status:** Implemented in `server/reaction_tracker.py` (`ReactionTracker`), hooked into `server/app.py`'s `/ws/ingest` handler alongside Task 13's telemetry call. Required extending the wire contract (documented in `server/app.py`'s module docstring, since Task 11 doesn't exist yet to define it): `"event"` messages now fire every frame, not just on alert, always carrying `ego_stationary` — Task 7's engine already needs that value as an input every frame, so it's a near-free addition. Single-pending-alert model: a new alert always replaces whatever was pending. Reuses Task 13's `TelemetryLogger.log_reaction_time()` and its `to_timestamptz()` float→`TIMESTAMPTZ` conversion (made public for this). Filtered Task 13's `log_event()` call to alerts/audio only, not every per-frame event — the extended contract would otherwise flood `detection_events` at ~15/sec with idle state. `scripts/fake_producer.py` extended to simulate a randomized driver-reaction delay per alert so the full round-trip is exercisable without Windows. Verified against live Tiger Cloud: `reaction_pending` flips true/false correctly in `/health`, logged `delta_ms` matches simulated delays with zero negative values, overlapping alerts correctly replace the pending one instead of double-logging, relay throughput unaffected, graceful degradation intact with no DSN.

---

## TASK 15: ElevenLabs Voice Alert (Mac)

**Objective:** Convert alert reason into a spoken audio cue and play/stream it on trigger.

**Technical Approach:** `elevenlabs` Python SDK, pre-generate short phrases per `reason` type (`"Light's green — go!"`, `"Car ahead is moving"`) at startup and cache audio bytes (avoid per-event API latency). On alert (Task 7), either (a) play server-side via `simpleaudio`/`playsound` for local demo, or (b) push base64 audio over the same WebSocket as a `{"type":"audio","data":...}` message for the browser to play via `<audio>`/Web Audio API.

**Interface:** Input: `reason: str` enum from Task 7. Output: audio bytes (WAV/MP3) cached in memory; WebSocket message `{"type":"audio","reason":str,"data":"<base64>"}`.

---

## TASK 16: Frontend HUD Page (Mac)

**Objective:** Browser page showing live annotated video feed + alert flash + audio playback, as the primary demo screen.

**Technical Approach:** Single `index.html` + vanilla JS. `WebSocket("ws://host/ws/stream")`; on `type:"frame"` draw base64 JPEG onto `<canvas>`; on `type:"event"` flash a CSS alert banner; on `type:"audio"` decode base64 and play via `new Audio()`. No frameworks, no chat UI (per Microsoft constraint).

**Interface:** Input: WebSocket messages (Task 10/15 schemas). Output: rendered HUD in browser — canvas video + overlay banner.

---

## TASK 17: Real-Time Metrics Dashboard (Mac)

**Objective:** Secondary panel/page showing telemetry charts (alert frequency, reaction times over session).

**Technical Approach:** `Chart.js` via CDN or vendored file, polling a small FastAPI REST endpoint `GET /api/metrics/summary` (reads from Tiger Data via `db_pool`) every few seconds, or reuse the WebSocket event stream to update charts live.

**Interface:** Input: `GET /api/metrics/summary` → Output: JSON `{"total_alerts": int, "avg_reaction_ms": float, "events_timeline": [...]}`. Frontend renders line/bar chart from this.

---

## TASK 18: End-to-End Integration & Demo Hardening (Mac)

**Objective:** Tie everything together, verify with sample footage, harden for live demo (no crashes on missing detections, graceful WebSocket reconnect).

**Technical Approach:** Sample driving video for repeatable demo. Try/except around each pipeline stage logging to console instead of crashing loop. Config flag to switch webcam↔file source without code change. Startup checklist script (`check_env.py`) verifying `.env`, DB reachable, ElevenLabs key valid. Verify LAN connectivity between the Windows PC and the Mac before demo start (correct `SERVER_URL`/IP, firewall allows the port); confirm Task 11's reconnect-with-backoff logic actually recovers if the Windows↔Mac link drops mid-demo, not just the browser↔server link.

**Interface:** Input: none. Output: `main.py` runs full stack (`uvicorn server.app:app` on the Mac, pipeline script on the Windows PC), demo works end-to-end from cold start.
