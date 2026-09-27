"""Main processing loop (pipeline orchestration): wires Tasks 1-9 into one loop -- read frame, detect,
track, decide, annotate -- and pushes the result to the Mac's Task 10 relay over WebSocket. Windows-side
only: this machine is a client, not a host, so no local FastAPI server is needed.

Supersedes scripts/fake_producer.py (Task 10's stand-in) as the real producer: same wire schema, same
reconnect-with-backoff shape, real frames instead of synthetic ones.

Task 18 hardening: every CV stage is wrapped individually (Pipeline._stage) so one bad frame degrades
that frame's overlay/event instead of killing the loop -- see the per-stage fallback table in
Plan.md's Task 18 section for why a single try/except around the whole pipeline would be worse, not
better (it would silently swallow one-shot alert edges like transitioned_to_green/accelerating).
run_pipeline's own reconnect loop is similarly widened so it can never die from a stage exception, a
malformed frame, or a websocket error that isn't a plain OSError.
"""

import argparse
import asyncio
import base64
import json
import logging

import cv2
import numpy as np
import websockets

import config
from cv.alert_engine import AlertEngine
from cv.annotator import annotate_frame
from cv.ego_stationary import EgoStationaryDetector
from cv.frame_source import FrameSource
from cv.lead_vehicle import LeadVehicleDetector
from cv.reference_vehicle import Direction, ReferenceVehicleDetector
from cv.traffic_light import Roi, TrafficLightDetector
from cv.tracker import VehicleTracker
from cv.vehicle_detector import VehicleDetector

logger = logging.getLogger("drivesense.pipeline")

JPEG_QUALITY = 70

# A stage that keeps failing logs a full traceback once, then a counted warning every
# STAGE_WARN_EVERY calls -- so a persistently broken stage can't flood the console at 15-30 FPS.
STAGE_WARN_EVERY = 50


def _encode_frame(frame: np.ndarray) -> str | None:
    """Returns None (never raises) on encode failure -- Task 18: the caller skips the frame but
    still sends the event, rather than losing both."""
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        logger.exception("frame encode failed, skipping frame")
        return None


class Pipeline:
    """Owns every stage's stateful detector, so run_pipeline() can treat one frame as a single
    process() call. Kept as a class instead of module-level globals so a caller could run two
    independent instances (e.g. two cameras) without their state bleeding together."""

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.4,
        light_roi: Roi | None = None,
        expected_direction: Direction = "straight",
        alert_debounce_s: float = 3.0,
    ):
        self.detector = VehicleDetector(model, conf=conf)
        self.tracker = VehicleTracker()
        self.lead_detector = LeadVehicleDetector()
        self.ego_detector = EgoStationaryDetector()
        self.light_detector = TrafficLightDetector(roi=light_roi)
        self.reference_detector = ReferenceVehicleDetector()
        self.alert_engine = AlertEngine(debounce_s=alert_debounce_s)
        self.expected_direction = expected_direction
        self._prev_gray: np.ndarray | None = None
        self._last_tracks: list[dict] = []
        self._stage_failures: dict[str, int] = {}

    def stats(self) -> dict:
        """Per-stage failure counters, so a demo operator can see degradation instead of
        guessing why an overlay looks stale (Task 18)."""
        return dict(self._stage_failures)

    def _stage(self, name: str, fn, fallback):
        """Run one CV stage. On failure, log and return `fallback` -- a bad frame degrades that
        frame's result, it never kills run_pipeline's loop. `fallback` may be a value or a
        zero-arg callable (for fallbacks that need to read current detector state, e.g. "hold last
        known reading")."""
        try:
            return fn()
        except Exception:
            count = self._stage_failures.get(name, 0) + 1
            self._stage_failures[name] = count
            if count == 1 or count % STAGE_WARN_EVERY == 0:
                logger.exception("stage %s failed (%d total failures)", name, count)
            else:
                logger.warning("stage %s failed (%d total failures)", name, count)
            return fallback() if callable(fallback) else fallback

    def process(self, frame: np.ndarray, timestamp: float) -> tuple[np.ndarray, dict]:
        """One call per frame. Returns (annotated_frame, event) where event is the exact shape
        server/app.py's `/ws/ingest` "event" message expects -- sent every frame, not only on alert,
        since Task 14's reaction-time capture needs a continuous ego_stationary reading to catch the
        True->False flip after an alert."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            logger.exception("cvtColor failed, skipping frame (prev_gray left untouched)")
            return frame, {
                "type": "event", "alert": False, "reason": None,
                "ego_stationary": self.ego_detector.stationary, "timestamp": timestamp,
            }

        # Advance _prev_gray immediately, before any stage below can raise -- Task 18 fix: the
        # original code advanced this *after* the ego-motion call, so an exception in any earlier
        # stage left _prev_gray stale by more than one frame, inflating the next optical-flow
        # measurement and risking a false ego-moving flip (see Plan.md Task 18).
        prev_gray = self._prev_gray
        self._prev_gray = gray

        tracks = self._stage(
            "detect_track",
            lambda: self.tracker.update(self.detector.detect(frame)),
            lambda: self._last_tracks,
        )
        self._last_tracks = tracks
        bboxes = [t["bbox"] for t in tracks]

        light_result = self._stage(
            "traffic_light",
            lambda: self.light_detector.detect(frame, timestamp=timestamp),
            lambda: {"state": self.light_detector.state, "transitioned_to_green": False, "timestamp": timestamp},
        )
        lead_result = self._stage(
            "lead_vehicle",
            lambda: self.lead_detector.update(tracks, frame.shape[1]),
            lambda: {"lead_vehicle_id": None, "accelerating": False, "speed_px_per_frame": 0.0},
        )

        if prev_gray is not None and prev_gray.shape == gray.shape:
            ego_result = self._stage(
                "ego_stationary",
                lambda: self.ego_detector.update(prev_gray, gray, bboxes),
                lambda: {"ego_stationary": self.ego_detector.stationary, "mean_flow_magnitude": 0.0},
            )
        else:
            # Cold start, or a resolution change mid-stream (prev_gray shape mismatch) -- neither
            # is a stage failure, so it isn't counted in self._stage_failures.
            ego_result = {"ego_stationary": True, "mean_flow_magnitude": 0.0}

        reference_result = self._stage(
            "reference_vehicle",
            lambda: self.reference_detector.update(tracks, self.expected_direction),
            lambda: {"reference_track_id": None, "highlight_color": "#39FF14", "label": ""},
        )
        alert_result = self._stage(
            "alert_engine",
            lambda: self.alert_engine.update(light_result, lead_result, ego_result, timestamp),
            lambda: {"alert": False, "reason": None, "timestamp": timestamp},
        )

        self._stage(
            "annotate",
            lambda: annotate_frame(
                frame, tracks, reference_result, light_result, alert_result, self.light_detector.last_roi
            ),
            lambda: frame,  # annotate_frame mutates in place; on failure just ship it un-annotated
        )

        event = {
            "type": "event",
            "alert": alert_result["alert"],
            "reason": alert_result["reason"],
            "ego_stationary": ego_result["ego_stationary"],
            "timestamp": timestamp,
        }
        return frame, event


async def run_pipeline(
    source: FrameSource,
    server_url: str,
    model: str = "yolov8n.pt",
    conf: float = 0.4,
    light_roi: Roi | None = None,
    expected_direction: Direction = "straight",
    alert_debounce_s: float = 3.0,
) -> None:
    """Reads frames from `source` forever, runs the full detect -> track -> decide -> annotate
    pipeline, and pushes {"type": "frame", ...} + {"type": "event", ...} JSON messages to the Mac's
    /ws/ingest (Task 10) for each one. Reconnects with exponential backoff (capped at 30s, matching
    scripts/fake_producer.py) if the connection drops; `source` keeps reading from wherever it left
    off, since FrameSource's own position isn't reset by a fresh websocket connection.

    Task 18: frame reads and CV processing run in worker threads (asyncio.to_thread) so the blocking
    time.sleep in FrameSource's FPS throttle and YOLO inference never stall the event loop -- a
    stalled loop delays websocket ping/pong and the close handshake, which is exactly the signal a
    mid-demo Windows<->Mac link drop needs in order to be noticed and reconnected promptly. The two
    awaits are sequential (not gathered) so detector state can't be mutated by two frames at once.
    """
    try:
        pipeline = Pipeline(
            model=model, conf=conf, light_roi=light_roi,
            expected_direction=expected_direction, alert_debounce_s=alert_debounce_s,
        )
    except Exception:
        logger.exception(
            "failed to construct Pipeline (model load / CUDA / warm-up inference) -- cannot start"
        )
        raise
    logger.info("model %s on %s", model, pipeline.detector.device)

    backoff = 1
    while True:
        try:
            async with websockets.connect(server_url) as ws:
                logger.info("connected to %s", server_url)
                backoff = 1
                source_iter = iter(source)
                while True:
                    frame_ts = await asyncio.to_thread(next, source_iter, None)
                    if frame_ts is None:
                        logger.info("frame source exhausted, stopping")
                        return
                    frame, ts = frame_ts

                    annotated, event = await asyncio.to_thread(pipeline.process, frame, ts)

                    encoded = _encode_frame(annotated)
                    if encoded is not None:
                        frame_msg = {"type": "frame", "data": encoded, "timestamp": ts}
                        await ws.send(json.dumps(frame_msg))
                    await ws.send(json.dumps(event))
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            logger.warning("connection lost (%r), retrying in %ds", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except Exception:
            # Catch-all so an unexpected error (a stage bug that slips past Pipeline._stage, a
            # transient decode error, etc.) never kills the producer mid-demo -- it just reconnects
            # like any other dropped link (Task 18: "no crashes on missing detections").
            logger.exception("unexpected error in pipeline loop, retrying in %ds", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DriveSense CV pipeline and stream it to the Mac.")
    parser.add_argument("--source", default=config.VIDEO_SOURCE, help="webcam index or video file path")
    parser.add_argument("--server", default=config.SERVER_URL, help="Mac's /ws/ingest URL")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--roi", type=str, default=None, help="traffic-light ROI as x,y,w,h")
    parser.add_argument("--direction", choices=["left", "right", "straight"], default="straight",
                         help="expected turn direction, e.g. from a stub GPS instruction")
    parser.add_argument("--debounce", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=config.TARGET_FPS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None

    with FrameSource(args.source, loop=True, target_fps=args.fps) as source:
        try:
            asyncio.run(run_pipeline(
                source, args.server, model=args.model, conf=args.conf,
                light_roi=roi, expected_direction=args.direction, alert_debounce_s=args.debounce,
            ))
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    main()
