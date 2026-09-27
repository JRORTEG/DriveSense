"""Main processing loop (pipeline orchestration): wires Tasks 1-9 into one loop -- read frame, detect,
track, decide, annotate -- and pushes the result to the Mac's Task 10 relay over WebSocket. Windows-side
only: this machine is a client, not a host, so no local FastAPI server is needed.

Supersedes scripts/fake_producer.py (Task 10's stand-in) as the real producer: same wire schema, same
reconnect-with-backoff shape, real frames instead of synthetic ones.
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


def _encode_frame(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


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

    def process(self, frame: np.ndarray, timestamp: float) -> tuple[np.ndarray, dict]:
        """One call per frame. Returns (annotated_frame, event) where event is the exact shape
        server/app.py's `/ws/ingest` "event" message expects -- sent every frame, not only on alert,
        since Task 14's reaction-time capture needs a continuous ego_stationary reading to catch the
        True->False flip after an alert."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracks = self.tracker.update(self.detector.detect(frame))
        bboxes = [t["bbox"] for t in tracks]

        light_result = self.light_detector.detect(frame, timestamp=timestamp)
        lead_result = self.lead_detector.update(tracks, frame.shape[1])
        if self._prev_gray is not None:
            ego_result = self.ego_detector.update(self._prev_gray, gray, bboxes)
        else:
            ego_result = {"ego_stationary": True, "mean_flow_magnitude": 0.0}
        self._prev_gray = gray

        reference_result = self.reference_detector.update(tracks, self.expected_direction)
        alert_result = self.alert_engine.update(light_result, lead_result, ego_result, timestamp)

        annotate_frame(frame, tracks, reference_result, light_result, alert_result, self.light_detector.last_roi)

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
    off, since FrameSource's own position isn't reset by a fresh websocket connection."""
    pipeline = Pipeline(
        model=model, conf=conf, light_roi=light_roi,
        expected_direction=expected_direction, alert_debounce_s=alert_debounce_s,
    )
    logger.info("model %s on %s", model, pipeline.detector.device)

    backoff = 1
    while True:
        try:
            async with websockets.connect(server_url) as ws:
                logger.info("connected to %s", server_url)
                backoff = 1
                for frame, ts in source:
                    annotated, event = pipeline.process(frame, ts)
                    frame_msg = {"type": "frame", "data": _encode_frame(annotated), "timestamp": ts}
                    await ws.send(json.dumps(frame_msg))
                    await ws.send(json.dumps(event))
        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            logger.warning("connection lost (%r), retrying in %ds", exc, backoff)
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
