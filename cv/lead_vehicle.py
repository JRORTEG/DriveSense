"""Lead-vehicle acceleration detector: picks the lead vehicle from Task 4's tracks (largest bbox area,
discounted by horizontal offset from frame center) and edge-detects it accelerating away from a stop.
"""

import json
import math
from collections import deque

Track = dict  # Task 4 shape: {"track_id": int, "bbox": [x1,y1,x2,y2], "class": str, "velocity": [vx,vy]}


def _centroid(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _area(bbox: list[int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


class LeadVehicleDetector:
    def __init__(
        self,
        buffer_len: int = 10,
        rest_thresh: float = 1.0,
        accel_thresh: float = 2.5,
        confirm_frames: int = 3,
        center_bias: float = 0.8,
    ):
        """
        buffer_len: centroids kept per track to compute speed (mean frame-to-frame displacement).
        rest_thresh: mean speed (px/frame) at/below this counts as "at rest".
        accel_thresh: mean speed above this counts as "moving". Kept above rest_thresh so a resting
            track must clear the gap between the two before firing again (hysteresis, no chatter at
            a single boundary value).
        confirm_frames: consecutive frames a raw rest/moving reading must persist before the tracked
            state flips (same debounce as TrafficLightDetector's red/green confirmation).
        center_bias: how much horizontal offset from the frame center discounts a track's bbox-area
            score when picking the lead vehicle (0 = area only, 1 = fully zeroed out at the frame edge).
        """
        if accel_thresh <= rest_thresh:
            raise ValueError("accel_thresh must be greater than rest_thresh")
        self.buffer_len = buffer_len
        self.rest_thresh = rest_thresh
        self.accel_thresh = accel_thresh
        self.confirm_frames = max(1, confirm_frames)
        self.center_bias = center_bias

        self._history: dict[int, deque] = {}
        self._moving: dict[int, bool] = {}
        self._candidate: dict[int, bool] = {}
        self._candidate_count: dict[int, int] = {}

    def reset(self) -> None:
        self._history.clear()
        self._moving.clear()
        self._candidate.clear()
        self._candidate_count.clear()

    def _select_lead(self, tracks: list[Track], frame_width: int) -> Track | None:
        if not tracks:
            return None
        center_x = frame_width / 2.0
        half_width = max(center_x, 1.0)

        def score(t: Track) -> float:
            cx, _ = _centroid(t["bbox"])
            offset_norm = min(abs(cx - center_x) / half_width, 1.0)
            return _area(t["bbox"]) * (1.0 - self.center_bias * offset_norm)

        return max(tracks, key=score)

    def _speed(self, track_id: int, centroid: tuple[float, float]) -> float:
        hist = self._history.setdefault(track_id, deque(maxlen=self.buffer_len))
        hist.append(centroid)
        if len(hist) < 2:
            return 0.0
        pts = list(hist)
        steps = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
        return sum(steps) / len(steps)

    def _edge(self, track_id: int, speed: float) -> bool:
        was_moving = self._moving.setdefault(track_id, False)  # a newly-seen track is assumed at rest
        raw_moving = speed > self.accel_thresh if not was_moving else speed > self.rest_thresh

        if raw_moving == self._candidate.get(track_id, was_moving):
            self._candidate_count[track_id] = self._candidate_count.get(track_id, 0) + 1
        else:
            self._candidate[track_id] = raw_moving
            self._candidate_count[track_id] = 1

        if self._candidate_count[track_id] < self.confirm_frames or raw_moving == was_moving:
            return False

        self._moving[track_id] = raw_moving
        return raw_moving  # True only for a confirmed rest -> moving transition, not moving -> rest

    def update(self, tracks: list[Track], frame_width: int) -> dict:
        """One call per frame. tracks: Task 4's VehicleTracker.update() output (current frame's matched
        tracks). frame_width: frame width in px, needed for the lead-vehicle center-offset heuristic."""
        seen = {t["track_id"] for t in tracks}
        for tid in [tid for tid in self._history if tid not in seen]:
            del self._history[tid]
            self._moving.pop(tid, None)
            self._candidate.pop(tid, None)
            self._candidate_count.pop(tid, None)

        # Update every visible track's speed buffer, not just the lead's: the lead can change frame to
        # frame, and a newly-picked lead needs its history already warmed up rather than starting cold.
        speeds = {t["track_id"]: self._speed(t["track_id"], _centroid(t["bbox"])) for t in tracks}

        lead = self._select_lead(tracks, frame_width)
        if lead is None:
            return {"lead_vehicle_id": None, "accelerating": False, "speed_px_per_frame": 0.0}

        lead_id = lead["track_id"]
        speed = speeds[lead_id]
        accelerating = self._edge(lead_id, speed)
        return {"lead_vehicle_id": lead_id, "accelerating": accelerating, "speed_px_per_frame": round(speed, 2)}

    def update_json(self, tracks: list[Track], frame_width: int) -> str:
        return json.dumps(self.update(tracks, frame_width))


def draw_overlay(frame, tracks: list[Track], result: dict) -> None:
    import cv2

    lead_id = result["lead_vehicle_id"]
    label = f"LEAD #{lead_id} {result['speed_px_per_frame']:.1f}px/f" if lead_id is not None else "LEAD: none"
    color = (0, 0, 255) if result["accelerating"] else (0, 200, 255)
    cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if result["accelerating"]:
        cv2.putText(frame, "ACCELERATING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    lead = next((t for t in tracks if t["track_id"] == lead_id), None)
    if lead is not None:
        x1, y1, x2, y2 = lead["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)


if __name__ == "__main__":
    import argparse
    import time

    import cv2

    import config
    from cv.frame_source import FrameSource
    from cv.tracker import VehicleTracker, draw_tracks
    from cv.vehicle_detector import VehicleDetector

    parser = argparse.ArgumentParser(description="Preview the lead-vehicle acceleration detector.")
    parser.add_argument("source", nargs="?", default=config.VIDEO_SOURCE, help="webcam index or video file path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--rest-thresh", type=float, default=1.0)
    parser.add_argument("--accel-thresh", type=float, default=2.5)
    parser.add_argument("--out", default=None, help="write annotated video here instead of opening a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = VehicleDetector(args.model, conf=args.conf)
    tracker = VehicleTracker()
    lead_detector = LeadVehicleDetector(rest_thresh=args.rest_thresh, accel_thresh=args.accel_thresh)
    print(f"Model {args.model} on {detector.device}")

    writer = None
    with FrameSource(args.source, loop=args.out is None, target_fps=0 if args.out else None) as src:
        last, fps, n = time.perf_counter(), 0.0, 0
        for frame, ts in src:
            tracks = tracker.update(detector.detect(frame))
            result = lead_detector.update(tracks, frame.shape[1])
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            n += 1
            if result["accelerating"]:
                print(json.dumps(result))

            draw_tracks(frame, tracks)
            draw_overlay(frame, tracks, result)
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src.native_fps or 30, (w, h))
                writer.write(frame)
            else:
                cv2.imshow("LeadVehicleDetector", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
