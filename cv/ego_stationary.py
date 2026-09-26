"""Ego-vehicle stationary detector: sparse optical flow on static background features (road surface,
lane markings), excluding tracked-vehicle boxes, to tell whether our own camera is moving without a
speed sensor.
"""

import json
from collections import deque

import cv2
import numpy as np

Bbox = list[int]  # [x1, y1, x2, y2]


class EgoStationaryDetector:
    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.05,
        min_distance: int = 7,
        road_region_top: float = 0.5,
        stationary_thresh: float = 0.75,
        buffer_len: int = 5,
        confirm_frames: int = 3,
        bbox_margin: int = 6,
    ):
        """
        max_corners/quality_level/min_distance: cv2.goodFeaturesToTrack params for picking static
            background features (road surface, lane markings) to track.
        road_region_top: fraction of frame height above which features are ignored (0.5 = only the
            bottom half of the frame is sampled -- sky/buildings/horizon are excluded, not just the
            moving vehicles).
        stationary_thresh: mean flow magnitude (px) at/below this counts as stationary for one frame.
        buffer_len: how many raw per-frame magnitudes are averaged before deciding stationary, so a
            single noisy frame (a missed match, a burst of tracked-point jitter) doesn't flip the flag.
        confirm_frames: consecutive frames a raw stationary/moving reading must persist before the
            reported state flips (same debounce as TrafficLightDetector's confirmation).
        bbox_margin: px padding added around excluded vehicle bboxes, since a feature just outside a
            box can still sit in LK's search window and pick up the vehicle's own motion.
        """
        self.feature_params = dict(
            maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance, blockSize=7
        )
        self.road_region_top = road_region_top
        self.stationary_thresh = stationary_thresh
        self.bbox_margin = bbox_margin
        self.confirm_frames = max(1, confirm_frames)

        self._history: deque[float] = deque(maxlen=buffer_len)
        self.stationary = True  # assume stationary until flow proves otherwise
        self._candidate = True
        self._candidate_count = 0

        # Exposed for overlay drawing / debugging: state of the most recent update() call.
        self.last_points: np.ndarray | None = None
        self.last_next_points: np.ndarray | None = None
        self.last_status: np.ndarray | None = None

    def reset(self) -> None:
        self._history.clear()
        self.stationary = True
        self._candidate = True
        self._candidate_count = 0
        self.last_points = self.last_next_points = self.last_status = None

    def _road_mask(self, shape: tuple[int, int], bboxes: list[Bbox]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * self.road_region_top):, :] = 255
        m = self.bbox_margin
        for x1, y1, x2, y2 in bboxes:
            x1, y1 = max(0, x1 - m), max(0, y1 - m)
            x2, y2 = min(w, x2 + m), min(h, y2 + m)
            mask[y1:y2, x1:x2] = 0
        return mask

    def update(self, prev_gray: np.ndarray, curr_gray: np.ndarray, vehicle_bboxes: list[Bbox]) -> dict:
        """One call per frame pair. prev_gray/curr_gray: consecutive single-channel frames.
        vehicle_bboxes: current vehicle boxes (e.g. Task 3/4 output's bboxes) to exclude from sampling."""
        mask = self._road_mask(prev_gray.shape[:2], vehicle_bboxes)
        points = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
        self.last_points = points

        magnitude = None
        if points is not None and len(points):
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None)
            status_mask = status.reshape(-1).astype(bool)
            self.last_next_points, self.last_status = next_points, status_mask
            if status_mask.any():
                displacement = (next_points[status_mask] - points[status_mask]).reshape(-1, 2)
                magnitude = float(np.mean(np.linalg.norm(displacement, axis=1)))
        else:
            self.last_next_points = self.last_status = None

        if magnitude is None:
            # No trackable static background this frame (fully occluded by vehicles, or a
            # featureless road) -- hold the last reading rather than guessing.
            mean_magnitude = self._history[-1] if self._history else 0.0
            return {"ego_stationary": self.stationary, "mean_flow_magnitude": round(mean_magnitude, 3)}

        self._history.append(magnitude)
        mean_magnitude = sum(self._history) / len(self._history)

        raw_stationary = mean_magnitude <= self.stationary_thresh
        if raw_stationary == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = raw_stationary, 1
        if self._candidate_count >= self.confirm_frames:
            self.stationary = raw_stationary

        return {"ego_stationary": self.stationary, "mean_flow_magnitude": round(mean_magnitude, 3)}

    def update_json(self, prev_gray: np.ndarray, curr_gray: np.ndarray, vehicle_bboxes: list[Bbox]) -> str:
        return json.dumps(self.update(prev_gray, curr_gray, vehicle_bboxes))


def draw_overlay(frame: np.ndarray, detector: EgoStationaryDetector, result: dict) -> None:
    color = (0, 200, 255) if result["ego_stationary"] else (0, 255, 0)
    label = f"EGO: {'STATIONARY' if result['ego_stationary'] else 'MOVING'} ({result['mean_flow_magnitude']:.2f}px)"
    cv2.putText(frame, label, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if detector.last_points is None:
        return
    pts = detector.last_points.reshape(-1, 2)
    status = detector.last_status if detector.last_status is not None else np.zeros(len(pts), dtype=bool)
    next_pts = detector.last_next_points.reshape(-1, 2) if detector.last_next_points is not None else None
    for i, (x, y) in enumerate(pts):
        ok = bool(status[i]) if i < len(status) else False
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0) if ok else (0, 0, 255), -1)
        if ok and next_pts is not None:
            nx, ny = next_pts[i]
            cv2.line(frame, (int(x), int(y)), (int(nx), int(ny)), (0, 255, 0), 1)


if __name__ == "__main__":
    import argparse
    import time

    import config
    from cv.frame_source import FrameSource
    from cv.vehicle_detector import VehicleDetector

    parser = argparse.ArgumentParser(description="Preview the ego-stationary detector.")
    parser.add_argument("source", nargs="?", default=config.VIDEO_SOURCE, help="webcam index or video file path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--stationary-thresh", type=float, default=0.75)
    parser.add_argument("--out", default=None, help="write annotated video here instead of opening a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = VehicleDetector(args.model, conf=args.conf)
    ego_detector = EgoStationaryDetector(stationary_thresh=args.stationary_thresh)
    print(f"Model {args.model} on {detector.device}")

    writer = None
    prev_gray = None
    with FrameSource(args.source, loop=args.out is None, target_fps=0 if args.out else None) as src:
        last, fps, n = time.perf_counter(), 0.0, 0
        for frame, ts in src:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vehicles = detector.detect(frame)
            bboxes = [v["bbox"] for v in vehicles]

            if prev_gray is not None:
                result = ego_detector.update(prev_gray, gray, bboxes)
            else:
                result = {"ego_stationary": True, "mean_flow_magnitude": 0.0}
            prev_gray = gray

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            n += 1
            if n % 30 == 0:
                print(f"frame {n}: {json.dumps(result)}")

            draw_overlay(frame, ego_detector, result)
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src.native_fps or 30, (w, h))
                writer.write(frame)
            else:
                cv2.imshow("EgoStationaryDetector", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
