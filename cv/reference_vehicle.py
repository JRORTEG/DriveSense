"""Reference-vehicle selection: picks a tracked vehicle (Task 4 output) whose recent trajectory shows
consistent lateral drift matching an expected turn direction, so the HUD can highlight "follow this car"
for contextual navigation.
"""

import json
from collections import deque

Track = dict  # Task 4 shape: {"track_id": int, "bbox": [x1,y1,x2,y2], "class": str, "velocity": [vx,vy]}
Direction = str  # "left" | "right" | "straight"

_DEFAULT_LABEL = "Follow this car"
_DEFAULT_COLOR = "#39FF14"  # neon green: distinct from draw_tracks' per-ID palette, reads as "look here"


def _centroid(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class ReferenceVehicleDetector:
    def __init__(
        self,
        buffer_len: int = 10,
        turn_thresh: float = 15.0,
        straight_thresh: float = 5.0,
        consistency_ratio: float = 0.7,
        highlight_color: str = _DEFAULT_COLOR,
        label: str = _DEFAULT_LABEL,
    ):
        """
        buffer_len: centroids kept per track to judge its trajectory.
        turn_thresh: net horizontal drift (px, oldest->newest in the buffer) a track must clear to
            count as "turning" for expected_direction "left"/"right".
        straight_thresh: net horizontal drift at/below this counts as "going straight", used for
            expected_direction "straight" instead of turn_thresh.
        consistency_ratio: fraction of frame-to-frame steps that must share the drift's own sign, so a
            track that's merely jittering side to side (net drift by chance) doesn't qualify as turning.
        highlight_color/label: styling metadata returned for whichever track is selected.
        """
        self.buffer_len = buffer_len
        self.turn_thresh = turn_thresh
        self.straight_thresh = straight_thresh
        self.consistency_ratio = consistency_ratio
        self.highlight_color = highlight_color
        self.label = label

        self._history: dict[int, deque] = {}
        self._current_id: int | None = None

    def reset(self) -> None:
        self._history.clear()
        self._current_id = None

    def _drift(self, track_id: int) -> tuple[float, float]:
        pts = list(self._history[track_id])
        if len(pts) < 2:
            return 0.0, 0.0
        net_dx = pts[-1][0] - pts[0][0]
        steps = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1)]
        if net_dx == 0 or not steps:
            return net_dx, 0.0
        agreeing = sum(1 for s in steps if (s > 0) == (net_dx > 0))
        return net_dx, agreeing / len(steps)

    def _qualifies(self, net_dx: float, consistency: float, expected_direction: Direction) -> bool:
        if expected_direction == "straight":
            return abs(net_dx) <= self.straight_thresh
        if abs(net_dx) < self.turn_thresh or consistency < self.consistency_ratio:
            return False
        drifting_left = net_dx < 0  # image x decreases toward the left of frame
        return drifting_left if expected_direction == "left" else not drifting_left

    def update(self, tracks: list[Track], expected_direction: Direction) -> dict:
        """One call per frame. tracks: Task 4's VehicleTracker.update() output. expected_direction:
        "left"/"right"/"straight", e.g. from a stub GPS instruction for the upcoming maneuver."""
        seen = {t["track_id"] for t in tracks}
        for tid in [tid for tid in self._history if tid not in seen]:
            del self._history[tid]

        for t in tracks:
            hist = self._history.setdefault(t["track_id"], deque(maxlen=self.buffer_len))
            hist.append(_centroid(t["bbox"]))

        candidates: list[tuple[float, int]] = []
        for t in tracks:
            net_dx, consistency = self._drift(t["track_id"])
            if self._qualifies(net_dx, consistency, expected_direction):
                score = abs(net_dx) if expected_direction == "straight" else abs(net_dx) * consistency
                candidates.append((score, t["track_id"]))

        # Sticky: keep the current reference as long as it still qualifies, rather than jumping to
        # whichever candidate scores highest this frame, so the HUD highlight doesn't flicker between
        # two similarly-turning cars.
        if self._current_id is not None and any(tid == self._current_id for _, tid in candidates):
            reference_id = self._current_id
        elif candidates:
            reference_id = max(candidates, key=lambda c: c[0])[1]
        else:
            reference_id = None
        self._current_id = reference_id

        if reference_id is None:
            return {"reference_track_id": None, "highlight_color": self.highlight_color, "label": ""}
        return {"reference_track_id": reference_id, "highlight_color": self.highlight_color, "label": self.label}

    def update_json(self, tracks: list[Track], expected_direction: Direction) -> str:
        return json.dumps(self.update(tracks, expected_direction))


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def draw_overlay(frame, tracks: list[Track], result: dict) -> None:
    import cv2

    track_id = result["reference_track_id"]
    if track_id is None:
        return
    track = next((t for t in tracks if t["track_id"] == track_id), None)
    if track is None:
        return
    color = _hex_to_bgr(result["highlight_color"])
    x1, y1, x2, y2 = track["bbox"]
    cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 3)
    cv2.putText(frame, result["label"], (x1 - 4, max(y1 - 12, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


if __name__ == "__main__":
    import argparse
    import time

    import cv2

    import config
    from cv.frame_source import FrameSource
    from cv.tracker import VehicleTracker, draw_tracks
    from cv.vehicle_detector import VehicleDetector

    parser = argparse.ArgumentParser(description="Preview the reference-vehicle selector.")
    parser.add_argument("source", nargs="?", default=config.VIDEO_SOURCE, help="webcam index or video file path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--direction", choices=["left", "right", "straight"], default="left")
    parser.add_argument("--out", default=None, help="write annotated video here instead of opening a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = VehicleDetector(args.model, conf=args.conf)
    tracker = VehicleTracker()
    ref_detector = ReferenceVehicleDetector()
    print(f"Model {args.model} on {detector.device}")

    writer = None
    with FrameSource(args.source, loop=args.out is None, target_fps=0 if args.out else None) as src:
        last, fps, n = time.perf_counter(), 0.0, 0
        for frame, ts in src:
            tracks = tracker.update(detector.detect(frame))
            result = ref_detector.update(tracks, args.direction)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            n += 1
            if result["reference_track_id"] is not None and n % 10 == 0:
                print(json.dumps(result))

            draw_tracks(frame, tracks)
            draw_overlay(frame, tracks, result)
            cv2.putText(frame, f"{fps:5.1f} FPS  expect={args.direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src.native_fps or 30, (w, h))
                writer.write(frame)
            else:
                cv2.imshow("ReferenceVehicleDetector", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
