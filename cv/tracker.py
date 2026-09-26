"""Simple multi-object tracker: greedy IoU/centroid matching, no external dependency (no scipy/filterpy).

Assigns persistent track_ids to Task 3's per-frame detection list so downstream tasks (lead-vehicle
acceleration, reference-vehicle highlight) can follow the same vehicle across frames.
"""

import json
import math
from collections import Counter, deque

Detection = dict  # Task 3 shape: {"id": None, "bbox": [x1,y1,x2,y2], "class": str, "conf": float}


def _centroid(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Track:
    """Internal per-object state. `class_votes` majority-votes the label so a single misclassified
    frame (car <-> truck jitter is common in YOLO) doesn't flicker the reported class."""

    def __init__(self, track_id: int, det: Detection, frame_idx: int, history_len: int):
        self.track_id = track_id
        self.bbox = det["bbox"]
        self.centroid = _centroid(det["bbox"])
        self.class_votes: Counter = Counter({det["class"]: 1})
        self.last_seen_frame = frame_idx
        self.first_seen_frame = frame_idx
        self.history: deque[tuple[float, float]] = deque([self.centroid], maxlen=history_len)
        self.velocity = (0.0, 0.0)

    @property
    def cls(self) -> str:
        return self.class_votes.most_common(1)[0][0]

    def update_from(self, det: Detection, frame_idx: int) -> None:
        new_centroid = _centroid(det["bbox"])
        self.bbox = det["bbox"]
        self.centroid = new_centroid
        self.class_votes[det["class"]] += 1
        self.last_seen_frame = frame_idx
        self.history.append(new_centroid)
        if len(self.history) >= 2:
            pts = list(self.history)
            dxs = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1)]
            dys = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
            # Averaged over the buffered window rather than last-frame-only, so one noisy detection
            # (a bbox that jitters a few px) doesn't spike the reported velocity.
            self.velocity = (sum(dxs) / len(dxs), sum(dys) / len(dys))

    def to_dict(self) -> dict:
        vx, vy = self.velocity
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "class": self.cls,
            "velocity": [round(vx, 2), round(vy, 2)],
        }


class VehicleTracker:
    def __init__(
        self,
        max_age: int = 15,
        max_center_dist: float = 120.0,
        min_iou: float = 0.15,
        velocity_window: int = 5,
    ):
        """
        max_age: frames a track may go unmatched before it's pruned (default ~0.5s at 30 FPS).
        max_center_dist: centroid-distance gate in px, used only as a fallback when boxes don't
            overlap enough for IoU (fast motion, low frame rate, or a missed detection in between).
        min_iou: an IoU at or above this always wins the match over any distance-based candidate.
        velocity_window: how many recent centroids are averaged into the reported velocity.
        """
        self.max_age = max_age
        self.max_center_dist = max_center_dist
        self.min_iou = min_iou
        self.velocity_window = velocity_window
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._frame_idx = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_idx = 0

    def _cost(self, track: _Track, det: Detection) -> float | None:
        iou = _iou(track.bbox, det["bbox"])
        if iou >= self.min_iou:
            return 1.0 - iou  # in [0, 1 - min_iou]: any real overlap beats any distance-only match below
        dist = math.hypot(*(a - b for a, b in zip(track.centroid, _centroid(det["bbox"]))))
        if dist > self.max_center_dist:
            return None  # outside the gate: not a candidate match at all
        return 1.0 + dist / self.max_center_dist  # in [1, 2]

    def update(self, detections: list[Detection]) -> list[dict]:
        """One call per frame. Returns only tracks matched *this* frame (see module docstring for why
        a track that goes briefly unseen isn't reported with a stale box); it's kept internally for
        `max_age` frames so its ID is reused if the same vehicle reappears nearby."""
        self._frame_idx += 1

        # Greedy nearest-match: cheapest (track, detection) pair first, skipping anything already
        # claimed. Not globally optimal like the Hungarian algorithm, but needs no external library
        # and is a fine approximation at vehicle counts/frame rates this pipeline sees.
        pairs = []
        for tid, track in self._tracks.items():
            for di, det in enumerate(detections):
                cost = self._cost(track, det)
                if cost is not None:
                    pairs.append((cost, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for cost, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            self._tracks[tid].update_from(detections[di], self._frame_idx)

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            self._tracks[self._next_id] = _Track(self._next_id, det, self._frame_idx, self.velocity_window)
            self._next_id += 1

        stale = [tid for tid, t in self._tracks.items() if self._frame_idx - t.last_seen_frame > self.max_age]
        for tid in stale:
            del self._tracks[tid]

        return [t.to_dict() for t in self._tracks.values() if t.last_seen_frame == self._frame_idx]

    def update_json(self, detections: list[Detection]) -> str:
        return json.dumps(self.update(detections))


def _track_color(track_id: int) -> tuple[int, int, int]:
    # Deterministic per-ID color so a track keeps its color across frames without storing a palette.
    h = (track_id * 2654435761) & 0xFFFFFFFF
    return h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF


def draw_tracks(frame, tracks: list[dict]) -> None:
    import cv2

    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        color = _track_color(t["track_id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"#{t['track_id']} {t['class']}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        vx, vy = t["velocity"]
        if abs(vx) > 0.5 or abs(vy) > 0.5:
            scale = 5  # exaggerate so slow motion is still visible
            cv2.arrowedLine(frame, (cx, cy), (int(cx + vx * scale), int(cy + vy * scale)), color, 2, tipLength=0.3)


if __name__ == "__main__":
    import argparse
    import time

    import cv2

    import config
    from cv.frame_source import FrameSource
    from cv.vehicle_detector import VehicleDetector

    parser = argparse.ArgumentParser(description="Preview the tracker on top of the vehicle detector.")
    parser.add_argument("source", nargs="?", default=config.VIDEO_SOURCE, help="webcam index or video file path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--max-age", type=int, default=15)
    parser.add_argument("--out", default=None, help="write annotated video here instead of opening a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = VehicleDetector(args.model, conf=args.conf)
    tracker = VehicleTracker(max_age=args.max_age)
    print(f"Model {args.model} on {detector.device}")

    writer = None
    with FrameSource(args.source, loop=args.out is None, target_fps=0 if args.out else None) as src:
        last, fps, n = time.perf_counter(), 0.0, 0
        for frame, ts in src:
            tracks = tracker.update(detector.detect(frame))
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            n += 1
            if n % 30 == 0:
                print(f"frame {n}: {len(tracks)} tracks, {fps:.1f} FPS  ids={[t['track_id'] for t in tracks]}")

            draw_tracks(frame, tracks)
            cv2.putText(frame, f"{fps:5.1f} FPS  {len(tracks)} tracks", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src.native_fps or 30, (w, h))
                writer.write(frame)
            else:
                cv2.imshow("VehicleTracker", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
