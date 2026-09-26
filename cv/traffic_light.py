"""Traffic light state detector: HSV color masking inside a light-housing ROI, with red->green edge detection."""

import json
import time
from typing import Literal

import cv2
import numpy as np

State = Literal["red", "yellow", "green", "unknown"]
Roi = tuple[int, int, int, int]  # (x, y, w, h)

# OpenCV hue is 0-179. Floors were tuned on data/samples/ampel_red_to_green.ogv: lit lamps overexpose to a
# near-white core, so only a thinner colored rim clears S/V — floors much above S80/V110 lose the lamp entirely.
_S_MIN, _V_MIN = 80, 110
_HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, _S_MIN, _V_MIN), (10, 255, 255)), ((165, _S_MIN, _V_MIN), (179, 255, 255))],  # wraps 0/180
    "yellow": [((15, _S_MIN, _V_MIN), (35, 255, 255))],
    "green": [((40, _S_MIN, _V_MIN), (100, 255, 255))],  # LED greens skew toward cyan
}

_COCO_TRAFFIC_LIGHT = 9
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_MIN_SIDE_FOR_OPEN = 100  # smaller ROIs have lamps only a few px wide; morphological opening would erase them


class TrafficLightDetector:
    def __init__(
        self,
        roi: Roi | None = None,
        min_pixels: int = 8,
        min_blob_frac: float = 0.002,
        confirm_frames: int = 3,
        yolo_model: str | None = None,
        yolo_conf: float = 0.3,
    ):
        """
        roi: default light-housing region, used when detect() gets no roi.
        min_pixels / min_blob_frac: a color wins only if its largest blob has at least this many pixels
            and covers at least this fraction of the ROI.
        confirm_frames: consecutive frames a raw reading must persist before the reported state changes.
        yolo_model: e.g. "yolov8n.pt" to locate the light automatically when no roi is available.
        """
        self.default_roi = roi
        self.min_pixels = min_pixels
        self.min_blob_frac = min_blob_frac
        self.confirm_frames = max(1, confirm_frames)
        self.yolo_conf = yolo_conf
        self._yolo = None
        if yolo_model:
            from ultralytics import YOLO  # deferred: heavy import, optional path

            self._yolo = YOLO(yolo_model)

        self.state: State = "unknown"
        self.last_roi: Roi | None = None
        self._last_known: State = "unknown"  # last confirmed non-unknown state
        self._candidate: State = "unknown"
        self._candidate_count = 0

    def reset(self) -> None:
        self.state = "unknown"
        self._last_known = "unknown"
        self._candidate = "unknown"
        self._candidate_count = 0

    def _find_roi_yolo(self, frame: np.ndarray) -> Roi | None:
        result = self._yolo(frame, classes=[_COCO_TRAFFIC_LIGHT], conf=self.yolo_conf, verbose=False)[0]
        if len(result.boxes) == 0:
            return None
        # Prefer the largest box: usually the nearest light, i.e. the one facing us.
        xyxy = result.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        x1, y1, x2, y2 = xyxy[int(np.argmax(areas))].astype(int)
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    @staticmethod
    def _clip_roi(roi: Roi, frame: np.ndarray) -> Roi | None:
        h, w = frame.shape[:2]
        x, y, rw, rh = roi
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1 - x0, y1 - y0

    def color_scores(self, roi_bgr: np.ndarray) -> dict[str, int]:
        """Largest-blob pixel area per color within the ROI (0 when below thresholds)."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        min_area = max(self.min_pixels, self.min_blob_frac * roi_bgr.shape[0] * roi_bgr.shape[1])
        scores = {}
        for color, ranges in _HSV_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            if min(mask.shape) >= _MIN_SIDE_FOR_OPEN:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                scores[color] = 0
                continue
            largest = max(contours, key=cv2.contourArea)
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [largest], -1, 255, cv2.FILLED)
            area = cv2.countNonZero(blob & mask)  # lit pixels in the blob; robust for tiny/ragged contours
            scores[color] = area if area >= min_area else 0
        return scores

    def classify(self, roi_bgr: np.ndarray) -> State:
        """Single-frame raw classification, no temporal smoothing."""
        if roi_bgr.size == 0:
            return "unknown"
        scores = self.color_scores(roi_bgr)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"

    def _update_state(self, raw: State) -> bool:
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = raw, 1
        if self._candidate_count < self.confirm_frames or raw == self.state:
            return False

        self.state = raw
        if raw == "unknown":
            return False
        # Compare against last *known* state so a brief unknown gap (lamp switching, occlusion) doesn't hide the edge.
        transitioned = raw == "green" and self._last_known == "red"
        self._last_known = raw
        return transitioned

    def detect(self, frame: np.ndarray, roi: Roi | None = None, timestamp: float | None = None) -> dict:
        ts = time.time() if timestamp is None else timestamp
        roi = roi or self.default_roi
        if roi is None and self._yolo is not None:
            roi = self._find_roi_yolo(frame)

        if roi is None:
            # No housing located. Scanning the whole frame would pick up brake lights, signs and foliage.
            self.last_roi = None
            raw: State = "unknown"
        else:
            self.last_roi = self._clip_roi(roi, frame)
            if self.last_roi is None:
                raw = "unknown"
            else:
                x, y, w, h = self.last_roi
                raw = self.classify(frame[y:y + h, x:x + w])

        transitioned = self._update_state(raw)
        return {"state": self.state, "transitioned_to_green": transitioned, "timestamp": ts}

    def detect_json(self, frame: np.ndarray, roi: Roi | None = None, timestamp: float | None = None) -> str:
        return json.dumps(self.detect(frame, roi, timestamp))


_STATE_BGR = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0), "unknown": (160, 160, 160)}


def draw_overlay(frame: np.ndarray, result: dict, roi: Roi | None) -> None:
    color = _STATE_BGR[result["state"]]
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"LIGHT: {result['state'].upper()}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview the traffic light detector on a webcam or video.")
    parser.add_argument("source", nargs="?", default="0", help="webcam index or video file path")
    parser.add_argument("--roi", type=str, default=None, help="fixed ROI as x,y,w,h")
    parser.add_argument("--select", action="store_true", help="drag to select the ROI on the first frame")
    parser.add_argument("--yolo", type=str, default=None, help="YOLO weights to auto-locate the light, e.g. yolov8n.pt")
    parser.add_argument("--fps", type=float, default=None, help="target FPS throttle")
    args = parser.parse_args()

    from cv.frame_source import FrameSource

    with FrameSource(args.source, loop=True, target_fps=args.fps) as src:
        roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
        if args.select:
            first = src.read()
            if first is None:
                raise SystemExit("Could not read a frame to select ROI from")
            sel = cv2.selectROI("Select traffic light", first, showCrosshair=False)
            cv2.destroyWindow("Select traffic light")
            roi = tuple(int(v) for v in sel) if sel[2] and sel[3] else None
            print(f"Selected ROI: {roi}")

        detector = TrafficLightDetector(roi=roi, yolo_model=args.yolo)
        for frame, ts in src:
            result = detector.detect(frame, timestamp=ts)
            if result["transitioned_to_green"]:
                print(json.dumps(result))
            draw_overlay(frame, result, detector.last_roi)
            cv2.imshow("TrafficLightDetector", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
