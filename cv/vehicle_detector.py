"""Vehicle detector: YOLOv8 filtered to car/truck/bus/motorcycle, one JSON-ready dict per detection."""

import json

import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")


def _default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class VehicleDetector:
    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.4,
        device: str | None = None,
        imgsz: int = 640,
        classes: tuple[str, ...] = VEHICLE_CLASSES,
    ):
        """model: weights path/name (ultralytics downloads yolov8n.pt on first use)."""
        self.model = YOLO(model)
        self.conf = conf
        self.device = device or _default_device()
        self.quantize = 16 if self.device.startswith("cuda") else None  # FP16 roughly doubles GPU throughput
        self.imgsz = imgsz

        name_to_id = {name: i for i, name in self.model.names.items()}
        missing = [c for c in classes if c not in name_to_id]
        if missing:
            raise ValueError(f"Model has no classes {missing}; available: {sorted(name_to_id)}")
        self.class_ids = [name_to_id[c] for c in classes]

        # First inference pays CUDA init + graph setup (~1 s); do it now rather than on frame 1 of the stream.
        self.detect(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> list[dict]:
        result = self.model.predict(
            frame,
            conf=self.conf,
            classes=self.class_ids,
            device=self.device,
            quantize=self.quantize,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().round().astype(int)
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        return [
            {
                "id": None,  # assigned by the tracker (Task 4)
                "bbox": [int(v) for v in box],
                "class": self.model.names[c],
                "conf": round(float(p), 2),
            }
            for box, p, c in zip(xyxy, confs, cls_ids)
        ]

    def detect_json(self, frame: np.ndarray) -> str:
        return json.dumps(self.detect(frame))


_CLASS_BGR = {"car": (255, 160, 0), "truck": (0, 140, 255), "bus": (0, 220, 255), "motorcycle": (255, 0, 200)}


def draw_detections(frame: np.ndarray, detections: list[dict]) -> None:
    import cv2

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = _CLASS_BGR.get(d["class"], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d['class']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":
    import argparse
    import time

    import cv2

    import config
    from cv.frame_source import FrameSource

    parser = argparse.ArgumentParser(description="Preview the vehicle detector on a webcam or video.")
    parser.add_argument("source", nargs="?", default=config.VIDEO_SOURCE, help="webcam index or video file path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda:0 (default: auto)")
    parser.add_argument("--out", default=None, help="write annotated video here instead of opening a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = VehicleDetector(args.model, conf=args.conf, device=args.device)
    print(f"Model {args.model} on {detector.device}")

    writer = None
    # Unthrottled when writing to a file: we want to measure inference speed, not play back in real time.
    with FrameSource(args.source, loop=args.out is None, target_fps=0 if args.out else None) as src:
        last, fps, n = time.perf_counter(), 0.0, 0
        for frame, ts in src:
            detections = detector.detect(frame)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            n += 1
            if n % 30 == 0:
                print(f"frame {n}: {len(detections)} vehicles, {fps:.1f} FPS  {json.dumps(detections[:2])}")

            draw_detections(frame, detections)
            cv2.putText(frame, f"{fps:5.1f} FPS  {len(detections)} vehicles", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src.native_fps or 30, (w, h))
                writer.write(frame)
            else:
                cv2.imshow("VehicleDetector", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
