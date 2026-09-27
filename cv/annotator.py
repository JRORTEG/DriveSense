"""Frame annotation renderer: composes every task's overlay (vehicle boxes, reference-vehicle highlight,
traffic-light badge, alert banner) onto one frame for the HUD. Pure function, no state of its own --
each task's own `draw_overlay`/`draw_tracks` already owns its drawing logic; this just calls them in the
right order. The annotated frame isn't shown locally -- Task 11 hands it to the network client instead.
"""

import numpy as np

from cv.alert_engine import draw_overlay as _draw_alert
from cv.reference_vehicle import draw_overlay as _draw_reference
from cv.traffic_light import Roi, draw_overlay as _draw_light
from cv.tracker import draw_tracks as _draw_tracks


def annotate_frame(
    frame: np.ndarray,
    tracks: list[dict],
    reference_result: dict,
    light_state: dict,
    alert_result: dict,
    light_roi: Roi | None = None,
) -> np.ndarray:
    """
    frame: BGR frame to draw onto (mutated in place; also returned for convenience).
    tracks: Task 4's VehicleTracker.update() output.
    reference_result: Task 8's ReferenceVehicleDetector.update() output.
    light_state: Task 2's TrafficLightDetector.detect() output.
    alert_result: Task 7's AlertEngine.update() output.
    light_roi: the traffic-light housing ROI actually used this frame (TrafficLightDetector.last_roi),
        so the badge draws its box in the right place; None draws just the text badge.
    """
    _draw_tracks(frame, tracks)
    _draw_reference(frame, tracks, reference_result)
    _draw_light(frame, light_state, light_roi)
    _draw_alert(frame, alert_result)
    return frame


if __name__ == "__main__":
    import argparse
    import json
    import time

    import cv2

    import config
    from cv.alert_engine import AlertEngine
    from cv.ego_stationary import EgoStationaryDetector
    from cv.frame_source import FrameSource
    from cv.lead_vehicle import LeadVehicleDetector
    from cv.reference_vehicle import ReferenceVehicleDetector
    from cv.traffic_light import TrafficLightDetector
    from cv.tracker import VehicleTracker
    from cv.vehicle_detector import VehicleDetector

    parser = argparse.ArgumentParser(description="Preview the full HUD render (Tasks 2-9 composed).")
    parser.add_argument("source", nargs="?", default=config.VIDEO_SOURCE, help="webcam index or video file path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--roi", type=str, default=None, help="traffic-light ROI as x,y,w,h")
    parser.add_argument("--select", action="store_true", help="drag to select the traffic-light ROI on frame 1")
    parser.add_argument("--direction", choices=["left", "right", "straight"], default="left")
    parser.add_argument("--debounce", type=float, default=3.0)
    parser.add_argument("--out", default=None, help="write annotated video here instead of opening a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = VehicleDetector(args.model, conf=args.conf)
    tracker = VehicleTracker()
    lead_detector = LeadVehicleDetector()
    ego_detector = EgoStationaryDetector()
    reference_detector = ReferenceVehicleDetector()
    alert_engine = AlertEngine(debounce_s=args.debounce)
    print(f"Model {args.model} on {detector.device}")

    writer = None
    prev_gray = None
    with FrameSource(args.source, loop=args.out is None, target_fps=0 if args.out else None) as src:
        roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
        if args.select:
            first = src.read()
            if first is None:
                raise SystemExit("Could not read a frame to select ROI from")
            sel = cv2.selectROI("Select traffic light", first, showCrosshair=False)
            cv2.destroyWindow("Select traffic light")
            roi = tuple(int(v) for v in sel) if sel[2] and sel[3] else None
            print(f"Selected ROI: {roi}")
        light_detector = TrafficLightDetector(roi=roi)

        last, fps, n = time.perf_counter(), 0.0, 0
        for frame, ts in src:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tracks = tracker.update(detector.detect(frame))
            bboxes = [t["bbox"] for t in tracks]

            light_result = light_detector.detect(frame, timestamp=ts)
            lead_result = lead_detector.update(tracks, frame.shape[1])
            ego_result = ego_detector.update(prev_gray, gray, bboxes) if prev_gray is not None else \
                {"ego_stationary": True, "mean_flow_magnitude": 0.0}
            prev_gray = gray
            reference_result = reference_detector.update(tracks, args.direction)
            alert_result = alert_engine.update(light_result, lead_result, ego_result, ts)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            n += 1
            if alert_result["alert"]:
                print(json.dumps(alert_result))

            annotate_frame(frame, tracks, reference_result, light_result, alert_result, light_detector.last_roi)
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src.native_fps or 30, (w, h))
                writer.write(frame)
            else:
                cv2.imshow("Annotator", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
