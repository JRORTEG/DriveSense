"""Distraction alert event engine: fuses Task 2 (traffic light), Task 5 (lead-vehicle acceleration) and
Task 6 (ego-stationary) into the core "you should be moving" trigger, debounced so it fires once per event.
"""

import json


class AlertEngine:
    def __init__(self, debounce_s: float = 3.0):
        """debounce_s: minimum gap between two alerts, regardless of reason -- guards against the light
        turning green and the lead vehicle accelerating within the same stopped period both firing their
        own alert, and against a re-trigger if an edge flag ever repeats across adjacent frames."""
        self.debounce_s = debounce_s
        self._last_alert_ts: float | None = None

    def reset(self) -> None:
        self._last_alert_ts = None

    def update(self, light_state: dict, lead_vehicle: dict, ego_stationary: dict, frame_timestamp: float) -> dict:
        """One call per frame. light_state: Task 2 output. lead_vehicle: Task 5 output.
        ego_stationary: Task 6 output. frame_timestamp: current frame's timestamp (same clock as
        debounce_s is measured in, e.g. FrameSource's `time.time()`)."""
        reason = None
        if ego_stationary.get("ego_stationary"):
            # Light takes priority when both fire on the same frame: it's the more decisive "go" signal.
            if light_state.get("transitioned_to_green"):
                reason = "light_green"
            elif lead_vehicle.get("accelerating"):
                reason = "lead_accelerating"

        alert = False
        if reason is not None:
            debounced = self._last_alert_ts is not None and frame_timestamp - self._last_alert_ts < self.debounce_s
            if not debounced:
                alert = True
                self._last_alert_ts = frame_timestamp

        return {"alert": alert, "reason": reason if alert else None, "timestamp": frame_timestamp}

    def update_json(self, light_state: dict, lead_vehicle: dict, ego_stationary: dict, frame_timestamp: float) -> str:
        return json.dumps(self.update(light_state, lead_vehicle, ego_stationary, frame_timestamp))


def draw_overlay(frame, result: dict) -> None:
    import cv2

    if not result["alert"]:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 255), cv2.FILLED)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    label = {"light_green": "LIGHT IS GREEN - GO", "lead_accelerating": "CAR AHEAD IS MOVING - GO"}[result["reason"]]
    cv2.putText(frame, label, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


if __name__ == "__main__":
    import argparse
    import time

    import cv2

    import config
    from cv.ego_stationary import EgoStationaryDetector
    from cv.ego_stationary import draw_overlay as draw_ego
    from cv.frame_source import FrameSource
    from cv.lead_vehicle import LeadVehicleDetector
    from cv.lead_vehicle import draw_overlay as draw_lead
    from cv.traffic_light import TrafficLightDetector, draw_overlay as draw_light
    from cv.tracker import VehicleTracker, draw_tracks
    from cv.vehicle_detector import VehicleDetector

    parser = argparse.ArgumentParser(description="Preview the full alert engine (Tasks 2+4+5+6+7 composed).")
    parser.add_argument("source", nargs="?", default=config.VIDEO_SOURCE, help="webcam index or video file path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--roi", type=str, default=None, help="traffic-light ROI as x,y,w,h")
    parser.add_argument("--select", action="store_true", help="drag to select the traffic-light ROI on frame 1")
    parser.add_argument("--debounce", type=float, default=3.0)
    parser.add_argument("--out", default=None, help="write annotated video here instead of opening a window")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = VehicleDetector(args.model, conf=args.conf)
    tracker = VehicleTracker()
    lead_detector = LeadVehicleDetector()
    ego_detector = EgoStationaryDetector()
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

            alert_result = alert_engine.update(light_result, lead_result, ego_result, ts)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            n += 1
            if alert_result["alert"]:
                print(json.dumps(alert_result))

            draw_tracks(frame, tracks)
            draw_light(frame, light_result, light_detector.last_roi)
            draw_lead(frame, tracks, lead_result)
            draw_ego(frame, ego_detector, ego_result)
            draw_overlay(frame, alert_result)
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src.native_fps or 30, (w, h))
                writer.write(frame)
            else:
                cv2.imshow("AlertEngine", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
