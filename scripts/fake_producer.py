"""Stand-in for the Windows CV pipeline (Plan.md Tasks 1-9 + 11), so Task 10's
relay can be exercised on the Mac alone before Windows is online.

Pushes webcam (or synthetic) JPEG frames as base64, plus a per-frame "event"
message carrying the fused alert/ego_stationary state Task 14 depends on, to
/ws/ingest. Every ~8s fires a fake alert, then after a randomized simulated
driver-reaction delay flips ego_stationary False (the signal Task 14 waits
for), then resets it back True shortly after for the next cycle. Uses the
same reconnect-with-backoff shape Task 11 needs on Windows -- meant to be
liftable almost as-is once that's built.

Usage:
    ./venv/bin/python scripts/fake_producer.py
    ./venv/bin/python scripts/fake_producer.py --synthetic
    ./venv/bin/python scripts/fake_producer.py --url ws://192.168.8.12:8000/ws/ingest
"""

import argparse
import asyncio
import base64
import itertools
import json
import random
import time

import numpy as np
import websockets

DEFAULT_URL = "ws://127.0.0.1:8000/ws/ingest"
FPS = 15
EVENT_INTERVAL_S = 8
JPEG_QUALITY = 70

# Simulated driver-reaction timing for exercising Task 14 end to end.
REACTION_DELAY_RANGE_S = (0.5, 2.0)
RESET_DELAY_S = 1.0

REASONS = itertools.cycle(["light_green", "lead_accelerating"])


class EgoState:
    """Shared mutable ego_stationary flag, read by the per-frame event
    sender and written by the background reaction simulator below."""

    def __init__(self) -> None:
        self.stationary = True


async def _simulate_reaction(state: EgoState) -> None:
    """Fired once per alert: waits a randomized delay (as if the driver just
    noticed the alert), flips ego_stationary False (the flip Task 14's
    ReactionTracker is watching for), then resets True after a beat so the
    next alert cycle starts from a clean stationary state."""
    delay = random.uniform(*REACTION_DELAY_RANGE_S)
    await asyncio.sleep(delay)
    state.stationary = False
    print(f"simulated driver reaction after {delay * 1000:.0f}ms")
    await asyncio.sleep(RESET_DELAY_S)
    state.stationary = True


def make_synthetic_frame(counter: int) -> np.ndarray:
    import cv2

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    hue = (counter * 2) % 255
    frame[:] = (hue, 255 - hue, 128)
    cv2.putText(
        frame, f"frame {counter}", (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
    )
    return frame


def encode_frame(frame: np.ndarray) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def run(url: str, synthetic: bool) -> None:
    import cv2

    cap = None
    if not synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("webcam unavailable, falling back to synthetic frames")
            cap = None

    ego = EgoState()
    backoff = 1
    frame_counter = 0
    while True:
        try:
            async with websockets.connect(url) as ws:
                print(f"connected to {url}")
                backoff = 1
                last_event = time.monotonic()
                while True:
                    if cap is not None:
                        ok, frame = cap.read()
                        if not ok:
                            frame = make_synthetic_frame(frame_counter)
                    else:
                        frame = make_synthetic_frame(frame_counter)
                    frame_counter += 1

                    msg = {
                        "type": "frame",
                        "data": encode_frame(frame),
                        "timestamp": time.time(),
                    }
                    await ws.send(json.dumps(msg))

                    now = time.monotonic()
                    if now - last_event >= EVENT_INTERVAL_S:
                        last_event = now
                        reason = next(REASONS)
                        event = {
                            "type": "event",
                            "alert": True,
                            "reason": reason,
                            "ego_stationary": ego.stationary,
                            "timestamp": time.time(),
                        }
                        await ws.send(json.dumps(event))
                        print(f"sent fake event: {reason}")
                        asyncio.create_task(_simulate_reaction(ego))
                    else:
                        # Per-frame fused state, per the wire contract Task 14
                        # depends on -- not just an event on the alert cadence.
                        event = {
                            "type": "event",
                            "alert": False,
                            "reason": None,
                            "ego_stationary": ego.stationary,
                            "timestamp": time.time(),
                        }
                        await ws.send(json.dumps(event))

                    await asyncio.sleep(1 / FPS)
        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            print(f"connection lost ({exc!r}), retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--synthetic", action="store_true", help="skip webcam, generate frames")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url, args.synthetic))
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
