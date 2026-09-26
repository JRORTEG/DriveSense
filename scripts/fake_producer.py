"""Stand-in for the Windows CV pipeline (Plan.md Tasks 1-9 + 11), so Task 10's
relay can be exercised on the Mac alone before Windows is online.

Pushes webcam (or synthetic) JPEG frames as base64, plus a periodic fake
alert event, to /ws/ingest. Uses the same reconnect-with-backoff shape Task
11 needs on Windows -- meant to be liftable almost as-is once that's built.

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
import time

import numpy as np
import websockets

DEFAULT_URL = "ws://127.0.0.1:8000/ws/ingest"
FPS = 15
EVENT_INTERVAL_S = 8
JPEG_QUALITY = 70

REASONS = itertools.cycle(["light_green", "lead_accelerating"])


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
                        event = {
                            "type": "event",
                            "alert": True,
                            "reason": next(REASONS),
                            "timestamp": time.time(),
                        }
                        await ws.send(json.dumps(event))
                        print(f"sent fake event: {event['reason']}")

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
