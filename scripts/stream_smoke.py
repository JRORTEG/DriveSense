"""Minimal /ws/stream consumer for proving Task 10's relay without waiting
on Task 16's browser HUD. Prints one line per message received.

Usage:
    ./venv/bin/python scripts/stream_smoke.py
    ./venv/bin/python scripts/stream_smoke.py --url ws://192.168.8.12:8000/ws/stream
"""

import argparse
import asyncio
import json
import time

import websockets

DEFAULT_URL = "ws://127.0.0.1:8000/ws/stream"


async def run(url: str) -> None:
    last = time.monotonic()
    async with websockets.connect(url) as ws:
        print(f"connected to {url}")
        async for raw in ws:
            now = time.monotonic()
            delta = now - last
            last = now
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type", "?")
            except json.JSONDecodeError:
                msg_type = "?"
            print(f"[+{delta:5.2f}s] type={msg_type:6s} bytes={len(raw)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url))
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
