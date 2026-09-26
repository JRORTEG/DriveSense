"""Unified frame source: webcam or pre-recorded file, optionally looped and FPS-throttled."""

import os
import time
from typing import Iterator

import cv2
import numpy as np

_WEBCAM_READ_RETRIES = 5
_WEBCAM_RETRY_DELAY_S = 0.02


class FrameSource:
    def __init__(self, source: int | str, loop: bool = True, target_fps: float | None = None):
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
        if isinstance(source, str) and not os.path.isfile(source):
            raise FileNotFoundError(f"Video file not found: {source}")

        self.source = source
        self.loop = loop
        self._cap = self._open()

        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if target_fps is None and self.is_file and self._native_fps > 0:
            target_fps = self._native_fps
        if target_fps is not None and not self.is_file:
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)  # best effort; many cams ignore it
        self.target_fps = target_fps
        self._next_deadline: float | None = None

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str)

    @property
    def native_fps(self) -> float:
        return self._native_fps

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _open(self) -> cv2.VideoCapture:
        if self.is_file:
            cap = cv2.VideoCapture(self.source)
        else:
            # DSHOW opens much faster than the default MSMF backend on Windows.
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            kind = "video file" if self.is_file else "webcam index"
            raise RuntimeError(f"Could not open {kind}: {self.source}")
        return cap

    def _throttle(self) -> None:
        if not self.target_fps:
            return
        period = 1.0 / self.target_fps
        now = time.perf_counter()
        if self._next_deadline is None:
            self._next_deadline = now + period
            return
        delay = self._next_deadline - now
        if delay > 0:
            time.sleep(delay)
            self._next_deadline += period
        else:
            # Fell behind; resync instead of bursting to catch up.
            self._next_deadline = now + period

    def _read_file(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if ok:
            return frame
        if not self.loop:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._cap.read()
        if ok:
            return frame
        # Some codecs ignore seeking; reopen from scratch.
        self._cap.release()
        self._cap = self._open()
        ok, frame = self._cap.read()
        return frame if ok else None

    def _read_webcam(self) -> np.ndarray | None:
        for _ in range(_WEBCAM_READ_RETRIES):
            ok, frame = self._cap.read()
            if ok:
                return frame
            time.sleep(_WEBCAM_RETRY_DELAY_S)
        return None

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        self._throttle()
        return self._read_file() if self.is_file else self._read_webcam()

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame, time.time()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview a FrameSource.")
    parser.add_argument("source", nargs="?", default="0", help="webcam index or video file path")
    parser.add_argument("--no-loop", action="store_true", help="stop at end of file")
    parser.add_argument("--fps", type=float, default=None, help="target FPS throttle")
    args = parser.parse_args()

    with FrameSource(args.source, loop=not args.no_loop, target_fps=args.fps) as src:
        print(f"Opened {args.source}: {src.width}x{src.height}, native {src.native_fps:.1f} FPS, "
              f"target {src.target_fps or 'unthrottled'}")
        last = time.perf_counter()
        fps = 0.0
        for frame, ts in src:
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("FrameSource", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
