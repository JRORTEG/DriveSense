"""Startup checklist (Plan.md Task 18): verifies .env, DB reachability, ElevenLabs key, YOLO
weights, sample footage, and the Windows<->Mac LAN path -- so all of that is discovered before the
demo starts, not during it.

Role-aware: the Mac (server/DB/audio) and the Windows PC (CV pipeline/GPU/LAN) need different
checks. Auto-detects role from sys.platform; override with --role. --offline skips every
network/API call (Tiger Cloud, ElevenLabs, the WebSocket handshake to the other machine) for a fast
local-only check.

Usage:
    python check_env.py                  # auto-detect role
    python check_env.py --role mac
    python check_env.py --role windows --offline
"""

import argparse
import asyncio
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent

# requirements.txt package name -> the name it's actually imported as.
_IMPORT_NAMES = {
    "opencv-python": "cv2",
    "ultralytics": "ultralytics",
    "fastapi": "fastapi",
    "uvicorn[standard]": "uvicorn",
    "websockets": "websockets",
    "asyncpg": "asyncpg",
    "python-dotenv": "dotenv",
    "elevenlabs": "elevenlabs",
    "numpy": "numpy",
}

_failed = 0


def report(status: str, label: str, detail: str = "") -> None:
    """status: 'OK' | 'WARN' | 'FAIL'. FAIL increments the exit-code counter."""
    global _failed
    if status == "FAIL":
        _failed += 1
    tag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    line = f"{tag} {label}"
    if detail:
        line += f" -- {detail}"
    print(line)


def lan_ip() -> str | None:
    """Best-effort primary LAN IP: a UDP "connect" just sets up routing, no packet is actually
    sent, so this works offline too."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def check_shared() -> None:
    print("-- shared --")
    if sys.version_info >= (3, 10):
        report("OK", f"Python {sys.version.split()[0]}")
    else:
        report("WARN", f"Python {sys.version.split()[0]}", "3.10+ recommended (match-case / X | Y unions used)")

    req_path = REPO_ROOT / "requirements.txt"
    packages = [
        line.strip() for line in req_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ] if req_path.is_file() else []
    for pkg in packages:
        module = _IMPORT_NAMES.get(pkg, pkg)
        try:
            __import__(module)
            report("OK", f"import {module}")
        except ImportError as exc:
            report("FAIL", f"import {module}", f"pip install -r requirements.txt ({exc})")

    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        report("OK", ".env exists")
    else:
        report("FAIL", ".env exists", "copy .env.example to .env and fill it in")


async def check_mac(offline: bool) -> None:
    print("-- mac --")
    import config

    ip = lan_ip()
    if ip:
        report("OK", f"LAN IP: {ip}", f"Windows .env needs SERVER_URL=ws://{ip}:{config.SERVER_PORT}/ws/ingest")
    else:
        report("WARN", "LAN IP", "could not determine outbound interface")

    if config.SERVER_HOST == "127.0.0.1":
        report("FAIL", f"SERVER_HOST={config.SERVER_HOST}", "must be 0.0.0.0 or Windows can't reach it over LAN")
    else:
        report("OK", f"SERVER_HOST={config.SERVER_HOST}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            in_use = s.connect_ex(("127.0.0.1", config.SERVER_PORT)) == 0
        if in_use:
            report("WARN", f"port {config.SERVER_PORT}", "already in use -- main.py may already be running")
        else:
            report("OK", f"port {config.SERVER_PORT} free")
    except OSError as exc:
        report("WARN", f"port {config.SERVER_PORT} check failed", str(exc))

    if config.TELEMETRY_FRAME_SAMPLE_N >= 1:
        report("OK", f"TELEMETRY_FRAME_SAMPLE_N={config.TELEMETRY_FRAME_SAMPLE_N}")
    else:
        report("FAIL", f"TELEMETRY_FRAME_SAMPLE_N={config.TELEMETRY_FRAME_SAMPLE_N}", "must be >= 1")

    frontend_dir = REPO_ROOT / "frontend"
    for name in ("index.html", "hud.js", "hud.css", "metrics.html", "metrics.js", "metrics.css"):
        if (frontend_dir / name).is_file():
            report("OK", f"frontend/{name}")
        else:
            report("FAIL", f"frontend/{name}", "missing")

    if offline:
        report("WARN", "Tiger Cloud DB", "skipped (--offline)")
        report("WARN", "ElevenLabs API", "skipped (--offline)")
        return

    from db.pool import _dsn_looks_configured, apply_schema, close_pool, create_pool, try_create_hypertable

    if not _dsn_looks_configured(config.TIGER_DATA_DSN):
        report("FAIL", "TIGER_DATA_DSN", "unset or still the .env.example placeholder")
    else:
        pool = await create_pool()
        if pool is None:
            report("FAIL", "TIGER_DATA_DSN", "configured but connection failed -- see logged traceback above")
        else:
            try:
                await apply_schema(pool)
                async with pool.acquire() as conn:
                    tables = await conn.fetch(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_name IN ('detection_events', 'reaction_times')"
                    )
                names = {r["table_name"] for r in tables}
                expected = {"detection_events", "reaction_times"}
                if expected <= names:
                    report("OK", "Tiger Cloud reachable, schema applied")
                else:
                    report("FAIL", "schema", f"missing tables: {expected - names}")
                hypertable = await try_create_hypertable(pool)
                report("OK" if hypertable else "WARN", f"hypertable={hypertable}")
            finally:
                await close_pool(pool)

    from audio.tts import _key_looks_configured

    if not _key_looks_configured(config.ELEVENLABS_API_KEY):
        report("FAIL", "ELEVENLABS_API_KEY", "unset or still the .env.example placeholder")
    else:
        from audio.tts import VoiceCache

        cache = VoiceCache()
        await cache.warm_up()
        stats = cache.stats()
        if stats["enabled"] and len(stats["cached_reasons"]) >= 2:
            report("OK", f"ElevenLabs key valid, cached reasons: {stats['cached_reasons']}")
        else:
            report("FAIL", "ElevenLabs key", f"warm_up did not populate the cache ({stats})")


async def check_windows(offline: bool) -> None:
    print("-- windows --")
    import config

    parsed = urlparse(config.SERVER_URL)
    host, port = parsed.hostname, parsed.port
    if host in ("127.0.0.1", "localhost"):
        report("WARN", f"SERVER_URL={config.SERVER_URL}", "loopback -- won't reach the Mac over LAN; set it to ws://<mac-ip>:8000/ws/ingest")
    elif host and port:
        report("OK", f"SERVER_URL={config.SERVER_URL}")
    else:
        report("FAIL", f"SERVER_URL={config.SERVER_URL}", "could not parse host:port")

    video_source = config.VIDEO_SOURCE
    try:
        from cv.frame_source import FrameSource

        with FrameSource(video_source, loop=False, target_fps=None) as src:
            frame, _ = next(iter(src))
            report("OK", f"VIDEO_SOURCE={video_source}", f"{src.width}x{src.height} @ {src.native_fps:.1f} FPS")
    except FileNotFoundError:
        hint = "run the curl commands in data/samples/SOURCES.md" if "data/samples" in str(video_source) else "check the path"
        report("FAIL", f"VIDEO_SOURCE={video_source}", f"not found -- {hint}")
    except Exception as exc:
        report("FAIL", f"VIDEO_SOURCE={video_source}", f"failed to open/read: {exc}")

    if config.TARGET_FPS is not None and config.TARGET_FPS <= 0:
        report("FAIL", f"TARGET_FPS={config.TARGET_FPS}", "must be positive or unset")
    else:
        report("OK", f"TARGET_FPS={config.TARGET_FPS or 'unthrottled (native)'}")

    try:
        from cv.vehicle_detector import VehicleDetector

        detector = VehicleDetector("yolov8n.pt")
        if detector.device == "cpu":
            report("WARN", f"YOLO device={detector.device}", "expected cuda on the NVIDIA box -- check the torch/CUDA install")
        else:
            report("OK", f"YOLO device={detector.device}")
    except Exception as exc:
        report("FAIL", "YOLO weights/model load", str(exc))

    if offline:
        report("WARN", "Mac LAN route", "skipped (--offline)")
        return

    if not (host and port):
        report("FAIL", "Mac LAN route", "cannot test -- SERVER_URL didn't parse")
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect((host, port))
        report("OK", f"TCP reachable: {host}:{port}")
    except OSError as exc:
        report("FAIL", f"TCP reachable: {host}:{port}", f"{exc} -- check the Mac is running main.py and its firewall allows this port")
        return

    try:
        import json

        import websockets

        async with asyncio.timeout(5):
            async with websockets.connect(config.SERVER_URL) as ws:
                await ws.send(json.dumps({
                    "type": "event", "alert": False, "reason": None,
                    "ego_stationary": True, "timestamp": 0.0,
                }))
        report("OK", f"WebSocket handshake to {config.SERVER_URL}")
    except Exception as exc:
        report("FAIL", f"WebSocket handshake to {config.SERVER_URL}", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["mac", "windows"], default=None,
                         help="defaults to auto-detect from the OS (darwin -> mac, win32 -> windows)")
    parser.add_argument("--offline", action="store_true", help="skip every network/API call")
    args = parser.parse_args()

    role = args.role or ("mac" if sys.platform == "darwin" else "windows" if sys.platform == "win32" else None)
    if role is None:
        print(f"Could not auto-detect role from platform {sys.platform!r} -- pass --role mac|windows")
        sys.exit(1)
    print(f"DriveSense environment check -- role={role}{' (offline)' if args.offline else ''}")

    check_shared()
    if role == "mac":
        asyncio.run(check_mac(args.offline))
    else:
        asyncio.run(check_windows(args.offline))

    print()
    if _failed:
        print(f"{_failed} check(s) failed.")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
