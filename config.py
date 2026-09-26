import os

from dotenv import load_dotenv

load_dotenv()

TIGER_DATA_DSN = os.getenv("TIGER_DATA_DSN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Log every Nth ingested frame as telemetry instead of every frame (Task 13) --
# at ~15 FPS, 30 is roughly one sampled frame every 2s.
TELEMETRY_FRAME_SAMPLE_N = int(os.getenv("TELEMETRY_FRAME_SAMPLE_N", "30"))

# No webcam yet: default to sample footage. Set VIDEO_SOURCE=0 for the first webcam.
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "data/samples/intersection_montreal_720p.webm")
TARGET_FPS = float(os.getenv("TARGET_FPS")) if os.getenv("TARGET_FPS") else None
