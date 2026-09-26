import os

from dotenv import load_dotenv

load_dotenv()

TIGER_DATA_DSN = os.getenv("TIGER_DATA_DSN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# No webcam yet: default to sample footage. Set VIDEO_SOURCE=0 for the first webcam.
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "data/samples/intersection_montreal_720p.webm")
TARGET_FPS = float(os.getenv("TARGET_FPS")) if os.getenv("TARGET_FPS") else None
