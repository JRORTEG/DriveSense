import os

from dotenv import load_dotenv

load_dotenv()

TIGER_DATA_DSN = os.getenv("TIGER_DATA_DSN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # webcam index or file path
TARGET_FPS = float(os.getenv("TARGET_FPS")) if os.getenv("TARGET_FPS") else None
