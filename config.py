import os

from dotenv import load_dotenv

load_dotenv()

TIGER_DATA_DSN = os.getenv("TIGER_DATA_DSN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # webcam index or file path
TARGET_FPS = float(os.getenv("TARGET_FPS")) if os.getenv("TARGET_FPS") else None
