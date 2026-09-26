"""Mac-side launcher: runs the FastAPI relay (server/app.py, Task 10).

Equivalent to: uvicorn server.app:app --host 0.0.0.0 --port 8000
Binding 0.0.0.0 (not 127.0.0.1) is required so the Windows PC can reach
/ws/ingest over LAN.
"""

import uvicorn

import config


def main() -> None:
    print("DriveSense config loaded.")
    print(f"TIGER_DATA_DSN set: {config.TIGER_DATA_DSN is not None}")
    print(f"ELEVENLABS_API_KEY set: {config.ELEVENLABS_API_KEY is not None}")
    print(f"Starting server on {config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run("server.app:app", host=config.SERVER_HOST, port=config.SERVER_PORT)


if __name__ == "__main__":
    main()
