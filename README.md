# DriveSense

AI-powered smart dashcam: contextual navigation + distraction alerts at intersections.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env` with your `TIGER_DATA_DSN` and `ELEVENLABS_API_KEY`.

Run: `python main.py`

## Layout

- `cv/` — vision pipeline (detection, tracking, alert logic)
- `server/` — FastAPI app + WebSocket streaming
- `db/` — Tiger Data (Postgres) schema + logging
- `audio/` — ElevenLabs voice alerts
- `frontend/` — HUD + metrics dashboard
