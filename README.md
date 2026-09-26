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

## Hardware / Network

Static LAN IPs:

| Machine | IP |
|---|---|
| Raspberry Pi | 192.168.8.10 |
| Windows | 192.168.8.11 |
| Mac | 192.168.8.12 |

## Layout

- `cv/` — vision pipeline (detection, tracking, alert logic)
- `server/` — FastAPI app + WebSocket streaming
- `db/` — Tiger Data (Postgres) schema + logging
- `audio/` — ElevenLabs voice alerts
- `frontend/` — HUD + metrics dashboard
