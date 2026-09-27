# DriveSense

AI-powered smart dashcam: contextual navigation + distraction alerts at intersections.

## Setup

This is a dual-machine stack (see "Hardware / Network" below) -- clone the repo
and set up a venv on both machines.

**Windows** (runs the CV pipeline, `cv/`):

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env`'s `SERVER_URL` with the Mac's `/ws/ingest` address (e.g.
`ws://192.168.8.12:8000/ws/ingest`), then `python check_env.py` before running
`python -m cv.pipeline`.

**Mac** (runs the server/DB/audio/frontend, `server/` `db/` `audio/` `frontend/`):

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`'s `TIGER_DATA_DSN` and `ELEVENLABS_API_KEY`, then
`python check_env.py` before running `python main.py`.

See `DEMO.md` for the full cold-start runbook and a failure playbook.

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
