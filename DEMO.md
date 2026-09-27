# DriveSense demo runbook (Task 18)

Two machines, two terminals. Run `check_env.py` on both before touching anything else.

## Machines

| Machine | Role | IP |
|---|---|---|
| Windows | CV pipeline (`cv/`), WebSocket client | 192.168.8.11 |
| Mac | FastAPI relay + DB + audio + frontend (`server/`, `db/`, `audio/`, `frontend/`) | 192.168.8.12 |
| Raspberry Pi | (reserved, not used by this stack) | 192.168.8.10 |

Both machines need the same repo, venv, and `.env` (copied from `.env.example`) --
see `README.md` "Setup". `SERVER_URL` in Windows's `.env` is the one value that
differs per machine; everything else can be identical.

## Cold start

### 1. Mac

```
source venv/bin/activate
python check_env.py
```

Fix anything `[FAIL]`. A `[WARN]` on `hypertable=False` is fine (plain Postgres
table, still works) -- Tiger Cloud DSN and ElevenLabs key must be `[ OK ]`.
`check_env.py` also prints the Mac's LAN IP and the exact `SERVER_URL=` line to
paste into Windows's `.env` -- use that instead of retyping the table above.

```
python main.py
```

Confirm: `curl -s localhost:8000/health` shows `"db":"connected"` and
`"voice":{"enabled":true,...}`.

Open `http://192.168.8.12:8000/` in a browser, click **START HUD** -- this click
is required, it's the only point a browser grants the autoplay exception audio
alerts need. Second tab: `http://192.168.8.12:8000/metrics.html`.

### 2. Windows

```
venv\Scripts\activate
```

Set `SERVER_URL=ws://192.168.8.12:8000/ws/ingest` in `.env` (the line
`check_env.py` printed on the Mac).

```
python check_env.py
```

Fix anything `[FAIL]`. `YOLO device=cpu` is a `[WARN]`, not fatal, but will be
slow -- confirm CUDA/torch install if it shows on the NVIDIA box. `[FAIL]` on the
TCP or WebSocket check almost always means either the Mac isn't running `main.py`
yet, or the Mac's firewall is blocking the port (see Failure Playbook below).

```
python -m cv.pipeline --roi 100,90,55,80 --direction straight
```

`--roi` is the traffic-light crop tuned for `ampel_red_to_green.ogv`
(`data/samples/SOURCES.md`) -- re-tune for a different clip or a live camera
angle. Omit `--source` to use `config.VIDEO_SOURCE` (defaults to
`data/samples/intersection_montreal_720p.webm`, downloaded per that same file).

### 3. Verify

Annotated frames appear on the Mac's HUD tab within a few seconds. When the
traffic light flips green or the lead vehicle accelerates while the ego vehicle
is stationary: the banner flashes, the voice line plays (only after the START
HUD click unlocked audio), and `metrics.html`'s alert count ticks up.

## Fallbacks

- **Windows/YOLO unavailable:** `./venv/bin/python scripts/fake_producer.py --synthetic`
  on the Mac drives the exact same wire schema against `localhost` -- proves the
  Mac stack end-to-end without any CV pipeline running.
- **Is the relay itself working, independent of the browser?**
  `./venv/bin/python scripts/stream_smoke.py` prints one line per message received
  on `/ws/stream`.

## Failure playbook

| Symptom | Check | Fix |
|---|---|---|
| HUD stays black after clicking START | Browser console; `/health`'s `stream_clients` | Confirm Windows is actually connected (`frames_relayed` climbing in `/health`); reload the HUD tab |
| No sound on alert | Did you click START HUD first? | Autoplay is blocked until that click unlocks the page's `Audio` element -- reload and click again |
| Banner never flashes / no alerts firing | `event` messages arriving? (`scripts/stream_smoke.py`) | Check the traffic-light `--roi` matches the actual light's position in frame; re-tune with `python -m cv.traffic_light <source> --roi x,y,w,h` |
| `/health` shows `"db":"unavailable"` | `check_env.py` on the Mac | `TIGER_DATA_DSN` wrong/expired -- demo still runs, just without DB logging or the metrics page's data |
| Windows `check_env.py` fails the TCP/WebSocket check | Is `main.py` running on the Mac? Same Wi-Fi/subnet? | Start `main.py`; on first LAN connection macOS prompts "Allow incoming connections" for Python -- click **Allow** (see below) |
| macOS firewall prompt appears mid-demo | System Settings > Network > Firewall | Click Allow once; it won't ask again for that binary |

## Recovery

Both reconnects are automatic and independent of each other:

- **Windows -> Mac** (`cv/pipeline.py`): exponential backoff, 1s doubling to a
  30s cap. Restarting `main.py` on the Mac does **not** require restarting the
  Windows process -- it reconnects on its own once the Mac is back.
- **Browser -> Mac** (`frontend/hud.js`): independent exponential backoff,
  500ms-8s. A HUD tab can be reloaded or a laptop put to sleep and it recovers
  without touching either producer.

To rehearse this: stop `main.py` on the Mac for ~30s mid-run, then restart it.
Windows should log `connection lost (...), retrying in Ns` with doubling
backoff and resume streaming without any Windows-side action; the browser tab
recovers the same way on its own.
