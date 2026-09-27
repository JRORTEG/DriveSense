# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git workflow

- Always create a new branch off `main` before starting work on a new `Plan.md` task. Never commit directly on `main`.
- Branch naming follows the existing pattern: `task-N-short-description` (e.g. `task-5-lead-vehicle-accel`, `task-6-ego-stationary`).
- Do not merge a task's PR into `main` yourself, even if asked with a short "merge"/"merge it" — leave every task PR open on its own branch until the user gives an explicit, deliberate instruction to merge that specific PR (not just "next step" or moving on to the next task). Each task's own branch is the source of truth; `main` only gets a task's code when the user has clearly decided it's ready.
- If a PR was merged into `main` and the user asks to undo it, use `git revert -m 1 <merge-commit>` (not a history rewrite / force-push) so the branch's own commits and PR history stay intact.
- Do not `git commit` automatically as part of finishing a task — implement and test the change, then stop and ask before committing. This applies on task branches too, not just `main`; committing to `main` needs even more caution (see above: never direct, never without an explicit ask).

## Task tracking

After completing any numbered task from `Plan.md` (fully implemented and verified), immediately update that task's `## TASK N` heading in `Plan.md` to mark it complete — append `(COMPLETE — <one-line pointer to the file(s)/branch/verification>)` to the heading, and add a `**Status:**` paragraph after the existing Objective/Technical Approach/Interface describing what was actually built, where, and how it was verified. Do this whether the work is merged to `main` or still on a feature/task branch — note the branch name in the pointer if unmerged. Do not wait to be asked.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Commands

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env` with `TIGER_DATA_DSN` and `ELEVENLABS_API_KEY`.

Run: `python main.py`

No test suite, linter, or formatter is configured yet — do not invent commands for these.

## Architecture

**Status:** `Plan.md`'s task-by-task build is well underway, not a scaffold — read `Plan.md` before writing code in any module; each `## TASK N` heading is marked `(COMPLETE — ...)` once implemented, noting whether it's merged to `main` or still on its own task branch, so check there for what actually exists before assuming a module is empty.

**Dual-machine split** (see "Hardware Infrastructure" in `Plan.md`): this is not a single-process app.
- **Windows PC (NVIDIA):** runs the CV pipeline — `cv/` (Tasks 1-9: video ingestion, traffic-light HSV detection, YOLO vehicle detection, tracking, lead-vehicle/ego-stationary logic, alert engine, frame annotation). It is a WebSocket *client* that pushes annotated frames + event JSON to the Mac; it does not host its own FastAPI server.
- **Mac (M2 Pro):** runs `server/` (FastAPI, Tasks 10-11), `db/` (Tiger Data/Postgres, Tasks 12-14), `audio/` (ElevenLabs, Task 15), `frontend/` (HUD + Chart.js dashboard, Tasks 16-17). FastAPI binds `0.0.0.0` with CORS enabled, and exposes `/ws/ingest` (Windows pushes here) separate from `/ws/stream` (browser clients consume here) so producer and browser traffic don't collide.

**Config:** `config.py` loads `.env` via `python-dotenv` at import time and exposes plain module-level constants (`TIGER_DATA_DSN`, `ELEVENLABS_API_KEY`) — no config class/singleton. `.env` differs per machine: Windows needs `SERVER_URL` pointing at the Mac's `/ws/ingest`; the Mac needs `TIGER_DATA_DSN` and `ELEVENLABS_API_KEY`.

**Data flow for the core alert feature:** `cv/` Task 2 (traffic light state) + Task 5 (lead-vehicle acceleration) + Task 6 (ego-stationary via optical flow) feed into Task 7's debounced state machine, which fires an alert when (light turns green OR lead vehicle accelerates) AND the ego vehicle is stationary. That alert event is what gets logged to Tiger Data (Task 13), triggers the ElevenLabs voice cue (Task 15), and flashes the HUD banner (Task 16).

**Build order** is risk/value-first, not file order — see the numbered chain at the top of `Plan.md` (CV core → streaming safety net → HUD polish → DB/audio sponsor tracks → frontend → hardening). Each `## TASK N` block in `Plan.md` is self-contained (Objective / Technical Approach / Interface) and is meant to be copy-pasted as its own implementation prompt.
