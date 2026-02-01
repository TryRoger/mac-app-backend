# Mouse Backend

Simple FastAPI service backed by Redis that exposes health and per-user pub/sub style endpoints.

## Requirements

- Docker with docker compose plugin, or a local Python 3.11 environment

## Running with Docker

Create a local `.env` file:

```bash
cp .env.example .env
```

```bash
docker compose up --build
```

The API becomes available on `http://localhost:3001`.

### Enable the LLM planner (OpenAI)

Set values in `.env` (recommended) or export env vars before running Docker:

```bash
export OPENAI_API_KEY="..."
export ROGER_PROVIDER="openai"
# optional:
export ROGER_MODEL="gpt-4o-2024-08-06"
docker compose up --build
```

## Local development (optional)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Overview

- `GET /health` – readiness probe.
- `POST /channels` – create a user-specific channel (UUID returned).
- `POST /channels/{channel_id}/publish` – publish arbitrary JSON data to a channel body `{"payload": ...}`.
- `GET /channels/{channel_id}/messages?limit=50&consume=false` – fetch queued messages. Set `consume=true` to remove returned items from the queue for basic pub/sub semantics.

## Roger (Agentic Cursor) API

### `POST /roger/ask`

Takes a full-screen screenshot + cursor/screen context + an optional "app map" (menus/shortcuts/coordinates) and returns a structured list of UI steps (click/drag/hotkey/wait).

Example request (JSON):

```json
{
  "prompt": "click export",
  "t": 12.34,
  "cursor": { "x": 640, "y": 360 },
  "screen": { "width": 1440, "height": 900 },
  "screenshot": {
    "media_type": "image/png",
    "data_base64": "<base64 bytes>"
  },
  "app_map": {
    "name": "example-editor",
    "json": {
      "shortcuts": { "save": "Cmd+S" },
      "menus": [
        { "label": "Export", "rect": { "x": 1200, "y": 40, "width": 90, "height": 28 } }
      ]
    }
  }
}
```

Example response:

```json
{
  "request_id": "3b0ee0d9-3f8b-4f8c-84ee-5db7d2941c2a",
  "created_at": "2026-01-31T00:00:00Z",
  "screenshot_sha256": "<sha256>",
  "steps": [
    {
      "step": {
        "description": "Click UI element: Export",
        "action": "click",
        "bounded_rectangle": { "x1": 1200, "y1": 40, "x2": 1290, "y2": 68 }
      }
    }
  ],
  "notes": []
}
```

Each channel is isolated and anchored by the UUID issued from the create endpoint; messages are stored in Redis lists to provide FIFO delivery semantics.

## Utilities

### Speech → text (multilingual)

Use `scripts/speech_to_text.py` with your `OPENAI_API_KEY` set:

```bash
export OPENAI_API_KEY="sk-..."
python scripts/speech_to_text.py --audio "/path/to/audio_or_video_file" --print-text-only
```

Record from microphone (requires local deps):

```bash
pip install sounddevice soundfile
python scripts/speech_to_text.py --mic --duration 4 --print-text-only
```
