# Mouse Backend

Simple FastAPI service backed by Redis that exposes health and per-user pub/sub style endpoints.

## Requirements

- Docker with docker compose plugin, or a local Python 3.11 environment

## Running with Docker

```bash
docker compose up --build
```

The API becomes available on `http://localhost:3001`.

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

Each channel is isolated and anchored by the UUID issued from the create endpoint; messages are stored in Redis lists to provide FIFO delivery semantics.
