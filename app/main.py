from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, List

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CHANNEL_REGISTRY_KEY = "channels:registry"


class ChannelResponse(BaseModel):
    channel_id: str


class MessagePayload(BaseModel):
    payload: Any


class Message(MessagePayload):
    message_id: str
    published_at: datetime


app = FastAPI(title="Mouse Backend", version="0.1.0")
redis_client: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return redis_client


def _channel_messages_key(channel_id: str) -> str:
    return f"channel:{channel_id}:messages"


@app.on_event("startup")
async def startup_event() -> None:
    client = await get_redis()
    try:
        await client.ping()
    except redis.RedisError as exc:  # pragma: no cover - defensive logging
        raise RuntimeError("Unable to connect to Redis") from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/channels", response_model=ChannelResponse, status_code=201)
async def create_channel() -> ChannelResponse:
    client = await get_redis()
    channel_id = str(uuid.uuid4())
    await client.sadd(CHANNEL_REGISTRY_KEY, channel_id)
    return ChannelResponse(channel_id=channel_id)


async def _ensure_channel_exists(client: redis.Redis, channel_id: str) -> None:
    exists = await client.sismember(CHANNEL_REGISTRY_KEY, channel_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Channel not found")


@app.post("/channels/{channel_id}/publish", response_model=Message)
async def publish_message(channel_id: str, body: MessagePayload) -> Message:
    client = await get_redis()
    await _ensure_channel_exists(client, channel_id)

    message = Message(
        message_id=str(uuid.uuid4()),
        payload=body.payload,
        published_at=datetime.now(timezone.utc),
    )
    await client.rpush(_channel_messages_key(channel_id), message.model_dump_json())
    return message


@app.get("/channels/{channel_id}/messages", response_model=List[Message])
async def fetch_messages(
    channel_id: str,
    limit: int = Query(50, ge=1, le=200),
    consume: bool = Query(False, description="If true, remove returned messages from the channel."),
) -> List[Message]:
    client = await get_redis()
    await _ensure_channel_exists(client, channel_id)

    key = _channel_messages_key(channel_id)
    raw_messages = await client.lrange(key, 0, limit - 1)

    messages = [Message(**json.loads(raw)) for raw in raw_messages]

    if consume and raw_messages:
        await client.ltrim(key, len(raw_messages), -1)

    return messages
