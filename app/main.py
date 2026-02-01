from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROGER_MODEL = os.getenv("ROGER_MODEL", "gpt-4o-2024-08-06")
ROGER_PROVIDER = os.getenv("ROGER_PROVIDER", "stub")  # "openai" | "stub"
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


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "docs": "/docs"}


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


class Point(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class Rect(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class ScreenInfo(BaseModel):
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class ScreenshotPayload(BaseModel):
    """
    Screenshot bytes encoded as base64.

    The API doesn't inspect pixels yet, but accepts and hashes bytes so we can
    later plug in a vision model / UI parser without changing the contract.
    """

    data_base64: str = Field(..., description="Base64-encoded screenshot bytes.")
    media_type: str = Field("image/png", description="e.g. image/png, image/jpeg")


class AppMapPayload(BaseModel):
    """
    Optional application map describing UI elements and shortcuts.

    This is intentionally flexible; it can be a JSON object (preferred) or raw text.
    """

    model_config = ConfigDict(populate_by_name=True)

    map_json: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="json",
        description="Arbitrary app map JSON.",
    )
    text: Optional[str] = Field(default=None, description="Raw app map text (optionally JSON).")
    name: Optional[str] = Field(default=None, description="Optional map name or app identifier.")

class BoundedRectangle(BaseModel):
    """
    Rectangle expressed as start/end coordinates (inclusive/exclusive depends on client).

    Convention we use:
    - x1,y1 = top-left corner
    - x2,y2 = bottom-right corner
    """

    x1: int = Field(..., ge=0)
    y1: int = Field(..., ge=0)
    x2: int = Field(..., ge=0)
    y2: int = Field(..., ge=0)


RogerAction = Literal["click", "drag", "hotkey", "wait"]


class RogerStep(BaseModel):
    """
    Step schema aligned with frontend contract:
    { "step": { "description", "action", "bounded_rectangle": {x1,x2,y1,y2}, ... } }
    """

    description: str
    action: RogerAction
    bounded_rectangle: Optional[BoundedRectangle] = None

    # Optional fields for non-click steps
    start: Optional[Point] = None
    end: Optional[Point] = None
    keys: Optional[str] = None
    ms: Optional[int] = Field(default=None, ge=0)


class RogerStepContainer(BaseModel):
    step: RogerStep


class RogerAskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User instruction from frontend.")
    t: Optional[float] = Field(
        default=None,
        description="Optional client timestamp (seconds) or timeline position, if applicable.",
    )
    cursor: Point
    screen: ScreenInfo
    screenshot: ScreenshotPayload
    app_map: Optional[AppMapPayload] = None


class RogerAskResponse(BaseModel):
    request_id: str
    created_at: datetime
    screenshot_sha256: str
    steps: List[RogerStepContainer]
    notes: List[str] = Field(default_factory=list)


def _b64decode_strict(data_base64: str) -> bytes:
    try:
        # validate=True rejects non-base64 alphabet characters
        return base64.b64decode(data_base64, validate=True)
    except Exception as exc:  # noqa: BLE001 - we want a consistent 400 response
        raise HTTPException(status_code=400, detail="Invalid screenshot base64 encoding") from exc


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _rect_around_cursor(cursor: Point, screen: ScreenInfo, radius: int = 12) -> Rect:
    x0 = _clamp(cursor.x - radius, 0, max(0, screen.width - 1))
    y0 = _clamp(cursor.y - radius, 0, max(0, screen.height - 1))
    x1 = _clamp(cursor.x + radius, 0, screen.width)
    y1 = _clamp(cursor.y + radius, 0, screen.height)
    return Rect(x=x0, y=y0, width=max(1, x1 - x0), height=max(1, y1 - y0))


def _to_bounded_rectangle(rect: Rect) -> BoundedRectangle:
    return BoundedRectangle(
        x1=rect.x,
        y1=rect.y,
        x2=rect.x + rect.width,
        y2=rect.y + rect.height,
    )


def _bounded_from_points(a: Point, b: Point) -> BoundedRectangle:
    return BoundedRectangle(
        x1=min(a.x, b.x),
        y1=min(a.y, b.y),
        x2=max(a.x, b.x),
        y2=max(a.y, b.y),
    )


def _iter_app_map_targets(app_map_json: Dict[str, Any]) -> Sequence[Tuple[str, Optional[Rect]]]:
    """
    Extract (label, rect) pairs from a loose "app map" structure.

    Supported shapes (best effort):
    - {"actions": [{"name": "...", "rect": {...}}, ...]}
    - {"menus": [{"label": "...", "rect": {...}}, ...]}
    - {"elements": [{"label": "...", "bounds": {...}}, ...]}
    """

    candidates: List[Tuple[str, Optional[Rect]]] = []

    def rect_from(obj: Any) -> Optional[Rect]:
        if not isinstance(obj, dict):
            return None
        raw = obj.get("rect") or obj.get("bounds") or obj.get("box")
        if not isinstance(raw, dict):
            return None
        try:
            # allow x/y/w/h or x/y/width/height
            x = raw.get("x")
            y = raw.get("y")
            width = raw.get("width", raw.get("w"))
            height = raw.get("height", raw.get("h"))
            if not all(isinstance(v, int) for v in [x, y, width, height]):
                return None
            return Rect(x=x, y=y, width=width, height=height)
        except Exception:
            return None

    for key in ("actions", "menus", "elements"):
        items = app_map_json.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            label = item.get("name") or item.get("label") or item.get("title")
            if not isinstance(label, str) or not label.strip():
                continue
            candidates.append((label.strip(), rect_from(item)))

    return candidates


def _find_shortcut(app_map_json: Dict[str, Any], prompt: str) -> Optional[Tuple[str, str]]:
    """
    Find a shortcut from a loose "shortcuts" map, using substring matching.
    Example shape: {"shortcuts": {"save": "Cmd+S", "undo": "Cmd+Z"}}
    Returns (shortcut_name, keys)
    """

    shortcuts = app_map_json.get("shortcuts")
    if not isinstance(shortcuts, dict):
        return None
    prompt_l = prompt.lower()
    for name, keys in shortcuts.items():
        if isinstance(name, str) and isinstance(keys, str) and name.lower() in prompt_l:
            return (name, keys)
    return None


def plan_roger_steps(
    *,
    prompt: str,
    cursor: Point,
    screen: ScreenInfo,
    app_map_json: Optional[Dict[str, Any]],
) -> Tuple[List[RogerStep], List[str]]:
    """
    Very small planner stub.

    Current behavior:
    - If prompt matches a known shortcut in app_map_json["shortcuts"], return a hotkey step.
    - If prompt mentions a known UI label in the app map and has coordinates, return a click step.
    - If prompt includes 'drag', return a short drag gesture starting at the cursor.
    - Otherwise, return a click centered around the current cursor location.
    """

    notes: List[str] = []
    prompt_l = prompt.lower()

    if app_map_json:
        shortcut = _find_shortcut(app_map_json, prompt)
        if shortcut:
            name, keys = shortcut
            return (
                [
                    RogerStep(
                        description=f"Trigger shortcut: {name}",
                        action="hotkey",
                        keys=keys,
                    )
                ],
                notes,
            )

        for label, rect in _iter_app_map_targets(app_map_json):
            if label.lower() in prompt_l and rect is not None:
                return (
                    [
                        RogerStep(
                            description=f"Click UI element: {label}",
                            action="click",
                            bounded_rectangle=_to_bounded_rectangle(rect),
                        )
                    ],
                    notes,
                )

        notes.append("No matching app-map target found; falling back to cursor-based action.")
    else:
        notes.append("No app_map provided; falling back to cursor-based action.")

    if "drag" in prompt_l or "slide" in prompt_l:
        end = Point(
            x=_clamp(cursor.x + 120, 0, max(0, screen.width - 1)),
            y=_clamp(cursor.y, 0, max(0, screen.height - 1)),
        )
        return (
            [
                RogerStep(
                    description="Drag gesture (stub)",
                    action="drag",
                    start=cursor,
                    end=end,
                    bounded_rectangle=_bounded_from_points(cursor, end),
                )
            ],
            notes,
        )

    return (
        [
            RogerStep(
                description="Click near cursor (stub)",
                action="click",
                bounded_rectangle=_to_bounded_rectangle(_rect_around_cursor(cursor, screen)),
            )
        ],
        notes,
    )


class RogerPlan(BaseModel):
    steps: List[RogerStepContainer]
    notes: List[str] = Field(default_factory=list)


def _openai_roger_json_schema() -> Dict[str, Any]:
    """
    JSON Schema used with OpenAI Structured Outputs.

    Notes:
    - Root schema must be an object.
    - All fields must be required; we emulate optional fields via anyOf with null.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "step": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "description": {"type": "string"},
                                "action": {
                                    "type": "string",
                                    "enum": ["click", "drag", "hotkey", "wait", "type"],
                                },
                                "bounded_rectangle": {
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "x1": {"type": "integer", "minimum": 0},
                                                "y1": {"type": "integer", "minimum": 0},
                                                "x2": {"type": "integer", "minimum": 0},
                                                "y2": {"type": "integer", "minimum": 0},
                                            },
                                            "required": ["x1", "y1", "x2", "y2"],
                                        },
                                        {"type": "null"},
                                    ]
                                },
                                "start": {
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "x": {"type": "integer", "minimum": 0},
                                                "y": {"type": "integer", "minimum": 0},
                                            },
                                            "required": ["x", "y"],
                                        },
                                        {"type": "null"},
                                    ]
                                },
                                "end": {
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "x": {"type": "integer", "minimum": 0},
                                                "y": {"type": "integer", "minimum": 0},
                                            },
                                            "required": ["x", "y"],
                                        },
                                        {"type": "null"},
                                    ]
                                },
                                "keys": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "ms": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
                                "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            },
                            "required": [
                                "description",
                                "action",
                                "bounded_rectangle",
                                "start",
                                "end",
                                "keys",
                                "ms",
                                "text",
                            ],
                        }
                    },
                    "required": ["step"],
                },
            },
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["steps", "notes"],
    }


def _openai_extract_output_text(response_json: Dict[str, Any]) -> str:
    """
    Extract text from a Responses API JSON payload (REST).
    """

    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = response_json.get("output")
    if not isinstance(output, list):
        raise ValueError("OpenAI response missing output array")

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                refusal = part.get("refusal")
                raise RuntimeError(f"Model refusal: {refusal}")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return part["text"]

    raise ValueError("OpenAI response missing output_text")


def call_openai_for_roger_plan(
    *,
    prompt: str,
    cursor: Point,
    screen: ScreenInfo,
    t: Optional[float],
    screenshot_media_type: str,
    screenshot_base64: str,
    app_map_json: Optional[Dict[str, Any]],
    app_map_text: Optional[str],
) -> RogerPlan:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    app_map_blob: str
    if app_map_json is not None:
        app_map_blob = json.dumps(app_map_json, ensure_ascii=False)
    elif app_map_text:
        app_map_blob = app_map_text
    else:
        app_map_blob = ""

    instructions = (
        "You are an agentic cursor planner.\n"
        "Return ONLY valid JSON that matches the provided JSON Schema.\n"
        "You are given: a full-screen screenshot (input_image), cursor coordinates, screen resolution, "
        "an optional timeline marker t, and an app map.\n"
        "Task: produce a list of UI automation steps to accomplish the user's prompt.\n"
        "Rules:\n"
        "- Use action='click' with bounded_rectangle {x1,y1,x2,y2} when clicking.\n"
        "- Use action='drag' with start/end points when dragging.\n"
        "- Use action='hotkey' with keys when keyboard shortcut.\n"
        "- Use action='wait' with ms for delays.\n"
        "- Use action='type' with text and optionally bounded_rectangle for target input field.\n"
        "- Prefer app_map coordinates when available; otherwise infer from screenshot.\n"
        "- Keep bounded_rectangle within the given screen resolution.\n"
        "- Keep descriptions one line.\n"
    )

    user_context = {
        "prompt": prompt,
        "cursor": cursor.model_dump(),
        "screen": screen.model_dump(),
        "t": t,
        "app_map": app_map_blob,
    }

    payload: Dict[str, Any] = {
        "model": ROGER_MODEL,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(user_context, ensure_ascii=False)},
                    {
                        "type": "input_image",
                        "image_url": f"data:{screenshot_media_type};base64,{screenshot_base64}",
                        "detail": "high",
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "roger_plan",
                "strict": True,
                "schema": _openai_roger_json_schema(),
            }
        },
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - intentional HTTPS call
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"OpenAI HTTPError {exc.code}: {body}") from exc
    except Exception as exc:  # noqa: BLE001 - surface as 502
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    response_json = json.loads(raw)
    out_text = _openai_extract_output_text(response_json)
    plan_json = json.loads(out_text)
    return RogerPlan.model_validate(plan_json)


@app.post("/roger/ask", response_model=RogerAskResponse)
async def roger_ask(body: RogerAskRequest) -> RogerAskResponse:
    screenshot_bytes = _b64decode_strict(body.screenshot.data_base64)

    # Best-effort parse app_map if provided as text JSON.
    app_map_json: Optional[Dict[str, Any]] = None
    if body.app_map:
        if body.app_map.map_json is not None:
            app_map_json = body.app_map.map_json
        elif body.app_map.text:
            try:
                parsed = json.loads(body.app_map.text)
                if isinstance(parsed, dict):
                    app_map_json = parsed
            except json.JSONDecodeError:
                # Keep non-JSON text as unstructured notes for now.
                pass

    plan_steps: List[RogerStepContainer]
    plan_notes: List[str]

    if ROGER_PROVIDER.lower() == "openai":
        try:
            plan = call_openai_for_roger_plan(
                prompt=body.prompt,
                cursor=body.cursor,
                screen=body.screen,
                t=body.t,
                screenshot_media_type=body.screenshot.media_type,
                screenshot_base64=body.screenshot.data_base64,
                app_map_json=app_map_json,
                app_map_text=body.app_map.text if body.app_map else None,
            )
            plan_steps = plan.steps
            plan_notes = plan.notes
        except Exception as exc:  # noqa: BLE001 - fall back to stub if model fails
            stub_steps, stub_notes = plan_roger_steps(
                prompt=body.prompt,
                cursor=body.cursor,
                screen=body.screen,
                app_map_json=app_map_json,
            )
            plan_steps = [RogerStepContainer(step=s) for s in stub_steps]
            plan_notes = [f"LLM planner failed; using stub. Error: {exc}", *stub_notes]
    else:
        stub_steps, stub_notes = plan_roger_steps(
            prompt=body.prompt,
            cursor=body.cursor,
            screen=body.screen,
            app_map_json=app_map_json,
        )
        plan_steps = [RogerStepContainer(step=s) for s in stub_steps]
        plan_notes = stub_notes

    return RogerAskResponse(
        request_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        screenshot_sha256=_sha256_hex(screenshot_bytes),
        steps=plan_steps,
        notes=plan_notes,
    )
