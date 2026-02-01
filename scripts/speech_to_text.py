#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_TRANSCRIBE_MODEL_ENV = "OPENAI_TRANSCRIBE_MODEL"


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _multipart_form_data(
    *,
    fields: Dict[str, str],
    file_field_name: str,
    file_path: Path,
    file_bytes: bytes,
    file_mime_type: str,
) -> Tuple[bytes, str]:
    """
    Build multipart/form-data body.
    Returns (body_bytes, content_type_header_value).
    """

    boundary = f"----roger-boundary-{uuid.uuid4().hex}"
    crlf = "\r\n"

    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}{crlf}".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'.encode("utf-8"))
        parts.append(value.encode("utf-8"))
        parts.append(crlf.encode("utf-8"))

    filename = file_path.name
    parts.append(f"--{boundary}{crlf}".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="{file_field_name}"; filename="{filename}"{crlf}'
            f"Content-Type: {file_mime_type}{crlf}{crlf}"
        ).encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(crlf.encode("utf-8"))

    parts.append(f"--{boundary}--{crlf}".encode("utf-8"))

    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def transcribe(
    *,
    api_key: str,
    audio_path: Path,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: Optional[float],
    output_format: str,
) -> Dict[str, Any]:
    """
    Call OpenAI Audio Transcriptions API.
    """

    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    audio_bytes = audio_path.read_bytes()
    audio_mime = _guess_mime_type(audio_path)

    fields: Dict[str, str] = {
        "model": model,
        "response_format": output_format,
    }
    if language:
        fields["language"] = language
    if prompt:
        fields["prompt"] = prompt
    if temperature is not None:
        fields["temperature"] = str(temperature)

    body, content_type = _multipart_form_data(
        fields=fields,
        file_field_name="file",
        file_path=audio_path,
        file_bytes=audio_bytes,
        file_mime_type=audio_mime,
    )

    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        method="POST",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - intentional HTTPS call
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8") if exc.fp else ""
        # Try to surface actionable errors (e.g. insufficient_quota vs rate_limit).
        try:
            parsed = json.loads(err_body) if err_body else {}
        except Exception:
            parsed = {}

        err = parsed.get("error") if isinstance(parsed, dict) else None
        code = err.get("code") if isinstance(err, dict) else None
        msg = err.get("message") if isinstance(err, dict) else None

        if exc.code == 429 and code == "insufficient_quota":
            raise RuntimeError(
                "OpenAI API quota/billing error (insufficient_quota).\n"
                "Fix:\n"
                "  1) Verify this is an OpenAI *Platform* API key (not ChatGPT Plus).\n"
                "  2) Enable billing / add credits in your OpenAI API project.\n"
                "  3) Re-run with the updated OPENAI_API_KEY.\n"
                f"Server message: {msg}\n"
            ) from exc

        raise RuntimeError(f"OpenAI HTTPError {exc.code}: {err_body}") from exc

    # response_format=text returns plain text, others return JSON
    if output_format == "text":
        return {"text": raw}
    return json.loads(raw)


def record_from_mic(
    *,
    duration_s: float,
    sample_rate: int,
    channels: int,
    device: Optional[str],
) -> Path:
    """
    Record audio from the default microphone into a temporary WAV file.

    Dependencies (install locally):
      pip install sounddevice soundfile

    On macOS you may need to grant Terminal/IDE microphone permissions.
    """

    try:
        import sounddevice as sd  # type: ignore[import-not-found]
        import soundfile as sf  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing mic recording dependencies.\n"
            "Install:\n"
            "  pip install sounddevice soundfile numpy\n"
            "Then re-run with --mic.\n"
        ) from exc

    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if channels <= 0:
        raise ValueError("channels must be > 0")

    # Record
    frames = int(duration_s * sample_rate)
    if frames <= 0:
        raise ValueError("duration too small")

    # Allow selecting device by name/index (string); sounddevice accepts int or str.
    sd.default.samplerate = sample_rate
    if device is not None:
        sd.default.device = device

    print(f"Recording {duration_s:.2f}s from mic…", file=sys.stderr)
    time.sleep(0.1)
    audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32")
    sd.wait()

    # Simple level meter (helps debug cases where the OS is "recording" silence).
    try:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        print(f"Mic levels: rms={rms:.4f}, peak={peak:.4f}", file=sys.stderr)
        if peak < 0.02:
            print(
                "Warning: audio peak is very low. Check mic permissions, input volume, and distance to mic.",
                file=sys.stderr,
            )
    except Exception:
        pass

    tmp = tempfile.NamedTemporaryFile(prefix="roger-mic-", suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    sf.write(str(tmp_path), audio, sample_rate, subtype="PCM_16")
    return tmp_path


def list_input_devices() -> str:
    """
    Return a human-readable list of audio input devices.

    Requires:
      pip install sounddevice
    """

    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "sounddevice is required to list devices.\n"
            "Install:\n"
            "  pip install sounddevice\n"
        ) from exc

    lines: list[str] = []
    devices = sd.query_devices()
    default_in, _default_out = sd.default.device
    for idx, d in enumerate(devices):
        if not isinstance(d, dict):
            continue
        max_in = d.get("max_input_channels", 0)
        if not isinstance(max_in, int) or max_in <= 0:
            continue
        name = d.get("name", "")
        marker = " (default)" if idx == default_in else ""
        lines.append(f"{idx}: {name} (max_input_channels={max_in}){marker}")

    if not lines:
        return "No input devices found."
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multilingual speech-to-text using OpenAI API (Audio Transcriptions)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=str, help="Path to audio/video file (mp3, wav, m4a, mp4, etc.)")
    src.add_argument("--mic", action="store_true", help="Record from microphone instead of a file.")
    src.add_argument("--list-devices", action="store_true", help="List available input devices and exit.")
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Mic recording duration in seconds (used with --mic).",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=44100,
        help="Mic sample rate in Hz (used with --mic).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Mic channels (used with --mic).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional mic device selector (index or substring) (used with --mic).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv(OPENAI_TRANSCRIBE_MODEL_ENV, "whisper-1"),
        help=f"Transcription model (default: env {OPENAI_TRANSCRIBE_MODEL_ENV} or whisper-1)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional BCP-47-ish language hint (e.g. en, hi, fr). Omit for auto-detect.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt to improve recognition (names/terms).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional decoding temperature (e.g. 0.0).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="verbose_json",
        choices=["json", "text", "verbose_json", "srt", "vtt"],
        help="Output format returned by the API. Use verbose_json to get detected language.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--print-detected-language",
        action="store_true",
        help="Print detected language (requires --format verbose_json).",
    )
    parser.add_argument(
        "--print-text-only",
        action="store_true",
        help="Print only the transcript text (best for piping into /roger/ask).",
    )
    parser.add_argument(
        "--save-wav",
        type=str,
        default=None,
        help="If using --mic, save the recorded WAV to this path (for debugging).",
    )
    args = parser.parse_args()

    if args.list_devices:
        try:
            print(list_input_devices())
            return 0
        except Exception as exc:
            print(str(exc).strip(), file=sys.stderr)
            return 1

    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        print(
            f"Missing {OPENAI_API_KEY_ENV}. Set it in your environment or .env (Docker).\n"
            f"Example:\n"
            f"  export {OPENAI_API_KEY_ENV}='sk-...'\n",
            file=sys.stderr,
        )
        return 2

    temp_to_delete: Optional[Path] = None
    try:
        if args.mic:
            audio_path = record_from_mic(
                duration_s=args.duration,
                sample_rate=args.rate,
                channels=args.channels,
                device=args.device,
            )
            temp_to_delete = audio_path
        else:
            audio_path = Path(args.audio).expanduser().resolve()

        if args.save_wav and args.mic:
            save_path = Path(args.save_wav).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audio_path, save_path)
            print(f"Saved mic recording to: {save_path}", file=sys.stderr)

        result = transcribe(
            api_key=api_key,
            audio_path=audio_path,
            model=args.model,
            language=args.language,
            prompt=args.prompt,
            temperature=args.temperature,
            output_format=args.format,
        )
    except Exception as exc:
        print(str(exc).strip(), file=sys.stderr)
        return 1
    finally:
        if temp_to_delete is not None:
            try:
                temp_to_delete.unlink(missing_ok=True)
            except Exception:
                pass

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        if isinstance(result, dict) and args.format == "text" and "text" in result:
            out_path.write_text(str(result["text"]), encoding="utf-8")
        else:
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if args.print_detected_language:
        if args.format != "verbose_json":
            print("--print-detected-language requires --format verbose_json", file=sys.stderr)
            return 2
        detected = result.get("language") if isinstance(result, dict) else None
        print(str(detected))
        return 0

    if args.print_text_only:
        # In verbose_json/json, transcript is under "text"
        if isinstance(result, dict) and "text" in result:
            print(str(result["text"]))
            return 0
        if args.format == "text" and isinstance(result, dict) and "text" in result:
            print(result["text"])
            return 0
        print("No transcript text found in response.", file=sys.stderr)
        return 2

    if args.format == "text" and "text" in result:
        print(result["text"])
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

