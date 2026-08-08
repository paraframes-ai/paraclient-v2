#!/usr/bin/env python3
"""
notes_schema.py — system prompt + image handling for ParaClient's handwritten
note / math transcription capability (served at /v1/notes).

A student photographs handwritten notes (prose + math) and gets back clean,
editable text — math as LaTeX. This is a VISION task: it routes to the Gemma 4V
base VLM (ParaClient-v4), NOT a text adapter and NOT a CPU HTR model (CPU-only
recognizers can't handle math + messy handwriting). Like the other capabilities
this module is shared between the server and any future dataset builder so the
model is served exactly the prompt it will be trained on.

Design rule baked into the output contract: transcription is NEVER trusted
blindly for a student. The route always returns needs_confirmation=True and the
prompt is instructed to mark anything illegible with [?] rather than guess.
"""
from __future__ import annotations

import base64
import re

NOTES_SYS = (
    "You are a careful transcription engine for a student's handwritten notes. "
    "Transcribe EXACTLY what is written — do not solve, correct, summarize, or "
    "add anything. Rules:\n"
    "- Output clean Markdown that preserves the layout: headings, bullet/numbered "
    "lists, and line breaks as written.\n"
    "- Render every mathematical expression as LaTeX: inline as $...$ and "
    "display/standalone equations as $$...$$. Transcribe math symbols faithfully "
    "(fractions, exponents, roots, integrals, subscripts, Greek letters).\n"
    "- If a word, symbol, or digit is unreadable, write [?] in its place instead "
    "of guessing. If a whole region is unreadable, write [illegible].\n"
    "- Do not wrap the output in code fences and do not add commentary. Output "
    "only the transcription."
)

# Accepted raster formats, by magic bytes -> MIME.
_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]
MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MB decoded — a phone photo, not a scan dump

_DATA_URI = re.compile(r"^data:(image/[a-zA-Z.+-]+);base64,(.*)$", re.DOTALL)


def _sniff_mime(raw: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if raw.startswith(magic):
            return mime
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def normalize_image(image: str) -> str:
    """Accept a data: URI OR bare base64 and return a validated data: URI.

    Validates that the bytes are a real raster image and enforces a size cap.
    Raises ValueError (-> 400 at the route) on anything malformed or too big.
    """
    if not isinstance(image, str) or not image.strip():
        raise ValueError("image required (base64 or data: URI)")
    image = image.strip()

    m = _DATA_URI.match(image)
    b64 = m.group(2) if m else image
    b64 = re.sub(r"\s+", "", b64)
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"image is not valid base64: {exc}") from exc

    if not raw:
        raise ValueError("image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image too large ({len(raw)} bytes, max {MAX_IMAGE_BYTES})")

    mime = _sniff_mime(raw)
    if mime is None:
        raise ValueError("unsupported image (expected png, jpeg, gif, or webp)")
    # Re-encode from the decoded bytes so the served data: URI is always clean.
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def notes_prompt(hint: str = "") -> str:
    """The user-turn text for a transcription call.

    Shared by every backend so the local VLM, Gemini and Claude are all asked
    for exactly the same thing — otherwise the three would drift apart and a
    transcription would depend on which one served it."""
    return ("Transcribe this handwritten page. " + (hint or "")).strip()


def notes_messages(data_uri: str, hint: str = "") -> list:
    """OpenAI-format multimodal messages for the VLM transcription call."""
    text = notes_prompt(hint)
    return [
        {"role": "system", "content": NOTES_SYS},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]},
    ]


_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.DOTALL)


def clean_transcription(text: str) -> str:
    """Strip an accidental outer code fence if the model wrapped its output."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    m = _FENCE.match(t)
    return m.group(1).strip() if m else t
