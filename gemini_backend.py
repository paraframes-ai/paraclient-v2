#!/usr/bin/env python3
"""
gemini_backend.py — thin adapter to Google's Gemini via Vertex AI, for the
gateway's /v1/gemini route.

This is the FIRST external-provider backend (the "add other models" option).
It runs on Google Cloud Vertex AI, authenticated by this VM's attached service
account (Application Default Credentials via the metadata server) — no API key
in the repo. Because calls go OFF-BOX to Google, the /v1/gemini route is gated
to consumer/internal audiences only; child/edu traffic must never reach it
(FERPA/COPPA), and Gemini has none of ParaClient's Socratic-withholding
behavior, so it is a direct assistant, not a tutor.
"""
from __future__ import annotations

import os

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "paraframes-app")

_client = None


def _get_client():
    """Lazily build the Vertex client (auth via the VM service account / ADC)."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(vertexai=True, project=VERTEX_PROJECT,
                               location=VERTEX_LOCATION)
    return _client


def to_gemini(messages: list):
    """OpenAI-style messages -> (system_instruction, contents). Gemini uses
    roles 'user'/'model' and carries the system prompt separately."""
    from google.genai import types
    system = None
    contents = []
    for m in messages:
        role = m.get("role")
        text = m.get("content", "") or ""
        if role == "system":
            system = f"{system}\n{text}" if system else text
            continue
        g_role = "model" if role == "assistant" else "user"
        contents.append(types.Content(role=g_role, parts=[types.Part(text=text)]))
    return system, contents


def _split_data_uri(data_uri: str):
    """'data:image/png;base64,AAA' -> ('image/png', raw_bytes)."""
    import base64
    header, _, payload = data_uri.partition(",")
    mime = header[5:].split(";")[0] or "image/png"   # strip the leading 'data:'
    return mime, base64.b64decode(payload)


def generate_vision(data_uri: str, prompt: str, system: str | None = None,
                    model: str | None = None, max_tokens: int = 1500,
                    temperature: float = 0.0):
    """Single-image + text call (used by /v1/notes transcription).

    Kept separate from generate() because the vision path takes one validated
    image and a prompt, not a chat history — folding it into the message
    converter would mean inventing an image-carrying message shape that nothing
    else uses."""
    from google.genai import types
    client = _get_client()
    mime, raw = _split_data_uri(data_uri)
    parts = [types.Part.from_bytes(data=raw, mime_type=mime),
             types.Part(text=prompt)]
    cfg = types.GenerateContentConfig(
        max_output_tokens=int(max_tokens),
        temperature=float(temperature),
        system_instruction=system,
    )
    used = model or DEFAULT_MODEL
    resp = client.models.generate_content(
        model=used, contents=[types.Content(role="user", parts=parts)], config=cfg)
    return (resp.text or ""), used


def generate(messages: list, model: str | None = None,
             max_tokens: int = 800, temperature: float = 0.7):
    """Call Gemini on Vertex. Returns (text, model_used). Raises on failure so
    the route can surface a clear error (e.g. API not enabled / no permission)."""
    from google.genai import types
    client = _get_client()
    system, contents = to_gemini(messages)
    if not contents:
        raise ValueError("no user/assistant messages to send")
    cfg = types.GenerateContentConfig(
        max_output_tokens=int(max_tokens),
        temperature=float(temperature),
        system_instruction=system,
    )
    used = model or DEFAULT_MODEL
    resp = client.models.generate_content(model=used, contents=contents, config=cfg)
    return (resp.text or ""), used
