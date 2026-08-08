#!/usr/bin/env python3
"""
claude_backend.py — adapter to Anthropic Claude via Google Vertex AI, for the
gateway's /v1/claude-free and /v1/claude-paid routes.

Uses the official Anthropic SDK's Vertex client (AnthropicVertex), authenticated
by this VM's GCP Application Default Credentials (the SAME ADC the Gemini route
needs — set it up once and both providers work). No API key in the repo.

Off-box like Gemini, so both routes are gated to consumer/internal (never edu)
at the gateway, and claude-paid additionally requires the paid/dev tier. Sonnet
backs the free route, Opus the paid route. NB: Opus 5 / Sonnet 5 reject
temperature/top_p/top_k with a 400 — this backend never sends sampling params.
"""
from __future__ import annotations

import os

# Vertex uses the BARE first-party model IDs (no provider prefix).
FREE_MODEL = os.environ.get("CLAUDE_FREE_MODEL", "claude-sonnet-5")
PAID_MODEL = os.environ.get("CLAUDE_PAID_MODEL", "claude-opus-5")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "paraframes-app")
# "global" is the recommended Vertex region for Claude; override per deployment.
CLAUDE_VERTEX_REGION = os.environ.get("CLAUDE_VERTEX_REGION", "global")

_client = None


def _get_client():
    """Lazily build the Vertex client (auth via GCP ADC / the VM service acct)."""
    global _client
    if _client is None:
        from anthropic import AnthropicVertex
        _client = AnthropicVertex(project_id=VERTEX_PROJECT,
                                  region=CLAUDE_VERTEX_REGION)
    return _client


def split_system(messages: list):
    """OpenAI-style messages -> (system_str_or_None, [user/assistant turns]).
    Claude carries the system prompt as a top-level arg, not a message."""
    system_parts, convo = [], []
    for m in messages:
        role = m.get("role")
        text = m.get("content", "") or ""
        if role == "system":
            if text:
                system_parts.append(text)
        else:
            convo.append({"role": "assistant" if role == "assistant" else "user",
                          "content": text})
    return ("\n".join(system_parts) or None), convo


def generate_vision(data_uri: str, prompt: str, system: str | None = None,
                    model: str | None = None, max_tokens: int = 1500):
    """Single-image + text call (used by /v1/notes transcription).

    Claude takes the image as a base64 content block alongside the text, so the
    data: URI is split back into media_type + payload. No sampling params, for
    the same reason as generate()."""
    client = _get_client()
    header, _, payload = data_uri.partition(",")
    media_type = header[5:].split(";")[0] or "image/png"
    content = [
        {"type": "image", "source": {"type": "base64",
                                     "media_type": media_type, "data": payload}},
        {"type": "text", "text": prompt},
    ]
    kwargs = {"model": model or FREE_MODEL, "max_tokens": int(max_tokens),
              "messages": [{"role": "user", "content": content}]}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return text, kwargs["model"]


def generate(messages: list, model: str, max_tokens: int = 2048):
    """Call Claude on Vertex. Returns (text, model_used). Raises on failure so
    the route can surface a clear error (e.g. Vertex auth / API not enabled).
    No sampling params are sent (rejected by Opus 5 / Sonnet 5); thinking is
    left at the model default (adaptive)."""
    client = _get_client()
    system, convo = split_system(messages)
    if not convo:
        raise ValueError("no user/assistant messages to send")
    kwargs = {"model": model, "max_tokens": int(max_tokens), "messages": convo}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return text, model
