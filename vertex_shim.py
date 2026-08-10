"""vertex_shim.py — a minimal OpenAI-shaped client over the Vertex backends.

generate_media's doc/slide generator speaks the OpenAI
`client.chat.completions.create(...).choices[0].message.content` interface. The
Vertex backends (gemini_backend, claude_backend) speak their own SDKs. This wraps
them in just enough of the OpenAI surface that the agentic loop drives Gemini or
Claude-on-Vertex with ZERO changes to generate_media.

Only the one method generate_media actually calls is implemented, on purpose —
this is a shim, not an OpenAI client.
"""
from __future__ import annotations

from gemini_backend import generate as _gemini_generate
from claude_backend import generate as _claude_generate


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, fn):
        self._fn = fn

    def create(self, model, messages, max_tokens=800, temperature=0.7, **_ignore):
        return _Response(self._fn(list(messages), model, int(max_tokens), float(temperature)))


class _Chat:
    def __init__(self, fn):
        self.completions = _Completions(fn)


class ShimClient:
    """Quacks like an OpenAI client for the one call generate_media makes."""
    def __init__(self, fn):
        self.chat = _Chat(fn)


def _gemini_fn(messages, model, max_tokens, temperature):
    text, _ = _gemini_generate(messages, model=model, max_tokens=max_tokens,
                               temperature=temperature)
    return text


def _claude_fn(messages, model, max_tokens, temperature):
    # Claude on Vertex (Opus/Sonnet 5) rejects sampling params, so temperature
    # is intentionally dropped — see claude_backend.
    text, _ = _claude_generate(messages, model=model, max_tokens=max_tokens)
    return text


def client_for(backend: str) -> ShimClient:
    """OpenAI-shaped client backed by the requested Vertex provider."""
    return ShimClient(_claude_fn if backend == "claude" else _gemini_fn)
