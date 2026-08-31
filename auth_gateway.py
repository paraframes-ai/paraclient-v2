#!/usr/bin/env python3
"""
auth_gateway.py — ParaFrames API gateway (DEV SCAFFOLD).

The one stable contract the apps code against. It sits IN FRONT of vLLM and:
  1. authenticates the caller by a PER-USER key (not the shared vLLM key),
  2. rate-limits per key,
  3. enforces which MODES an audience may use,
  4. runs the content-safety filter (screen input + output),
  5. maps (mode, subject) -> system prompt + model, forwards to vLLM,
  6. exposes /v1/generate for documents/slideshows.

Clients NEVER see the vLLM key or hit vLLM directly. vLLM stays bound to
localhost; only this gateway reaches it.

  App ──(per-user key)──▶ auth_gateway (this) ──(server-held vLLM key)──▶ vLLM

⚠️ DEV SCAFFOLD, NOT PRODUCTION. The content filter is the heuristic backend
(content_filter.py says: do NOT ship to real minors). Per-user keys live in a
local JSON keystore with an in-memory rate limiter. Before prod: real moderation
model, a real key store / IdP, durable rate limiting, TLS, and COPPA/FERPA
review. Run on the tailnet only.

Run:
  export VLLM_API_KEY=...            # the key the gateway uses to reach vLLM
  python auth_gateway.py --tutor-url http://127.0.0.1:8000/v1 --port 8080
"""
import argparse
import base64
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content_filter import ContentFilter, Action  # noqa: E402
import generate_media as gm  # noqa: E402
from cad_schema import sys_for, clean_sketch, clean_solid  # noqa: E402
from circuit_schema import (CIRCUIT_SYS, erc as circuit_erc,  # noqa: E402
                            clean as circuit_clean)

KEYSTORE = Path(__file__).with_name("gateway_keys.json")

# --------------------------------------------------------------------------
# Audience -> which modes it may use. This is where the product rule lives:
#   EDU (child-facing) may ONLY tutor (socratic / graduated_hint).
#   CONSUMER (adult/supervised) also gets the direct "normal" assistant.
# --------------------------------------------------------------------------
AUDIENCE_MODES = {
    "edu":      frozenset({"socratic", "graduated_hint"}),
    "consumer": frozenset({"socratic", "graduated_hint", "normal"}),
    "internal": frozenset({"socratic", "graduated_hint", "normal"}),  # dev/TUI
}

AUDIENCE_ALLOWED_MODES_SORTED = {
    aud: sorted(modes) for aud, modes in AUDIENCE_MODES.items()
}
_EMPTY_SET = frozenset()
_EMPTY_LIST = []

SUBJECT_MODEL = {
    "math":          "ParaFrames/ParaClient-math-v2.2",
    "language_arts": "ParaFrames/ParaClient-language-v2.2",
    "general":       "ParaFrames/ParaClient-v2.2",
}


def _build_system_prompt(mode: str, subject: str) -> str:
    subj = {"math": "math", "language_arts": "language arts",
            "general": "study"}.get(subject, subject)
    if mode == "socratic":
        return (f"You are a Socratic {subj} tutor for a K-12 student. Never state "
                f"the final answer. Guide with one clear question at a time.")
    if mode == "graduated_hint":
        return (f"You are a {subj} homework tutor for a K-12 student. Guide with "
                f"questions first; if the student stays stuck, escalate support "
                f"step by step so they can finish. Never just dump the full answer.")
    # normal: direct assistant, tailored to the subject (consumer only)
    return (f"You are a helpful, accurate {subj} assistant. Answer the user's "
            f"question directly and clearly.")


def _build_model_for(mode: str, subject: str) -> str:
    # 'normal' uses the base (adapters are tuned to withhold; don't fight them)
    if mode == "normal":
        return "ParaFrames/ParaClient-v2.2"
    return SUBJECT_MODEL.get(subject, "ParaFrames/ParaClient-v2.2")


# Pre-compute system prompt and model mapping tables for all standard (mode, subject) pairs
_KNOWN_MODES = ("socratic", "graduated_hint", "normal")
_KNOWN_SUBJECTS = ("math", "language_arts", "general")

SYSTEM_PROMPT_TABLE = {
    (m, s): _build_system_prompt(m, s)
    for m in _KNOWN_MODES
    for s in _KNOWN_SUBJECTS
}

MODEL_FOR_TABLE = {
    (m, s): _build_model_for(m, s)
    for m in _KNOWN_MODES
    for s in _KNOWN_SUBJECTS
}


def system_prompt(mode: str, subject: str) -> str:
    val = SYSTEM_PROMPT_TABLE.get((mode, subject))
    if val is not None:
        return val
    return _build_system_prompt(mode, subject)


def model_for(mode: str, subject: str) -> str:
    val = MODEL_FOR_TABLE.get((mode, subject))
    if val is not None:
        return val
    return _build_model_for(mode, subject)


# The CAD routes (/v1/sketch = 2D, /v1/3d = 3D) run entirely on the L4 via a
# dedicated paraclient LoRA adapter trained for CAD geometry — NO cloud. The
# system prompts + validators live in cad_schema.py (shared with the dataset
# builder so the adapter is served exactly what it was trained on).
CAD_MODEL = os.environ.get("CAD_MODEL", "ParaFrames/ParaClient-cad-v2.2")
BASE_MODEL = "ParaFrames/ParaClient-v2.2"

# The circuit route runs on a SEPARATE CPU llama.cpp server (a Qwen2.5-3B model
# specialized for netlists), never the GPU. Schema/ERC live in circuit_schema.py.
CIRCUIT_URL = os.environ.get("CIRCUIT_URL", "http://127.0.0.1:8001/v1")

# Embedding model for per-user RAG (the Knowledge Library). Small + CPU so it
# never contends with vLLM's GPU reservation. 384-dim, cosine via normalized dot.
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


# --------------------------------------------------------------------------
# Keystore + rate limiter (scaffold-grade)
# --------------------------------------------------------------------------

def load_keys() -> dict:
    if KEYSTORE.exists():
        return json.loads(KEYSTORE.read_text())
    # seed two demo keys so the scaffold is testable out of the box
    seed = {
        "pk-consumer-demo": {"user": "demo-consumer", "audience": "consumer", "rpm": 30},
        "pk-edu-demo":      {"user": "demo-edu",      "audience": "edu",      "rpm": 60},
    }
    KEYSTORE.write_text(json.dumps(seed, indent=2))
    return seed


class RateLimiter:
    """In-memory sliding-window per key. Scaffold only — not durable."""
    def __init__(self):
        self._hits = defaultdict(deque)

    def allow(self, key: str, rpm: int, now: float) -> bool:
        dq = self._hits[key]
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= rpm:
            return False
        dq.append(now)
        return True


# --------------------------------------------------------------------------

def build_app(tutor_url: str, tutor_key: str):
    from fastapi import FastAPI, Request, Header, HTTPException
    from fastapi.responses import JSONResponse
    import httpx
    from openai import AsyncOpenAI

    app = FastAPI(title="ParaFrames API Gateway (dev scaffold)")
    tutor_transport = httpx.AsyncHTTPTransport(
        limits=httpx.Limits(
            max_connections=128,
            max_keepalive_connections=64,
            keepalive_expiry=120.0,
        ),
        retries=1,
    )
    tutor_client = httpx.AsyncClient(transport=tutor_transport, trust_env=False, timeout=httpx.Timeout(60.0, connect=5.0))
    tutor = AsyncOpenAI(base_url=tutor_url, api_key=tutor_key, http_client=tutor_client)
    filt = ContentFilter()
    keys = load_keys()
    rl = RateLimiter()

    print(f"[*] CAD routes (/v1/sketch, /v1/3d) -> LOCAL adapter {CAD_MODEL}")
    print("[*] docs/slides -> LOCAL agentic (paraclient + web search/fetch)")

    # RAG embedder (CPU). Loaded once; /v1/embed serves it to e2 for indexing +
    # retrieval over each user's Knowledge Library.
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        print(f"[*] /v1/embed -> {EMBED_MODEL_NAME} (CPU)")
    except Exception as e:  # noqa: BLE001
        print(f"[!] embedder load failed ({e}); /v1/embed disabled")

    async def gen_cad_json(mode: str, prompt: str, units: str) -> dict:
        """Generate CAD JSON on the local paraclient CAD adapter (mode is
        'sketch' for 2D or '3d' for 3D). Falls back to the base paraclient if
        the adapter isn't loaded yet; retries a few times on a parse failure."""
        sys_ins = sys_for(mode, units)
        msgs = [{"role": "system", "content": sys_ins},
                {"role": "user", "content": prompt}]
        last = None
        for model in (CAD_MODEL, BASE_MODEL):
            for _ in range(3):
                try:
                    r = await tutor.chat.completions.create(
                        model=model, messages=msgs,
                        max_tokens=3000, temperature=0.2)
                    return gm.parse_json(r.choices[0].message.content.strip())
                except Exception as e:  # noqa: BLE001
                    last = e
        raise RuntimeError(f"CAD ({mode}) generation failed: {last}")

    circuit_transport = httpx.AsyncHTTPTransport(
        limits=httpx.Limits(
            max_connections=128,
            max_keepalive_connections=64,
            keepalive_expiry=120.0,
        ),
        retries=1,
    )
    circuit_client = httpx.AsyncClient(transport=circuit_transport, trust_env=False, timeout=httpx.Timeout(60.0, connect=5.0))
    circuit = AsyncOpenAI(base_url=CIRCUIT_URL, api_key="none", http_client=circuit_client)
    print(f"[*] /v1/circuit -> CPU llama.cpp {CIRCUIT_URL}")

    async def gen_circuit(prompt: str) -> dict:
        """Netlist on the CPU circuit model with ERC-gated rejection sampling:
        try a few, return the first electrically-valid one (else the best-effort
        structurally-clean last result)."""
        msgs = [{"role": "system", "content": CIRCUIT_SYS},
                {"role": "user", "content": prompt}]
        last = None
        for _ in range(3):
            try:
                r = await circuit.chat.completions.create(
                    model="circuit", messages=msgs,
                    max_tokens=900, temperature=0.4)
                nl = circuit_clean(gm.parse_json(r.choices[0].message.content.strip()))
                last = nl
                if circuit_erc(nl)[0]:
                    return nl
            except Exception:  # noqa: BLE001
                pass
        if isinstance(last, dict):
            return last          # ERC-imperfect but structurally clean
        raise RuntimeError("no parseable netlist from circuit model")

    def lookup(authorization: str | None):
        """Validate the bearer key only (no rate-limit consumption)."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer key")
        tok = authorization.split(" ", 1)[1].strip()
        rec = keys.get(tok)
        if not rec:
            raise HTTPException(401, "invalid key")
        return tok, rec

    def auth(authorization: str | None):
        """Validate + rate-limit (for the model/generation routes)."""
        tok, rec = lookup(authorization)
        if not rl.allow(tok, rec.get("rpm", 30), time.time()):
            raise HTTPException(429, "rate limit exceeded")
        return rec

    @app.post("/v1/auth/validate")
    async def validate(authorization: str | None = Header(None)):
        """Handshake for an upstream login service (e.g. the e2 site): after it
        authenticates the user (email whitelist), it calls this with its service
        key to confirm success + learn the identity/permissions. The service key
        is held ONLY by that server — never sent to browsers."""
        _, rec = lookup(authorization)
        return {"ok": True, "user": rec["user"], "audience": rec["audience"],
                "allowed_modes": AUDIENCE_ALLOWED_MODES_SORTED.get(rec["audience"], _EMPTY_LIST)}

    @app.post("/v1/chat")
    async def chat(request: Request, authorization: str | None = Header(None)):
        rec = auth(authorization)
        body = await request.json()
        mode = body.get("mode", "socratic")
        subject = body.get("subject", "general")
        messages = body.get("messages", [])

        if mode not in AUDIENCE_MODES.get(rec["audience"], _EMPTY_SET):
            raise HTTPException(
                403, f"mode '{mode}' not allowed for audience '{rec['audience']}'")

        user_turns = [m for m in messages if m.get("role") == "user"]
        latest = user_turns[-1]["content"] if user_turns else ""

        din = filt.screen_input(latest)
        if din.action != Action.ALLOW:
            return JSONResponse({"role": "assistant", "content": din.student_message,
                                 "safety": {"action": din.action.value,
                                            "categories": din.categories}})

        model = model_for(mode, subject)
        sys_msg = {"role": "system", "content": system_prompt(mode, subject)}
        convo = [sys_msg] + [m for m in messages if m.get("role") != "system"]
        try:
            resp = await tutor.chat.completions.create(
                model=model, messages=convo,
                max_tokens=body.get("max_tokens", 400),
                temperature=body.get("temperature", 0.3))
            answer = resp.choices[0].message.content
        except Exception:
            raise HTTPException(502, "tutor backend unavailable")

        dout = filt.screen_output(answer)
        if dout.action != Action.ALLOW:
            answer = dout.student_message
        return JSONResponse({"role": "assistant", "content": answer,
                             "model": model,
                             "safety": {"action": dout.action.value}})

    @app.post("/v1/generate")
    async def generate(request: Request, authorization: str | None = Header(None)):
        rec = auth(authorization)
        # Docs/slides are a NON-SOCRATIC capability (direct generation), so they
        # follow the same audience rule as the 'normal' mode: consumer/internal
        # only, never EDU.
        if "normal" not in AUDIENCE_MODES.get(rec["audience"], _EMPTY_SET):
            raise HTTPException(
                403, f"document/slide generation (non-socratic) not allowed "
                     f"for audience '{rec['audience']}'")
        body = await request.json()
        kind = body.get("kind")
        if kind not in ("slides", "doc"):
            raise HTTPException(400, "kind must be 'slides' or 'doc'")
        prompt = body.get("prompt", "")
        # web=False -> single-pass generation (fast, grounded in provided source);
        # web=True (default) -> the agentic web-research generator.
        web = bool(body.get("web", True))
        din = filt.screen_input(prompt)
        if din.action != Action.ALLOW:
            return JSONResponse({"error": "blocked", "safety": din.categories}, 403)
        try:
            data = (await gm.generate_agentic(tutor, BASE_MODEL, kind, prompt) if web
                    else await gm.generate(tutor, BASE_MODEL, kind, prompt))
            name, theme = gm.pick_theme(data, body.get("theme"))
            ext = "pptx" if kind == "slides" else "docx"
            out = Path("out") / f"gen-{int(time.time()*1000)}.{ext}"
            (gm.render_pptx if kind == "slides" else gm.render_docx)(data, theme, out)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"generation failed: {e}")
        b = base64.b64encode(out.read_bytes()).decode()
        ctype = ("application/vnd.openxmlformats-officedocument."
                 + ("presentationml.presentation" if kind == "slides"
                    else "wordprocessingml.document"))
        return JSONResponse({"kind": kind, "theme": name, "filename": out.name,
                             "content_type": ctype, "file_base64": b})

    async def _cad_route(mode: str, rec: dict, body: dict):
        # Non-socratic capability -> consumer/internal only (same rule as generate)
        if "normal" not in AUDIENCE_MODES.get(rec["audience"], _EMPTY_SET):
            raise HTTPException(
                403, f"CAD generation (non-socratic) not allowed for "
                     f"audience '{rec['audience']}'")
        prompt = body.get("prompt", "")
        units = str(body.get("units", "mm"))
        if not prompt:
            raise HTTPException(400, "prompt required")
        din = filt.screen_input(prompt)
        if din.action != Action.ALLOW:
            return JSONResponse({"error": "blocked", "safety": din.categories}, 403)
        try:
            data = await gen_cad_json(mode, prompt, units)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"CAD ({mode}) generation failed: {e}")
        out = clean_sketch(data, units) if mode == "sketch" else clean_solid(data, units)
        return JSONResponse(out)

    @app.post("/v1/sketch")
    async def sketch(request: Request, authorization: str | None = Header(None)):
        rec = auth(authorization)
        return await _cad_route("sketch", rec, await request.json())

    @app.post("/v1/3d")
    async def three_d(request: Request, authorization: str | None = Header(None)):
        rec = auth(authorization)
        return await _cad_route("3d", rec, await request.json())

    @app.post("/v1/circuit")
    async def circuit_route(request: Request,
                            authorization: str | None = Header(None)):
        rec = auth(authorization)
        if "normal" not in AUDIENCE_MODES.get(rec["audience"], _EMPTY_SET):
            raise HTTPException(
                403, f"circuit generation (non-socratic) not allowed for "
                     f"audience '{rec['audience']}'")
        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(400, "prompt required")
        din = filt.screen_input(prompt)
        if din.action != Action.ALLOW:
            return JSONResponse({"error": "blocked", "safety": din.categories}, 403)
        try:
            data = await gen_circuit(prompt)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"circuit generation failed: {e}")
        return JSONResponse(data)

    @app.post("/v1/embed")
    async def embed(request: Request, authorization: str | None = Header(None)):
        """Batch text -> normalized embeddings, for the RAG Knowledge Library.
        Internal use (e2 indexing + retrieval); rate-limited by the caller's key."""
        auth(authorization)
        if embedder is None:
            raise HTTPException(503, "embedder unavailable")
        body = await request.json()
        texts = body.get("texts")
        if (not isinstance(texts, list) or not texts
                or not all(isinstance(t, str) for t in texts)):
            raise HTTPException(400, "texts must be a non-empty list of strings")
        if len(texts) > 256:
            raise HTTPException(400, "too many texts (max 256 per call)")
        vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=32)
        return JSONResponse({"model": EMBED_MODEL_NAME, "dim": int(vecs.shape[1]),
                             "embeddings": vecs.tolist()})

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "audiences": list(AUDIENCE_MODES)}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutor-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    tutor_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    import uvicorn
    app = build_app(args.tutor_url, tutor_key)
    print(f"[*] ParaFrames gateway (DEV SCAFFOLD) on {args.host}:{args.port}")
    print(f"    -> tutor {args.tutor_url}  (vLLM key held server-side)")
    print(f"    keystore: {KEYSTORE}   safety: content_filter (heuristic)")
    print(f"    audiences/modes: {AUDIENCE_MODES}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
