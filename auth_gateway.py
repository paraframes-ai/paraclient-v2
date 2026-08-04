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
from cad_schema import (sys_for, clean_sketch, clean_solid,  # noqa: E402
                        repair_sketch, floorplan_issues)
from circuit_schema import (CIRCUIT_SYS, erc as circuit_erc,  # noqa: E402
                            clean as circuit_clean)
from notes_schema import (normalize_image, notes_messages,  # noqa: E402
                          clean_transcription)
from civics_schema import sys_for as civics_sys, is_time_varying  # noqa: E402
from civics_agent import answer_time_varying  # noqa: E402

KEYSTORE = Path(__file__).with_name("gateway_keys.json")

# --------------------------------------------------------------------------
# Audience -> which modes it may use. This is where the product rule lives:
#   EDU (child-facing) may ONLY tutor (socratic / graduated_hint).
#   CONSUMER (adult/supervised) also gets the direct "normal" assistant.
# --------------------------------------------------------------------------
AUDIENCE_MODES = {
    "edu":      {"socratic", "graduated_hint"},
    "consumer": {"socratic", "graduated_hint", "normal"},
    "internal": {"socratic", "graduated_hint", "normal"},  # dev/TUI
}

SUBJECT_MODEL = {
    "math":          "ParaFrames/ParaClient-math-v2.2",
    "language_arts": "ParaFrames/ParaClient-language-v2.2",
    "general":       "ParaFrames/ParaClient-v2.2",
}


def system_prompt(mode: str, subject: str) -> str:
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


def model_for(mode: str, subject: str) -> str:
    # 'normal' uses the base (adapters are tuned to withhold; don't fight them)
    if mode == "normal":
        return "ParaFrames/ParaClient-v2.2"
    return SUBJECT_MODEL.get(subject, "ParaFrames/ParaClient-v2.2")


# The CAD routes (/v1/sketch = 2D, /v1/3d = 3D) run entirely on the L4 via a
# dedicated paraclient LoRA adapter trained for CAD geometry — NO cloud. The
# system prompts + validators live in cad_schema.py (shared with the dataset
# builder so the adapter is served exactly what it was trained on).
CAD_MODEL = os.environ.get("CAD_MODEL", "ParaFrames/ParaClient-cad-v2.2")
BASE_MODEL = "ParaFrames/ParaClient-v2.2"

# /v1/notes (handwriting + math transcription) is a VISION call — it needs the
# multimodal base VLM, which is the ParaClient-v4 (Gemma 4V) base served WITHOUT
# a subject adapter. The served-model name is the same base string; it only
# becomes vision-capable after the v4 cutover, so until then the route returns a
# clear 503 instead of a wrong answer.
NOTES_MODEL = os.environ.get("NOTES_MODEL", BASE_MODEL)

# Civics has TWO answer paths (see /v1/civics): static civic knowledge is served
# by the dedicated civics adapter; time-varying facts (current office-holders,
# a user's own representatives) are NOT memorized -- they route to civics_agent,
# which runs the ReAct loop against official .gov/.mil sources. The agent loop
# runs on the general base model (BASE_MODEL); only the static path needs the
# civics adapter, which arrives with the ParaClient-v4 cutover.
CIVICS_MODEL = os.environ.get("CIVICS_MODEL", "ParaFrames/ParaClient-civics-v2.2")

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
    from openai import OpenAI

    app = FastAPI(title="ParaFrames API Gateway (dev scaffold)")
    tutor = OpenAI(base_url=tutor_url, api_key=tutor_key)
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

    def gen_cad_json(mode: str, prompt: str, units: str) -> dict:
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
                    r = tutor.chat.completions.create(
                        model=model, messages=msgs,
                        max_tokens=3000, temperature=0.2)
                    return gm.parse_json(r.choices[0].message.content.strip())
                except Exception as e:  # noqa: BLE001
                    last = e
        raise RuntimeError(f"CAD ({mode}) generation failed: {last}")

    circuit = OpenAI(base_url=CIRCUIT_URL, api_key="none")
    print(f"[*] /v1/circuit -> CPU llama.cpp {CIRCUIT_URL}")

    def gen_circuit(prompt: str) -> dict:
        """Netlist on the CPU circuit model with ERC-gated rejection sampling:
        try a few, return the first electrically-valid one (else the best-effort
        structurally-clean last result)."""
        msgs = [{"role": "system", "content": CIRCUIT_SYS},
                {"role": "user", "content": prompt}]
        last = None
        for _ in range(3):
            try:
                r = circuit.chat.completions.create(
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
                "allowed_modes": sorted(AUDIENCE_MODES.get(rec["audience"], []))}

    @app.post("/v1/chat")
    async def chat(request: Request, authorization: str | None = Header(None)):
        rec = auth(authorization)
        body = await request.json()
        mode = body.get("mode", "socratic")
        subject = body.get("subject", "general")
        messages = body.get("messages", [])

        if mode not in AUDIENCE_MODES.get(rec["audience"], set()):
            raise HTTPException(
                403, f"mode '{mode}' not allowed for audience '{rec['audience']}'")

        user_turns = [m for m in messages if m.get("role") == "user"]
        latest = user_turns[-1]["content"] if user_turns else ""

        din = filt.screen_input(latest)
        if din.action != Action.ALLOW:
            return JSONResponse({"role": "assistant", "content": din.student_message,
                                 "safety": {"action": din.action.value,
                                            "categories": din.categories}})

        sys_msg = {"role": "system", "content": system_prompt(mode, subject)}
        convo = [sys_msg] + [m for m in messages if m.get("role") != "system"]
        try:
            resp = tutor.chat.completions.create(
                model=model_for(mode, subject), messages=convo,
                max_tokens=body.get("max_tokens", 400),
                temperature=body.get("temperature", 0.3))
            answer = resp.choices[0].message.content
        except Exception:
            raise HTTPException(502, "tutor backend unavailable")

        dout = filt.screen_output(answer)
        if dout.action != Action.ALLOW:
            answer = dout.student_message
        return JSONResponse({"role": "assistant", "content": answer,
                             "model": model_for(mode, subject),
                             "safety": {"action": dout.action.value}})

    @app.post("/v1/generate")
    async def generate(request: Request, authorization: str | None = Header(None)):
        rec = auth(authorization)
        # Docs/slides are a NON-SOCRATIC capability (direct generation), so they
        # follow the same audience rule as the 'normal' mode: consumer/internal
        # only, never EDU.
        if "normal" not in AUDIENCE_MODES.get(rec["audience"], set()):
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
            data = (gm.generate_agentic(tutor, BASE_MODEL, kind, prompt) if web
                    else gm.generate(tutor, BASE_MODEL, kind, prompt))
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

    def _cad_route(mode: str, rec: dict, body: dict):
        # Non-socratic capability -> consumer/internal only (same rule as generate)
        if "normal" not in AUDIENCE_MODES.get(rec["audience"], set()):
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
            data = gen_cad_json(mode, prompt, units)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"CAD ({mode}) generation failed: {e}")
        if mode == "sketch":
            ok, issues = floorplan_issues(clean_sketch(data, units))
            out = repair_sketch(data, units)   # relayouts only if the plan is invalid
            if not ok:
                print(f"[cad] floor-plan auto-repaired (model geometry invalid: {issues})")
        else:
            out = clean_solid(data, units)
        return JSONResponse(out)

    @app.post("/v1/sketch")
    async def sketch(request: Request, authorization: str | None = Header(None)):
        rec = auth(authorization)
        return _cad_route("sketch", rec, await request.json())

    @app.post("/v1/3d")
    async def three_d(request: Request, authorization: str | None = Header(None)):
        rec = auth(authorization)
        return _cad_route("3d", rec, await request.json())

    @app.post("/v1/circuit")
    async def circuit_route(request: Request,
                            authorization: str | None = Header(None)):
        rec = auth(authorization)
        if "normal" not in AUDIENCE_MODES.get(rec["audience"], set()):
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
            data = gen_circuit(prompt)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"circuit generation failed: {e}")
        return JSONResponse(data)

    @app.post("/v1/civics")
    async def civics_route(request: Request,
                           authorization: str | None = Header(None)):
        """K-12 civics Q&A with a split answer path:
          * time-varying questions (current office-holders, a user's own reps) ->
            civics_agent: a live ReAct lookup constrained to official .gov/.mil
            sources, returned WITH a citation. Never answered from memory.
          * static civic knowledge -> the dedicated civics adapter (CIVICS_MODEL).
        Civics is a tutoring capability, so it follows the same audience/mode gate
        as /v1/chat (edu may use it via socratic/graduated_hint)."""
        rec = auth(authorization)
        body = await request.json()
        question = (body.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "question required")
        mode = body.get("mode", "graduated_hint")
        if mode not in AUDIENCE_MODES.get(rec["audience"], set()):
            raise HTTPException(
                403, f"mode '{mode}' not allowed for audience '{rec['audience']}'")
        din = filt.screen_input(question)
        if din.action != Action.ALLOW:
            return JSONResponse({"error": "blocked", "safety": din.categories}, 403)

        source = None
        if is_time_varying(question):
            # Live, officially-sourced answer — runs on the general base model.
            try:
                res = answer_time_varying(tutor, BASE_MODEL, question)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(502, f"civics live lookup failed: {e}")
            answer = res.get("answer", "")
            if res.get("source_url"):
                source = {"title": res.get("source_title", ""),
                          "url": res["source_url"], "verified": res.get("verified", False)}
            answer_route = "live_agent"
        else:
            # Static civic knowledge -> the civics adapter (arrives with v4).
            try:
                resp = tutor.chat.completions.create(
                    model=CIVICS_MODEL,
                    messages=[{"role": "system", "content": civics_sys()},
                              {"role": "user", "content": question}],
                    max_tokens=body.get("max_tokens", 400),
                    temperature=body.get("temperature", 0.3))
                answer = resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    503, "civics adapter not serving yet (arrives with the "
                         f"ParaClient-v4 cutover): {e}")
            answer_route = "static"

        dout = filt.screen_output(answer)
        if dout.action != Action.ALLOW:
            answer = dout.student_message
        return JSONResponse({"answer": answer, "route": answer_route,
                             "source": source,
                             "safety": {"action": dout.action.value}})

    @app.post("/v1/notes")
    async def notes_route(request: Request,
                          authorization: str | None = Header(None)):
        """Transcribe a photo of handwritten notes (prose + math) -> editable
        Markdown, math as LaTeX. Vision call to the ParaClient-v4 (Gemma 4V) base.
        Non-socratic capability -> consumer/internal only. The transcription is
        NEVER auto-trusted: needs_confirmation is always true so the app forces a
        student confirm/edit step before the text is used or saved."""
        rec = auth(authorization)
        if "normal" not in AUDIENCE_MODES.get(rec["audience"], set()):
            raise HTTPException(
                403, f"note transcription (non-socratic) not allowed for "
                     f"audience '{rec['audience']}'")
        body = await request.json()
        try:
            data_uri = normalize_image(body.get("image", ""))
        except ValueError as e:
            raise HTTPException(400, str(e))
        hint = str(body.get("hint", "")).strip()
        if hint:
            din = filt.screen_input(hint)
            if din.action != Action.ALLOW:
                return JSONResponse({"error": "blocked", "safety": din.categories}, 403)
        try:
            resp = tutor.chat.completions.create(
                model=NOTES_MODEL, messages=notes_messages(data_uri, hint),
                max_tokens=int(body.get("max_tokens", 1500)),
                temperature=float(body.get("temperature", 0.0)))
            text = clean_transcription(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            # The base model is not vision-capable until the ParaClient-v4
            # (Gemma 4V) cutover — surface that clearly rather than a 502.
            raise HTTPException(
                503, "note transcription needs the multimodal ParaClient-v4 "
                     f"(Gemma 4V) backend, which is not serving yet: {e}")
        dout = filt.screen_output(text)
        if dout.action != Action.ALLOW:
            text = dout.student_message
        return JSONResponse({"text": text, "format": "markdown+latex",
                             "needs_confirmation": True, "model": NOTES_MODEL,
                             "safety": {"action": dout.action.value}})

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
