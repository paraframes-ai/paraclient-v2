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
from fastapi import HTTPException  # noqa: E402  (module-level: used by resolve_version)
# The Vertex SDKs are blocking; run them off the event loop so one slow
# third-party transcription cannot stall every other in-flight request.
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from content_filter import ContentFilter, Action, HeuristicBackend  # noqa: E402
from moderation import ShieldGemmaBackend, CompositeBackend  # noqa: E402
import accounts  # noqa: E402  (SQLite account store: age gate, signup, login)
import generate_media as gm  # noqa: E402
from cad_schema import (sys_for, clean_sketch, clean_solid,  # noqa: E402
                        repair_sketch, floorplan_issues,
                        guided_schema as cad_guided_schema)
from circuit_schema import (CIRCUIT_SYS, erc as circuit_erc,  # noqa: E402
                            clean as circuit_clean,
                            guided_schema as circuit_guided_schema)
from notes_schema import (normalize_image, notes_messages,  # noqa: E402
                          notes_prompt, clean_transcription, NOTES_SYS)
from civics_schema import sys_for as civics_sys, is_time_varying  # noqa: E402
from civics_agent import answer_time_varying  # noqa: E402
from spreadsheet_schema import (SPREADSHEET_SYS, spreadsheet_issues,  # noqa: E402
                                clean_spreadsheet, evaluate_sheet,
                                guided_schema as spreadsheet_guided_schema)
from gemini_backend import (generate as gemini_generate,  # noqa: E402
                            generate_vision as gemini_vision)
from claude_backend import (generate as claude_generate,  # noqa: E402
                            generate_vision as claude_vision,
                            FREE_MODEL as CLAUDE_FREE_MODEL,
                            PAID_MODEL as CLAUDE_PAID_MODEL)

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


# --------------------------------------------------------------------------
# Model-version tiers. A key's TIER decides which ParaClient generation it may
# use. The single 24 GB L4 serves ONE version live on the GPU (v4, Gemma-4,
# multimodal); v2 (7B) and v3 (14B) are served on CPU (llama.cpp) for the free
# tier. v4 is premium -> paid/dev only. edu (schools) is treated as a PAID tier.
# --------------------------------------------------------------------------
# Plans: free -> plus -> premiere, plus an internal dev tier. "paid" is the
# ORIGINAL name of the single paid tier and is kept as a working alias for plus
# so existing keys (gateway_keys.json, and the edu->paid derivation below) keep
# their access; nothing has to be re-issued.
TIERS = ("free", "plus", "premiere", "dev")
TIER_LABEL = {"free": "Free", "plus": "Plus", "premiere": "Premiere",
              "dev": "Dev", "paid": "Plus"}
# Every tier above free may reach the GPU (v4 / Kalvi 4).
PAID_TIERS = {"plus", "premiere", "dev", "paid"}

# The off-box, third-party models (Gemini / Claude on Vertex) are a Premiere
# perk, NOT a general paid perk: Plus reaches v4 on the GPU but not the external
# providers. "paid" is the legacy alias of Plus, so it is deliberately excluded.
EXTERNAL_MODEL_TIERS = {"premiere", "dev"}


def can_use_external(rec: dict) -> bool:
    return tier_of(rec) in EXTERNAL_MODEL_TIERS

VERSION_ACCESS = {
    "v2": {"free"} | PAID_TIERS,
    "v3": {"free"} | PAID_TIERS,
    "v4": set(PAID_TIERS),          # premium: GPU + multimodal (not free)
}
PREFERRED_VERSION_ORDER = ("v4", "v3", "v2")   # best first, for defaulting
# Per-tier default: paid tiers get v4 (GPU); free defaults to v2 (7B, responsive
# on CPU) rather than v3 (14B, ~2-4 tok/s) — v3 is opt-in for free users.
# CPU-only: v4 (GPU/Gemma-4) is gone, so every tier defaults to v2 (the 7B
# CPU model on llama.cpp). v4 stays in VERSION_ACCESS but only answers if a GPU
# vLLM is ever running again; on the CPU box a request for it 503s by design.
DEFAULT_VERSION_BY_TIER = {"free": "v2", "plus": "v2", "premiere": "v2",
                           "paid": "v2", "dev": "v2"}
VERSION_MODEL = {"v2": "paraclient-v2", "v3": "paraclient-v3"}  # CPU llama.cpp names

# Knowledge Library (per-user RAG) storage allowance per tier. Decimal units,
# the way storage is quoted to users: 1 TB = 1000 GB, not 1024 GiB.
_GB = 1000 ** 3
_TB = 1000 ** 4
TIER_STORAGE_BYTES = {
    "free":     128 * _GB,
    "plus":       1 * _TB,
    "premiere":   2 * _TB,
    "dev":        4 * _TB,
    "paid":       1 * _TB,          # legacy alias of plus
}

# Rate limiting: EVERY tier is capped except dev, which is exempt so internal
# tooling and load tests are never throttled by their own gateway.
# Per-minute request caps. These are deliberately generous — the L4 generates at
# ~16 tok/s, so a real 400-token answer already takes ~25s and no human
# approaches these numbers. They exist to stop a runaway loop or an abusive
# signup from monopolising the single GPU, not to shape normal use.
UNLIMITED_TIERS = {"dev"}
TIER_RPM = {"free": 20, "plus": 60, "premiere": 120, "paid": 60}
DEFAULT_RPM = 20


# Monthly USAGE quota — distinct from the per-minute rate limit above. The rate
# limit stops a burst; this caps what a plan is worth over a month.
#   Free      1x   (baseline)
#   Plus      2x free
#   Premiere  10x plus  = 20x free
#   Dev       exempt
# Metered in REQUESTS (a tutoring turn is the unit a user understands). The
# meter also accumulates tokens for capacity planning; only requests are
# enforced, so switching the enforced unit later needs no schema change.
FREE_MONTHLY_REQUESTS = 500
TIER_MONTHLY_REQUESTS = {
    "free":     FREE_MONTHLY_REQUESTS,           #    500
    "plus":     FREE_MONTHLY_REQUESTS * 2,       #  1,000
    "premiere": FREE_MONTHLY_REQUESTS * 20,      # 10,000  (= 10x plus)
    "paid":     FREE_MONTHLY_REQUESTS * 2,       # legacy alias of plus
}


# Concurrent active sessions per user. A session is claimed via
# /v1/session/open and released on close or after SESSION_IDLE_TTL.
TIER_MAX_SESSIONS = {"free": 50, "plus": 100, "premiere": 1000, "paid": 100}

# Context window per tier. The v4 server is started with --max-model-len 16384,
# so premiere gets the full window and lower tiers are clamped below it.
TIER_MAX_CONTEXT = {"free": 4096, "plus": 8192, "premiere": 16384,
                    "paid": 8192, "dev": 16384}

# vLLM scheduling priority — LOWER IS HANDLED EARLIER. Only has an effect when
# the server runs with --scheduling-policy priority (see serve_g4_fp8.sh);
# under the default fcfs policy the field is accepted and ignored, so this is
# safe to send either way.
TIER_PRIORITY = {"premiere": 0, "dev": 0, "plus": 5, "paid": 5, "free": 10}


def max_sessions_for(rec: dict) -> int | None:
    tier = tier_of(rec)
    if tier in UNLIMITED_TIERS:
        return None
    return TIER_MAX_SESSIONS.get(tier, TIER_MAX_SESSIONS["free"])


def max_context_for(rec: dict) -> int:
    return TIER_MAX_CONTEXT.get(tier_of(rec), TIER_MAX_CONTEXT["free"])


def priority_for(rec: dict) -> int:
    """Queue priority for this key. Premiere (and dev) jump ahead of plus,
    which jumps ahead of free, whenever requests contend for the single GPU."""
    return TIER_PRIORITY.get(tier_of(rec), TIER_PRIORITY["free"])


# NOTE: /v1/generate deliberately runs ON-BOX (the Munivar model), NOT Vertex —
# see the route. The earlier Vertex tiering was removed to honour the
# sustainability commitment: the efficient local model authors docs/slides, and
# web search/fetch stay on-box. Vertex remains only for /v1/gemini, /v1/claude-*
# and note-vision, which have no on-box replacement yet.
def monthly_quota_for(rec: dict) -> int | None:
    """Requests-per-month allowance, or None if the tier is exempt."""
    tier = tier_of(rec)
    if tier in UNLIMITED_TIERS:
        return None
    return TIER_MONTHLY_REQUESTS.get(tier, FREE_MONTHLY_REQUESTS)


def rpm_for(rec: dict) -> int | None:
    """Per-minute cap for this key, or None if the tier is exempt.

    An explicit `rpm` on the key record wins (that is how the legacy keystore
    sets per-key limits); otherwise the tier's default applies."""
    tier = tier_of(rec)
    if tier in UNLIMITED_TIERS:
        return None
    return rec.get("rpm") or TIER_RPM.get(tier, DEFAULT_RPM)


def tier_of(rec: dict) -> str:
    """A key's tier. Explicit `tier` wins; else derive: internal->dev,
    edu->paid (schools pay), everything else->free."""
    t = rec.get("tier")
    if t in TIERS or t == "paid":
        return t
    return {"internal": "dev", "edu": "paid"}.get(rec.get("audience"), "free")


def storage_bytes_for(rec: dict) -> int:
    """Knowledge Library allowance for this key, in bytes."""
    return TIER_STORAGE_BYTES.get(tier_of(rec), TIER_STORAGE_BYTES["free"])


def human_bytes(n: int) -> str:
    """Render a quota the way it is sold ('128 GB', '2 TB')."""
    return f"{n // _TB} TB" if n >= _TB else f"{n // _GB} GB"


# Product naming. The EDU line is branded KALVI; consumer stays ParaClient.
# Generations line up one-for-one, so v2/v3/v4 read as Kalvi 2/3/4 to a school
# and ParaClient 2/3/4 to a consumer -- the same served weights either way.
#
# This is a DISPLAY layer only. The vLLM --served-model-name and the LoRA module
# names (ParaFrames/ParaClient-*-v2.2) are the serving contract and are
# deliberately untouched: renaming them would require re-serving the GPU and
# re-pointing every adapter for what is a branding change.
EDU_PRODUCT = "Kalvi"
CONSUMER_PRODUCT = "ParaClient"


def product_for(rec: dict) -> str:
    return EDU_PRODUCT if rec.get("audience") == "edu" else CONSUMER_PRODUCT


def display_version(rec: dict, version: str) -> str:
    """'v4' -> 'Kalvi 4' for a school, 'ParaClient 4' for a consumer."""
    return f"{product_for(rec)} {str(version).lstrip('v')}"


def display_versions(rec: dict, versions) -> list:
    return [display_version(rec, v) for v in versions]


def versions_for_tier(tier: str) -> list:
    return sorted(v for v, tiers in VERSION_ACCESS.items() if tier in tiers)


def versions_for(rec: dict) -> list:
    return versions_for_tier(tier_of(rec))


def resolve_version(rec: dict, requested: str | None) -> str:
    """Pick the ParaClient version for this request, enforcing the tier gate.
    402 (payment required) if the tier can't use the requested version."""
    tier = tier_of(rec)
    allowed = {v for v, tiers in VERSION_ACCESS.items() if tier in tiers}
    if requested:
        rv = str(requested).lower()
        if rv not in VERSION_ACCESS:
            raise HTTPException(400, f"unknown model version {requested!r}")
        if rv not in allowed:
            raise HTTPException(
                402, f"model {rv} requires a paid plan (your tier: {tier})")
        return rv
    default = DEFAULT_VERSION_BY_TIER.get(tier)
    if default and default in allowed:
        return default
    for v in PREFERRED_VERSION_ORDER:   # fallback: best the tier allows
        if v in allowed:
            return v
    raise HTTPException(403, "no model versions available for this account")


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

# /v1/notes may be served by the local VLM or, for paid consumer keys, by a
# third-party VLM on Vertex. "paraclient" is the default because it is the only
# one where the page never leaves the box; the others are opt-in per request.
NOTES_BACKENDS = {"paraclient", "gemini", "claude"}

# Civics has TWO answer paths (see /v1/civics): static civic knowledge is served
# by the dedicated civics adapter; time-varying facts (current office-holders,
# a user's own representatives) are NOT memorized -- they route to civics_agent,
# which runs the ReAct loop against official .gov/.mil sources. The agent loop
# runs on the general base model (BASE_MODEL); only the static path needs the
# civics adapter, which arrives with the ParaClient-v4 cutover.
CIVICS_MODEL = os.environ.get("CIVICS_MODEL", "ParaFrames/ParaClient-civics-v2.2")

# /v1/spreadsheet: 'generate' produces a {title, cells} sheet that is
# formula-gated (spreadsheet_issues recomputes every formula before it's
# served); 'tutor' explains a spreadsheet concept in plain language.
SPREADSHEET_MODEL = os.environ.get("SPREADSHEET_MODEL",
                                   "ParaFrames/ParaClient-spreadsheet-v2.2")
SPREADSHEET_TUTOR_SYS = (
    "You are a friendly, clear spreadsheet tutor for students. Explain concepts "
    "and formulas simply, with a concrete example, and keep it short.")

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


def save_keys(keys: dict) -> None:
    """Persist the keystore after a mutation (e.g. account deletion)."""
    KEYSTORE.write_text(json.dumps(keys, indent=2))


def erasure_audit(user: str, keys_removed: int, requested_by: str,
                  kl_status: str) -> None:
    """Append a metadata-only record of a data-deletion (right-to-erasure /
    under-13 takedown). No content — just that an erasure happened, for whom,
    when, and by whom. Doubles as the deletion audit trail."""
    rec = {"ts": time.time(), "event": "account_erasure", "user": user,
           "keys_removed": keys_removed, "requested_by": requested_by,
           "knowledge_library_purge": kl_status}
    try:
        log = Path(__file__).with_name("logs") / "erasure_audit.jsonl"
        log.parent.mkdir(exist_ok=True)
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 — audit logging must never break the request
        pass


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
    # Real on-prem moderation: instant heuristic (self-harm/abuse escalation) +
    # ShieldGemma-2B per-policy classifier (nuanced block-harms). Fails closed.
    filt = ContentFilter(backend=CompositeBackend(
        [HeuristicBackend(), ShieldGemmaBackend()]))
    keys = load_keys()
    rl = RateLimiter()

    # Real account store (signup/login/age gate). The legacy JSON keystore above
    # still backs the internal/service + demo keys; see lookup().
    accounts.init_db()
    print(f"[*] accounts db -> {accounts.DB_PATH} "
          f"({accounts.count_accounts()} accounts) | new signups get tier "
          f"'{accounts.default_tier()}' (PARACLIENT_ENV="
          f"{os.environ.get('PARACLIENT_ENV', 'dev')})")

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
        # Guided decoding first: `response_format: json_schema` constrains output
        # to a valid CAD envelope with a known entity/solid `type`, so the model
        # physically can't emit malformed JSON and the geometry gate + auto-repair
        # only ever see structurally-clean input. One unguided retry as a
        # backend-compat fallback (if a llama.cpp build ever rejects the schema).
        rf = {"type": "json_schema",
              "json_schema": {"name": f"cad_{mode}",
                              "schema": cad_guided_schema(mode)}}
        last = None
        for kw in ({"response_format": rf}, {}):
            try:
                r = tutor.chat.completions.create(
                    model=CAD_MODEL, messages=msgs,
                    max_tokens=3000, temperature=0.2, **kw)
                return gm.parse_json(r.choices[0].message.content.strip())
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"CAD ({mode}) generation failed: {last}")

    circuit = OpenAI(base_url=CIRCUIT_URL, api_key="none")
    print(f"[*] /v1/circuit -> CPU llama.cpp {CIRCUIT_URL}")

    # Older ParaClient generations for the free tier, served on CPU (llama.cpp).
    # v4 is the GPU `tutor` above; v2/v3 clients point at their CPU servers.
    V2_URL = os.environ.get("V2_URL", "http://127.0.0.1:8002/v1")
    V3_URL = os.environ.get("V3_URL", "http://127.0.0.1:8003/v1")
    version_client = {"v2": OpenAI(base_url=V2_URL, api_key="none"),
                      "v3": OpenAI(base_url=V3_URL, api_key="none")}
    print(f"[*] free-tier versions -> v2 {V2_URL} | v3 {V3_URL} (CPU)")

    def gen_circuit(prompt: str) -> dict:
        """Netlist on the CPU circuit model with ERC-gated rejection sampling:
        try a few, return the first electrically-valid one (else the best-effort
        structurally-clean last result)."""
        msgs = [{"role": "system", "content": CIRCUIT_SYS},
                {"role": "user", "content": prompt}]
        # Guided decoding makes every sample valid JSON with known component
        # types, so ERC rejection sampling spends all 3 tries on *electrical*
        # validity instead of losing some to malformed JSON.
        rf = {"type": "json_schema",
              "json_schema": {"name": "circuit_netlist",
                              "schema": circuit_guided_schema()}}
        last = None
        for _ in range(3):
            try:
                r = circuit.chat.completions.create(
                    model="circuit", messages=msgs,
                    max_tokens=900, temperature=0.4, response_format=rf)
                nl = circuit_clean(gm.parse_json(r.choices[0].message.content.strip()))
                last = nl
                if circuit_erc(nl)[0]:
                    return nl
            except Exception:  # noqa: BLE001
                pass
        if last is None:
            # Guided never parsed -> backend may not support it; unguided fallback.
            try:
                r = circuit.chat.completions.create(
                    model="circuit", messages=msgs,
                    max_tokens=900, temperature=0.4)
                last = circuit_clean(gm.parse_json(r.choices[0].message.content.strip()))
            except Exception:  # noqa: BLE001
                pass
        if isinstance(last, dict):
            return last          # ERC-imperfect but structurally clean
        raise RuntimeError("no parseable netlist from circuit model")

    def gen_spreadsheet(prompt: str) -> dict:
        """Spreadsheet JSON on the spreadsheet adapter with a formula-gated
        rejection sample: return the first sheet whose formulas all compute
        (spreadsheet_issues), else the best-effort structurally-clean last one."""
        msgs = [{"role": "system", "content": SPREADSHEET_SYS},
                {"role": "user", "content": prompt}]
        # Guided decoding guarantees the {title, cells:[{ref,...}]} envelope
        # parses, so both tries go to the formula gate (spreadsheet_issues), not
        # to malformed JSON. Unguided fallback if the backend rejects the schema.
        rf = {"type": "json_schema",
              "json_schema": {"name": "spreadsheet",
                              "schema": spreadsheet_guided_schema()}}
        last = None
        for _ in range(2):
            try:
                r = tutor.chat.completions.create(
                    model=SPREADSHEET_MODEL, messages=msgs,
                    max_tokens=1500, temperature=0.3, response_format=rf)
                sheet = clean_spreadsheet(
                    gm.parse_json(r.choices[0].message.content.strip()))
                last = sheet
                if spreadsheet_issues(sheet)[0]:
                    return sheet
            except Exception:  # noqa: BLE001
                pass
        if last is None:
            try:
                r = tutor.chat.completions.create(
                    model=SPREADSHEET_MODEL, messages=msgs,
                    max_tokens=1500, temperature=0.3)
                last = clean_spreadsheet(
                    gm.parse_json(r.choices[0].message.content.strip()))
            except Exception:  # noqa: BLE001
                pass
        if isinstance(last, dict):
            return last          # formula-imperfect but structurally clean
        raise RuntimeError("no parseable spreadsheet from model")

    def lookup(authorization: str | None):
        """Validate the bearer key only (no rate-limit consumption).

        Two key sources, checked in this order:
          1. accounts.db — real user accounts (signup/login). Keys are stored
             hashed; a suspended account resolves to nothing.
          2. gateway_keys.json — the legacy scaffold keystore, which still holds
             the internal/service and demo keys. Kept so the e2 handshake and
             existing tooling do not break; new users never land here.
        """
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer key")
        tok = authorization.split(" ", 1)[1].strip()
        rec = accounts.record_for_key(tok) or keys.get(tok)
        if not rec:
            raise HTTPException(401, "invalid key")
        return tok, rec

    def auth(authorization: str | None):
        """Validate + rate-limit (for the model/generation routes).

        Two independent limits, both exempt for dev (UNLIMITED_TIERS):
          * RATE  — requests/minute, per key. Stops a burst (TIER_RPM).
          * USAGE — requests/month, per user. Caps what a plan is worth
                    (TIER_MONTHLY_REQUESTS). Metered here because auth() is the
                    one choke point every model route funnels through; that
                    counts a request at admission, so a turn later refused by
                    the safety filter still consumes quota.
        """
        tok, rec = lookup(authorization)
        rpm = rpm_for(rec)
        if rpm is not None and not rl.allow(tok, rpm, time.time()):
            raise HTTPException(
                429, f"rate limit exceeded ({rpm}/min for the "
                     f"{TIER_LABEL.get(tier_of(rec), 'Free')} plan)")

        quota = monthly_quota_for(rec)
        if quota is not None:
            user = rec.get("user")
            used = accounts.usage_for(user)["requests"]
            if used >= quota:
                plan = TIER_LABEL.get(tier_of(rec), "Free")
                raise HTTPException(
                    402, f"monthly usage limit reached ({used}/{quota} requests "
                         f"on the {plan} plan). It resets at the start of next "
                         f"month, or upgrade for a higher limit.")
            accounts.record_usage(user, requests=1)
        return rec

    @app.post("/v1/auth/validate")
    async def validate(authorization: str | None = Header(None)):
        """Handshake for an upstream login service (e.g. the e2 site): after it
        authenticates the user (email whitelist), it calls this with its service
        key to confirm success + learn the identity/permissions. The service key
        is held ONLY by that server — never sent to browsers."""
        _, rec = lookup(authorization)
        return {"ok": True, "user": rec["user"], "audience": rec["audience"],
                "allowed_modes": sorted(AUDIENCE_MODES.get(rec["audience"], [])),
                "tier": tier_of(rec), "allowed_versions": versions_for(rec),
                "product": product_for(rec),
                "allowed_model_names": display_versions(rec, versions_for(rec)),
                "plan": TIER_LABEL.get(tier_of(rec), "Free"),
                # e2 owns the Knowledge Library files, so it enforces the quota;
                # the gateway is the single source of truth for what it IS.
                "library_storage_bytes": storage_bytes_for(rec),
                "library_storage": human_bytes(storage_bytes_for(rec)),
                "monthly_request_limit": monthly_quota_for(rec),
                "rpm": rpm_for(rec),
                "max_sessions": max_sessions_for(rec),
                "max_context": max_context_for(rec),
                "queue_priority": priority_for(rec)}

    # ---------------- Accounts: age gate -> signup -> login ----------------
    # The gate is a SEPARATE endpoint on purpose: compliance/01 §1 requires the
    # age check to happen before any other data is collected, so the client must
    # be able to ask it without having gathered an email or password yet.

    def _acct_error(e: accounts.AccountError):
        return JSONResponse({"error": e.message}, status_code=e.status)

    @app.post("/v1/auth/age-gate")
    async def auth_age_gate(request: Request):
        """Neutral date-of-birth gate. Body: {"dob": "YYYY-MM-DD"}.

        13+ -> {allowed: true, gate_token}. Under 13 -> {allowed: false} with
        the school message, and NOTHING is stored (compliance/01 §2). The DOB
        is used for one comparison and discarded; it is never persisted and is
        deliberately never written to any log line."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "expected a JSON body with 'dob'")
        try:
            return JSONResponse(accounts.age_gate(body.get("dob", "")))
        except accounts.AccountError as e:
            return _acct_error(e)

    @app.post("/v1/auth/signup")
    async def auth_signup(request: Request):
        """Create a consumer account. Body: {email, password, gate_token}.

        Requires a live gate token, so there is no path to an account that
        skipped the age check. Returns the API key ONCE — it is stored hashed
        and cannot be shown again."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "expected a JSON body")
        try:
            out = accounts.create_account(
                email=body.get("email", ""), password=body.get("password", ""),
                gate_token=body.get("gate_token", ""))
        except accounts.AccountError as e:
            return _acct_error(e)
        return JSONResponse({**out,
                             "allowed_versions": versions_for_tier(out["tier"]),
                             "note": "Save this api_key now — it cannot be "
                                     "retrieved again."}, status_code=201)

    @app.post("/v1/auth/login")
    async def auth_login(request: Request):
        """Email + password -> a fresh API key."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "expected a JSON body")
        try:
            out = accounts.login(body.get("email", ""),
                                 body.get("password", ""))
        except accounts.AccountError as e:
            return _acct_error(e)
        return JSONResponse({**out,
                             "allowed_versions": versions_for_tier(out["tier"])})

    @app.post("/v1/auth/logout")
    async def auth_logout(authorization: str | None = Header(None)):
        """Revoke the presented key (other sessions keep working)."""
        tok, _ = lookup(authorization)
        return {"revoked": accounts.revoke_key(tok)}

    @app.get("/v1/account/me")
    async def account_me(authorization: str | None = Header(None)):
        _, rec = lookup(authorization)
        acct = accounts.get_account(rec.get("user")) or {}
        return {"user": rec.get("user"), "email": acct.get("email"),
                "audience": rec.get("audience"), "tier": tier_of(rec),
                "plan": TIER_LABEL.get(tier_of(rec), "Free"),
                "status": acct.get("status", "active"),
                "allowed_modes": sorted(AUDIENCE_MODES.get(rec["audience"], [])),
                "allowed_versions": versions_for(rec),
                "product": product_for(rec),
                "allowed_model_names": display_versions(rec, versions_for(rec)),
                "library_storage_bytes": storage_bytes_for(rec),
                "library_storage": human_bytes(storage_bytes_for(rec)),
                "rate_limited": rpm_for(rec) is not None,
                "rpm": rpm_for(rec),
                "max_context": max_context_for(rec),
                "queue_priority": priority_for(rec),
                "external_models": can_use_external(rec),
                "sessions": {"active": accounts.active_sessions(rec.get("user")),
                             "limit": max_sessions_for(rec)},
                "usage": _usage_block(rec)}

    def _usage_block(rec: dict) -> dict:
        """This period's meter vs the plan's allowance."""
        u = accounts.usage_for(rec.get("user"))
        quota = monthly_quota_for(rec)
        return {"period": u["period"], "requests_used": u["requests"],
                "requests_limit": quota,
                "requests_remaining": (None if quota is None
                                       else max(0, quota - u["requests"])),
                "tokens_used": u["tokens"], "unlimited": quota is None}

    @app.post("/v1/session/open")
    async def session_open(authorization: str | None = Header(None)):
        """Claim one of this user's concurrent session slots (see
        TIER_MAX_SESSIONS). 429 when the tier's cap is already in use."""
        _, rec = lookup(authorization)
        try:
            out = accounts.open_session(rec.get("user"), max_sessions_for(rec))
        except accounts.AccountError as e:
            return _acct_error(e)
        return {**out, "plan": TIER_LABEL.get(tier_of(rec), "Free")}

    @app.post("/v1/session/heartbeat")
    async def session_heartbeat(request: Request,
                                authorization: str | None = Header(None)):
        """Keep a session alive. Without a heartbeat a session idles out after
        SESSION_IDLE_TTL and frees its slot."""
        lookup(authorization)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        sid = body.get("session_id")
        if not sid:
            raise HTTPException(400, "pass {'session_id': '...'}")
        if not accounts.touch_session(sid):
            raise HTTPException(404, "unknown or expired session")
        return {"session_id": sid, "alive": True}

    @app.post("/v1/session/close")
    async def session_close(request: Request,
                            authorization: str | None = Header(None)):
        lookup(authorization)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        sid = body.get("session_id")
        if not sid:
            raise HTTPException(400, "pass {'session_id': '...'}")
        return {"session_id": sid, "closed": accounts.close_session(sid)}

    @app.get("/v1/session/list")
    async def session_list(authorization: str | None = Header(None)):
        _, rec = lookup(authorization)
        return {"active": accounts.active_sessions(rec.get("user")),
                "limit": max_sessions_for(rec),
                "idle_ttl": accounts.SESSION_IDLE_TTL}

    @app.post("/v1/account/suspend")
    async def account_suspend(request: Request,
                              authorization: str | None = Header(None)):
        """Step 1 of the discovered-under-13 takedown (compliance/01 §4):
        suspend immediately (revoking every live key) so access stops now, then
        run /v1/account/delete for the data erasure. Internal callers only."""
        _, rec = lookup(authorization)
        if rec.get("audience") != "internal":
            raise HTTPException(403, "internal/service callers only")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        user = body.get("user")
        if not user:
            raise HTTPException(400, "pass {'user': '<id>'}")
        status = body.get("status", "suspended")
        try:
            changed = accounts.set_status(user, status)
        except accounts.AccountError as e:
            return _acct_error(e)
        if not changed:
            raise HTTPException(404, f"no such account: {user}")
        return {"user": user, "status": status}

    @app.post("/v1/account/delete")
    async def account_delete(request: Request,
                             authorization: str | None = Header(None)):
        """Right-to-erasure / under-13 takedown. Deletes what the L4 holds for a
        user (their access key(s) + rate-limit state), records a metadata-only
        audit entry, and triggers the e2 Knowledge-Library purge if configured.

        By default a caller deletes their OWN account (the key they present). An
        internal/service caller (e.g. e2) may delete a named user by passing
        {"user": "<id>"} — used for the coordinated erasure and the under-13
        takedown. A normal user cannot delete anyone else."""
        _, rec = lookup(authorization)   # authenticate; deletion is not rate-limited
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 — empty body = self-delete
            body = {}
        target = body.get("user")
        if target and target != rec.get("user"):
            if rec.get("audience") != "internal":
                raise HTTPException(
                    403, "only an internal/service caller may delete another "
                         "user's account")
            user = target
        else:
            user = rec.get("user")
        if not user:
            raise HTTPException(400, "no user associated with this request")

        removed = [k for k, v in keys.items() if v.get("user") == user]
        for k in removed:
            keys.pop(k, None)
            rl._hits.pop(k, None)
        save_keys(keys)
        # Real accounts live in accounts.db; erase the row and every key there
        # too, or the "deletion" would only clear the legacy scaffold keystore.
        db_keys_removed = accounts.delete_account(user)

        # Best-effort: tell e2 to purge this user's Knowledge Library + record.
        kl_status = "e2_purge_not_configured"
        kl_url = os.environ.get("KL_DELETE_URL")
        if kl_url:
            try:
                import httpx
                httpx.post(kl_url, json={"user": user}, timeout=15)
                kl_status = "e2_purge_requested"
            except Exception as e:  # noqa: BLE001
                kl_status = f"e2_purge_failed: {e}"

        total_removed = len(removed) + db_keys_removed
        erasure_audit(user, total_removed, requested_by=rec.get("user"),
                      kl_status=kl_status)
        return JSONResponse({
            "deleted": True, "user": user, "keys_removed": total_removed,
            "knowledge_library_purge": kl_status,
            "note": ("L4 revoked access and cleared local state. The Knowledge "
                     "Library embeddings and account record live on e2 and must "
                     "be purged there — set KL_DELETE_URL or have e2 complete the "
                     "erasure.")})

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

        # Tier gate: which ParaClient version this key may use (v4 = paid/dev).
        version = resolve_version(rec, body.get("version"))

        user_turns = [m for m in messages if m.get("role") == "user"]
        latest = user_turns[-1]["content"] if user_turns else ""

        din = filt.screen_input(latest)
        if din.action != Action.ALLOW:
            return JSONResponse({"role": "assistant", "content": din.student_message,
                                 "safety": {"action": din.action.value,
                                            "categories": din.categories}})

        sys_msg = {"role": "system", "content": system_prompt(mode, subject)}
        convo = [sys_msg] + [m for m in messages if m.get("role") != "system"]
        # v4 -> GPU adapter model; v2/v3 -> CPU llama.cpp base for that generation
        if version == "v4":
            client, model = tutor, model_for(mode, subject)
        else:
            client, model = version_client[version], VERSION_MODEL[version]
        # Tier perks: premiere gets the full context window and jumps the GPU
        # queue. `priority` is a vLLM extra -- only send it to the v4 (vLLM)
        # backend; the CPU llama.cpp servers would reject an unknown field.
        max_tok = max(1, min(int(body.get("max_tokens", 400) or 400),
                             max_context_for(rec)))
        extra = ({"priority": priority_for(rec)} if version == "v4" else None)
        try:
            resp = client.chat.completions.create(
                model=model, messages=convo,
                max_tokens=max_tok,
                temperature=body.get("temperature", 0.3),
                extra_body=extra)
            answer = resp.choices[0].message.content
        except Exception:
            raise HTTPException(
                503, f"ParaClient {version} backend not available"
                     + (" (CPU free-tier server not running yet)"
                        if version != "v4" else ""))

        dout = filt.screen_output(answer)
        if dout.action != Action.ALLOW:
            answer = dout.student_message
        return JSONResponse({"role": "assistant", "content": answer,
                             "version": version, "model": model,
                             # What the student/teacher should see. `model` above
                             # stays the internal serving id.
                             "product": product_for(rec),
                             "model_name": display_version(rec, version),
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
        # Docs/slides author ON-BOX, on the Munivar model — NOT Vertex. This is
        # the sustainability commitment: the efficient local model does the
        # work, web search/fetch run on-box (DuckDuckGo + httpx), so the whole
        # loop stays on-prem and off the frontier.
        #   web=false -> single-pass, SCHEMA-GUIDED (the grammar keystone):
        #                the decoder is constrained to the doc/slide schema so a
        #                small model can't emit malformed structure.
        #   web=true  -> the agentic search/fetch loop, also on the local model.
        # `tutor` now points at the CPU model server; it ignores the model name.
        gen_client = tutor
        gen_model = VERSION_MODEL.get("v2", "paraclient-v2")
        try:
            data = (gm.generate_agentic(gen_client, gen_model, kind, prompt) if web
                    else gm.generate_guided(gen_client, gen_model, kind, prompt))
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
                             "content_type": ctype, "file_base64": b,
                             "authored_by": gen_model, "provider": "paraclient (on-prem)",
                             "on_prem": True})

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

    @app.post("/v1/spreadsheet")
    async def spreadsheet_route(request: Request,
                                authorization: str | None = Header(None)):
        """Generate a formula-gated spreadsheet (mode='generate', default) or
        explain a spreadsheet concept (mode='tutor'). Generation is a
        non-socratic capability -> consumer/internal only (same as CAD/circuit).
        Generated sheets are returned with their computed values and a `valid`
        flag from the same evaluator the dataset was built with."""
        rec = auth(authorization)
        if "normal" not in AUDIENCE_MODES.get(rec["audience"], set()):
            raise HTTPException(
                403, f"spreadsheet generation (non-socratic) not allowed for "
                     f"audience '{rec['audience']}'")
        body = await request.json()
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(400, "prompt required")
        din = filt.screen_input(prompt)
        if din.action != Action.ALLOW:
            return JSONResponse({"error": "blocked", "safety": din.categories}, 403)

        if body.get("mode") == "tutor":
            try:
                resp = tutor.chat.completions.create(
                    model=SPREADSHEET_MODEL,
                    messages=[{"role": "system", "content": SPREADSHEET_TUTOR_SYS},
                              {"role": "user", "content": prompt}],
                    max_tokens=body.get("max_tokens", 400), temperature=0.3)
                answer = resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    503, "spreadsheet adapter not serving yet (arrives with the "
                         f"ParaClient-v4 cutover): {e}")
            dout = filt.screen_output(answer)
            if dout.action != Action.ALLOW:
                answer = dout.student_message
            return JSONResponse({"mode": "tutor", "answer": answer,
                                 "safety": {"action": dout.action.value}})

        try:
            sheet = gen_spreadsheet(prompt)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"spreadsheet generation failed: {e}")
        ok, issues = spreadsheet_issues(sheet)
        values, _errs = evaluate_sheet(sheet.get("cells", []))
        return JSONResponse({"mode": "generate", "title": sheet.get("title"),
                             "cells": sheet.get("cells", []),
                             "computed": values, "valid": ok, "issues": issues})

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
        # COPPA/FERPA: the live lookup calls out to the web (DuckDuckGo + .gov),
        # so a child's question would leave the box. EDU (child-facing) must stay
        # fully on-prem — time-varying questions fall through to the civics
        # adapter, which is trained to deflect ("that changes; check an official
        # source") instead of asserting a stale fact. Live sourcing is for the
        # adult consumer/internal audiences only.
        live_ok = is_time_varying(question) and rec["audience"] != "edu"
        if live_ok:
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

        backend = str(body.get("backend", "gemini")).strip().lower()
        if backend not in NOTES_BACKENDS:
            raise HTTPException(
                400, f"unknown backend {backend!r} (choose one of "
                     f"{', '.join(sorted(NOTES_BACKENDS))})")

        # A photo of a student's own notebook is about as personal as this
        # product gets, so sending it off-box is an explicit, gated choice.
        #
        # POLICY: note transcription is the ONE external capability open to
        # everyone -- every tier gets it, because it's the core scan feature and
        # there is no on-prem VLM in the CPU-only world. The general external
        # models (/v1/gemini, /v1/claude-*) stay Premiere/Dev-only; notes is the
        # deliberate exception.
        #
        # The ONE hard exclusion is edu: a child's handwriting must never leave
        # the box (COPPA/FERPA). `edu` already cannot reach this route (it lacks
        # the 'normal' mode checked above); this is the belt-and-braces block so
        # that stays true even if the mode policy is ever loosened.
        if backend != "paraclient":
            if rec.get("audience") == "edu":
                raise HTTPException(
                    403, "edu note transcription stays on-prem; the third-party "
                         "backends are not available for school accounts")

        try:
            if backend == "gemini":
                text, used = await run_in_threadpool(
                    gemini_vision, data_uri, notes_prompt(hint), NOTES_SYS,
                    None, int(body.get("max_tokens", 1500)), 0.0)
            elif backend == "claude":
                text, used = await run_in_threadpool(
                    claude_vision, data_uri, notes_prompt(hint), NOTES_SYS,
                    CLAUDE_PAID_MODEL if body.get("model") == "opus"
                    else CLAUDE_FREE_MODEL,
                    int(body.get("max_tokens", 1500)))
            else:
                resp = tutor.chat.completions.create(
                    model=NOTES_MODEL, messages=notes_messages(data_uri, hint),
                    max_tokens=int(body.get("max_tokens", 1500)),
                    temperature=float(body.get("temperature", 0.0)))
                text, used = resp.choices[0].message.content, NOTES_MODEL
            text = clean_transcription(text)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            if backend == "paraclient":
                # The base model is not vision-capable until the ParaClient-v4
                # (Gemma 4V) cutover — surface that clearly rather than a 502.
                raise HTTPException(
                    503, "note transcription needs the multimodal ParaClient-v4 "
                         f"(Gemma 4V) backend, which is not serving yet: {e}")
            msg = str(e)
            # Same diagnosis the /v1/gemini and /v1/claude-* routes give, so a
            # missing Vertex setup reads the same wherever it is hit.
            if any(s in msg for s in ("SCOPE", "PERMISSION_DENIED", "403",
                                      "credentials", "default credentials")):
                raise HTTPException(
                    503, f"{backend} (Vertex) auth not configured. Needs ADC with "
                         "cloud-platform scope, the aiplatform API enabled, and "
                         f"roles/aiplatform.user. Detail: {msg[:200]}")
            raise HTTPException(502, f"{backend} transcription failed: {msg[:200]}")

        dout = filt.screen_output(text)
        if dout.action != Action.ALLOW:
            text = dout.student_message
        return JSONResponse({"text": text, "format": "markdown+latex",
                             "needs_confirmation": True, "model": used,
                             "backend": backend,
                             "on_prem": backend == "paraclient",
                             "safety": {"action": dout.action.value}})

    @app.post("/v1/gemini")
    async def gemini_route(request: Request,
                           authorization: str | None = Header(None)):
        """Proxy to Google Gemini on Vertex AI — the first external-provider
        option. CONSUMER/INTERNAL ONLY: calls go OFF-BOX to Google, so the 'edu'
        (child-facing) audience is blocked (student data stays on-prem), and
        Gemini is a direct assistant with none of ParaClient's Socratic
        withholding. Input + output are still run through the safety filter."""
        rec = auth(authorization)
        if rec["audience"] not in ("consumer", "internal"):
            raise HTTPException(
                403, "the Gemini route is not available to the 'edu' audience "
                     "(student data must stay on-prem; use the local tutor)")
        if not can_use_external(rec):
            raise HTTPException(
                402, "external models (Gemini) are a Premiere feature "
                     f"(your plan: {TIER_LABEL.get(tier_of(rec), 'Free')})")
        body = await request.json()
        messages = body.get("messages")
        if not messages:
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                raise HTTPException(400, "messages or prompt required")
            messages = [{"role": "user", "content": prompt}]
        if body.get("system"):
            messages = ([{"role": "system", "content": body["system"]}]
                        + [m for m in messages if m.get("role") != "system"])
        user_turns = [m for m in messages if m.get("role") == "user"]
        latest = user_turns[-1]["content"] if user_turns else ""
        din = filt.screen_input(latest if isinstance(latest, str) else "")
        if din.action != Action.ALLOW:
            return JSONResponse({"error": "blocked", "safety": din.categories}, 403)
        try:
            text, used = gemini_generate(
                messages, model=body.get("model"),
                max_tokens=body.get("max_tokens", 800),
                temperature=body.get("temperature", 0.7))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(s in msg for s in ("SCOPE", "PERMISSION_DENIED", "403",
                                      "credentials", "default credentials")):
                raise HTTPException(
                    503, "Gemini (Vertex) auth not configured. Needs ADC with "
                         "cloud-platform scope (gcloud auth application-default "
                         "login), the aiplatform API enabled, and roles/"
                         f"aiplatform.user. Detail: {msg[:200]}")
            raise HTTPException(502, f"Gemini call failed: {msg[:200]}")
        dout = filt.screen_output(text)
        if dout.action != Action.ALLOW:
            text = dout.student_message
        return JSONResponse({"role": "assistant", "content": text, "model": used,
                             "provider": "google-vertex",
                             "safety": {"action": dout.action.value}})

    async def _claude_route(request: Request, rec: dict, model: str, tier_label: str):
        """Shared handler for the Claude-via-Vertex routes. Off-box, so edu is
        blocked; input + output still pass the safety filter."""
        if rec["audience"] not in ("consumer", "internal"):
            raise HTTPException(
                403, "Claude routes are not available to the 'edu' audience "
                     "(student data must stay on-prem; use the local tutor)")
        if not can_use_external(rec):
            raise HTTPException(
                402, "external Claude models are a Premiere feature "
                     f"(your plan: {TIER_LABEL.get(tier_of(rec), 'Free')})")
        body = await request.json()
        messages = body.get("messages")
        if not messages:
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                raise HTTPException(400, "messages or prompt required")
            messages = [{"role": "user", "content": prompt}]
        if body.get("system"):
            messages = ([{"role": "system", "content": body["system"]}]
                        + [m for m in messages if m.get("role") != "system"])
        user_turns = [m for m in messages if m.get("role") == "user"]
        latest = user_turns[-1]["content"] if user_turns else ""
        din = filt.screen_input(latest if isinstance(latest, str) else "")
        if din.action != Action.ALLOW:
            return JSONResponse({"error": "blocked", "safety": din.categories}, 403)
        try:
            text, used = claude_generate(
                messages, model=model,
                max_tokens=int(body.get("max_tokens", 2048)))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(s in msg for s in ("SCOPE", "PERMISSION_DENIED", "403",
                                      "credentials", "default credentials")):
                raise HTTPException(
                    503, "Claude (Vertex) auth not configured — same setup as "
                         "/v1/gemini (ADC with cloud-platform scope + aiplatform "
                         f"API enabled). Detail: {msg[:200]}")
            raise HTTPException(502, f"Claude call failed: {msg[:200]}")
        dout = filt.screen_output(text)
        if dout.action != Action.ALLOW:
            text = dout.student_message
        return JSONResponse({"role": "assistant", "content": text, "model": used,
                             "provider": "anthropic-vertex", "tier": tier_label,
                             "safety": {"action": dout.action.value}})

    @app.post("/v1/claude-free")
    async def claude_free(request: Request,
                          authorization: str | None = Header(None)):
        """Claude Sonnet via Vertex — the free external-Claude tier. Any
        non-edu audience may use it."""
        rec = auth(authorization)
        return await _claude_route(request, rec, CLAUDE_FREE_MODEL, "free")

    @app.post("/v1/claude-paid")
    async def claude_paid(request: Request,
                          authorization: str | None = Header(None)):
        """Claude Opus via Vertex — the premium external-Claude model. Paid/dev
        only (enforced in _claude_route, like every off-box model)."""
        rec = auth(authorization)
        return await _claude_route(request, rec, CLAUDE_PAID_MODEL, "paid")

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
