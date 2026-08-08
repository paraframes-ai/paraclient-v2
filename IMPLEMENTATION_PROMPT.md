# Implementation prompt — wire the ParaFrames app to the new backend

Paste everything below into local Claude Code, in the app/frontend repo.

---

You are implementing client-side support for a batch of backend changes that
already shipped. **The backend is done and live — do not modify it.** Your job is
the app: accounts, plans, limits, sessions, teams, and the age gate.

Read this whole brief before writing code. Where a decision isn't specified,
match the existing app's patterns rather than inventing a new one.

## The two services

| Service | Base URL | Role |
|---|---|---|
| **Gateway (L4)** | `http://100.122.196.7:8080` | Auth, plans, limits, all model routes. Tailnet-only. |
| **e2 gateway** | `http://e2-gateway.tail7bcdf0.ts.net` | Browser-facing web app: Google sign-in, teams, Knowledge Library. Tailnet-only. |

Both are reachable only over the Tailscale tailnet — assume the user is on it.

Auth to the **gateway** is `Authorization: Bearer <api_key>`. Auth to **e2** is a
session cookie (`e2_session`) plus a CSRF header (`x-csrf-token`) on writes.

---

## 1. Accounts (gateway)

Keys are issued once and stored **hashed** — they cannot be retrieved again. Save
the key on receipt; if lost, the user logs in again for a new one.

### `POST /v1/auth/age-gate`
Neutral date-of-birth gate. **Must be called before signup** — signup fails without
a token from here.
```jsonc
// → {"dob": "2005-03-15"}
// 13+  ← 200 {"allowed": true, "gate_token": "…", "expires_in": 1800}
// <13  ← 200 {"allowed": false, "message": "ParaFrames for under-13 is available through your school…"}
```
**UI requirements (these are compliance constraints, not preferences):**
- An **open date field**. Never "Are you 13 or older? [Yes]" — a yes/no that nudges toward yes is not a valid age screen.
- Ask **before** collecting name, email, or anything else.
- On `allowed: false` — **hard stop**. Show the message, offer the school path, and do **not** offer a retry or a "go back and change your birthday" affordance.
- Never log or persist the DOB client-side. Don't put it in analytics.

### `POST /v1/auth/signup`
```jsonc
// → {"email": "...", "password": "...", "gate_token": "..."}   // password ≥ 10 chars
// ← 201 {"user","email","audience":"consumer","tier","api_key","allowed_versions",...}
// ← 400 short password / bad email · 403 missing or used gate_token · 409 email unavailable
```
Gate tokens are **single-use** and expire in 30 min.

### `POST /v1/auth/login`
```jsonc
// → {"email","password"}   ← 200 {"user","email","tier","api_key","allowed_versions"}
// ← 401 wrong credentials · 403 suspended · 429 too many attempts
```
401 is intentionally identical for unknown-email and wrong-password. **Surface it verbatim** — don't "helpfully" tell the user the account doesn't exist; that reintroduces user enumeration.

### `POST /v1/auth/logout`
Revokes only the presented key; other devices stay signed in. Say so in the UI.

### `GET /v1/account/me`
The single source of truth for plan state. **Drive the UI from this — never hardcode plan limits.**
```jsonc
{
  "user":"u_…","email":"…","audience":"consumer","tier":"plus","plan":"Plus","status":"active",
  "allowed_modes":["graduated_hint","normal","socratic"],
  "allowed_versions":["v2","v3","v4"],
  "library_storage_bytes":1000000000000,"library_storage":"1 TB",
  "rate_limited":true,"rpm":60,"max_context":8192,"queue_priority":5,
  "sessions":{"active":3,"limit":100},
  "usage":{"period":"2026-08","requests_used":142,"requests_limit":1000,
           "requests_remaining":858,"tokens_used":98213,"unlimited":false}
}
```

### `POST /v1/account/delete`
Right to erasure. Empty body = delete self. **Require an explicit typed confirmation** — it is irreversible and purges the Knowledge Library too.

---

## 2. Plans

| | Free | Plus | Premiere | Dev |
|---|---|---|---|---|
| Library storage | 128 GB | 1 TB | 2 TB | 4 TB |
| Requests / month | 500 | 1,000 | 10,000 | unlimited |
| Rate limit | 20/min | 60/min | 120/min | exempt |
| Concurrent sessions | 50 | 100 | 1,000 | unlimited |
| Context window | 4k | 8k | 16k | 16k |
| Queue priority | 10 | 5 | 0 (first) | 0 |
| Model versions | v2, v3 (CPU) | + v4 (GPU) | + v4 (GPU) | all |

Build a plan/usage screen from `/v1/account/me`. Show a usage meter with the reset
date (`usage.period` is `YYYY-MM`; it resets at the start of next month). Handle
`unlimited: true` (dev) — render "Unlimited", not "NaN%" or a full bar.

**There is no billing yet.** Do not build checkout, card entry, or an "Upgrade"
button that takes money. An upgrade CTA may link to a contact/waitlist page only.

---

## 3. Errors you must handle distinctly

These are different conditions and deserve different UI. Do not collapse them into "something went wrong".

| Status | Meaning | UI |
|---|---|---|
| **401** | Bad/expired key | Send to sign-in |
| **402** | *Two cases — read the message:* monthly quota exhausted, **or** the plan can't use the requested model version | Quota: show usage + reset date. Version: explain the plan gate. |
| **403** | Mode not allowed for audience, or suspended | Explain; don't retry |
| **429** | Rate limit (`…/min for the … plan`) **or** all concurrent sessions in use | Back off and retry with jitter; distinguish the two by message |
| **503** | Backend/model not running | "Temporarily unavailable", offer retry |

Implement **exponential backoff with jitter** on 429. Never retry 402 automatically.

---

## 4. Concurrent sessions (gateway)

```
POST /v1/session/open       ← {"session_id","active","limit","idle_ttl","plan"} · 429 if at cap
POST /v1/session/heartbeat  → {"session_id"}   keeps it alive
POST /v1/session/close      → {"session_id"}
GET  /v1/session/list       ← {"active","limit","idle_ttl"}
```
Open on app start, **heartbeat at half the `idle_ttl`** (default 1800s → every ~15 min), close on sign-out and on `beforeunload`. Sessions idle out server-side, so a crashed client self-heals — but an app that never closes sessions will burn a user's slots.

---

## 5. Model routes (gateway)

All take `Authorization: Bearer <key>`; all are subject to rate limit + quota.

```
POST /v1/chat         {mode, subject, version?, messages[], max_tokens?, temperature?}
                      mode: socratic | graduated_hint | normal   (normal is consumer-only)
                      subject: math | language_arts | general
                      ← {role, content, version, model, safety:{action}}
POST /v1/generate     documents / slideshows
POST /v1/sketch       2D CAD → JSON, renderable + DXF export
POST /v1/3d           3D CAD → positioned primitives / CSG
POST /v1/circuit      netlist (grammar-constrained + ERC-checked)
POST /v1/spreadsheet  {mode: generate|tutor} — formula-gated
POST /v1/civics       static adapter + live .gov-sourced answers
POST /v1/notes        handwriting / math transcription (image; needs v4)
POST /v1/gemini       third-party — paid tiers only
POST /v1/claude-free  Sonnet — paid tiers only
POST /v1/claude-paid  Opus — paid tiers only
POST /v1/embed        embeddings for the Knowledge Library
GET  /healthz
```

**Handling `safety`:** every chat response includes `safety.action`. When it is not
`allow`, the `content` has already been replaced with a supportive message — render
it plainly. Do **not** show raw moderation categories to a student, and never
display the original blocked text.

**Version selection:** offer only `allowed_versions` from `/v1/account/me`. Requesting
one outside the plan returns **402**.

---

## 6. e2 web app: Google sign-in + teams

```
GET  /auth/google/start      → redirects to Google (503 until credentials are configured)
GET  /auth/google/callback   ← Google returns here; sets session, or redirects to the age gate
GET  /auth/age-gate          neutral DOB form for first-time users (403 without a pending sign-in)
POST /auth/age-gate          {dob, csrf_token} → signs in, or hard-stops under-13
GET  /api/teams              ← {"email","is_dev":bool,"teams":[{domain,name,role,members[]}]}
POST /logout
GET  /api/session            current session info
GET  /api/backend/validate   backend reachability
POST /api/generate · /api/chat · /api/sketch · /api/3d · /api/circuit
POST /api/library · GET /api/library · GET /api/library/{artifact_id} · GET /api/storage
```

**Team rule** — derived from the email domain automatically:

| Signs in as | Team |
|---|---|
| `alice@paraframes.org` | **dev** |
| `bob@eng.paraframes.org` | **eng** (created on first sign-in) |
| `carol@a.b.paraframes.org` | **a.b** |
| `dave@gmail.com` | none — ordinary user |

Build: a **"Sign in with Google"** button, a team badge in the header when
`is_dev` or a team is present, and a team page listing members from `/api/teams`.
Users with no team must see a coherent UI — not an empty "Teams" shell.

**CSRF:** e2 writes require the `x-csrf-token` header matching the `e2_csrf` cookie.
Send it on every POST.

---

## 7. Build these screens

1. **Sign up** — age gate → email/password → "save your API key" (shown once).
2. **Sign in** — email/password, plus Google on e2.
3. **Account / plan** — plan, usage meter + reset date, storage, limits, sessions, sign-out, delete account.
4. **Teams** (e2) — team name, members, dev badge.
5. **Chat** — mode + subject + version pickers (gated by plan), safety-aware rendering.
6. **Limit states** — quota exhausted, rate limited, session cap reached, plan-gated model.

## Definition of done
- No hardcoded plan limits anywhere — all from `/v1/account/me`.
- The API key is shown exactly once, with a copy button and an explicit warning.
- 402 / 429 / 403 render distinct, actionable states.
- Sessions open, heartbeat, and close correctly (verify the count in `/v1/session/list`).
- The age gate is an open date field, ask-first, with a genuine hard stop and no retry path.
- The DOB never reaches storage, logs, or analytics.
- Tests for: the age-gate branches, each error status, and plan-gated version selection.

## Ask before assuming
- Which framework/state library to use if the repo doesn't already fix it.
- Whether the app should target the tailnet URLs above or a proxy.
- Anything requiring a payment flow — that does not exist yet; stop and ask.
