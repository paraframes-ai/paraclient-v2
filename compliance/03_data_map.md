# Data Map — What ParaFrames Collects, Where It Lives, Who Touches It

> **⚠️ DRAFT — not legal advice.** A data inventory / "record of processing" for
> clinic + school review. Verify each row against the live system before relying
> on it. `<FILL>` = confirm/complete.

## Deployment context

ParaClient runs **on-premise**: a GCP NVIDIA L4 GPU VM (the model server + API
gateway) and an e2 auth VM, joined on a private Tailscale tailnet (WireGuard-
encrypted, not publicly exposed). The on-prem models (ParaClient v2/v3/v4) do the
tutoring; external LLM providers are used **only** on the paid consumer lane and
**never** for edu.

## Data inventory

| # | Data element | Collected from | Purpose | Where it lives | Retention | Processors / who touches it |
|---|---|---|---|---|---|---|
| 1 | Account identity (email; name if provided) | User / school roster | Authenticate, route to lane | e2 auth VM (keystore / IdP) | Life of account; deleted on request | On-prem only (e2) |
| 2 | Age / "13+ verified" flag | Consumer sign-up age gate | Enforce the 13/edu boundary | e2 | Store the boolean + timestamp, not raw DOB | On-prem only (e2) |
| 3 | Chat messages / tutoring prompts | User (student) | Generate the tutor response | Processed in memory on the L4; **not persisted server-side** (gateway is stateless — client sends full history each turn) | Not stored on the L4 | On-prem model (ParaClient). **Consumer-paid only:** may go to Google Vertex (Gemini) / Anthropic (Claude). **Edu: on-prem only, never external.** |
| 4 | Uploaded files → Knowledge Library | User | RAG / "knowledge library" retrieval | Embeddings stored on **e2**; the L4 `/v1/embed` computes vectors and returns them (does not store) | Until user deletes | On-prem embedder (L4) + storage (e2) |
| 5 | Handwriting / note images (`/v1/notes`) | User (photo) | Transcribe to text (Gemma-4V, on-prem) | Processed in memory on the L4; **not persisted** | Not stored | On-prem model only |
| 6 | Safety-moderation signal | Derived from #3 | Child-safety screening | ShieldGemma-2B on the L4 (on-prem); a decision log at `logs/content_filter.jsonl` | ⚠️ see gap below | On-prem only |
| 7 | Civics live-lookup queries (consumer only) | Derived from a consumer civics question | Fetch a current .gov fact | Sent to DuckDuckGo + .gov via the L4 | Not stored | External web — **consumer only; edu is blocked from this path** |
| 8 | API/access logs | System | Ops, rate-limiting, audit | L4 / e2 (journald) | `<FILL: retention>` | On-prem only |

## Egress summary (the compliance-critical view)

- **Edu (under-13) data NEVER leaves the box.** External models (Gemini/Claude)
  are paid-tier + edu-blocked; the docs/slides web search is edu-blocked; the
  civics live-web lookup is edu-blocked; embeddings are computed on-prem.
- **Consumer-paid (13+)** traffic *may* leave to Google Vertex (Gemini) and
  Anthropic (Claude) and to web search — this is fine for 13+ and must be
  disclosed in the privacy policy (it is).

## No secondary use

- **Student/user content is not used to train models** and is not sold or used for
  advertising. `<FILL: confirm and state this as policy.>` (Model fine-tuning uses
  separately-sourced, license-clean datasets — not user data.)

## ⚠️ Known gaps to remediate (flag for the clinic + fix)

1. **Moderation content logging — ✅ REMEDIATED (2026-08-06).** `content_filter.py`
   previously logged a 300-char preview of student text. It now logs **decision
   metadata only** (timestamp, surface, `text_len`, action, categories) — no
   student content at rest. Content logging is an explicit dev-only opt-in
   (`log_content=True`), off by default and never enabled on the edu/production
   path. `logs/#6` in the table above is now content-free.
2. **Retention & deletion.** Define concrete retention for #1, #4, #6, #8 and a
   deletion path (ties to the takedown runbook in `01_...`). The Knowledge Library
   (#4, on e2) is the main student-data store needing a delete endpoint.
3. **Access control / audit.** Per-user keys are a static JSON keystore; there's
   no per-access audit trail (FERPA expects one). *(Offered as a follow-on build.)*
4. **Encryption at rest.** Transit is WireGuard-encrypted; confirm `<FILL: disk
   encryption on the L4 / e2 volumes and the e2 embedding store>`.
