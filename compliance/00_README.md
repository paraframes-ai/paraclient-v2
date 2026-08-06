# ParaFrames — Compliance Working Drafts

> **⚠️ DRAFT — internal working documents. NOT legal advice.**
> These were assembled from the ParaFrames/ParaClient system architecture to
> give a law-school privacy clinic or counsel a running start — they are *inputs
> to* a legal review, not a substitute for one. Do not rely on them, publish the
> privacy policy, or sign a DPA based on them until a qualified reviewer has
> signed off. Fill every `<FILL: ...>` placeholder with real values.

## The compliance model in one paragraph

ParaFrames runs **two lanes**, deliberately split to minimize regulatory burden:

- **Consumer (age 13+)** — COPPA does not apply (it covers under-13). Governed by
  a standard consumer privacy policy + a neutral age gate.
- **Edu (under-13, and schools)** — under-13 users reach the product **only**
  through a school, which acts as a **"school official"** under FERPA and consents
  on parents' behalf under COPPA's school-consent path. The school signs a Data
  Processing Agreement; ParaFrames never collects verifiable parental consent
  directly.

The whole model depends on **enforcing the boundary**: the consumer age gate must
be real, and an `edu` credential must mean "a school authenticated this user and a
DPA is in place." See `01_age_gate_and_routing.md`.

## Documents

| File | Purpose | Audience |
|---|---|---|
| `01_age_gate_and_routing.md` | The age gate + edu/consumer routing + under-13 takedown — the hinge the model depends on | Engineering (e2 signup) |
| `02_privacy_policy_draft.md` | Consumer + school-facing privacy policy | Public / schools |
| `03_data_map.md` | Data inventory: what's collected per lane, where it lives, who touches it, retention | Clinic / schools / internal |
| `04_sdpc_ndpa_and_pledge_prep.md` | How to adopt the free SDPC National DPA + sign the Student Privacy Pledge, with your values pre-filled | Schools / counsel |

## Free help this unlocks (no lawyer budget needed)

- **Harvard Law School Cyberlaw Clinic** / **MIT Martin Trust Center, Sandbox, delta v** — pro-bono privacy/edtech legal help; your MIT affiliation is the way in.
- **SDPC National Data Privacy Agreement** — the free, standard vendor↔school DPA (don't draft one).
- **Student Privacy Pledge** (Future of Privacy Forum) — free public commitment / trust signal.
- **FTC "COPPA: Six-Step Compliance Plan for Your Business"** — free, official, plain-English.

Hand a clinic `03_data_map.md` + this README first — it turns their free hours into *review* instead of discovery.
