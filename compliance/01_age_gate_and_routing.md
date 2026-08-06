# Age Gate & Edu/Consumer Routing — Spec

> **⚠️ DRAFT — not legal advice.** This is the enforcement design the two-lane
> compliance model depends on. Review with counsel/clinic.

## Why this is the hinge

The entire compliance posture rests on one fact being true: **no under-13 user is
ever handled on the consumer lane, and every `edu` credential is backed by a real
school + a signed DPA.** If a 12-year-old can sign up as a consumer, COPPA
("actual knowledge") applies and the whole model collapses. This spec is how the
boundary is enforced. Enforcement lives in **e2** (the auth/login service that
issues per-user API keys) — the gateway only *trusts* the `audience` on the key.

## 1. Neutral age gate at consumer sign-up

- **Collect date of birth (or age) with a NEUTRAL prompt** — an open date field,
  not "Are you 13 or older? [Yes]". A yes/no that nudges toward "yes" is not a
  valid COPPA age screen.
- **Ask before collecting anything else.** No name, email, or content until the
  age gate is answered. If the user abandons at the gate, nothing is stored.
- **Do not let the user retry to get a different answer** (no "you must be 13,
  go back and change your birthday"). Persist the first answer for the session.

## 2. Routing decision

```
age = ageFromDOB(dob)
if age >= 13:
    → CONSUMER lane. Issue a consumer key (audience="consumer", tier per plan).
else:  # under 13
    → DO NOT create a consumer account. Do NOT collect data.
    → Show: "ParaFrames for under-13 is available through your school.
             Ask your teacher, or have your school contact us."
    → No edu access is granted here — edu keys come only via the school flow (§3).
```

Under-13 self-signup on consumer is a **hard stop**, not a downgrade. The only
under-13 path is school-provisioned edu.

## 3. Edu credential issuance (school-gated)

An `edu` key must **only** be minted when both hold:

1. **A school authenticated the user** — via the school's SSO / roster
   integration / district-controlled login, not a self-serve form.
2. **A signed DPA is on file for that school/district** (the SDPC NDPA — see
   `04_...`). No DPA → no edu keys for that school.

The gateway's `/v1/auth/validate` handshake already trusts e2 to have
authenticated the user and to assert the audience; **e2 must not issue an `edu`
key outside this school-authenticated + DPA-on-file path.** Document this as an
invariant and test it.

> The FERPA/COPPA "school official" coverage is *only* valid when the school is
> genuinely in the loop. A self-claimed "edu" account with no school behind it is
> not covered — it's an unconsented under-13 account, i.e. a COPPA violation.

## 4. Discovered-under-13 takedown (the "actual knowledge" safety valve)

If ParaFrames ever gains actual knowledge that a **consumer** account belongs to
an under-13 (support ticket, self-report, a message that reveals age):

1. **Suspend** the account immediately.
2. **Delete** the associated data: account record, chat history, uploaded files +
   their Knowledge-Library embeddings (on e2), any moderation logs referencing the
   user. (See `03_data_map.md` for every store.)
3. **Redirect** to the school path.
4. **Log the takedown** (metadata only — that a deletion occurred, when — not the
   content) for your own audit.

Provide a standing, easy channel for anyone to report a suspected under-13 account
(`<FILL: privacy@paraframes...>`).

## 5. Data minimization at the gate

- The consumer age gate stores only what routing requires (an age band or the DOB,
  per your retention choice) — ideally store just a boolean "13+ verified" +
  timestamp, not the raw DOB, once the decision is made.
- Never log the DOB in application/telemetry logs.

## 6. Teen (13–17) note

13+ removes core COPPA, but not everything: proposed COPPA 2.0 (→ under-16) and
state laws (e.g. California Age-Appropriate Design Code) add duties for minors
13–17. Keep the consumer defaults privacy-protective (no behavioral ads, no data
sale, minimal collection) so you're already aligned if those apply. This is a
policy paragraph, not a rebuild.

## Implementation checklist (e2)

- [ ] Neutral DOB gate before any data collection on consumer sign-up.
- [ ] Under-13 → hard stop + school messaging; no data stored.
- [ ] `edu` keys minted only via school-authenticated login **and** DPA-on-file.
- [ ] Invariant test: an `edu` key cannot be obtained through the consumer flow.
- [ ] Takedown runbook + a public report channel.
- [ ] DOB not persisted raw / not logged.
