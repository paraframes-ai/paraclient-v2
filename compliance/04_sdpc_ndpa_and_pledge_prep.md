# SDPC National DPA + Student Privacy Pledge — Prep Pack

> **⚠️ DRAFT — not legal advice.** This tells you which free, standard instruments
> to adopt for the edu lane and pre-fills your answers so a school (or clinic) can
> move fast. Do not sign anything until reviewed. `<FILL>` = confirm/complete.

## A. The SDPC National Data Privacy Agreement (NDPA) — the edu DPA

**Don't draft a DPA.** The **Student Data Privacy Consortium** publishes the
**National Data Privacy Agreement (NDPA)** — the standard vendor↔school contract
used across US districts. You adopt it; schools recognize it; the FERPA
"school official" designation lives inside it.

- Get the current NDPA (Standard Version) from the SDPC (`privacyframework.org` /
  SDPC resource site). `<FILL: confirm latest version.>`
- The core terms (school owns the education record; vendor is a "school official"
  under FERPA's direct-control exception; no secondary use; no sale; deletion on
  request; breach notice; sub-processor limits) come pre-written. **Your job is the
  exhibits**, pre-filled below.
- Consider joining SDPC as a vendor and listing your signed agreements — districts
  search there.

### NDPA Exhibit "E" — Data Elements you collect (pre-filled from the data map)

| Category | Collected? | Notes |
|---|---|---|
| Name | `<FILL: if provided>` | Account |
| Email / login ID | Yes | Auth |
| Student work / content (chat, uploads, note images) | Yes | Core feature; processed on-prem |
| Grades / assessment | No | `<FILL: confirm>` |
| Behavioral / advertising data | No | Not collected |
| Biometric | No | Note images are transcribed, not used as biometrics |
| Geolocation | No | |

### Exhibit — Sub-processors (edu lane)

- **None.** Student (edu) data is processed **entirely on ParaFrames' on-premise
  infrastructure**; no third-party AI provider or external web service receives
  edu data. *(This is a strong selling point to districts — state it plainly.)*
- (For completeness, external providers Google Vertex / Anthropic are used **only
  on the paid consumer 13+ lane**, never edu.)

### Exhibit — Security practices (pre-filled)

- On-premise processing; private WireGuard-encrypted network, not publicly exposed.
- Per-user authentication; audience-scoped access (edu isolated from external routes).
- Automated on-prem content moderation (child-safety classifier).
- `<FILL: at-rest disk encryption; access controls / audit log — see remediation
  items in 03_data_map.md; breach-response contact.>`

### Deletion / data return

- On school or user request, delete the student's content and Knowledge-Library
  embeddings (on e2). `<FILL: implement the deletion endpoint; state turnaround.>`

## B. Student Privacy Pledge (free trust signal)

The **Student Privacy Pledge** (Future of Privacy Forum + SIIA) is a free, public
set of commitments for edtech vendors. Signing is a recognized signal to schools
and costs nothing.

- Review the current Pledge commitments at the Student Privacy Pledge site.
- Confirm you meet them (you already do most): not selling student data, not
  behavioral-advertising to students, not building non-educational profiles, using
  data only for authorized educational purposes, supporting access/correction,
  retention limits, security, and transparency.
- Sign and list ParaFrames. `<FILL: submit the org.>`

## C. Other free/official references

- **FTC — "COPPA: Six-Step Compliance Plan for Your Business"** (official, free).
  Use it to sanity-check the consumer age gate + the under-13 handling.
- **FTC COPPA FAQ** — the "schools can consent" section supports your edu model.
- **Future of Privacy Forum — Protecting Student Privacy** — free guides.
- **Common Sense Privacy Program** — a district-facing evaluation you can pursue
  later for a public privacy rating.

## D. Do-this-order checklist

1. Fill the `<FILL>`s across these docs (mostly your legal name, contact, jurisdiction).
2. Fix the data-map remediation items (edu content-log redaction; deletion path).
3. Adopt the **SDPC NDPA** with the exhibits above; get one friendly pilot district to sign.
4. Sign the **Student Privacy Pledge**.
5. Take this whole `compliance/` folder to the **Harvard Cyberlaw Clinic / MIT**
   entrepreneurship legal resources for a free review before any real student data.
