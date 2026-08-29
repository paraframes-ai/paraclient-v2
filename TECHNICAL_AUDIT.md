# Technical audit brief

## Scope

This repository contains the Python model gateway, account store, deterministic
validators, media generation, moderation, and model-training utilities. The
client application and the e2 authentication service are separate repositories.
There is no Go service in this checkout.

## Configuration and credentials

- Runtime secrets are supplied through environment variables.
- `.env` and `.env.*` are excluded from version control.
- `.env.example` documents configuration names without values.
- Account API keys and passwords are stored as hashes in SQLite.
- `gateway_keys.json` is excluded and retained only for internal service keys.
- Model weights, generated artifacts, training data, and account databases are
  excluded from Git.

Audit command:

```bash
git check-ignore -v .env .env.dev accounts.db gateway_keys.json data out models
git grep -n -E 'BEGIN .* PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]+'
```

The second command should return no credential material. Secret rotation and
repository-history scanning should also be enforced in CI before launch.

## Privacy boundary

Education traffic cannot select normal-assistant or external-model routes. The
gateway screens input and output using models hosted on the L4 node. Prompt
content is not persisted by the gateway, and moderation logs contain decision
metadata and text length by default.

The preferred live demonstration is the actual routing boundary:

1. Submit an education request to an on-premise tutoring route.
2. Show the successful request and local model-server log.
3. Attempt an external-model route with the same education credential.
4. Show the policy rejection before any external request is created.
5. Show that `logs/content_filter.jsonl` contains no prompt text.

Do not print raw student payloads during a demonstration. A redaction example
using fabricated text is not evidence that production routing is safe.

## Concurrency and cancellation

FastAPI handles concurrent requests, but several model SDK calls are synchronous.
The application currently uses explicit downstream timeouts for moderation, web
fetches, and deletion callbacks. Full propagation of client disconnects into
all model backends is not implemented. This is open work; do not represent it as
complete during an audit.

The current deployment also uses a process-local rate limiter and one SQLite
account database. Horizontal scaling requires shared rate-limit state, a managed
transactional store, idempotency controls, and coordinated artifact storage.

## Model weights and evaluation

Adapters are trained separately from the base model and stored outside Git. The
training CLI records its base model, dataset path, optimizer settings, LoRA
configuration, final metrics, log history, and a loss-curve SVG with each run.
See `FINE_TUNING.md` for the current configurations and acceptance gates.

## Live verification

```bash
./venv/bin/python test_accounts.py
./venv/bin/python test_guard_distill.py
./venv/bin/python -m compileall -q \
  accounts.py auth_gateway.py generate_media.py moderation.py guard_distill
systemctl is-active \
  paraclient-gateway paraclient-v2 paraclient-circuit paraclient-guard
curl --fail http://100.122.196.7:8080/healthz
```

## Open production work

- Shared rate limiting and production account storage
- High availability, monitoring, alerting, backups, and restore drills
- End-to-end cancellation and bounded request queues
- CI secret scanning, linting, dependency scanning, and test execution
- Distilled-guard shadow evaluation before cutover
- Independent privacy and security review
