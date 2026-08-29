# ParaFrames L4 runbook

## Services

| Unit | Port | Purpose |
| --- | ---: | --- |
| `paraclient-gateway` | 8080 | Tailnet API gateway |
| `paraclient-v2` | 8002 | Qwen2.5-7B llama.cpp server |
| `paraclient-circuit` | 8001 | Circuit specialist |
| `paraclient-guard` | 8004 | ShieldGemma moderation |

The gateway listens on the machine's Tailscale address. Model servers listen on
localhost and are not directly reachable from clients.

## Health checks

```bash
systemctl is-active \
  paraclient-gateway paraclient-v2 paraclient-circuit paraclient-guard
curl --fail http://100.122.196.7:8080/healthz
curl --fail http://127.0.0.1:8002/v1/models
curl --fail http://127.0.0.1:8004/health
```

The first request after a model restart may fail while weights are loading.
Confirm service readiness before treating the failure as an incident.

## Logs

```bash
journalctl -u paraclient-gateway --since "30 minutes ago"
journalctl -u paraclient-v2 --since "30 minutes ago"
journalctl -u paraclient-guard --since "30 minutes ago"
```

Moderation decision logs contain metadata by default, not prompt content. Do not
enable content logging for education traffic.

## Restart procedure

Restart the narrowest affected service and wait for readiness:

```bash
sudo systemctl restart paraclient-gateway
curl --retry 10 --retry-delay 2 --retry-connrefused \
  --fail http://100.122.196.7:8080/healthz
```

Restarting the gateway does not revoke API keys or sessions. Restarting a model
server temporarily interrupts requests routed to that model.

## Theme adapter

The slideshow theme LoRA is loaded by `paraclient-v2` at global scale zero.
Theme-selection requests activate adapter ID 0 at request scope. Verify the
global scale after a restart:

```bash
curl --fail http://127.0.0.1:8002/lora-adapters
```

The response must report `scale: 0.0` outside an active selection request.

## Verification after changes

```bash
./venv/bin/python test_accounts.py
./venv/bin/python test_guard_distill.py
./venv/bin/python -m compileall -q \
  accounts.py auth_gateway.py generate_media.py moderation.py guard_distill
```

## Known operational constraints

- The gateway rate limiter is process-local.
- The deployment is a single node without automatic failover.
- ShieldGemma moderation is CPU-bound; long inputs may take tens of seconds.
- The distilled guard must pass shadow evaluation before activation.
- Account and artifact metadata use local SQLite storage.
