# ParaFrames

ParaFrames is an on-premise AI workspace for tutoring and structured content
creation. The serving stack is designed for private deployment on efficient
CPU hardware: prompts, student data, retrieval, moderation, and generation can
remain inside the customer's network boundary.

The model and training program is developed under the Munivar name.

## System overview

The L4 service is the authenticated API boundary. It routes requests to local
llama.cpp model servers, applies audience and plan policy, meters usage, and
moderates model input and output. A separate authentication service hosts the
login BFF and Knowledge Library. Clients connect to both services through a
private Tailscale network.

| Component | Responsibility |
| --- | --- |
| `auth_gateway.py` | Authentication, authorization, metering, safety, and model routing |
| `accounts.py` | Accounts, sessions, plans, generated artifacts, and sharing |
| `generate_media.py` | Schema-guided document and presentation generation |
| `cad_schema.py` | Deterministic geometry validation |
| `circuit_schema.py` | Circuit normalization and electrical-rule checks |
| `spreadsheet_schema.py` | Spreadsheet normalization and formula evaluation |
| `moderation.py` | On-premise ShieldGemma moderation backend |
| `guard_distill/` | Training and evaluation for the low-latency guard |

The API exposes tutoring, document and presentation generation, CAD, circuits,
spreadsheets, civics, notes transcription, embeddings, and optional external
model routes. Education traffic is restricted to on-premise models.

## Design principles

- Private by architecture: education data is processed locally.
- Correct by construction: structured outputs use constrained schemas and
  deterministic validators.
- Fail closed: unavailable moderation blocks protected workflows.
- Capability per watt: the default serving path targets quantized models on
  commodity CPU infrastructure.
- Stable client contract: applications integrate through one versioned gateway.

## Local setup

Python 3.11 or newer is recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Runtime configuration belongs in `.env.dev`, which is excluded from version
control. Model weights, account databases, generated files, and training data
are also excluded.

Start the local model services before the gateway:

```bash
sudo systemctl start paraclient-v2 paraclient-circuit paraclient-guard
sudo systemctl start paraclient-gateway
```

The deployed gateway binds to its Tailscale address rather than localhost.

## Verification

```bash
./venv/bin/python test_accounts.py
./venv/bin/python test_guard_distill.py
./venv/bin/python -m py_compile \
  accounts.py auth_gateway.py generate_media.py moderation.py
```

`test_accounts.py` covers age gating, account lifecycle, plan policy, usage,
sessions, OAuth provisioning, generated artifacts, and sharing. The guard tests
cover fail-closed behavior and response validation.

## Training

Dataset builders emit JSONL into the ignored `data/` directory. Adapter jobs run
on ORCD through Slurm. For example:

```bash
python build_slideshow_theme_dataset.py
sbatch train_slideshow_theme.sbatch
sbatch eval_slideshow_theme.sbatch
```

Adapters are evaluated before conversion to GGUF. The slideshow theme adapter is
loaded by llama.cpp at scale zero and enabled only for theme-selection requests.

## Operational status

The current deployment is an internal, single-node environment. Production work
still includes high availability, centralized monitoring, credential rotation,
billing integration, and completion of the distilled-guard shadow evaluation.
See `RUNBOOK.md` and the compliance directory for operational and policy details.
