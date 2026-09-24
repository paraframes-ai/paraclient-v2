# ParaClient Muse — CSAIL research

This branch ports the subject-adapter tutoring pipeline to
`meta-models/Muse-Glimmer-30B`, while retaining Qwen2.5 QLoRA. Each subject adapter
learns **socratic** and **graduated_hint** behavior, selected by its system prompt.
Socratic mode withholds the answer; graduated hints can provide worked steps after
the student remains stuck. This is a research pipeline, not a student deployment.

The authorized smoke run **2197693 completed successfully** in 1m51s: 404
examples (202 per mode), SIGUSR1 save at step 2, automatic resume to step 6,
62.8 GB peak allocated CUDA memory, and a 419 MB adapter. All 24 CPU regression
tests passed. Compact evidence is in `artifacts/smoke_result.json` and
[artifacts/VALIDATION.md](artifacts/VALIDATION.md). These are pipeline checks,
not a measurement of tutoring quality; live behavioral evaluation remains pending.

## Remote workspace and environment

All project work is on CSAIL NFS, never the laptop or AFS home:

- `NFS_ROOT=/data/scratch/ashwin`
- Clone: `/data/scratch/ashwin/paraclient-muse`, branch `muse-glimmer`
- ARM environment: `/data/scratch/ashwin/envs/muse-arm`
- Hugging Face cache: `/data/scratch/ashwin/hf`
- Slurm logs: `/data/scratch/ashwin/logs`

GH200 benchmark job 2197295 measured a 2,000 MiB direct write at **577 MB/s**
on `/data/scratch`, versus **166 MB/s** on `/data/scratch-oc40`. Scratch is **not
backed up**. Commit code in the remote clone and copy compact adapter/evaluation
artifacts to durable storage separately. Do not put credentials in files.

The login node is for editing and submitting only. Setup, model loading, dataset
preparation, and training belong in compute allocations. All remote commands use
`ssh -o BatchMode=yes slurm-agent`; stop on authentication prompts or a stalled SSH
connection. The scripts use account `quanta`, QoS `quanta-main`, and
`quanta-gh200` (one aarch64 GH200 per task).

```bash
ssh -o BatchMode=yes slurm-agent 'cd /data/scratch/ashwin/paraclient-muse && sbatch --job-name=muse-setup --account=quanta --partition=quanta-gh200 --qos=quanta-main --gpus=1 --cpus-per-task=16 --mem=200000M --time=01:00:00 --chdir=/data/scratch/ashwin/paraclient-muse --output=/data/scratch/ashwin/logs/%x_%j.out slurm/setup_env_gh200.sh'
ssh -o BatchMode=yes slurm-agent 'cd /data/scratch/ashwin/paraclient-muse && sbatch slurm/inspect_muse.sbatch'
```

Setup installs PyTorch/torchvision aarch64 wheels from the **cu128** index, then
Transformers, PEFT, TRL, Accelerate, Datasets, and testing/API dependencies. It
verifies `torch.cuda.is_available()`, bf16 support, and a CUDA tensor operation.
Resolved versions are in `slurm/requirements-gh200.lock`; hardware/results are in
`artifacts/environment.json`. It bootstraps pip inside the venv if the system
Python lacks `ensurepip`; no root install is needed.

NVIDIA cuSPARSELt 0.7.1 uses the nonstandard `manylinux2014_sbsa` tag. The setup
script accepts **only that exact pip-check warning**, verifies the library is
AArch64 ELF, and still requires successful CUDA checks. Other conflicts fail.
If an ARM package fails to build/load, retain the error log and use an ARM NVIDIA
NGC PyTorch image through **Apptainer `--nv`**, binding the same NFS root and
setting `HF_HOME` inside the container. Select and verify a compatible image on
GH200; no container installation or image has been tested in this branch.

## Verified model integration

`inspect_muse.py` downloads public metadata/weights, instantiates
`AutoModelForImageTextToText`, checks a finite text-only forward pass, and exercises
`tokenizer.apply_chat_template` with reasoning, tools, and tool results. The
inspection report pins the HF commit for dataset building, training, and serving.
No `trust_remote_code` or Hugging Face token is required for the public model.
If access returns 401/403, stop and ask for access; never write a token.

- Config architecture: `MuseGlimmerForConditionalGeneration`. Runtime inspection
  counted 29,776,626,688 total parameters and 1,852,639,744 in the vision tower
  (the model card gives approximate sizes). There are 416 selected language
  linear modules. The verified BF16 forward peaked at 59.6 GB allocated CUDA memory.
- Language modules: `model.language_model.layers.<n>`. LoRA selects full paths to
  attention `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, and MLP
  `gate_proj`, `up_proj`, `down_proj`.
- Vision modules: `model.vision_tower`, `model.vision_adapter`, and
  `model.vision_projection`. All remain frozen. Suffix-only targets are unsafe
  here because vision and language share names such as `q_proj`.
- Native chat protocol: reasoning is `assistant to=self`, student text is
  `assistant to=user`, and function calls use `assistant to=<function>` followed
  by `<atem:function_calls>` / `<atem:invoke>` / `<atem:parameter>` blocks.
  Tool results use `<tool_output name="...">`. Message endings are `<|eom|>`
  and `<|eot|>`; the generation prefix ends at `<|start|>assistant`.
- HF message dictionaries use `reasoning_content`, `tool_calls`, and tool-role
  messages with `tool_call_id`/`name`. **HF tool arguments must be dictionaries**;
  OpenAI API tool arguments are JSON strings. Do not interchange the two.
- The native template defaults to high reasoning strength. Dataset formatting
  uses low by default; inference can request `Reasoning strength: low.` in the
  system prompt. The two tutoring modes remain controlled by the system prompt.

## Data and training

The original JSONL keys remain `subject`, `mode`, and `messages`. The system turn
comes first. Plain dialogues alternate user/assistant and end on assistant.
Each GSM8K-socratic source problem produces one example of each mode; thus
`--limit 200` produces 400 dialogues before optional tool examples. Synthesized
subjects use `data/raw/<subject>.synth.jsonl` after human review.

`build_dataset.py --base qwen|muse` uses the selected tokenizer's native template.
It adds `text`, `base_model`, and `base_revision` without replacing messages.
Training re-renders messages and checks these fields to reject mismatched data.
Existing output files are protected: choose a new `--out-dir` for a rebuild.

For Muse, add a top-level OpenAI-style `tools` schema list and native structured
messages. Merge reviewed files with repeated `--tool-data PATH` options or place
them at `data/raw/<subject>.tools.jsonl`. Validation rejects unknown tools,
invalid arguments, duplicate IDs, unmatched results, and incomplete tool turns.
`examples/math.tools.jsonl` contains four small illustrative dialogues covering
both tutoring modes and tool errors; review/expand these before a full study.
They are fixtures, not a representative tool-use corpus.

Inside a GH200 allocation after sourcing `slurm/env.sh`:

```bash
python build_dataset.py --base muse --subject math --limit 200 \
  --tool-data examples/math.tools.jsonl --out-dir data/my-trial
python train_adapter.py --base muse --subject math --data data/my-trial/math.jsonl \
  --out-dir adapters/my-trial --max-steps 6 --max-seq-len 512
```

Muse defaults: BF16 **without quantization**, LoRA r=16 / alpha=32 / dropout=0.05,
SDPA, gradient checkpointing, batch size 1, accumulation 16, sequence length 2048,
and a checkpoint every 50 optimizer steps. Only language LoRA parameters are
trainable; runtime assertions enforce this. PEFT may minimize long target lists
during injection; the trainer restores the verified full paths and pinned base
revision in saved adapter metadata. Training uses Transformers Trainer
with explicit chat rendering and attention-mask-based label padding; TRL is
installed but its version-dependent SFT formatting API is not required.

The newest **complete** checkpoint resumes automatically, including optimizer,
scheduler, RNG, and trainer state. SIGUSR1 requests a save and stop after the next
optimizer step; Python exits 75 only after saving. The Slurm wrapper then
requeues that array task. Keep the 10-minute warning window sufficient for an
optimizer step plus NFS save. At most two checkpoints are retained by default.

`slurm/smoke_muse.sbatch` runs one 45-minute math allocation: `--limit 200`, four
tool fixtures, six optimizer steps, and a real SIGUSR1 save/resume exercise. Its
outputs have job-specific directories and cannot be confused with full adapters.
Submit another smoke job only with explicit authorization if the one allowed run
has already been used.

`slurm/train_muse.sbatch` is the prepared **full-training** array, with indices
`0=math`, `1=language_arts`, `2=civics`, `3=general`. It requests one GH200 each,
`--requeue`, `--signal=USR1@600`, a four-day limit, and
`logs/%x_%A_%a.out` under NFS_ROOT. **It has not been submitted. Obtain explicit
approval before submitting it or any additional GPU workload.** Missing reviewed
subject data causes that task to fail clearly.

| Subject | Source/readiness |
| --- | --- |
| math | GSM8K-socratic; builder ready |
| language_arts | reviewed synthesis required |
| civics | reviewed synthesis required; retain mandatory review gate |
| general | reviewed synthesis required |

Keep source licenses, attribution, and provenance with research data. GSM8K's
source metadata identifies MIT licensing; additional datasets and generator
outputs need their own checks. Research status does not remove those obligations.

## Qwen compatibility

The default builder is still Qwen. Existing Qwen HF IDs, including the circuit/CAD
builders' custom base IDs, remain accepted by `train_adapter.py`. Qwen retains
NF4 QLoRA by default and requires bitsandbytes in its original compatible GPU
environment. Keep `requirements.txt` for that legacy stack; do not install it over
`muse-arm`. Muse's setup intentionally does not require bitsandbytes or flash-attn.
`merge_lora.py` remains the legacy Qwen CPU merge utility; do not use it for Muse.

## Serving and evaluation

`serve_tardy.sh` is for later use on **tardy**, with a separate **x86_64** vLLM
environment. The ARM venv is not portable there. The script fixes
`CUDA_VISIBLE_DEVICES=2,3`, tensor parallelism 2, BF16, four named LoRA modules,
and localhost binding. It checks the installed architecture's `SupportsLoRA`,
HF weight mapper, Muse reasoning/tool parsers, and each adapter's base/rank/targets
before starting. It fails clearly on missing support or missing subject adapters.
Only the script has been prepared; tardy has not been contacted or validated.

Copy final adapter files (`adapter_model.safetensors`, `adapter_config.json`,
tokenizer files), the inspection report, and compact evaluation summaries for
archival/serving. Checkpoint directories contain resumable optimizer state and
are larger; do not include them in a serving copy. Do not copy base weights into
each adapter directory. `HF_HOME` and `ADAPTER_ROOT` can be set on the serving host.

Once an authorized server exists:

```bash
python eval_adapter.py --base muse --adapter math --base-url http://127.0.0.1:8000/v1 \
  --json-out artifacts/math-eval.json
```

The eval retains answer-leak and graduated-hint escalation checks and adds schema
validity, appropriate tool use, recovery from a deterministic tool error, refusal
to pass along a final answer embedded in tool output, and reasoning isolation.
Tool results are fixtures; no real tools execute. JSON output stores compact
pass/fail summaries, not private reasoning transcripts. Failures return nonzero.
Live behavioral evaluation requires an authorized running backend and remains a
heuristic; follow it with human transcript review.

**The safety layer stays independent of the adapter.** `safe_tutor_proxy.py`
first screens input, calls the tutor, extracts only `to=user` content through
`tutor_channels.py`, then invokes `content_filter.py` on that text. Private
reasoning and tool payloads never enter output-filter logs or student responses.
Malformed/truncated channels fail closed. Already-parsed API `content` is the
student channel; separate `reasoning_content`/`tool_calls` fields are discarded.
This framing boundary cannot detect arbitrary unmarked reasoning copied into
plain text; behavioral review is still necessary. The proxy does not execute
student-supplied tools; application tool orchestration remains separate work.
Keep the vLLM backend private and expose the proxy. The heuristic content filter
is a research scaffold, not sufficient protection for real student deployment.

Tests include real FastAPI proxy requests with mocked model responses and assert
that the content filter receives only student text, plus tool-schema, checkpoint,
Qwen-path, output-blocking, and escalation regressions:

```bash
python -m pytest -q tests
```
