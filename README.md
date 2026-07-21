# ParaFrames Tutor — Adapter Training Pipeline

One shared base (`Qwen/Qwen2.5-7B-Instruct`, Apache-2.0) + one LoRA adapter per
subject. Each adapter learns **two modes** from the data:

- **socratic** — withholds the answer, guides with questions.
- **graduated_hint** — guides first, then escalates to hints/worked steps so a
  stuck student can finish (homework mode).

Mode is chosen at inference via the system prompt; both are in the training data
so the adapter can do either on command.

## Run order

```bash
# 0. install (on your L4 VM, in a venv)
pip install -r requirements.txt

# 1. build data for a subject -> data/<subject>.jsonl
python scripts/build_dataset.py --subject math          # full set
python scripts/build_dataset.py --subject math --limit 200   # quick test

# 2. train the adapter (QLoRA, fits a single 24GB L4) -> adapters/<subject>/
python scripts/train_adapter.py --subject math --data data/math.jsonl

# 3. serve with vLLM (separate terminal)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules math=adapters/math \
    --max-model-len 4096 --gpu-memory-utilization 0.9

# 4. behavioral eval (does it withhold? does it escalate?)
python scripts/eval_adapter.py --adapter math
```

Serve several subjects at once by listing more `--lora-modules` and raising
`--max-loras`. Select the subject per request via the `model` field.

## Per-subject data status

| Subject        | Source                              | Status                         |
|----------------|-------------------------------------|--------------------------------|
| math           | GSM8K-socratic (MIT)                | ✅ ready, builds today          |
| civics         | synthesized from knowledge Q&A      | ⛔ needs synthesis + review     |
| language_arts  | synthesized from knowledge Q&A      | ⛔ needs synthesis + review     |
| general        | mix / synthesized                   | ⛔ needs synthesis + review     |

For the non-math subjects, produce dialogues via the LLM-propose / human-review
loop, save them as `data/raw/<subject>.synth.jsonl` in the **same schema**
`build_dataset.py` emits, then run `build_dataset.py --subject <subject>` to
validate + merge.

## Licensing gate (read before training a sellable model)

- **GSM8K-socratic** — MIT. Clear for commercial use. ✅
- **MathDial** — confirm the license before adding it. Not wired in yet.
- **Eedi tutoring dialogues** — NON-COMMERCIAL. Do not put in this pipeline. ⛔

## Honest constraints

- **L4 training is slow.** QLoRA of 7B fits in 24GB but expect hours per epoch.
  Adapters are portable — train on a bigger GPU later if you want speed; serve
  on the L4 either way.
- **If you OOM:** drop `--max-seq-len` to 1024, keep `--batch-size 1`, raise
  `--grad-accum`.
- **bitsandbytes + LoRA in vLLM** can be version-sensitive. If serving misbehaves,
  serve the base in bf16 (a 7B fits ~15GB on L4) with the adapter, or use an
  AWQ base. Pin versions and test the exact combo early.
- **The eval is a smoke test, not a safety guarantee.** The answer-leak check is
  a heuristic. For a child-facing product, human-review real transcripts before
  shipping, and keep the separate input/output safety layer we discussed OUTSIDE
  the tutoring model — do not rely on the adapter alone to keep students safe.
```
