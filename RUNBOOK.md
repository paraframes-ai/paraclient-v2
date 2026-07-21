# ParaFrames Tutor — Runbook

One base model (`Qwen/Qwen2.5-7B-Instruct`, Apache-2.0) + one LoRA adapter per
subject. Each adapter learns two MODES from its data:
  - socratic        : withholds the answer, guides with questions
  - graduated_hint  : guides first, then escalates to hints/worked steps
Mode is chosen at inference via the system prompt.

## Files
  scripts/build_dataset.py        Build/validate/merge training JSONL per subject
  scripts/train_adapter.py        QLoRA fine-tune one adapter (fits a 24GB L4)
  scripts/eval_adapter.py         Behavioral eval (withholds? escalates?)
  scripts/synthesize_dialogues.py Seed content -> dialogues (Civics + LA)
  scripts/review_dialogues.py     Human review gate (mandatory for Civics)
  requirements.txt                Pinned stack

## Subject status
  math          GSM8K-socratic (MIT, human-written)  -> READY, no synthesis
  language_arts FairytaleQA seed -> synthesize        -> needs seed + generator
  civics        NAEP/CivEd seed  -> synthesize        -> needs seed + generator + review
  general       synthesize                            -> later

===========================================================================
## TRACK A — MATH (do this first; it proves the whole pipeline)
===========================================================================
On your L4 VM:

  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt

  # 1. build data (small trial first)
  python scripts/build_dataset.py --subject math --limit 200
  python scripts/build_dataset.py --subject math          # full

  # 2. train (slow on L4 — hours; that's expected)
  python scripts/train_adapter.py --subject math --data data/math.jsonl
  #    OOM? add:  --max-seq-len 1024   (keep batch-size 1, raise --grad-accum)

  # 3. serve (separate terminal)
  vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
      --lora-modules math=adapters/math \
      --max-model-len 4096 --gpu-memory-utilization 0.9

  # 4. eval behavior
  python scripts/eval_adapter.py --adapter math
  #    then READ some transcripts yourself. The eval is a smoke test, not proof.

Math done end-to-end = pipeline validated. Only then move on.

===========================================================================
## TRACK B — LANGUAGE ARTS, then CIVICS (data work, not ML work)
===========================================================================
Prereq you own: a generator model whose OUTPUT license permits training a
commercial model, served at an OpenAI-compatible URL.

  # 1. get licensed seed -> JSONL with documented keys:
  #    data/seed/fairytaleqa.jsonl   (LA;  CC BY 4.0 — attribute)
  #    data/seed/naep_civics.jsonl   (Civics; confirm item reuse terms)

  # 2. synthesize (TRIAL of 10 first — check quality before scaling)
  python scripts/synthesize_dialogues.py --subject language_arts \
      --seed data/seed/fairytaleqa.jsonl \
      --generator-url <YOUR_CLEARED_ENDPOINT> \
      --generator-model <YOUR_MODEL> \
      --out data/raw/language_arts.synth.jsonl --limit 10

  # 3. human-review (full read first time; civics = mandatory, every row)
  python scripts/review_dialogues.py --in data/raw/language_arts.synth.jsonl
  #    -> writes data/raw/language_arts.synth.reviewed.jsonl (accepted only)
  #    rename to language_arts.synth.jsonl so build_dataset picks it up.

  # 4. validate + merge, then train + serve + eval exactly like math
  python scripts/build_dataset.py --subject language_arts
  python scripts/train_adapter.py --subject language_arts --data data/language_arts.jsonl

Repeat for civics (heaviest review). Order: LA -> Civics.

===========================================================================
## SERVING ALL SUBJECTS AT ONCE
===========================================================================
  vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-loras 4 \
      --lora-modules math=adapters/math language_arts=adapters/language_arts \
                     civics=adapters/civics general=adapters/general \
      --max-model-len 4096 --gpu-memory-utilization 0.9
Select subject per request via the OpenAI `model` field.

===========================================================================
## OPEN ITEMS YOU OWN (not code)
===========================================================================
  [ ] Confirm generator model output-license allows commercial training use
  [ ] Confirm FairytaleQA license on the copy you download (expect CC BY 4.0)
  [ ] Confirm NAEP/CivEd item reuse terms (govt ≠ automatically clear for items)
  [ ] Safety layer OUTSIDE the model: input/output filtering, mental-health
      flagging, teacher guardrail toggles — the adapter is NOT your safety net
  [ ] Privacy/DPA + COPPA (consumer) / FERPA (schools) — legal review before sale
