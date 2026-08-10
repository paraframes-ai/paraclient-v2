# ORCD training plan — the 3–4B specialist + a distilled fast guard

**Why this exists.** We can't train on the CPU serving box (`paraclient-l4`,
c3d-standard-4, no GPU). This is the runnable plan to execute on ORCD (MIT's GPU
cluster) and ship the results back as GGUF for CPU serving. It targets the two
things the speed pass on 2026-08-10 identified as the real CPU bottlenecks:

1. **Decode speed on the structured routes** — they run on the 7B v2 model.
   A specialized **3–4B** model decodes ~2× faster on CPU *and* is more accurate
   on structure, so we get faster *and* better at once.
2. **The safety guard is the dominant chat latency** — measured ~9.5s per
   `screen_output` on the CPU box (4 ShieldGemma policy passes, prefill-bound at
   ~68 tok/s; not reducible by caching/`--parallel`/slot-pinning because content
   precedes the policy clause in ShieldGemma's trained template, which we won't
   reorder). The fix is a *distilled* single-pass classifier, which is training
   work — hence it lives here.

Everything below reuses the toolchain already in this repo. Nothing new to build
before training: `build_*_dataset.py` (correct-by-construction synth data),
`train_adapter.py` (QLoRA), `eval_adapter.py`, `merge_lora.py`,
`push_adapters_hf.py`. The **grammar keystone is already wired**: every
structured route serves with `response_format: json_schema` from each module's
`guided_schema()` (`cad_schema`, `circuit_schema`, `spreadsheet_schema`), so a
smaller model is guaranteed valid structure at inference and only has to get the
*content* right — which is exactly what SFT on validated data teaches.

---

## Part A — the 3–4B structured specialist

### Goal
One `Qwen2.5-3B-Instruct` model, QLoRA-fine-tuned on the union of the structured
tasks, that replaces the 7B v2 base for **/v1/sketch, /v1/3d, /v1/circuit,
/v1/spreadsheet** (and optionally the schema-guided half of /v1/generate).

Why 3B and why Qwen2.5:
- **~2× CPU decode** vs the 7B (we measured 7B at ~7 tok/s on 4 vCPUs; a 3B Q4
  should land ~13–15 tok/s), so CAD ~57s → ~30s, spreadsheet ~35s → ~18s, etc.
- **Same family as the 0.5B draft** already on the box
  (`models/qwen2.5-0.5b-gguf`), so speculative decoding becomes viable *if* we
  ever put this model on a GPU/bigger box (it was a measured wash on 4 CPUs — see
  `serve_v2.sh` header — but the draft is family-matched and kept for that path).
- Structure is guaranteed by guided decoding regardless of model size, so the
  only thing size costs us is *content* quality, which SFT on validated data
  recovers.

### Data (already correct-by-construction)
Run the existing builders — each emits `<subject>.jsonl` where every sample was
validated by the SAME gate that serves it (CAD `floorplan_issues`/`repair_sketch`,
circuit `erc`, spreadsheet `spreadsheet_issues`). Train on exactly the system
prompt + schema we serve.

```bash
python build_cad_dataset.py         --n 12000 --out data/cad.jsonl
python build_circuit_dataset.py     --n 10000 --out data/circuit.jsonl
python build_spreadsheet_dataset.py --n 10000 --out data/spreadsheet.jsonl
# concatenate into one multi-task set (shuffle); keep a 5% holdout per task
cat data/cad.jsonl data/circuit.jsonl data/spreadsheet.jsonl \
  | shuf > data/structured.jsonl
```
(The builders already default `--base Qwen/Qwen2.5-3B-Instruct` in their own
usage examples, so the 3B target is the intended path, not a downgrade.)
Target ~30k samples total, balanced across the three tasks. Because the data
is synthetic + gate-validated, we can generate as much as we want; bias the mix
toward the failure modes the gates most often catch today (floorplan overlaps,
LED-without-resistor, circular spreadsheet refs) so the model learns to avoid
them, not just rely on repair.

### Train (QLoRA, one multi-task adapter)
`train_adapter.py` already does QLoRA on Qwen2.5 with `--base` overridable:

```bash
python train_adapter.py \
  --subject structured \
  --data data/structured.jsonl \
  --base Qwen/Qwen2.5-3B-Instruct \
  --epochs 3 --rank 32 --alpha 64 \
  --max-seq-len 2048 --batch-size 8 --grad-accum 4
```
Notes vs the current 7B defaults: bump `--rank` to 32 (a 3B has less capacity to
spare; a fatter adapter helps multi-task), and raise batch size (a 3B QLoRA fits
easily on one A100-40GB / H100). Expect **<2 GPU-hours on one A100** for 3 epochs
over ~30k short samples. This is a single-GPU job — no multi-node, no DeepSpeed.

### Ship back to CPU
```bash
python merge_lora.py --base Qwen/Qwen2.5-3B-Instruct \
  --adapter adapters/structured --out models/structured-merged
# convert + quantize with the llama.cpp already on the box
python llama.cpp/convert_hf_to_gguf.py models/structured-merged \
  --outfile models/structured-f16.gguf
llama.cpp/build/bin/llama-quantize \
  models/structured-f16.gguf models/structured-q4_k_m.gguf Q4_K_M
```
Serve it exactly like `serve_v2.sh` (llama.cpp on a new port, `-t 4`), then point
`CAD_MODEL` / `SPREADSHEET_MODEL` / the circuit server at it. **No gateway code
changes** — the routes already send `response_format` and run the gates. Validate
with `eval_adapter.py` against the per-task holdout before flipping the env vars.

### Acceptance gate before cutover
- Per-task gate-pass rate on the holdout **≥** current 7B v2 (measure both).
- Median route latency drops (target the ~2× decode win end-to-end).
- Roll back = repoint the env vars at v2; keep v2 warm during the bake.

---

## Part B — the distilled fast guard (the real chat-latency fix)

### The problem, quantified (2026-08-10, on the c3d-standard-4 box)
`screen_output` = ~9.5s. It runs 4 ShieldGemma-2B policy passes; each is a full
~170-token prefill (content **+** policy clause) at ~68 tok/s on 4 vCPUs.
`--parallel 4`, sequential+`cache_prompt`, and `id_slot` pinning were all tested
— none help, because the varying content sits *before* the policy clause in
ShieldGemma's trained prompt, so there's no reusable cross-request prefix and we
won't reorder a safety classifier's trained template. So on chat, the guard is
roughly *half* the wall-clock (generation ~20s + guard ~10s). It also
false-positives on partial sentences, which is why streaming can't moderate
per-segment (see `auth_gateway.py` `_raw_chat_stream`).

### Goal
A **single-pass, multi-label** K-12 safety classifier, ~0.5–1B params, distilled
from ShieldGemma-2B, that emits all 4 policy probabilities in **one forward pass
over one copy of the content** — i.e. 1 prefill on a smaller model instead of 4
prefills on a 2B. Expected: **<1s** on the same box (≈10× faster), same
fail-closed semantics, no accuracy regression on our K-12 policies.

> **Implemented.** The full pipeline is built in `guard_distill/` (see its
> README): `build_corpus.py` → `label_teacher.py` → `train_student.py` (ORCD GPU)
> → `eval_student.py` → `serve_student.py` (`:8005`). Cutover is one env var:
> `GUARD_BACKEND=distilled` (`moderation.model_backend()` +
> `DistilledGuardBackend`). The teacher-labeling and serving/fail-closed paths
> are verified on the box; only the GPU training step is left to run on ORCD.

### Method (teacher → student distillation)
1. **Teacher labels.** Run current ShieldGemma-2B over a large corpus of *our
   own* traffic-shaped text: tutor outputs from every route (safe), plus a
   curated hard-negative set per policy (jailbreak attempts, borderline
   self-harm, weapons/drug instructions, harassment) so the student sees the
   decision boundary. Store per-policy P(yes) (soft labels — distillation, not
   just hard labels).
2. **Student.** Fine-tune a small base (candidates: `Qwen2.5-0.5B` we already
   have, or a `MobileBERT`/`DeBERTa-v3-small` encoder) with a **4-head
   multi-label** output (sexual / violence / hate / dangerous), BCE against the
   teacher's soft P(yes). One content prefill → 4 sigmoids.
3. **Calibrate fail-closed.** Pick per-policy thresholds on a validation set so
   the student's recall on the hard negatives **≥** ShieldGemma's (never trade
   child-safety recall for speed). Keep the `_moderation_unavailable` →
   fail-closed path. When in doubt, block.
4. **Serve.** GGUF on llama.cpp (`n_predict:0` + read the head logits) or a tiny
   ONNX/torch server on :8004 replacing ShieldGemma. `moderation.py` changes from
   4 concurrent `_score` calls to 1 `classify` call returning all 4 — a small,
   contained edit behind the existing `ShieldGemmaBackend` interface.

### Why this is the highest-leverage item
It's the only change that speeds up **every moderated turn for every audience**
(edu included — where per-segment streaming is impossible). Faster guard also
*unlocks* child-safe streaming: at <1s/segment, the moderated-streaming design we
shelved (`auth_gateway.py`) becomes viable and edu could finally stream.

### Acceptance gate before cutover
- Per-policy **recall ≥ ShieldGemma-2B** on the hard-negative validation set
  (hard requirement — safety first).
- `screen_output` p50 **<1.5s** on the CPU box.
- Shadow-run in parallel with ShieldGemma for a week (log both verdicts, alert on
  any case where the student is more permissive) before removing the teacher.

---

## Suggested order
1. **Part B first** — biggest, broadest latency win, and it's safety-positive.
2. **Part A** — the structured specialist; ~2× on the structured routes.
3. Revisit speculative decoding *only if* Part A ever lands on a GPU box (it was a
   measured wash on 4 CPUs; the family-matched 0.5B draft is already staged).

## What NOT to change
- Don't reorder ShieldGemma's prompt to chase prefix caching — off-distribution
  for a safety model. The distilled student is the right fix.
- Don't drop ShieldGemma to "run only when the keyword heuristic is unsure" —
  nuanced harms have no keywords; that's exactly what the neural guard is for.
- Don't stream unmoderated output to `consumer`/`edu`. Streaming stays
  `internal`-only until Part B makes moderated streaming fast enough.
