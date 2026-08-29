# Fine-tuning and evaluation

## Shared adapter pipeline

| Setting | Default |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Method | QLoRA, NF4 with double quantization |
| Compute dtype | bfloat16 |
| Epochs | 3 |
| Learning rate | 2e-4, cosine schedule |
| Warmup | 3% |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, and down projections |
| Gradient checkpointing | enabled |

`train_adapter.py` accepts overrides for the base model, sequence length, batch
size, gradient accumulation, epochs, learning rate, rank, and alpha. Each run
writes `training_run.json`, `trainer_state.json`, and `loss_curve.svg` beside the
adapter weights.

## Slideshow theme selector

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Training examples | 2,250 |
| Synthetic holdout | 250 |
| Sequence length | 512 |
| Per-device batch | 8 |
| Gradient accumulation | 4 |
| Epochs | 3 |
| LoRA rank / alpha | 16 / 32 |

The ORCD run completed 213 steps with aggregate training loss 0.2524 and mean
token accuracy approximately 0.973. It scored 250/250 on the template-disjoint
synthetic holdout and 23/25 on a separately written challenge set when served
through llama.cpp on the production CPU runtime.

The challenge-set result is the relevant acceptance signal. The two errors were
audience-versus-topic ambiguities. User-selected themes override the selector,
and selector failures fall back to the content model's schema-valid theme.

## Distilled moderation model

The student is a four-label sequence classifier trained against ShieldGemma soft
labels. Current defaults are four epochs, learning rate 2e-5, batch size 32, and
class-weighted binary cross entropy. The first RoBERTa-family candidate passed a
recall gate on BeaverTails but has insufficient precision and has not completed
traffic-shaped shadow evaluation.

It is not approved for production. `GUARD_BACKEND` must remain `shieldgemma`
until per-policy recall, precision, latency, and shadow-disagreement gates pass.

## Evidence retention

Preserve the following together for each training run:

- Git commit and exact training command
- Dataset manifest, license, checksum, and train/validation split
- Base-model revision and license
- Adapter configuration and weight checksum
- `training_run.json`, `trainer_state.json`, and `loss_curve.svg`
- Evaluation outputs, challenge cases, and serving-runtime benchmark
- Approval decision and rollback target

Training data and model weights contain large or sensitive artifacts and belong
in controlled object storage or a model registry, not the source repository.
