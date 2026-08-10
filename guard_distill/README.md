# guard_distill — the distilled fast safety guard (ORCD plan Part B)

Turns the ~9.5s ShieldGemma guard (4 full-content prefills on a 2B) into a
**single-pass, sub-second** multi-label classifier, without weakening the
fail-closed K-12 safety guarantee. This is the biggest chat-latency win on the
CPU box (the guard is ~half of chat wall-clock) and it also unlocks child-safe
streaming. See `../ORCD_TRAINING_PLAN.md` Part B for the why.

**Drop-in by design:** the student satisfies the same `classify(text, surface) ->
set[str]` contract as ShieldGemma, so cutover is one env var —
`GUARD_BACKEND=distilled` (see `moderation.model_backend()`). No gateway or
`content_filter` changes.

## Pipeline

| stage | script | where | notes |
|-------|--------|-------|-------|
| 1. corpus | `build_corpus.py` | box | procedural safe text + `--from-jsonl` real logs / red-team set |
| 2. teacher labels | `label_teacher.py` | box (slow) / **ORCD** (fast) | ShieldGemma soft P(yes) per policy |
| 3. train student | `train_student.py` | **ORCD GPU** | BCE-distill a tiny encoder, 4-label head |
| 4. accept + thresholds | `eval_student.py` | either | fail-closed gate: recall ≥ teacher |
| 5. serve | `serve_student.py` + `../serve_guard_student.sh` | box | CPU, `:8005`, one forward pass |

```bash
# 1. corpus  (add real coverage on ORCD: logged outputs + a public red-team set)
python -m guard_distill.build_corpus --n-safe 4000 \
    --from-jsonl data/tutor_logs.jsonl --from-jsonl data/redteam.jsonl \
    --out data/guard_corpus.jsonl

# 2. teacher soft-labels  (run on ORCD GPU for the full set; --limit to smoke-test on the box)
python -m guard_distill.label_teacher --in data/guard_corpus.jsonl \
    --out data/guard_labeled.jsonl --workers 8

# 3. distill  (ORCD GPU; ~minutes on one card)
python -m guard_distill.train_student --data data/guard_labeled.jsonl \
    --base distilbert-base-uncased --out models/guard-student

# 4. fail-closed acceptance gate + per-policy thresholds
python -m guard_distill.eval_student --data data/guard_labeled.jsonl \
    --model models/guard-student --target-recall 0.99

# 5. serve on the box, then flip the gateway
GUARD_STUDENT_DIR=models/guard-student ./serve_guard_student.sh   # :8005
#   set GUARD_BACKEND=distilled in the gateway env and restart
```

## Safety rules (non-negotiable)
- **Recall ≥ ShieldGemma on every policy** before cutover (`eval_student.py`
  gate). A false-allow for a child is far worse than a false-block.
- **Shadow-run** the student beside ShieldGemma (log both verdicts, alert on any
  case where the student is more permissive) before removing the teacher.
- Fail-closed everywhere: an unloaded/unreachable student → `:8005` 503 →
  `DistilledGuardBackend` returns `_moderation_unavailable` → BLOCK.
- The `HeuristicBackend` self-harm/abuse **escalation** signals stay in the
  composite regardless — the student only replaces the neural *block-harm* layer.
- Do **not** hand-author graphic harm in this repo. Real unsafe coverage comes
  from `--from-jsonl` public benchmarks on ORCD.
