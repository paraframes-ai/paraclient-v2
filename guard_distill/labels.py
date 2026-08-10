#!/usr/bin/env python3
"""
guard_distill/labels.py — the shared contract for the distilled K-12 safety guard.

One source of truth for the label set + I/O shapes used by every stage of the
distillation pipeline (build_corpus -> label_teacher -> train_student ->
eval_student -> serve_student) AND by moderation.DistilledGuardBackend. The label
order here IS the model's output-head order, so DO NOT reorder it after training.

The four labels match moderation.POLICIES and content_filter.BLOCK_CATEGORIES, so
a distilled student is a drop-in for ShieldGemmaBackend with no gateway changes.
"""
from __future__ import annotations

# Head order == POLICIES order in moderation.py. Frozen once a model is trained.
LABELS = ["sexual", "violence", "hate", "dangerous"]

# Surfaces the guard screens (same as ShieldGemma): student INPUT vs tutor OUTPUT.
SURFACES = ["input", "output"]

# Encoder input: we prepend a surface tag so one model handles both directions
# (ShieldGemma varies the role line by surface; the student learns it from this
# tag). Keep this function identical in training and serving.
def student_input(text: str, surface: str) -> str:
    surface = surface if surface in SURFACES else "output"
    return f"[{surface}] {text}"


# A corpus row (build_corpus output) — unlabeled text to be teacher-scored.
#   {"text": str, "surface": "input"|"output", "source": str}
# A labeled row (label_teacher output) — adds teacher soft labels in LABELS order:
#   {..., "labels": {label: p_yes_float, ...}}
MAX_LEN = 320          # tokens; tutor outputs + questions fit comfortably
