#!/usr/bin/env bash
# Distilled single-pass K-12 safety guard on :8005 (CPU). Replaces the 4-prefill
# ShieldGemma path when the gateway runs with GUARD_BACKEND=distilled. Trained by
# guard_distill/train_student.py on ORCD; see ORCD_TRAINING_PLAN.md Part B.
set -euo pipefail
cd "$(dirname "$0")"
export GUARD_STUDENT_DIR="${GUARD_STUDENT_DIR:-models/guard-student}"
export GUARD_STUDENT_THREADS="${GUARD_STUDENT_THREADS:-2}"
exec ./venv/bin/uvicorn guard_distill.serve_student:app \
  --host 127.0.0.1 --port 8005
