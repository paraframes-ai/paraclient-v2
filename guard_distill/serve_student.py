#!/usr/bin/env python3
"""
guard_distill/serve_student.py — CPU inference server for the distilled guard.

    GUARD_STUDENT_DIR=models/guard-student \
    uvicorn guard_distill.serve_student:app --host 127.0.0.1 --port 8005

Loads the distilled encoder once and exposes ONE endpoint that
moderation.DistilledGuardBackend calls:

    POST /classify  {"text": str, "surface": "input"|"output"}
      -> {"scores": {"sexual": p, "violence": p, "hate": p, "dangerous": p}}

One forward pass, all four policy probabilities at once — the whole point of the
distillation (vs ShieldGemma's four prefills). torch stays on CPU with grad off
and threads capped so it doesn't fight the tutor for the box's 4 vCPUs.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from .labels import LABELS, MAX_LEN, student_input
except ImportError:
    from labels import LABELS, MAX_LEN, student_input

MODEL_DIR = os.environ.get("GUARD_STUDENT_DIR", "models/guard-student")
THREADS = int(os.environ.get("GUARD_STUDENT_THREADS", "2"))

app = FastAPI(title="ParaFrames distilled guard")
_state: dict = {"tok": None, "model": None, "labels": LABELS, "err": None}


class Req(BaseModel):
    text: str
    surface: str = "output"


@app.on_event("startup")
def _load():
    try:
        import json
        from pathlib import Path
        import torch
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        torch.set_num_threads(THREADS)
        _state["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
        _state["model"] = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR).eval()
        lp = Path(MODEL_DIR) / "labels.json"
        if lp.exists():
            _state["labels"] = json.loads(lp.read_text())
        print(f"[guard-student] loaded {MODEL_DIR} labels={_state['labels']} "
              f"threads={THREADS}")
    except Exception as e:  # noqa: BLE001
        _state["err"] = str(e)
        print(f"[guard-student] MODEL NOT LOADED ({e}). /classify will 503 so the "
              f"backend fails CLOSED.")


@app.get("/health")
def health():
    return {"ok": _state["model"] is not None, "model_dir": MODEL_DIR,
            "labels": _state["labels"], "error": _state["err"]}


@app.post("/classify")
def classify(req: Req):
    import torch
    from fastapi import HTTPException
    if _state["model"] is None:
        # 503 -> DistilledGuardBackend returns _moderation_unavailable -> BLOCK.
        raise HTTPException(503, f"guard student not loaded: {_state['err']}")
    enc = _state["tok"]([student_input(req.text, req.surface)],
                        return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = _state["model"](**enc).logits[0]
        probs = torch.sigmoid(logits).tolist()
    return {"scores": {lab: round(float(p), 6)
                       for lab, p in zip(_state["labels"], probs)}}
