#!/usr/bin/env python3
"""
review_service.py — shared civics review console for a small team.

FastAPI + SQLite. Three reviewers work one live queue: decisions are stored
centrally, everyone sees live progress, and there is no double-work. Time-varying
items (the highest-risk civics category) require 2-of-3 approval; everything else
needs one. Approved items export to a JSONL that feeds
build_civics_dataset.py --promote.

Run (tailnet-only):
    uvicorn review_service:app --host <tailnet-ip> --port 8090
Env:
    CANDIDATES=data/civics_candidates.jsonl   source candidates
    REVIEW_DB=review.db                        sqlite decision store
    DOUBLE_REVIEW=time_varying                 'time_varying' | 'all' | 'none'
"""
import json
import os
import sqlite3
import time
from contextlib import closing

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

CANDIDATES_PATH = os.environ.get("CANDIDATES", "data/civics_candidates.jsonl")
DB_PATH = os.environ.get("REVIEW_DB", "review.db")
DOUBLE_REVIEW = os.environ.get("DOUBLE_REVIEW", "time_varying")  # time_varying|all|none

SYS = ""            # civics tutor system prompt, filled from civics_schema at startup
ITEMS = []          # ordered candidate list (dicts)
BY_KEY = {}         # (source, grade_band) -> item


def key(source, band):
    return f"{source}#{band}"


def approvals_needed(item):
    if DOUBLE_REVIEW == "all":
        return 2
    if DOUBLE_REVIEW == "none":
        return 1
    return 2 if item["time_varying"] else 1


def load_candidates():
    global SYS, ITEMS, BY_KEY
    try:
        import civics_schema as cs
        SYS = cs.CIVICS_SYS
    except Exception:
        SYS = ""
    ITEMS, BY_KEY = [], {}
    with open(CANDIDATES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = {m["role"]: m["content"] for m in r["messages"]}
            it = {
                "source": r["source"], "grade_band": r["grade_band"],
                "strand": r["strand"], "time_varying": bool(r["time_varying"]),
                "license": r["license"], "attribution": r.get("attribution", ""),
                "q": msgs.get("user", ""), "a": msgs.get("assistant", ""),
            }
            ITEMS.append(it)
            BY_KEY[key(it["source"], it["grade_band"])] = it


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with closing(db()) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS decisions(
            source TEXT, grade_band TEXT, reviewer TEXT,
            status TEXT, answer TEXT, strand TEXT, band TEXT, tv INTEGER,
            note TEXT, ts REAL,
            PRIMARY KEY(source, grade_band, reviewer))""")
        con.commit()


app = FastAPI(title="Civics Review")


@app.on_event("startup")
def _startup():
    load_candidates()
    init_db()


def item_state(it, reviewer):
    k = key(it["source"], it["grade_band"])
    with closing(db()) as con:
        rows = con.execute(
            "SELECT * FROM decisions WHERE source=? AND grade_band=?",
            (it["source"], it["grade_band"])).fetchall()
    approvals = [r for r in rows if r["status"] == "approved"]
    rejections = [r for r in rows if r["status"] == "rejected"]
    mine = next((r for r in rows if r["reviewer"] == reviewer), None)
    return {
        "approvals": len(approvals), "rejections": len(rejections),
        "needed": approvals_needed(it),
        "approvers": [r["reviewer"] for r in approvals],
        "my_status": mine["status"] if mine else "pending",
        "my_answer": mine["answer"] if mine else it["a"],
        "my_note": mine["note"] if mine else "",
        "my_strand": mine["strand"] if mine else it["strand"],
        "my_band": mine["band"] if mine else it["grade_band"],
    }


@app.get("/api/items")
def api_items(reviewer: str):
    out = []
    for it in ITEMS:
        st = item_state(it, reviewer)
        out.append({**it, **st})
    return {"items": out, "double_review": DOUBLE_REVIEW}


class Decision(BaseModel):
    source: str
    grade_band: str
    reviewer: str
    status: str            # approved | rejected | pending
    answer: str = ""
    strand: str = ""
    band: str = ""
    tv: bool = False
    note: str = ""


@app.post("/api/decision")
def api_decision(d: Decision):
    if d.status not in ("approved", "rejected", "pending"):
        raise HTTPException(400, "bad status")
    if key(d.source, d.grade_band) not in BY_KEY:
        raise HTTPException(404, "unknown item")
    if not d.reviewer.strip():
        raise HTTPException(400, "reviewer required")
    with closing(db()) as con:
        con.execute("""INSERT INTO decisions
            (source,grade_band,reviewer,status,answer,strand,band,tv,note,ts)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(source,grade_band,reviewer) DO UPDATE SET
              status=excluded.status, answer=excluded.answer, strand=excluded.strand,
              band=excluded.band, tv=excluded.tv, note=excluded.note, ts=excluded.ts""",
            (d.source, d.grade_band, d.reviewer, d.status, d.answer, d.strand,
             d.band, int(d.tv), d.note, time.time()))
        con.commit()
    it = BY_KEY[key(d.source, d.grade_band)]
    return item_state(it, d.reviewer)


@app.get("/api/stats")
def api_stats():
    finalized = pending = 0
    with closing(db()) as con:
        for it in ITEMS:
            rows = con.execute(
                "SELECT status,reviewer FROM decisions WHERE source=? AND grade_band=?",
                (it["source"], it["grade_band"])).fetchall()
            ok = len({r["reviewer"] for r in rows if r["status"] == "approved"})
            if ok >= approvals_needed(it):
                finalized += 1
            else:
                pending += 1
    return {"total": len(ITEMS), "finalized": finalized, "pending": pending}


@app.get("/api/export", response_class=PlainTextResponse)
def api_export():
    """Emit approved items (meeting their threshold) as JSONL for --promote."""
    lines = []
    with closing(db()) as con:
        for it in ITEMS:
            rows = con.execute(
                "SELECT * FROM decisions WHERE source=? AND grade_band=? AND status='approved' "
                "ORDER BY ts DESC", (it["source"], it["grade_band"])).fetchall()
            approvers = {r["reviewer"] for r in rows}
            if len(approvers) < approvals_needed(it):
                continue
            top = rows[0]  # latest approving edit wins
            lines.append(json.dumps({
                "subject": "civics", "grade_band": top["band"] or it["grade_band"],
                "strand": top["strand"] or it["strand"], "time_varying": bool(top["tv"]),
                "source": it["source"], "license": it["license"],
                "attribution": it["attribution"], "review_status": "approved",
                "reviewer": ", ".join(sorted(approvers)),
                "review_notes": "; ".join(r["note"] for r in rows if r["note"]),
                "messages": [{"role": "system", "content": SYS},
                             {"role": "user", "content": it["q"]},
                             {"role": "assistant", "content": top["answer"]}],
            }))
    return "\n".join(lines) + ("\n" if lines else "")


FRONTEND_PATH = os.environ.get("FRONTEND", "review_frontend.html")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(FRONTEND_PATH) as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    load_candidates()
    init_db()
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8090")))
