"""
civics_schema.py — shared schema, system prompt, and validators for the K-12
civics tutor dataset.

Civics is NOT built "correct by construction" like CAD/circuit. It is knowledge-
grounded and license-sensitive, so every row is a *candidate* until a human
reviewer approves it. This module defines the row shape, the tutor system
prompt, the coverage taxonomy (grade band x NAEP civics strand), and the flags
that reviewers rely on — especially `time_varying`, for facts whose answer
changes over time (current officials, etc.) which must NOT be memorized into
weights.

Pipeline: ingest (license-clean sources) -> draft candidates (build_civics_dataset.py)
-> HUMAN review gate -> approved rows only -> data/civics.jsonl.
"""
from __future__ import annotations

# --- Tutor system prompt -----------------------------------------------------
# Neutral, factual, age-aware. Explicitly refuses to assert volatile facts from
# memory (those are handled via retrieval, not baked in).
CIVICS_SYS = (
    "You are ParaClient, a K-12 civics tutor for U.S. students. Teach the "
    "structure and principles of U.S. government, the Constitution, rights and "
    "responsibilities, and how citizens participate. Be accurate, age-"
    "appropriate, and strictly non-partisan: present civic facts and multiple "
    "perspectives on contested political questions without advocating a side. "
    "Do NOT state time-varying facts (who currently holds an office, current "
    "counts, recent election outcomes) from memory — say the answer changes and "
    "explain how to look it up. Encourage reasoning, not memorization."
)

# --- Coverage taxonomy -------------------------------------------------------
# Grade bands for full K-12 coverage.
GRADE_BANDS = ("elementary", "middle", "high")  # K-5, 6-8, 9-12

# NAEP Civics content strands (the framework we align coverage to).
NAEP_STRANDS = {
    "civic_life": "Civic life, politics, and government",
    "foundations": "Foundations of the American political system",
    "constitution": "How the constitutional government embodies democratic "
                    "purposes, values, and principles",
    "world_affairs": "The United States and world affairs",
    "citizen_roles": "The roles of citizens in American democracy",
}

# --- License provenance (for the NOTICE file + DD) ---------------------------
LICENSES = {
    "public_domain": "U.S. Government work (17 U.S.C. 105) — public domain",
    "cc_by_sa_4": "CC BY-SA 4.0 — requires attribution + ShareAlike",
}

# --- Row schema --------------------------------------------------------------
# One row (candidate or approved):
# {
#   "subject": "civics",
#   "grade_band": "elementary" | "middle" | "high",
#   "strand": <one of NAEP_STRANDS keys>,
#   "time_varying": bool,          # True => teach concept / retrieval, not memorize
#   "source": "<short source id, e.g. 'USCIS-2008-Q1'>",
#   "license": <one of LICENSES keys>,
#   "attribution": "<attribution string; required for cc_by_sa_4>",
#   "review_status": "candidate" | "approved" | "rejected",
#   "reviewer": "<name/email or ''>",
#   "review_notes": "<free text>",
#   "messages": [ {role: system}, {role: user}, {role: assistant} ],
# }

REQUIRED_FIELDS = (
    "subject", "grade_band", "strand", "time_varying", "source",
    "license", "review_status", "messages",
)


class CivicsSchemaError(ValueError):
    pass


def sys_for() -> str:
    """The civics tutor system message content."""
    return CIVICS_SYS


def validate(row: dict, *, require_approved: bool = False) -> dict:
    """Structurally validate a civics row. Does NOT judge factual/neutrality
    quality — that is the human reviewer's job. Raises CivicsSchemaError on a
    structural problem; returns the row unchanged on success.

    require_approved=True additionally enforces that only human-approved rows
    pass (used when building the training file, so nothing unreviewed leaks in).
    """
    for f in REQUIRED_FIELDS:
        if f not in row:
            raise CivicsSchemaError(f"missing field: {f}")
    if row["subject"] != "civics":
        raise CivicsSchemaError(f"subject must be 'civics', got {row['subject']!r}")
    if row["grade_band"] not in GRADE_BANDS:
        raise CivicsSchemaError(f"bad grade_band: {row['grade_band']!r}")
    if row["strand"] not in NAEP_STRANDS:
        raise CivicsSchemaError(f"bad strand: {row['strand']!r}")
    if not isinstance(row["time_varying"], bool):
        raise CivicsSchemaError("time_varying must be a bool")
    if row["license"] not in LICENSES:
        raise CivicsSchemaError(f"unknown license: {row['license']!r}")
    if row["license"] == "cc_by_sa_4" and not row.get("attribution"):
        raise CivicsSchemaError("cc_by_sa_4 rows require a non-empty attribution")
    if row["review_status"] not in ("candidate", "approved", "rejected"):
        raise CivicsSchemaError(f"bad review_status: {row['review_status']!r}")

    msgs = row["messages"]
    if not (isinstance(msgs, list) and len(msgs) >= 2):
        raise CivicsSchemaError("messages must be a list of >= 2 turns")
    if msgs[0].get("role") != "system":
        raise CivicsSchemaError("first message must be role=system")
    roles = [m.get("role") for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        raise CivicsSchemaError("messages need at least one user and one assistant turn")

    # Safety rail: a row that bakes in a volatile fact must be flagged so it is
    # routed to concept/retrieval handling instead of memorization.
    if require_approved and row["review_status"] != "approved":
        raise CivicsSchemaError(
            f"training rows must be human-approved; got {row['review_status']!r}"
        )
    return row


# --- Serving-time time-varying detector --------------------------------------
# The civics adapter is trained NOT to assert volatile facts from memory. At
# serving time we still need to decide which incoming questions are volatile so
# the gateway can route them to civics_agent (live official-source lookup)
# instead of the adapter. This is a heuristic: it fires on questions about
# current office-holders / a user's own representatives, or an office noun paired
# with a "current/now/today" temporal cue. Over-firing is safe (it just triggers
# a cited .gov lookup); under-firing is the risk, so the patterns are generous.
import re  # noqa: E402

# Offices whose holder changes over time.
_OFFICE = (r"president|vice[-\s]?president|speaker(?:\s+of\s+the\s+house)?|"
           r"chief\s+justice|senators?|representatives?|congress(?:wo)?m[ae]n|"
           r"governors?|secretary\s+of\s+state|attorney\s+general|"
           r"majority\s+leader|minority\s+leader")
_TEMPORAL = r"now|current(?:ly)?|today|right\s+now|these\s+days|at\s+present|this\s+year"

_TV_PATTERNS = [
    # "who is (the current) <office>", "what is the name of the <office>"
    re.compile(rf"\bwho\s+is\s+(?:the\s+)?(?:current\s+)?(?:{_OFFICE})\b", re.I),
    re.compile(rf"\b(?:name|what.*name)\s+of\s+the\s+(?:{_OFFICE})\b", re.I),
    # "name your U.S. representative", "who is one of your state's senators"
    re.compile(rf"\b(?:name|who\s+is)\b.*\byour\b.*\b(?:{_OFFICE})\b", re.I),
    re.compile(rf"\byour\s+(?:u\.?s\.?\s+|state'?s?\s+)?(?:{_OFFICE})\b", re.I),
    re.compile(r"\bwho\s+represents\s+you\b", re.I),
    # an office noun explicitly tied to a "current/now/today" cue
    re.compile(rf"\b(?:{_OFFICE})\b.*\b(?:{_TEMPORAL})\b", re.I),
    re.compile(rf"\b(?:{_TEMPORAL})\b.*\b(?:{_OFFICE})\b", re.I),
    # recent/upcoming elections and current counts
    re.compile(r"\b(?:most\s+recent|last|latest|upcoming|next)\s+(?:\w+\s+)?election\b", re.I),
    re.compile(rf"\bhow\s+many\b.*\b(?:{_TEMPORAL})\b", re.I),
]


def is_time_varying(question: str) -> bool:
    """True if a civics question asks for a fact that changes over time and so
    should be answered by live official-source lookup, not from the adapter."""
    if not isinstance(question, str) or not question.strip():
        return False
    q = question.strip()
    return any(p.search(q) for p in _TV_PATTERNS)
