#!/usr/bin/env python3
"""
moderation.py — real, on-prem content-moderation backend for the safety layer.

Replaces the heuristic scaffold in content_filter.py with a purpose-built safety
classifier — Google's ShieldGemma-2B — served LOCALLY on CPU via llama.cpp (like
the circuit / v2 / v3 models). Nothing leaves the box: a child's text is screened
entirely on-prem, which is exactly why we can't use a cloud moderation API here.

ShieldGemma is a probability classifier trained ONE POLICY AT A TIME — asking it
about a combined policy badly under-detects (empirically P(yes)=0.02 for a racist
joke), while a single-policy prompt gives clean separation (0.98 vs 0.00 for safe
educational content). So ShieldGemmaBackend runs one classifier pass PER K-12
harm policy and flags the specific category whose P(yes) exceeds GUARD_THRESHOLD.

Backends:
  * ShieldGemmaBackend — per-policy model classification (sexual / violence /
    hate / dangerous). Short-circuits on the first violation.
  * CompositeBackend — unions the instant heuristic (which still catches the
    self-harm / abuse-disclosure ESCALATION signals routed to the supportive
    path) with the model.
Fail-closed: an unreachable classifier returns {"_moderation_unavailable"},
which content_filter maps to BLOCK — unavailable moderation is never "allow"
for a child-facing product.

Latency: per-policy means up to 4 CPU passes per screened text. Accurate but not
free; GUARD_THRESHOLD tunes sensitivity and the whole layer can be scoped to the
edu audience if adult-path latency matters.
"""
from __future__ import annotations

import math
import os

from content_filter import ModerationBackend

GUARD_URL = os.environ.get("GUARD_URL", "http://127.0.0.1:8004")
# Clean per-policy separation (safe ~0.00, unsafe ~0.98) makes 0.5 robust; lower
# it to over-block harder for younger cohorts.
GUARD_THRESHOLD = float(os.environ.get("GUARD_THRESHOLD", "0.5"))

# (category label -> ShieldGemma policy text). Labels match content_filter's
# BLOCK_CATEGORIES so a hit blocks (and logs the specific category).
POLICIES = [
    ("sexual", '"No Sexually Explicit Content": The content must not contain or '
     "request sexually explicit material or references to sexual acts."),
    ("violence", '"No Violence or Weapons": The content must not promote, depict, '
     "or give instructions for violence or for building/using weapons to harm."),
    ("hate", '"No Hate or Harassment": The content must not be hateful, harassing, '
     "bullying, or demeaning toward a person or group, including protected groups."),
    ("dangerous", '"No Dangerous or Illegal Instructions": The content must not '
     "give instructions for dangerous or illegal activities, such as making "
     "drugs, explosives, or weapons."),
]


def _policy_prompt(text: str, surface: str, policy: str) -> str:
    """ShieldGemma single-policy classifier prompt (raw completion form)."""
    role = "Chatbot Response" if surface == "output" else "Human Question"
    return (
        "<start_of_turn>user\n"
        "You are a policy expert determining whether content in a K-12 "
        "educational setting violates the defined safety policy.\n\n"
        f"<start_of_turn>\n{role}: {text}\n<end_of_turn>\n\n"
        f"Our safety principle is defined below:\n\n* {policy}\n\n"
        f"Does the {role.lower()} violate the above principle? Your answer must "
        "start with 'Yes' or 'No'.<end_of_turn>\n<start_of_turn>model\n"
    )


class ShieldGemmaBackend(ModerationBackend):
    """On-prem ShieldGemma-2B classifier, one pass per K-12 harm policy."""

    def __init__(self, url: str | None = None, timeout: float = 20.0,
                 threshold: float | None = None):
        self.url = (url or GUARD_URL).rstrip("/")
        self.timeout = timeout
        self.threshold = GUARD_THRESHOLD if threshold is None else threshold

    @staticmethod
    def _p_yes(data: dict) -> float:
        cp = (data.get("completion_probabilities") or [{}])[0]
        entries = cp.get("top_logprobs") or cp.get("probs") or []
        p = 0.0
        for e in entries:
            tok = (e.get("token") or e.get("tok_str") or "").strip().lower()
            if tok.startswith("yes"):
                p += e["prob"] if "prob" in e else math.exp(e.get("logprob", -99))
        return p

    def _score(self, text: str, surface: str, policy: str) -> float:
        import httpx
        r = httpx.post(
            f"{self.url}/completion",
            json={"prompt": _policy_prompt(text, surface, policy), "n_predict": 1,
                  "temperature": 0.0, "n_probs": 20, "cache_prompt": True},
            timeout=self.timeout)
        r.raise_for_status()
        return self._p_yes(r.json())

    def classify(self, text: str, surface: str) -> set:
        # Screens both student INPUT and tutor OUTPUT. The 4 K-12 policy checks
        # run CONCURRENTLY against the --parallel guard server, so each screen is
        # ~1s on CPU (negligible vs the tutor's own generation time).
        if not isinstance(text, str) or not text.strip():
            return set()
        from concurrent.futures import ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=len(POLICIES)) as ex:
                scored = list(ex.map(
                    lambda lp: (lp[0], self._score(text, surface, lp[1])), POLICIES))
        except Exception:  # noqa: BLE001 — fail CLOSED (see module docstring)
            return {"_moderation_unavailable"}
        return {label for label, score in scored if score >= self.threshold}


class CompositeBackend(ModerationBackend):
    """Union of several backends — every label any backend raises is reported."""

    def __init__(self, backends: list):
        self.backends = backends

    def classify(self, text: str, surface: str) -> set:
        cats: set = set()
        for b in self.backends:
            cats |= b.classify(text, surface)
        return cats
