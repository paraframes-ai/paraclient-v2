#!/usr/bin/env python3
"""
content_filter.py — Independent content-safety layer that wraps the tutor.

WHY THIS IS A SEPARATE LAYER (not baked into the adapter):
  A fine-tuned tutor can drift or be jailbroken. The safety layer must sit
  OUTSIDE the model so a misbehaving adapter cannot bypass it. This module is
  that layer: it screens student INPUT before the tutor sees it, and tutor
  OUTPUT before the student sees it. It never depends on the tutor to police
  itself.

WHAT IT DOES:
  screen_input(text)  -> Decision(allow/block/escalate, categories, message)
  screen_output(text) -> Decision(allow/block, categories, message)
  A Decision of ESCALATE (e.g. self-harm signals) means: do not answer normally,
  surface a safe supportive message, AND emit an escalation event for the
  teacher-alert path.

MODERATION BACKEND (pluggable):
  The actual classification is delegated to a MODERATION BACKEND so you can swap
  implementations without touching the wrapper logic. Two are provided:
    - HeuristicBackend: keyword/pattern rules. NO external calls. Good enough to
      stand up R1 and start collecting signal TODAY. NOT sufficient for real
      students — it will miss things.
    - ModerationAPIBackend: stub that shows where to call a real content-
      moderation service (OpenAI moderation, Perspective API, Azure Content
      Safety, Llama Guard served locally, etc.). Fill in the call for production.

  For R1 with 10 researchers: HeuristicBackend is fine to start — the POINT of
  R1 is to log what it catches/misses and learn the real filter's spec. Tune it
  toward OVER-blocking so you see failure modes, then dial back.

IMPORTANT (honesty): the heuristic backend is a scaffold, not a guarantee. Do
NOT ship it to real minors. Before student exposure, swap in a real moderation
model/service and validate it. This file gives you the ARCHITECTURE and a
working-today default; it does not give you a certified safety classifier.
"""
from __future__ import annotations
import re
import json
import time
import os
import queue
import threading
import atexit
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable


class Action(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"   # block normal answer + fire teacher alert


@dataclass
class Decision:
    action: Action
    categories: list[str] = field(default_factory=list)
    # message shown to the student when not ALLOW (kept safe + supportive)
    student_message: str | None = None
    # free-form detail for logging / the teacher alert
    detail: str = ""


# --------------------------------------------------------------------------
# Categories we care about for a K-12 tutor. Escalation categories get special
# handling (supportive response + teacher alert) rather than a bare block.
# --------------------------------------------------------------------------
ESCALATE_CATEGORIES = {"self_harm", "abuse_disclosure"}
BLOCK_CATEGORIES = {
    "sexual", "violence", "harassment", "hate", "dangerous",
    "self_harm", "abuse_disclosure",  # also block the normal flow
    # screen_output() rewrites an ESCALATE hit to "self_harm_content" (escalation
    # is an input-side concept). Without that label here, _decide() matched
    # neither the escalate set nor the block set and fell through to ALLOW —
    # so self-harm content in TUTOR OUTPUT reached the student, and the
    # Action.ESCALATE guard below the _decide() call was unreachable.
    "self_harm_content",
}


# --------------------------------------------------------------------------
# Moderation backends
# --------------------------------------------------------------------------

class ModerationBackend:
    """Interface: classify(text, surface) -> set[str] of category labels."""
    def classify(self, text: str, surface: str) -> set[str]:
        raise NotImplementedError


class HeuristicBackend(ModerationBackend):
    """
    Pattern-based scaffold. Deliberately broad (favor over-flagging for R1).
    surface is "input" or "output" — some checks differ by direction.

    NOTE: self-harm / abuse patterns are intentionally NOT exhaustively listed
    here and are matched at a coarse, non-graphic level — enough to route to the
    escalation path without this file enumerating sensitive phrasing.
    """
    def __init__(self):
        # Coarse signals only. Real detection belongs in a trained classifier.
        self._patterns = {
            # First-person forms catch a student DISCLOSING distress (the input
            # surface). Second-person forms catch a tutor DIRECTING harm at the
            # student (the output surface) -- the original pattern was
            # first-person only and reused verbatim for output screening, so
            # "you should kill yourself" from the tutor matched nothing at all.
            "self_harm": re.compile(
                r"\b(hurt (myself|yourself)|kill (myself|yourself)|"
                r"harm (myself|yourself)|want to die|"
                r"end (it|my life|your life)|"
                r"suicidal|self[- ]harm)\b", re.IGNORECASE),
            "abuse_disclosure": re.compile(
                r"\b(someone (hurts|touches) me|being abused|"
                r"afraid to go home|hits me)\b", re.IGNORECASE),
            "sexual": re.compile(
                r"\b(sexual|explicit|porn|nude)\b", re.IGNORECASE),
            "violence": re.compile(
                r"\b(how to (make|build) a (bomb|weapon)|kill (someone|people))\b",
                re.IGNORECASE),
            "hate": re.compile(
                r"\b(racial slur|kill all|subhuman)\b", re.IGNORECASE),
            "dangerous": re.compile(
                r"\b(how to (make|synthesize) (drugs|meth|explosives))\b",
                re.IGNORECASE),
        }

    def classify(self, text: str, surface: str) -> set[str]:
        hits = set()
        for cat, pat in self._patterns.items():
            if pat.search(text):
                hits.add(cat)
        return hits


class ModerationAPIBackend(ModerationBackend):
    """
    Production stub. Wire this to a real content-moderation service or a locally
    served safety model (e.g. Llama Guard on your infra). Keep it INDEPENDENT of
    the tutor model. Fill in `_call` and map its response to your category labels.
    """
    def __init__(self, call_fn: Callable[[str, str], set[str]]):
        # inject the real call so this module stays dependency-free
        self._call = call_fn

    def classify(self, text: str, surface: str) -> set[str]:
        try:
            return self._call(text, surface)
        except Exception:
            # FAIL CLOSED: if moderation is unavailable, treat as unsafe rather
            # than letting unscreened content through. For a child-facing
            # product, unavailable moderation must not mean "allow".
            return {"_moderation_unavailable"}


# --------------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------------

SUPPORTIVE_SELF_HARM_MSG = (
    "It sounds like you might be going through something really hard right now, "
    "and I want to make sure you get support from someone who can help. Please "
    "talk to a trusted adult — a parent, teacher, or school counselor. If you "
    "ever feel unsafe, you can reach people who care and can help right away. "
    "You're not alone."
)
GENERIC_BLOCK_MSG = (
    "I can't help with that here — let's keep our work focused on your lesson. "
    "If you have a question about the material, I'm happy to help with that."
)


class ContentFilter:
    def __init__(self, backend: ModerationBackend | None = None,
                 log_path: str | None = "logs/content_filter.jsonl"):
        self.backend = backend or HeuristicBackend()
        self.log_path = log_path
        self._queue: queue.Queue | None = None
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        if self.log_path:
            self._queue = queue.Queue()
            self._worker_thread = threading.Thread(
                target=self._logger_loop, daemon=True, name="ContentFilterLogger"
            )
            self._worker_thread.start()
            atexit.register(self.flush)

    def _logger_loop(self):
        dir_created = False
        file_handle = None
        current_path = self.log_path

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                rec = self._queue.get(timeout=0.05)
            except queue.Empty:
                if file_handle and not file_handle.closed:
                    try:
                        file_handle.flush()
                    except Exception:
                        pass
                continue

            records = [rec]
            while True:
                try:
                    records.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            try:
                if not dir_created and current_path:
                    dirname = os.path.dirname(current_path)
                    if dirname:
                        os.makedirs(dirname, exist_ok=True)
                    dir_created = True

                if file_handle is None or file_handle.closed:
                    file_handle = open(current_path, "a", encoding="utf-8")

                for r in records:
                    file_handle.write(json.dumps(r, ensure_ascii=False) + "\n")
                file_handle.flush()
            except Exception:
                pass
            finally:
                for _ in records:
                    self._queue.task_done()

        if file_handle and not file_handle.closed:
            try:
                file_handle.flush()
                file_handle.close()
            except Exception:
                pass

    def flush(self):
        if self._queue:
            self._queue.join()
    # -- logging: every decision, allow or block, for R1 signal --------------
    def _log(self, surface: str, text: str, decision: Decision):
        if not self.log_path or not self._queue:
            return
        rec = {
            "ts": time.time(),
            "surface": surface,
            # store a truncated preview, not necessarily the full text
            "text_preview": text[:300],
            "decision": asdict(decision) | {"action": decision.action.value},
        }
        try:
            self._queue.put_nowait(rec)
        except Exception:
            pass  # logging must never break the request path

    def _decide(self, cats: set[str]) -> Decision:
        if "_moderation_unavailable" in cats:
            return Decision(Action.BLOCK, ["moderation_unavailable"],
                            GENERIC_BLOCK_MSG,
                            "moderation backend unavailable — failed closed")
        escalate = cats & ESCALATE_CATEGORIES
        if escalate:
            msg = (SUPPORTIVE_SELF_HARM_MSG if "self_harm" in escalate
                   else SUPPORTIVE_SELF_HARM_MSG)  # same supportive path
            return Decision(Action.ESCALATE, sorted(cats), msg,
                            "escalation categories triggered")
        block = cats & BLOCK_CATEGORIES
        if block:
            return Decision(Action.BLOCK, sorted(cats), GENERIC_BLOCK_MSG,
                            "block categories triggered")
        return Decision(Action.ALLOW, [], None, "")

    def screen_input(self, text: str) -> Decision:
        cats = self.backend.classify(text, "input")
        d = self._decide(cats)
        self._log("input", text, d)
        return d

    def screen_output(self, text: str) -> Decision:
        cats = self.backend.classify(text, "output")
        # output never "escalates" (that's an input-side concept); downgrade
        if cats & ESCALATE_CATEGORIES:
            cats = (cats - ESCALATE_CATEGORIES) | {"self_harm_content"}
        d = self._decide(cats)
        if d.action == Action.ESCALATE:
            d = Decision(Action.BLOCK, d.categories, GENERIC_BLOCK_MSG,
                         "unsafe tutor output blocked")
        self._log("output", text, d)
        return d


# --------------------------------------------------------------------------
# Self-test (offline, no tutor needed)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    f = ContentFilter(log_path=None)

    cases_input = [
        ("How do I add fractions?", Action.ALLOW),
        ("i want to hurt myself", Action.ESCALATE),
        ("how to build a bomb", Action.BLOCK),
        ("What caused the American Revolution?", Action.ALLOW),
    ]
    print("INPUT screening:")
    for text, expected in cases_input:
        d = f.screen_input(text)
        ok = "OK" if d.action == expected else "MISMATCH"
        print(f"  [{ok}] {d.action.value:9s} <- {text!r}")
        assert d.action == expected, f"{text!r} -> {d.action} expected {expected}"

    cases_output = [
        ("To add fractions, find a common denominator...", Action.ALLOW),
        ("Here's how to make explosives at home", Action.BLOCK),
    ]
    print("OUTPUT screening:")
    for text, expected in cases_output:
        d = f.screen_output(text)
        ok = "OK" if d.action == expected else "MISMATCH"
        print(f"  [{ok}] {d.action.value:9s} <- {text!r}")
        assert d.action == expected

    print("SELF-TEST PASSED")
