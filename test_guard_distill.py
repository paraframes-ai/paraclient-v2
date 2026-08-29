#!/usr/bin/env python3
"""Unit tests for the distilled-guard cutover and shadow safety gates."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from guard_distill.shadow_test import compare
from moderation import DistilledGuardBackend


class _Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class _Backend:
    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)

    def classify(self, _text, _surface):
        return set(next(self.verdicts))


class DistilledBackendTests(unittest.TestCase):
    def test_artifact_thresholds_are_applied_per_policy(self):
        data = {
            "scores": {"sexual": 0.2, "violence": 0.1,
                       "hate": 0.2, "dangerous": 0.2},
            "thresholds": {"sexual": 0.1, "violence": 0.2,
                           "hate": 0.25, "dangerous": 0.1},
        }
        with patch("httpx.post", return_value=_Response(data)):
            verdict = DistilledGuardBackend().classify("probe", "input")
        self.assertEqual(verdict, {"sexual", "dangerous"})

    def test_partial_label_response_fails_closed(self):
        data = {"scores": {"sexual": 0.0}}
        with patch("httpx.post", return_value=_Response(data)):
            verdict = DistilledGuardBackend().classify("probe", "input")
        self.assertEqual(verdict, {"_moderation_unavailable"})

    def test_partial_threshold_response_fails_closed(self):
        data = {
            "scores": {"sexual": 0.0, "violence": 0.0,
                       "hate": 0.0, "dangerous": 0.0},
            "thresholds": {"sexual": 0.1},
        }
        with patch("httpx.post", return_value=_Response(data)):
            verdict = DistilledGuardBackend().classify("probe", "input")
        self.assertEqual(verdict, {"_moderation_unavailable"})


class ShadowGateTests(unittest.TestCase):
    def test_student_false_allow_fails_gate(self):
        rows = [{"text": "one", "surface": "input"}]
        report, passed = compare(
            rows, _Backend([{"dangerous"}]), _Backend([set()]))
        self.assertFalse(passed)
        self.assertEqual(report["counts"]["student_more_permissive"], 1)
        self.assertNotIn("one", str(report))

    def test_equal_or_stricter_student_passes_gate(self):
        rows = [{"text": "one"}, {"text": "two"}]
        report, passed = compare(
            rows, _Backend([set(), {"hate"}]),
            _Backend([{"violence"}, {"hate"}]))
        self.assertTrue(passed)
        self.assertEqual(report["counts"]["student_more_restrictive"], 1)
        self.assertEqual(report["counts"]["exact_match"], 1)

    def test_unavailable_student_fails_gate(self):
        rows = [{"text": "one"}]
        _, passed = compare(
            rows, _Backend([set()]),
            _Backend([{"_moderation_unavailable"}]))
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
