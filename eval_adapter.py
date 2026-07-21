#!/usr/bin/env python3
"""
eval_adapter.py — Behavioral eval for a trained tutor adapter.

This does NOT measure accuracy. It measures the two behaviors that make or
break a K-12 tutor product:

  1. SOCRATIC mode must WITHHOLD the final answer and ask a guiding question.
  2. GRADUATED_HINT mode must eventually HELP a stuck student progress.

It talks to a running vLLM server (OpenAI-compatible) with your adapter loaded.

Start vLLM first, e.g.:
  vllm serve Qwen/Qwen2.5-7B-Instruct \
      --enable-lora \
      --lora-modules math=adapters/math \
      --max-model-len 4096 --gpu-memory-utilization 0.9

Then:
  python scripts/eval_adapter.py --adapter math \
      --base-url http://localhost:8000/v1

NOTE: The answer-leak check here is a heuristic (looks for the known numeric
answer in the tutor's reply). It is a smoke test, not a guarantee. For a
child-facing product, pair this with human review before shipping any adapter.
"""
import argparse
import json
import re

# Small hand-built probe set. Each item: a question, its final answer (for the
# leak check), and which mode we're probing. Expand this with your own items —
# the bigger and more representative this set, the more useful the signal.
PROBES = [
    {"q": "A pack has 12 pencils. I buy 4 packs. How many pencils total?",
     "answer": "48", "topic": "multiplication"},
    {"q": "I have 3/4 of a pizza and eat 1/4. How much is left?",
     "answer": "1/2", "topic": "fractions"},
    {"q": "A rectangle is 5 cm by 8 cm. What is its area?",
     "answer": "40", "topic": "area"},
    {"q": "Sam saved $6 a week for 7 weeks. How much did he save?",
     "answer": "42", "topic": "multiplication"},
]

QUESTION_MARK = "?"


def chat(client, model, messages, max_tokens=400):
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.3)
    return resp.choices[0].message.content.strip()


def leaks_answer(reply: str, answer: str) -> bool:
    # normalize simple fraction / number matches
    ans = answer.strip().lower()
    r = reply.lower()
    # direct token match on a word boundary where possible
    if re.search(rf"(?<!\d){re.escape(ans)}(?!\d)", r):
        return True
    return False


def run_socratic(client, model):
    print("\n=== SOCRATIC mode: must NOT give the answer, must ask a question ===")
    passed = 0
    for p in PROBES:
        msgs = [
            {"role": "system",
             "content": ("You are a Socratic math tutor for a K-12 student. "
                         "Never state the final answer. Guide with one question "
                         "at a time.")},
            {"role": "user", "content": p["q"]},
        ]
        reply = chat(client, model, msgs)
        leaked = leaks_answer(reply, p["answer"])
        asked = QUESTION_MARK in reply
        ok = (not leaked) and asked
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {p['topic']:15s} "
              f"leaked={leaked} asked_question={asked}")
        if not ok:
            print(f"        reply: {reply[:160]}...")
    print(f"Socratic: {passed}/{len(PROBES)} passed "
          f"(withheld answer AND asked a guiding question)")
    return passed


def run_graduated(client, model):
    print("\n=== GRADUATED_HINT mode: stuck student must eventually get help ===")
    passed = 0
    for p in PROBES:
        msgs = [
            {"role": "system",
             "content": ("You are a math homework tutor for a K-12 student. "
                         "Guide with questions first, but if the student stays "
                         "stuck, escalate to hints and worked steps so they can "
                         "finish.")},
            {"role": "user", "content": p["q"]},
            {"role": "assistant", "content": "What operation do you think this needs?"},
            {"role": "user", "content": "I don't know, I'm really stuck and confused."},
            {"role": "user", "content": "Please just help me work through it."},
        ]
        reply = chat(client, model, msgs)
        # After explicit stuck signals, a good tutor gives a concrete worked
        # step (heuristic: mentions the answer OR shows an operation/number).
        helped = leaks_answer(reply, p["answer"]) or bool(re.search(r"\d", reply))
        print(f"[{'PASS' if helped else 'FAIL'}] {p['topic']:15s} escalated_help={helped}")
        if not helped:
            print(f"        reply: {reply[:160]}...")
        passed += helped
    print(f"Graduated-hint: {passed}/{len(PROBES)} passed "
          f"(gave concrete help after student was stuck)")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="lora module name as registered in vLLM (e.g. 'math')")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    s = run_socratic(client, args.adapter)
    g = run_graduated(client, args.adapter)

    print("\n--- SUMMARY ---")
    print(f"Socratic withholding : {s}/{len(PROBES)}")
    print(f"Graduated escalation : {g}/{len(PROBES)}")
    print("\nReminder: heuristic smoke test only. Human-review a sample of real "
          "transcripts before shipping any adapter to students.")


if __name__ == "__main__":
    main()
