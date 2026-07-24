#!/usr/bin/env python3
"""
safe_tutor_proxy.py — Put the content filter IN FRONT of the tutor.

This is the component your R1 researchers actually talk to. It exposes an
OpenAI-compatible /v1/chat/completions endpoint, and for every request it:

  1. screens the student's latest input   (content_filter.screen_input)
     - BLOCK    -> return a safe refusal, never call the tutor
     - ESCALATE -> return a supportive message, never call the tutor, AND
                   emit an escalation event (teacher-alert path)
  2. if allowed, calls the real tutor (vLLM) with the adapter selected
  3. screens the tutor's output          (content_filter.screen_output)
     - BLOCK    -> replace the answer with a safe refusal
  4. returns the (screened) answer

The tutor (vLLM) should NOT be exposed directly to clients — only this proxy
should reach it. That is what makes the filter unbypassable: clients can't skip
it. In your deployment, bind vLLM to localhost and expose only this proxy.

R1 usage:
  # 1. serve the tutor on localhost (not public):
  vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
      --lora-modules language_arts=adapters/language_arts \
      --host 127.0.0.1 --port 8000 --max-model-len 4096

  # 2. run this proxy (researchers hit :8080, never :8000 directly):
  pip install fastapi uvicorn openai
  python scripts/safe_tutor_proxy.py --tutor-url http://127.0.0.1:8000/v1 \
      --host 0.0.0.0 --port 8080

  # 3. escalation events are appended to logs/escalations.jsonl — in R1 this is
  #    your stand-in for the teacher-alert path. Wire it to real alerting later.
"""
import argparse
import json
import os
import time

# content_filter is a sibling module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content_filter import ContentFilter, Action  # noqa: E402


def emit_escalation(event: dict, path="logs/escalations.jsonl"):
    """R1 stand-in for the teacher-alert path. Later: push to real alerting."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def build_app(tutor_url: str, api_key: str):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from openai import OpenAI

    app = FastAPI(title="ParaFrames Safe Tutor Proxy")
    tutor = OpenAI(base_url=tutor_url, api_key=api_key)
    filt = ContentFilter()

    def _completion_shell(content: str, model: str) -> dict:
        """Shape a minimal OpenAI-compatible chat completion response."""
        return {
            "id": f"safeproxy-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "_safety": {"filtered": True},
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        model = body.get("model", "unknown")
        messages = body.get("messages", [])

        # find the latest user turn to screen
        user_turns = [m for m in messages if m.get("role") == "user"]
        latest = user_turns[-1]["content"] if user_turns else ""

        # 1. screen input
        din = filt.screen_input(latest)
        if din.action == Action.ESCALATE:
            emit_escalation({
                "ts": time.time(), "model": model,
                "categories": din.categories,
                "student_input_preview": latest[:300],
                "note": "input escalation — supportive message returned, tutor NOT called",
            })
            return JSONResponse(_completion_shell(din.student_message, model))
        if din.action == Action.BLOCK:
            return JSONResponse(_completion_shell(din.student_message, model))

        # 2. call the real tutor
        try:
            resp = tutor.chat.completions.create(
                model=model, messages=messages,
                max_tokens=body.get("max_tokens", 512),
                temperature=body.get("temperature", 0.3))
            answer = resp.choices[0].message.content
        except Exception as e:
            return JSONResponse(
                _completion_shell(
                    "Sorry — I'm having trouble right now. Please try again.",
                    model),
                status_code=200)

        # 3. screen output
        dout = filt.screen_output(answer)
        if dout.action != Action.ALLOW:
            return JSONResponse(_completion_shell(dout.student_message, model))

        # 4. safe answer through
        return JSONResponse(_completion_shell(answer, model))

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutor-url", default="http://127.0.0.1:8000/v1",
                    help="the vLLM tutor endpoint (keep it localhost-only)")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    import uvicorn
    app = build_app(args.tutor_url, args.api_key)
    print(f"[*] Safe tutor proxy on {args.host}:{args.port} -> tutor {args.tutor_url}")
    print(f"[*] Researchers hit THIS proxy, never the tutor directly.")
    print(f"[*] Filter decisions -> logs/content_filter.jsonl")
    print(f"[*] Escalations       -> logs/escalations.jsonl")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
