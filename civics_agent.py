#!/usr/bin/env python3
"""
civics_agent.py — live, official-source answers for TIME-VARYING civics
questions ("Who is the President now?", "Name your U.S. Representative").

Such facts must never be answered from a model's memory: they change and go
stale. Instead we run the SAME ReAct search/fetch loop used by the docs/slides
agent (generate_media), but constrained to official U.S. government sources and
required to cite one. The current fact is fetched fresh at serving time, so the
answer is always up to date and traceable to a .gov page.

Contract: answer_time_varying(client, model, question, ...) ->
    {"answer": str, "source_title": str, "source_url": str, "verified": bool}
`verified` is True only when the cited source is on an official domain.
"""
from __future__ import annotations

from urllib.parse import urlparse

from generate_media import _web_search, _web_fetch, parse_json

# Official U.S. government domains we trust for civics facts. Any *.gov / *.mil
# host qualifies; the explicit list documents the canonical ones.
_OFFICIAL_SUFFIXES = (".gov", ".mil")
_OFFICIAL_HOSTS = {
    "congress.gov", "whitehouse.gov", "senate.gov", "house.gov", "usa.gov",
    "uscis.gov", "supremecourt.gov", "state.gov", "usemb.gov",
}


def is_official(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
    except Exception:  # noqa: BLE001
        return False
    if not host:
        return False
    return (host in _OFFICIAL_HOSTS
            or any(host == s.lstrip(".") or host.endswith(s)
                   for s in _OFFICIAL_SUFFIXES))


_CIVICS_AGENT_SYS = (
    "You answer a single U.S. civics question whose answer CHANGES OVER TIME "
    "(current office-holders, representatives, dates). Never answer from memory. "
    "You have two tools; output EXACTLY ONE JSON object per turn and nothing "
    "else:\n"
    '  search — {"tool":"search","query":"..."}\n'
    '  fetch  — {"tool":"fetch","url":"..."}\n'
    "Rules:\n"
    "- You MUST fetch and read at least one OFFICIAL U.S. government source (a "
    ".gov or .mil site — e.g. congress.gov, whitehouse.gov, senate.gov, "
    "house.gov, supremecourt.gov) before answering. Prefer adding 'site:.gov' to "
    "searches.\n"
    "- When you have verified the current fact from an official page, finish with "
    'EXACTLY: {"final":{"answer":"<short, K-12-appropriate answer>",'
    '"source_title":"<page title>","source_url":"<the .gov url you used>"}}\n'
    "- If you cannot verify it from an official source, finish with "
    '{"final":{"answer":"I could not verify this from an official government '
    'source right now.","source_title":"","source_url":""}}\n'
    "Keep the answer to one or two plain sentences."
)


def answer_time_varying(client, model, question: str,
                        max_steps: int = 8, min_official_fetch: int = 1) -> dict:
    """Run the constrained ReAct loop and return a cited, verified answer dict."""
    messages = [{"role": "system", "content": _CIVICS_AGENT_SYS},
                {"role": "user", "content": f"Question: {question}"}]
    official_fetches = 0
    for step in range(max_steps):
        last = step == max_steps - 1
        if last:
            messages.append({"role": "user", "content":
                             'Stop. Output ONLY the {"final":...} JSON now, '
                             "using the best official source you have read."})
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, max_tokens=700, temperature=0.1)
        except Exception:  # noqa: BLE001 — context/backend issue: bail out cleanly
            break
        raw = (r.choices[0].message.content or "").strip()
        try:
            obj = parse_json(raw)
        except Exception:  # noqa: BLE001
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content":
                          "Not valid JSON. Reply with ONE JSON object only."}]
            continue

        if isinstance(obj, dict) and "final" in obj:
            fin = obj["final"] if isinstance(obj["final"], dict) else {}
            src = str(fin.get("source_url", ""))
            verified = is_official(src)
            # Don't let it finalize a positive answer before reading an official
            # source — push it back to research once.
            if not verified and official_fetches < min_official_fetch and not last:
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content":
                              "You have not cited an official .gov/.mil source "
                              "yet. Search with 'site:.gov' and fetch the official "
                              "page, then finalize with its URL."}]
                continue
            return {"answer": str(fin.get("answer", "")).strip(),
                    "source_title": str(fin.get("source_title", "")).strip(),
                    "source_url": src.strip(), "verified": verified}

        if isinstance(obj, dict) and obj.get("tool") == "search":
            obs = _web_search(obj.get("query", ""))
        elif isinstance(obj, dict) and obj.get("tool") == "fetch":
            url = obj.get("url", "")
            obs = _web_fetch(url)
            if is_official(url) and not obs.startswith("[fetch error"):
                official_fetches += 1
        else:
            obs = '[unknown action — use "search", "fetch", or "final"]'
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "OBSERVATION:\n" + obs}]

    return {"answer": "I could not verify this from an official government "
                      "source right now.",
            "source_title": "", "source_url": "", "verified": False}
