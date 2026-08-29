#!/usr/bin/env python3
"""
generate_media.py — Turn a prompt into a document (.docx) or a themed
slideshow (.pptx) using the ParaFrames base model (ParaClient-v2.2).

This is an APPLICATION layer on top of the served base model — no adapter or
training involved. The model returns STRICT JSON describing the doc/deck (and,
for slides, picks an appropriate THEME from the palette below based on the
topic/audience); this script renders that JSON into a real Office file.

Usage (against the running vLLM endpoint):
  python generate_media.py --kind slides \
      --prompt "A 6-slide intro to the water cycle for 4th graders" \
      --out out/water_cycle.pptx
  python generate_media.py --kind doc \
      --prompt "A one-page explainer on photosynthesis for middle school" \
      --out out/photosynthesis.docx

  # theme is auto-chosen for slides; override with --theme <name> if you like.
"""
import argparse
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import docx
from docx.shared import Pt as DocxPt, RGBColor as DocxRGB

DOC_THEMES = {
    "light":     dict(bg="FFFFFF", title="111111", body="333333", accent="666666",
                      font_head="Georgia", font_body="Calibri", dark=False),
    "slate":     dict(bg="F2F2F2", title="1A1A1A", body="3A3A3A", accent="707070",
                      font_head="Calibri", font_body="Calibri", dark=False),
    "newsprint": dict(bg="FAFAFA", title="0A0A0A", body="2B2B2B", accent="595959",
                      font_head="Georgia", font_body="Georgia", dark=False),
    "contrast":  dict(bg="FFFFFF", title="000000", body="1A1A1A", accent="000000",
                      font_head="Calibri", font_body="Calibri", dark=False),
    "dark":      dict(bg="121212", title="FFFFFF", body="D0D0D0", accent="8A8A8A",
                      font_head="Segoe UI", font_body="Segoe UI", dark=True),
}

SLIDE_THEMES = {
    "academic": dict(bg="F7F3E8", bg_alt="E9EEF5", title="172A46", body="26364A",
                     accent="B8860B", accent2="315A8A", font_head="Aptos Display",
                     font_body="Aptos", layout="frame", dark=False),
    "playful": dict(bg="FFF7E6", bg_alt="EAF7FF", title="55286F", body="34304A",
                    accent="FF6B6B", accent2="2EC4B6", font_head="Aptos Display",
                    font_body="Aptos", layout="blocks", dark=False),
    "corporate": dict(bg="F7F9FC", bg_alt="EDF2F8", title="102A43", body="334E68",
                      accent="1677FF", accent2="00A3A3", font_head="Aptos Display",
                      font_body="Aptos", layout="band", dark=False),
    "nature": dict(bg="F3F7EE", bg_alt="E3EFE0", title="214E34", body="35543F",
                   accent="4C956C", accent2="D68C45", font_head="Georgia",
                   font_body="Aptos", layout="organic", dark=False),
    "tech": dict(bg="101827", bg_alt="172238", title="F4F7FF", body="D4DCEF",
                 accent="38BDF8", accent2="A78BFA", font_head="Aptos Display",
                 font_body="Aptos", layout="tech", dark=True),
}

THEMES = {**DOC_THEMES, **SLIDE_THEMES}
DEFAULT_DOC_THEME = "light"
DEFAULT_SLIDE_THEME = "corporate"

def _theme_guidance(kind: str) -> str:
    hints = ({
        "academic": "scholarly / formal / older students",
        "playful": "young children (K-5), fun, colorful",
        "corporate": "business, professional, neutral default",
        "nature": "science, environment, biology, outdoors",
        "tech": "computing, engineering, coding (dark palette)",
    } if kind == "slides" else {
        "light": "clean general-purpose document",
        "slate": "modern neutral report",
        "newsprint": "editorial or humanities writing",
        "contrast": "high-contrast formal handout",
        "dark": "dark-background digital document",
    })
    return "; ".join(f'"{k}" = {v}' for k, v in hints.items())


THEME_SELECTOR_SYS = (
    "You are ParaFrames' slideshow art director. Choose the single visual theme "
    "that best fits the topic, audience, purpose, and requested tone. Return "
    "STRICT JSON only as {\"theme\":\"<name>\"}. Valid themes: "
    + ", ".join(SLIDE_THEMES) + ". Guidance: " + _theme_guidance("slides") + "."
)


def theme_selector_messages(user_prompt: str) -> list:
    """Build the slideshow-theme adapter input."""
    return [{"role": "system", "content": THEME_SELECTOR_SYS},
            {"role": "user", "content": user_prompt}]


def select_slideshow_theme(client, model: str, user_prompt: str) -> str:
    """Choose a renderer-backed theme with the request-scoped LoRA."""
    response = client.chat.completions.create(
        model=model,
        messages=theme_selector_messages(user_prompt),
        max_tokens=20,
        temperature=0,
        extra_body={"lora": [{"id": 0, "scale": 1.0}]},
    )
    raw = response.choices[0].message.content or ""
    data = json.loads(raw)
    theme = data.get("theme")
    if theme not in SLIDE_THEMES:
        raise ValueError(f"invalid slideshow theme: {theme!r}")
    return theme


_MATH_NOTE = (
    "Write any math or equations in plain text or Unicode (e.g. θ, v², ×, √, "
    "≈, π, ½) — do NOT use LaTeX backslash commands (no backslash-frac, "
    "backslash-theta, etc.), because the output must be valid JSON."
)


def build_prompt(kind: str, user_prompt: str) -> list:
    if kind == "slides":
        schema = (
            '{"title": "...", "subtitle": "...", "theme": "<one of: '
            + ", ".join(SLIDE_THEMES) + '>", "slides": [{"title": "...", '
            '"bullets": ["...", "..."], "notes": "..."}]}'
        )
        sys = (
            "You are a presentation generator. Return STRICT JSON only (no prose, "
            "no markdown fences) matching exactly:\n" + schema + "\n"
            "Choose the SINGLE most appropriate theme for the topic and audience — "
            f"{_theme_guidance(kind)}. 4-10 slides, 2-5 short bullets each; `notes` is "
            "optional speaker notes. Keep it accurate and age-appropriate. " + _MATH_NOTE
        )
    else:  # doc
        schema = (
            '{"title": "...", "theme": "<one of: ' + ", ".join(DOC_THEMES) + '>", '
            '"sections": [{"heading": "...", "paragraphs": ["...", "..."]}]}'
        )
        sys = (
            "You are a document generator. Return STRICT JSON only (no prose, no "
            "markdown fences) matching exactly:\n" + schema + "\n"
            "Choose the SINGLE most appropriate theme for the topic and audience — "
            f"{_theme_guidance(kind)}. Use clear headings and well-formed paragraphs. "
            "Keep it accurate and age-appropriate. " + _MATH_NOTE
        )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user_prompt}]


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


_BAD_ESCAPE = re.compile(r'\\(?![\\/"bfnrtu])')  # backslash not starting a valid JSON escape


def parse_json(raw: str) -> dict:
    """Parse model JSON; tolerate code fences and INVALID backslash escapes
    (e.g. LaTeX like backslash-frac / backslash-theta inside string values,
    which otherwise raise 'Invalid \\escape'). Repair = double any stray
    backslash, applied only as a fallback so valid JSON is untouched."""
    txt = _FENCE.sub("", raw.strip())

    def _try(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
            return None

    d = _try(txt)
    if d is not None:
        return d
    d = _try(_BAD_ESCAPE.sub(r"\\\\", txt))   # repair stray backslashes, retry
    if d is not None:
        return d
    return json.loads(txt)  # raise the original, informative error


# XML/Office-illegal control chars (keep tab 0x09, newline 0x0a, return 0x0d).
# LaTeX like backslash-frac / backslash-theta can decode to 0x0c/0x08 etc.,
# which python-docx/pptx reject — strip them so rendering can't crash.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(obj):
    if isinstance(obj, str):
        return _CTRL.sub("", obj)
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    return obj


def generate(client, model, kind, user_prompt, retries=3):
    last = None
    for _ in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=build_prompt(kind, user_prompt),
                max_tokens=4000, temperature=0.4)
            return _clean(parse_json(r.choices[0].message.content.strip()))
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"generation/parse failed after {retries} tries: {last}")


# --------------------------------------------------------------------------
# Grammar-constrained generation.
#
# The whole point of the efficient on-box generator: a SMALL model doesn't have
# to be smart enough to emit perfect JSON by luck — the decoder is CONSTRAINED
# to the schema, so every token it can produce keeps the output valid. Same
# idea as the circuit route's GBNF grammar, expressed as a JSON Schema so it
# works unchanged on llama.cpp (`response_format: json_schema`) and vLLM
# (guided decoding). Structure is guaranteed; a fine-tune only has to improve
# the words, not the shape.
#
# This is what lets doc/slide generation come HOME to a small on-prem model and
# off the frontier — while web search/fetch stay on-box (DuckDuckGo + httpx),
# so nothing in the loop needs Vertex.
# --------------------------------------------------------------------------

def guided_schema(kind: str) -> dict:
    """JSON Schema the on-box generator is constrained to. Theme enum is derived
    from the relevant palette so it never drifts from the renderer."""
    themes = sorted(SLIDE_THEMES if kind == "slides" else DOC_THEMES)
    if kind == "slides":
        return {
            "type": "object", "additionalProperties": False,
            "required": ["title", "subtitle", "theme", "slides"],
            "properties": {
                "title": {"type": "string"},
                "subtitle": {"type": "string"},
                "theme": {"type": "string", "enum": themes},
                "slides": {
                    "type": "array", "minItems": 4, "maxItems": 10,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["title", "bullets"],
                        "properties": {
                            "title": {"type": "string"},
                            "bullets": {"type": "array", "minItems": 2,
                                        "maxItems": 5, "items": {"type": "string"}},
                            "notes": {"type": "string"},
                        },
                    },
                },
            },
        }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["title", "theme", "sections"],
        "properties": {
            "title": {"type": "string"},
            "theme": {"type": "string", "enum": themes},
            "sections": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["heading", "paragraphs"],
                    "properties": {
                        "heading": {"type": "string"},
                        "paragraphs": {"type": "array", "minItems": 1,
                                       "items": {"type": "string"}},
                    },
                },
            },
        },
    }


def generate_guided(client, model, kind, user_prompt, retries=2):
    """Schema-constrained single-pass generation for an on-box model.

    Passes the schema as the OpenAI-standard `response_format: {type:
    json_schema}` — honoured by llama.cpp's server and vLLM's guided decoding —
    so the model physically cannot emit malformed structure. Falls back to
    unconstrained generate() if a backend doesn't support guided decoding, so
    this is always safe to call."""
    schema = guided_schema(kind)
    rf = {"type": "json_schema",
          "json_schema": {"name": f"paraframes_{kind}", "schema": schema}}
    last = None
    for _ in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=build_prompt(kind, user_prompt),
                max_tokens=4000, temperature=0.3, response_format=rf)
            return _clean(parse_json(r.choices[0].message.content.strip()))
        except Exception as e:  # noqa: BLE001 -- backend may not support it
            last = e
    # Backend can't guide (or kept failing) -> unconstrained path still works.
    return generate(client, model, kind, user_prompt)


# --------------------------------------------------------------------------
# Agentic generation (DEV): OUR OWN agent. The local paraclient model runs a
# search/fetch ReAct loop (tools executed here in Python), then authors the
# deck/doc. Same output shape as generate() -> renders unchanged. No cloud —
# model on the L4, web tools on the box.
# NOTE: dev-only. For prod/edu we will add source-vetting / credibility gating.
# --------------------------------------------------------------------------
import html  # noqa: E402

_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _web_search(query: str, n: int = 5) -> str:
    try:
        from ddgs import DDGS
        rows = list(DDGS().text(query, max_results=n))
    except Exception as e:  # noqa: BLE001
        return f"[search error: {e}]"
    if not rows:
        return "[no results]"
    return "\n".join(
        f"[{i}] {r.get('title','')}\n    {r.get('href','')}\n    {r.get('body','')[:300]}"
        for i, r in enumerate(rows[:n], 1))


def _web_fetch(url: str, limit: int = 5000) -> str:
    """Fetch a page: extract the MAIN content (headings/paragraphs/list items,
    not nav chrome) and surface same-site links so the agent can go deeper."""
    try:
        import httpx
        r = httpx.get(url, timeout=25, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (ParaFramesAgent)"})
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"[fetch error: {e}]"
    # PDFs (papers!) are binary — extract their text, don't HTML-parse them.
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf") or r.content[:5] == b"%PDF-":
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(r.content))
            text = _WS.sub(" ", "\n".join(
                (p.extract_text() or "") for p in reader.pages[:15])).strip()
            return ("[PDF] " + text)[:max(limit, 9000)] if text \
                else "[PDF: no extractable text]"
        except Exception as e:  # noqa: BLE001
            return f"[pdf parse error: {e}]"
    raw = r.text
    try:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse
        soup = BeautifulSoup(raw, "html.parser")
        # collect same-site links FIRST (nav links are how we go deeper)
        base = urlparse(url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"].split("#")[0]).rstrip("/")
            if href and urlparse(href).netloc == base and href not in links \
                    and href.rstrip("/") != url.rstrip("/"):
                links.append(href)
            if len(links) >= 10:
                break
        for t in soup(["script", "style", "noscript", "form"]):
            t.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup
        parts = [el.get_text(" ", strip=True)
                 for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li"])]
        text = "\n".join(p for p in parts if len(p) >= 30)[:limit]
        if not text:
            text = _WS.sub(" ", root.get_text(" ", strip=True))[:limit]
        if links:
            text += "\n\nOTHER PAGES ON THIS SITE (fetch to dig deeper):\n" \
                    + "\n".join(links)
        return text
    except Exception:  # noqa: BLE001 -- bs4 missing / parse failure: crude fallback
        txt = _WS.sub(" ", html.unescape(_TAG.sub(" ", _TAGS.sub(" ", raw)))).strip()
        return txt[:limit]


_AGENT_SYS = (
    "You are a research agent that authors well-researched, substantive documents "
    "and slideshows. You have two tools:\n"
    '  search — {"tool":"search","query":"..."}\n'
    '  fetch  — {"tool":"fetch","url":"..."}\n'
    "On EACH turn output EXACTLY ONE JSON object and nothing else: a tool call, or "
    'the finished deliverable as {"final": <document JSON>}.\n'
    "RESEARCH DEEPLY before writing:\n"
    "- If the request names a website, lab, organization, product, or paper, that "
    "is your STARTING POINT, not the subject. Identify the actual work, research, "
    "findings, and people behind it and present THAT. NEVER just describe the "
    "website, its navigation, or its sections.\n"
    "- Fetch the page, then FOLLOW its links (research / projects / publications / "
    "people / about) and SEARCH for the underlying studies, papers, and specifics. "
    "Gather at least 3 solid sources before finishing.\n"
    "- Capture concrete specifics: what the work actually is, how it works, key "
    "results and numbers, who does it, and why it matters. No generic filler.\n"
    'When you genuinely have depth, return {"final": ...}. The document JSON MUST '
    "match this schema exactly:\n"
)


def generate_agentic(client, model, kind, user_prompt, max_steps=6, min_research=2):
    """Our own local agent: the paraclient model runs a search/fetch loop (tools
    run here), then returns the doc/slides JSON. Enforces a minimum amount of
    real research before it may finalize; falls back to a single-shot generation
    if the loop can't produce a valid document."""
    schema_sys = build_prompt(kind, user_prompt)[0]["content"]
    messages = [{"role": "system", "content": _AGENT_SYS + schema_sys},
                {"role": "user", "content": f"Task: {user_prompt}"}]
    tool_calls = 0
    for step in range(max_steps):
        last = step == max_steps - 1
        if last:
            messages.append({"role": "user", "content":
                'Stop researching. Output ONLY {"final": <document JSON>} now.'})
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, max_tokens=3000, temperature=0.3)
        except Exception:  # noqa: BLE001 -- e.g. context overflow: bail to fallback
            break
        raw = r.choices[0].message.content.strip()
        try:
            obj = parse_json(raw)
        except Exception:  # noqa: BLE001
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content":
                          "Not valid JSON. Reply with ONE JSON object only."}]
            continue
        wants_final = isinstance(obj, dict) and (
            "final" in obj or "slides" in obj or "sections" in obj)
        # don't let it finalize until it has actually researched
        if wants_final and not last and tool_calls < min_research:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content":
                          f"Not yet — you have only used {tool_calls} source(s). "
                          "Dig deeper: fetch this site's research/publication pages "
                          "and search for the specific studies and findings, then "
                          "finalize with concrete details."}]
            continue
        if isinstance(obj, dict) and "final" in obj:
            return _clean(obj["final"])
        if isinstance(obj, dict) and ("slides" in obj or "sections" in obj):
            return _clean(obj)
        if isinstance(obj, dict) and obj.get("tool") == "search":
            obs = _web_search(obj.get("query", "")); tool_calls += 1
        elif isinstance(obj, dict) and obj.get("tool") == "fetch":
            obs = _web_fetch(obj.get("url", "")); tool_calls += 1
        else:
            obs = '[unknown action — use "search", "fetch", or "final"]'
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "OBSERVATION:\n" + obs}]
    return generate(client, model, kind, user_prompt)  # last-resort single shot


def pick_theme(data: dict, override: str | None) -> dict:
    palettes = SLIDE_THEMES if "slides" in data else DOC_THEMES
    default = DEFAULT_SLIDE_THEME if "slides" in data else DEFAULT_DOC_THEME
    name = override or data.get("theme") or default
    if name not in palettes:
        name = default
    return name, palettes[name]


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------

def _rgb(hexstr):
    return RGBColor.from_string(hexstr)


def render_pptx(data: dict, theme: dict, out: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def bg(slide, color=None):
        f = slide.background.fill
        f.solid(); f.fore_color.rgb = _rgb(color or theme["bg"])

    def rect(slide, left, top, width, height, color):
        shape = slide.shapes.add_shape(
            1, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.fill.solid(); shape.fill.fore_color.rgb = _rgb(color)
        shape.line.fill.background()
        return shape

    def textbox(slide, left, top, width, height):
        tb = slide.shapes.add_textbox(Inches(left), Inches(top),
                                      Inches(width), Inches(height))
        tb.text_frame.word_wrap = True
        return tb.text_frame

    # ---- title slide ----
    s = prs.slides.add_slide(blank); bg(s)
    rect(s, 0, 0, 13.333, 0.18, theme["accent"])
    rect(s, 0, 7.32, 13.333, 0.18, theme.get("accent2", theme["accent"]))
    if theme.get("layout") in ("blocks", "tech"):
        rect(s, 11.6, 0.18, 1.73, 0.48, theme.get("accent2", theme["accent"]))
        rect(s, 0, 6.84, 1.73, 0.48, theme["accent"])
    tf = textbox(s, 1, 2.4, 11.3, 2)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = data.get("title", "Untitled")
    run.font.size = Pt(44); run.font.bold = True
    run.font.name = theme["font_head"]; run.font.color.rgb = _rgb(theme["title"])
    if data.get("subtitle"):
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run(); r2.text = data["subtitle"]
        r2.font.size = Pt(22); r2.font.name = theme["font_body"]
        r2.font.color.rgb = _rgb(theme["accent"])

    # ---- content slides ----
    for index, slide in enumerate(data.get("slides", []), start=1):
        s = prs.slides.add_slide(blank)
        bg(s, theme.get("bg_alt") if index % 2 == 0 else theme["bg"])
        layout = theme.get("layout", "band")
        accent2 = theme.get("accent2", theme["accent"])
        if layout == "band":
            rect(s, 0, 0, 0.18, 7.5, theme["accent"])
        elif layout == "blocks":
            rect(s, 0, 0, 1.55, 0.22, theme["accent"])
            rect(s, 11.78, 7.28, 1.55, 0.22, accent2)
        elif layout == "frame":
            rect(s, 0, 0, 0.12, 7.5, theme["accent"])
            rect(s, 0, 7.38, 13.333, 0.12, accent2)
        elif layout == "organic":
            rect(s, 0, 0, 13.333, 0.12, theme["accent"])
            rect(s, 12.98, 0, 0.35, 7.5, accent2)
        elif layout == "tech":
            rect(s, 0, 0, 13.333, 0.12, theme["accent"])
            rect(s, 13.21, 0, 0.12, 7.5, accent2)
        # accent bar under the title
        bar = s.shapes.add_shape(1, Inches(0.8), Inches(1.35), Inches(3.2), Pt(4))
        bar.fill.solid(); bar.fill.fore_color.rgb = _rgb(
            accent2 if index % 2 == 0 else theme["accent"])
        bar.line.fill.background()
        ttf = textbox(s, 0.8, 0.5, 11.7, 1)
        tp = ttf.paragraphs[0]; tr = tp.add_run()
        tr.text = slide.get("title", ""); tr.font.size = Pt(32); tr.font.bold = True
        tr.font.name = theme["font_head"]; tr.font.color.rgb = _rgb(theme["title"])
        btf = textbox(s, 1.0, 1.7, 11.3, 5.2)
        first = True
        for bullet in slide.get("bullets", []):
            p = btf.paragraphs[0] if first else btf.add_paragraph()
            first = False
            r = p.add_run(); r.text = "•  " + str(bullet)
            r.font.size = Pt(20); r.font.name = theme["font_body"]
            r.font.color.rgb = _rgb(theme["body"])
            p.space_after = Pt(10)
        ntf = textbox(s, 11.9, 6.85, 0.6, 0.35)
        np = ntf.paragraphs[0]; np.alignment = PP_ALIGN.RIGHT
        nr = np.add_run(); nr.text = str(index)
        nr.font.size = Pt(10); nr.font.name = theme["font_body"]
        nr.font.color.rgb = _rgb(accent2)
        if slide.get("notes"):
            s.notes_slide.notes_text_frame.text = str(slide["notes"])

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))


def render_docx(data: dict, theme: dict, out: Path):
    d = docx.Document()
    normal = d.styles["Normal"]
    normal.font.name = theme["font_body"]; normal.font.size = DocxPt(11)
    normal.font.color.rgb = DocxRGB.from_string(theme["body"])

    title = d.add_heading(data.get("title", "Untitled"), level=0)
    for run in title.runs:
        run.font.color.rgb = DocxRGB.from_string(theme["title"])
        run.font.name = theme["font_head"]

    for sec in data.get("sections", []):
        h = d.add_heading(sec.get("heading", ""), level=1)
        for run in h.runs:
            run.font.color.rgb = DocxRGB.from_string(theme["accent"])
            run.font.name = theme["font_head"]
        for para in sec.get("paragraphs", []):
            d.add_paragraph(str(para))

    out.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(out))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["slides", "doc"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="ParaFrames/ParaClient-v2.2")
    ap.add_argument("--theme", default=None, choices=list(THEMES),
                    help="override the model's auto-chosen theme")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"[*] Generating {args.kind} via {args.model}...")
    data = generate(client, args.model, args.kind, args.prompt)
    name, theme = pick_theme(data, args.theme)
    out = Path(args.out)
    if args.kind == "slides":
        render_pptx(data, theme, out)
        n = len(data.get("slides", []))
        print(f"[✓] Wrote {n}-slide deck -> {out}  (theme: {name})")
    else:
        render_docx(data, theme, out)
        n = len(data.get("sections", []))
        print(f"[✓] Wrote {n}-section document -> {out}  (theme: {name})")


if __name__ == "__main__":
    main()
