#!/usr/bin/env python3
"""Build balanced SFT data for slideshow-theme selection."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from generate_media import SLIDE_THEMES, theme_selector_messages

R = random.Random(29)

TOPICS = {
    "academic": [
        "a thesis defense on urban migration", "a seminar on constitutional law",
        "an analysis of Shakespearean tragedy", "a graduate research proposal",
        "the historiography of the Silk Road", "a formal mathematics symposium",
        "a literature review on language acquisition", "an archaeology lecture",
        "a philosophy colloquium on ethics", "a university admissions lecture",
        "a museum talk about Renaissance art", "a scholarly economics briefing",
    ],
    "playful": [
        "the alphabet for kindergarten", "friendly dinosaurs for first graders",
        "a classroom birthday celebration", "learning shapes through a treasure hunt",
        "a colorful introduction to fractions", "an elementary school talent show",
        "ocean animals for young children", "a choose-your-own fairy tale",
        "a fun spelling bee kickoff", "healthy snacks for second graders",
        "a school field-day announcement", "a playful tour of the solar system",
    ],
    "corporate": [
        "a quarterly business review", "an investor update for a seed-stage company",
        "a product launch plan", "a customer success operating review",
        "a consulting recommendation", "an annual strategy offsite",
        "a sales pipeline review", "a board meeting summary",
        "a market-entry proposal", "a project status report",
        "a partnership pitch", "a professional development workshop",
    ],
    "nature": [
        "rainforest biodiversity", "the water cycle", "local watershed conservation",
        "plant cell biology", "renewable energy and ecosystems", "pollinator habitats",
        "ocean currents", "sustainable agriculture", "forest succession",
        "wildlife conservation", "climate adaptation", "the geology of national parks",
    ],
    "tech": [
        "a cybersecurity architecture", "an introduction to robotics",
        "a machine-learning system design", "a coding bootcamp kickoff",
        "a cloud migration architecture", "the future of quantum computing",
        "an API platform overview", "a developer conference talk",
        "a data-center network", "a spaceflight engineering review",
        "a mobile-app technical demo", "an electronics and circuits lesson",
    ],
}

AUDIENCES = {
    "academic": ["university faculty", "graduate students", "a scholarly review panel"],
    "playful": ["kindergarten students", "children ages 6–9", "an elementary classroom"],
    "corporate": ["executives", "investors", "customers", "a cross-functional team"],
    "nature": ["middle-school science students", "community members", "park visitors"],
    "tech": ["software engineers", "STEM students", "technical decision-makers"],
}

TONES = {
    "academic": ["scholarly and authoritative", "formal and evidence-led", "timeless"],
    "playful": ["joyful and energetic", "friendly and colorful", "curious and whimsical"],
    "corporate": ["polished and confident", "clean and persuasive", "modern and concise"],
    "nature": ["organic and calm", "hopeful and earthy", "fresh and accessible"],
    "tech": ["futuristic and precise", "bold and high-contrast", "sleek and technical"],
}

PURPOSES = ["teach the fundamentals", "win support", "explain the key ideas",
            "open a workshop", "summarize the plan", "tell a memorable story"]

TEMPLATES = [
    "Create a {length}-slide presentation about {topic} for {audience}. Make it {tone} and {purpose}.",
    "I need a {tone} deck on {topic}. The audience is {audience}; its goal is to {purpose}.",
    "Design direction for {topic}: {length} slides, aimed at {audience}, with a {tone} feeling. It should {purpose}.",
    "Build a presentation for {audience} about {topic}. Keep the visual language {tone}; use it to {purpose}.",
]

OOD_CASES = {
    "academic": [
        "Turn my archival findings into a restrained faculty colloquium deck.",
        "Present a peer-reviewed argument about algorithmic bias to a dissertation committee.",
        "I am defending a proof before mathematicians; make the visuals serious and timeless.",
        "Build lecture slides comparing primary sources for an upper-level history seminar.",
        "Summarize this ecology paper for journal club, with citations taking visual priority.",
    ],
    "playful": [
        "Help a room of six-year-olds meet the planets as funny characters.",
        "Make a bright show-and-tell deck about my puppy for first grade.",
        "Teach counting to kindergarteners through a pirate treasure adventure.",
        "Create an energetic classroom game about healthy food for children aged seven.",
        "Explain how computers think to young kids using friendly robots and big visual moments.",
    ],
    "corporate": [
        "Brief our board on climate exposure, mitigation costs, and next-quarter decisions.",
        "Turn customer retention numbers into a crisp executive operating review.",
        "Pitch a hospital partnership to its procurement and leadership teams.",
        "Give investors a concise update on runway, traction, risks, and hiring.",
        "Recommend whether the company should enter Japan; this is for the CEO and CFO.",
    ],
    "nature": [
        "Invite local families to restore the creek and protect native wildlife.",
        "Tell park visitors how fire helps a forest renew itself.",
        "Explain coral bleaching to a community group with an earthy, hopeful visual voice.",
        "Create a field-guide-style lesson on pollinators for middle school science.",
        "Show how a seed becomes a mature tree in a calm, organic visual story.",
    ],
    "tech": [
        "Walk engineers through our zero-trust migration and threat model.",
        "Demo a new developer API at a launch event; make it sharp and high contrast.",
        "Explain a spacecraft guidance stack to an avionics review panel.",
        "Present the architecture and latency budget of our inference platform.",
        "Teach high-school robotics students how sensors, control loops, and motors connect.",
    ],
}


def example(theme: str) -> dict:
    prompt = R.choice(TEMPLATES).format(
        length=R.randint(5, 10), topic=R.choice(TOPICS[theme]),
        audience=R.choice(AUDIENCES[theme]), tone=R.choice(TONES[theme]),
        purpose=R.choice(PURPOSES))
    messages = theme_selector_messages(prompt)
    messages.append({"role": "assistant",
                     "content": json.dumps({"theme": theme}, separators=(",", ":"))})
    return {"subject": "slideshow_theme", "mode": "select",
            "theme": theme, "messages": messages}


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def ood_rows() -> list[dict]:
    rows = []
    for theme, prompts in OOD_CASES.items():
        for prompt in prompts:
            messages = theme_selector_messages(prompt)
            messages.append({"role": "assistant", "content": json.dumps(
                {"theme": theme}, separators=(",", ":"))})
            rows.append({"subject": "slideshow_theme", "mode": "select",
                         "theme": theme, "messages": messages})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", default="data/slideshow_theme.jsonl")
    ap.add_argument("--holdout-out", default="data/slideshow_theme_holdout.jsonl")
    ap.add_argument("--ood-out", default="data/slideshow_theme_ood.jsonl")
    ap.add_argument("--holdout-fraction", type=float, default=0.1)
    args = ap.parse_args()
    themes = sorted(SLIDE_THEMES)
    rows = [example(themes[i % len(themes)]) for i in range(args.n)]
    R.shuffle(rows)
    split = max(1, round(len(rows) * args.holdout_fraction))
    holdout, train = rows[:split], rows[split:]
    write(Path(args.out), train)
    write(Path(args.holdout_out), holdout)
    write(Path(args.ood_out), ood_rows())
    train_counts = {theme: sum(r["theme"] == theme for r in train) for theme in themes}
    print(f"wrote train={len(train)} {train_counts} -> {args.out}")
    print(f"wrote holdout={len(holdout)} -> {args.holdout_out}")
    print(f"wrote hand-authored OOD={len(ood_rows())} -> {args.ood_out}")


if __name__ == "__main__":
    main()
