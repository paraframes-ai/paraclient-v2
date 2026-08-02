#!/usr/bin/env python3
"""Parse USCIS civics-test PDFs -> structured civics source JSONL.

Emits items in the civics_sources schema (q, a, answers, strand, grade_bands,
time_varying, source, license). Output feeds build_civics_dataset.py, then the
HUMAN review gate. Nothing here is training-ready.
"""
import json
import re

from pypdf import PdfReader

BULLETS = ("▪", "•")  # ▪ (100q), • (128q)

STRAND_KEYWORDS = [
    ("Principles of American Democracy", "foundations"),
    ("System of Government", "constitution"),
    ("Rights and Responsibilities", "citizen_roles"),
    ("Colonial Period", "foundations"),
    ("1800s", "foundations"),
    ("Recent American History", "civic_life"),
    ("American History", "foundations"),
    ("Geography", "civic_life"),
    ("Symbols", "civic_life"),
    ("Holidays", "civic_life"),
    ("Integrated Civics", "civic_life"),
    ("American Government", "constitution"),
]

TV_TRIGGERS = [
    "now", "current", "name your", "your u.s. senator", "your senator",
    "your u.s. representative", "your representative", "your state",
    "governor of your state", "capital of your state",
    "political party of the president", "speaker of the house",
    "who is the president", "who is the vice president", "serving",
    "will vary", "visit", "answers will vary", "testupdates",
]


# Real section headers only: "A: Principles ...", "AMERICAN GOVERNMENT", etc.
# Question-number lines are matched BEFORE this, so imperative questions like
# "Name one war ... in the 1800s." are never mistaken for headers.
def is_header(line: str) -> bool:
    return bool(re.match(r"^[A-Z]:\s", line)) or (line.isupper() and 3 < len(line) < 60)


def strand_for(header: str) -> str:
    for k, s in STRAND_KEYWORDS:
        if k.lower() in header.lower():
            return s
    return "constitution"


# uscis.gov anchored to the WHOLE line so answer bullets like
# "Visit uscis.gov/citizenship/testupdates ..." are preserved.
BOILER = re.compile(
    r"^(-?\d+-?|\d+ of \d+|\*|(www\.)?uscis\.gov(/citizenship)?/?|M-\d.*|"
    r".*If you are 65.*|.*\(rev.*|65/20.*|.*Civics \(History and Government\).*|"
    r".*128 Civics Questions.*|.*Listed below are.*|.*The 100 civics.*)$",
    re.IGNORECASE,
)


def clean_lines(txt: str):
    for raw in txt.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw.replace("\t", " ")).strip()
        if not line or BOILER.match(line):
            continue
        yield line


def parse(path: str, year: str):
    txt = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    recs, cur, strand = [], None, "constitution"
    for line in clean_lines(txt):
        m = re.match(r"^(\d{1,3})\.\s+(.*)", line)
        if m:                                        # question first — wins over header
            if cur:
                recs.append(cur)
            cur = {"num": int(m.group(1)), "q": m.group(2).rstrip(" *").strip(),
                   "answers": [], "strand": strand}
        elif is_header(line):
            strand = strand_for(line)
        elif line[0] in BULLETS:
            if cur is not None:
                cur["answers"].append(line.lstrip("".join(BULLETS) + " ").strip())
        elif cur is not None:                        # wrapped question or answer
            if cur["answers"]:
                cur["answers"][-1] += " " + line
            else:
                cur["q"] += " " + line
    if cur:
        recs.append(cur)

    items = []
    for r in recs:
        blob = (r["q"] + " " + " ".join(r["answers"])).lower()
        tv = any(t in blob for t in TV_TRIGGERS)
        items.append({
            "q": r["q"],
            "a": "" if tv else "; ".join(r["answers"]),
            "answers": r["answers"],
            "strand": r["strand"],
            "grade_bands": ["middle", "high"],
            "time_varying": tv,
            "concept": "",
            "source": f"USCIS-{year}-Q{r['num']}",
            "license": "public_domain",
            "attribution": "",
        })
    return items


def main():
    jobs = [
        ("civics_sources/raw/uscis_100_2008.pdf", "2008", "civics_sources/uscis_100.jsonl", 100),
        ("civics_sources/raw/uscis_128_2025.pdf", "2025", "civics_sources/uscis_128.jsonl", 128),
    ]
    for pdf, year, out, expected in jobs:
        items = parse(pdf, year)
        with open(out, "w") as w:
            for it in items:
                w.write(json.dumps(it) + "\n")
        nums = {int(i["source"].split("Q")[1]) for i in items}
        missing = [x for x in range(1, expected + 1) if x not in nums]
        tv = sum(i["time_varying"] for i in items)
        gaps = [i["source"] for i in items if not i["answers"] and not i["time_varying"]]
        print(f"{out}: {len(items)}/{expected} questions, {tv} time_varying; "
              f"missing={missing}; no-answer&not-tv={gaps}")


if __name__ == "__main__":
    main()
