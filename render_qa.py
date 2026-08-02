#!/usr/bin/env python3
"""
render_qa.py — dependency-free SVG montage of CAD training examples, so we can
eyeball the procedurally-generated ground truth (2D sketch rows) before/while
the adapter trains. No matplotlib/CAD libs required.

    python render_qa.py --data data/cad.jsonl --out out/cad_qa.svg
"""
import argparse
import json
import math

CELL_W, CELL_H = 380, 300          # cell footprint
DRAW_W, DRAW_H = 340, 235          # drawing area inside a cell
CAPTION_H = 46
COLS = 2


def bbox(ents):
    xs, ys = [], []
    for e in ents:
        t = e["type"]
        if t == "line":
            xs += [e["x1"], e["x2"]]; ys += [e["y1"], e["y2"]]
        elif t == "rect":
            xs += [e["x"], e["x"] + e["w"]]; ys += [e["y"], e["y"] + e["h"]]
        elif t in ("circle", "arc"):
            xs += [e["cx"] - e["r"], e["cx"] + e["r"]]
            ys += [e["cy"] - e["r"], e["cy"] + e["r"]]
        elif t == "polyline":
            for px, py in e["points"]:
                xs.append(px); ys.append(py)
        elif t == "text":
            xs.append(e["x"]); ys.append(e["y"])
    if not xs:
        return 0, 0, 1, 1
    return min(xs), min(ys), max(xs), max(ys)


def arc_pts(cx, cy, r, a0, a1, n=24):
    if a1 < a0:
        a1 += 360
    return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in [a0 + (a1 - a0) * i / n for i in range(n + 1)]]


def draw_entities(ents):
    """Return SVG element strings in CAD coordinates (placed under a flip group)."""
    S = []
    ST = ('fill="none" stroke="#12335b" stroke-width="1" '
          'vector-effect="non-scaling-stroke"')
    for e in ents:
        t = e["type"]
        if t == "line":
            S.append(f'<line x1="{e["x1"]}" y1="{e["y1"]}" x2="{e["x2"]}" '
                     f'y2="{e["y2"]}" {ST}/>')
        elif t == "rect":
            S.append(f'<rect x="{e["x"]}" y="{e["y"]}" width="{e["w"]}" '
                     f'height="{e["h"]}" {ST}/>')
        elif t == "circle":
            S.append(f'<circle cx="{e["cx"]}" cy="{e["cy"]}" r="{e["r"]}" {ST}/>')
        elif t == "arc":
            pts = arc_pts(e["cx"], e["cy"], e["r"], e["start_deg"], e["end_deg"])
            d = " ".join(f'{"M" if i == 0 else "L"}{x:.1f},{y:.1f}'
                         for i, (x, y) in enumerate(pts))
            S.append(f'<path d="{d}" {ST}/>')
        elif t == "polyline":
            pts = e["points"]
            d = " ".join(f'{"M" if i == 0 else "L"}{x},{y}'
                         for i, (x, y) in enumerate(pts))
            if e.get("closed"):
                d += " Z"
            S.append(f'<path d="{d}" {ST}/>')
    return S


def cell_svg(ents, col, row, caption):
    ox, oy = col * CELL_W, row * CELL_H
    x0, y0, x1, y1 = bbox(ents)
    bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    s = min(DRAW_W / bw, DRAW_H / bh) * 0.9
    padx = ox + (CELL_W - s * bw) / 2
    pady = oy + CAPTION_H + (DRAW_H - s * bh) / 2
    # flip Y (CAD up -> SVG down): screen = padx + s*(x-x0), pady + s*(y1-y)
    g = (f'<g transform="translate({padx:.1f},{pady:.1f}) scale({s:.4f},{-s:.4f}) '
         f'translate({-x0:.1f},{-y1:.1f})">')
    body = "".join(draw_entities(ents))
    # caption + cell border in screen coords
    cap = caption if len(caption) <= 62 else caption[:59] + "..."
    frame = (f'<rect x="{ox+4}" y="{oy+4}" width="{CELL_W-8}" height="{CELL_H-8}" '
             f'fill="#fff" stroke="#d0d7de" stroke-width="1"/>'
             f'<text x="{ox+14}" y="{oy+26}" font-family="Segoe UI,Arial" '
             f'font-size="13" fill="#0b2545">{escape(cap)}</text>'
             f'<text x="{ox+14}" y="{oy+42}" font-family="Segoe UI,Arial" '
             f'font-size="11" fill="#6b7684">{len(ents)} entities</text>')
    return frame + g + body + "</g>"


def escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cad.jsonl")
    ap.add_argument("--out", default="out/cad_qa.svg")
    args = ap.parse_args()

    # pick a diverse set of 2D sketch rows by prompt keyword
    wanted = ["floor plan", "floor plan", "flange", "plate", "bracket", "gasket"]
    picks = []
    rows = [json.loads(l) for l in open(args.data)]
    used = set()
    for kw in wanted:
        for i, r in enumerate(rows):
            if i in used or r["mode"] != "sketch":
                continue
            if kw in r["messages"][1]["content"].lower():
                picks.append(r); used.add(i); break

    cells = []
    for i, r in enumerate(picks):
        data = json.loads(r["messages"][2]["content"])
        cells.append(cell_svg(data["entities"], i % COLS, i // COLS,
                              r["messages"][1]["content"]))
    rows_n = (len(picks) + COLS - 1) // COLS
    W, H = COLS * CELL_W, rows_n * CELL_H
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#eef2f7"/>'
           + "".join(cells) + "</svg>")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg)
    print(f"[✓] Wrote {len(picks)} QA cells -> {args.out}")


if __name__ == "__main__":
    main()
