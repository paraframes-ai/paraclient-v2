#!/usr/bin/env python3
"""
build_cad_dataset.py — synthesize the CAD adapter's training set for ParaClient.

Why procedural (not LLM-generated): the base model's weakness IS geometry —
overlaps, wrong dimensions, labels instead of shapes. Asking a cloud LLM to make
the training data would bake those same errors in. Instead we GENERATE exact,
valid, real-dimension 2D sketches and 3D solids with a parametric generator and
pair each with a natural-language request. The adapter learns real CAD geometry
from ground truth, not from another model's guesses.

Output rows match the tutor training schema exactly (subject/mode + messages),
so train_adapter.py consumes it unchanged:

    python train_adapter.py --subject cad --data data/cad.jsonl

Each row's system prompt is the SAME production prompt the gateway serves
(cad_schema.sys_for), so the adapter is tuned to the instruction it will see.

    python build_cad_dataset.py --n 2400 --out data/cad.jsonl
"""
import argparse
import json
import math
import random
from pathlib import Path

from cad_schema import sys_for, clean_sketch, clean_solid

R = random.Random()

# --------------------------------------------------------------------------
# tiny entity/solid constructors
# --------------------------------------------------------------------------

def line(x1, y1, x2, y2):
    return {"type": "line", "x1": r0(x1), "y1": r0(y1), "x2": r0(x2), "y2": r0(y2)}


def circle(cx, cy, r):
    return {"type": "circle", "cx": r0(cx), "cy": r0(cy), "r": r0(r)}


def arc(cx, cy, r, s, e):
    return {"type": "arc", "cx": r0(cx), "cy": r0(cy), "r": r0(r),
            "start_deg": r0(s), "end_deg": r0(e)}


def rect(x, y, w, h):
    return {"type": "rect", "x": r0(x), "y": r0(y), "w": r0(w), "h": r0(h)}


def poly(points, closed=True):
    return {"type": "polyline", "closed": closed,
            "points": [[r0(px), r0(py)] for px, py in points]}


def text(x, y, h, value):
    return {"type": "text", "x": r0(x), "y": r0(y), "h": r0(h), "value": value}


def box(x, y, z, w, d, h):
    return {"type": "box", "x": r0(x), "y": r0(y), "z": r0(z),
            "w": r0(w), "d": r0(d), "h": r0(h)}


def cyl(cx, cy, z, r, h):
    return {"type": "cylinder", "cx": r0(cx), "cy": r0(cy), "z": r0(z),
            "r": r0(r), "h": r0(h)}


def sphere(cx, cy, cz, r):
    return {"type": "sphere", "cx": r0(cx), "cy": r0(cy), "cz": r0(cz), "r": r0(r)}


def extrude(profile, z, h):
    return {"type": "extrude", "profile": [[r0(px), r0(py)] for px, py in profile],
            "z": r0(z), "h": r0(h)}


def hole3(cx, cy, z, r, depth):
    return {"type": "hole", "cx": r0(cx), "cy": r0(cy), "z": r0(z),
            "r": r0(r), "depth": r0(depth)}


def r0(v):
    v = round(float(v), 1)
    return int(v) if v == int(v) else v


def pick(*opts):
    return R.choice(opts)


# --------------------------------------------------------------------------
# 2D — floor plans (guillotine partition into rectangular rooms)
# --------------------------------------------------------------------------
ROOM_KINDS = ["bedroom", "living room", "kitchen", "bathroom", "dining room",
              "office", "closet", "hallway"]


def _split(rooms, min_side):
    """Split the largest room in-place once; return True if a split happened."""
    rooms.sort(key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = rooms[0]
    if w >= h and w > 2 * min_side:
        f = R.uniform(0.4, 0.6)
        cut = w * f
        rooms[0] = (x, y, cut, h)
        rooms.append((x + cut, y, w - cut, h))
        return True
    if h > 2 * min_side:
        f = R.uniform(0.4, 0.6)
        cut = h * f
        rooms[0] = (x, y, w, cut)
        rooms.append((x, y + cut, w, h - cut))
        return True
    return False


def _furnish(x, y, w, h, kind, ents):
    """Place one clean furniture piece against a wall, inside the room."""
    m = 250  # keep clear of walls
    ix, iy, iw, ih = x + m, y + m, w - 2 * m, h - 2 * m
    if iw < 400 or ih < 400:
        return
    cx, cy = x + w / 2, y + h / 2
    if kind == "bedroom":
        bw, bh = min(1500, iw), min(2000, ih)
        bx, by = x + m, y + h - m - bh
        ents.append(rect(bx, by, bw, bh))                 # bed
        ents.append(rect(bx + bw * 0.1, by + bh - 350, bw * 0.8, 300))  # pillow
    elif kind == "living room":
        sw, sh = min(2200, iw), 850
        sx, sy = cx - sw / 2, y + m
        ents.append(rect(sx, sy, sw, sh))                 # sofa body
        ents.append(rect(sx, sy, sw, 180))                # backrest
    elif kind == "kitchen":
        cw = min(iw, w - 2 * m)
        ents.append(rect(x + m, y + h - m - 600, cw, 600))  # counter
        ents.append(circle(x + m + cw * 0.3, y + h - m - 300, 200))  # sink
    elif kind == "bathroom":
        tw, th = min(1700, iw), 750
        ents.append(rect(x + m, y + m, tw, th))           # tub
        ents.append(circle(x + w - m - 250, y + m + 250, 220))  # sink
    elif kind == "dining room":
        tw, th = min(1400, iw * 0.7), min(900, ih * 0.7)
        tx, ty = cx - tw / 2, cy - th / 2
        ents.append(rect(tx, ty, tw, th))                 # table
        for dx, dy in [(tw * 0.25, -260), (tw * 0.75, -260),
                       (tw * 0.25, th + 260), (tw * 0.75, th + 260)]:
            ents.append(circle(tx + dx, ty + dy, 200))    # chairs
    elif kind == "office":
        ents.append(rect(x + m, y + m, min(1400, iw), 700))   # desk
        ents.append(circle(x + m + 500, y + m + 900, 220))    # chair


def gen_floorplan():
    W = R.randrange(5000, 10000, 250)
    D = R.randrange(4000, 8000, 250)
    n_rooms = R.randint(2, 5)
    rooms = [(0, 0, W, D)]
    while len(rooms) < n_rooms:
        if not _split(rooms, 2200):
            break
    kinds = R.sample(ROOM_KINDS, k=min(len(rooms), len(ROOM_KINDS)))
    while len(kinds) < len(rooms):
        kinds.append(R.choice(ROOM_KINDS))

    ents = [rect(0, 0, W, D)]  # exterior wall
    for (x, y, w, h), kind in zip(rooms, kinds):
        ents.append(rect(x, y, w, h))                     # room partition
        # door: quarter-arc swing on the longest interior wall segment
        if w >= h:
            dx = x + w * R.uniform(0.3, 0.6)
            ents.append(arc(dx, y + h, min(800, h * 0.4), 180, 270))
        else:
            dy = y + h * R.uniform(0.3, 0.6)
            ents.append(arc(x + w, dy, min(800, w * 0.4), 90, 180))
        _furnish(x, y, w, h, kind, ents)
        ents.append(text(x + 200, y + h - 220, 240, kind.upper()))

    room_list = ", ".join(kinds)
    prompt = pick(
        f"Draw a floor plan for a {W/1000:.1f} m by {D/1000:.1f} m home with "
        f"{len(rooms)} rooms: {room_list}.",
        f"Sketch a {len(rooms)}-room floor plan ({room_list}) that fits in "
        f"{W} x {D} mm, with furniture and doors.",
        f"I need a 2D floor plan, about {W}mm wide and {D}mm deep, laid out as "
        f"{room_list}. Include basic furniture.",
    )
    return prompt, {"units": "mm", "entities": ents}


# --------------------------------------------------------------------------
# 2D — mechanical
# --------------------------------------------------------------------------

def gen_plate():
    w = R.randrange(80, 320, 10)
    h = R.randrange(60, 240, 10)
    inset = R.choice([10, 12, 15, 20])
    hr = R.choice([2.5, 3, 4, 5, 6])
    ents = [rect(0, 0, w, h)]
    for cx, cy in [(inset, inset), (w - inset, inset),
                   (inset, h - inset), (w - inset, h - inset)]:
        ents.append(circle(cx, cy, hr))
    if R.random() < 0.4:
        ents.append(circle(w / 2, h / 2, R.choice([8, 10, 12, 15])))  # center bore
    prompt = pick(
        f"A rectangular mounting plate {w} x {h} mm with 4 corner holes "
        f"(radius {hr} mm), inset {inset} mm.",
        f"Draw a {w}x{h} mm plate, four bolt holes {hr}mm radius near the corners.",
        f"2D drawing of a {w} by {h} millimetre bracket plate with corner "
        f"mounting holes of {hr} mm radius.",
    )
    return prompt, {"units": "mm", "entities": ents}


def gen_flange():
    od = R.randrange(80, 220, 10)
    R_out = od / 2
    bore = R.choice([20, 25, 30, 40, 50])
    n = R.choice([4, 6, 8])
    bc = R_out * R.uniform(0.62, 0.78)
    hr = R.choice([3, 4, 5, 6])
    ents = [circle(R_out, R_out, R_out), circle(R_out, R_out, bore / 2)]
    for i in range(n):
        a = 2 * math.pi * i / n
        ents.append(circle(R_out + bc * math.cos(a), R_out + bc * math.sin(a), hr))
    prompt = pick(
        f"A circular flange, {od} mm outer diameter, {bore} mm bore, with {n} "
        f"bolt holes ({hr} mm radius) on the bolt circle.",
        f"Draw a {od}mm OD flange with a {bore}mm center bore and {n} equally "
        f"spaced mounting holes.",
        f"2D flange: outer diameter {od} mm, {n} holes of radius {hr} mm around "
        f"a {2*bc:.0f} mm bolt circle, central bore {bore} mm.",
    )
    return prompt, {"units": "mm", "entities": ents}


def gen_bracket2d():
    a = R.randrange(60, 160, 10)   # leg length
    t = R.randrange(20, 50, 5)     # leg thickness
    b = R.randrange(60, 160, 10)
    pts = [(0, 0), (a, 0), (a, t), (t, t), (t, b), (0, b)]
    ents = [poly(pts, closed=True)]
    hr = R.choice([3, 4, 5])
    ents.append(circle(a - 15, t / 2, hr))
    ents.append(circle(t / 2, b - 15, hr))
    prompt = pick(
        f"An L-shaped bracket outline, {a} mm x {b} mm legs, {t} mm thick, with "
        f"a mounting hole ({hr} mm) on each leg.",
        f"Draw an L bracket: {a}mm horizontal leg, {b}mm vertical leg, {t}mm "
        f"wide, one hole per leg.",
    )
    return prompt, {"units": "mm", "entities": ents}


def gen_gasket():
    w = R.randrange(100, 260, 10)
    h = R.randrange(80, 200, 10)
    wall = R.choice([12, 15, 20, 25])
    ents = [rect(0, 0, w, h), rect(wall, wall, w - 2 * wall, h - 2 * wall)]
    hr = R.choice([3, 4, 5])
    for cx, cy in [(wall / 2, wall / 2), (w - wall / 2, wall / 2),
                   (wall / 2, h - wall / 2), (w - wall / 2, h - wall / 2)]:
        ents.append(circle(cx, cy, hr))
    prompt = pick(
        f"A rectangular gasket {w} x {h} mm with a {wall} mm border (open "
        f"center) and four corner bolt holes.",
        f"Draw a flat gasket: {w}x{h}mm outer, {wall}mm wall, cut-out middle, "
        f"holes at the corners.",
    )
    return prompt, {"units": "mm", "entities": ents}


def gen_disc():
    d = R.randrange(60, 200, 10)
    R_out = d / 2
    bore = R.choice([10, 12, 16, 20])
    ents = [circle(R_out, R_out, R_out), circle(R_out, R_out, bore / 2)]
    # keyway
    kw = bore * 0.4
    ents.append(rect(R_out - kw / 2, R_out + bore / 2, kw, bore * 0.35))
    prompt = pick(
        f"A pulley disc, {d} mm diameter, with a {bore} mm bore and a keyway.",
        f"Draw a {d}mm circular disc with a central {bore}mm shaft hole and keyway slot.",
    )
    return prompt, {"units": "mm", "entities": ents}


# --------------------------------------------------------------------------
# 3D — solids
# --------------------------------------------------------------------------

def gen_block():
    w = R.randrange(30, 160, 5)
    d = R.randrange(20, 120, 5)
    h = R.randrange(10, 80, 5)
    sol = [box(0, 0, 0, w, d, h)]
    if R.random() < 0.55:
        hr = R.choice([3, 4, 5, 6, 8])
        sol.append(hole3(w / 2, d / 2, 0, hr, h))
    prompt = pick(
        f"A rectangular block {w} x {d} x {h} mm"
        + (", with a through hole in the center." if len(sol) > 1 else "."),
        f"Model a {w}x{d}x{h} mm solid block"
        + (f" bored through the middle." if len(sol) > 1 else "."),
    )
    return prompt, {"units": "mm", "solids": sol}


def gen_shaft():
    r = R.choice([8, 10, 12, 15, 20, 25])
    h = R.randrange(30, 160, 5)
    sol = [cyl(0, 0, 0, r, h)]
    if R.random() < 0.5:  # stepped shaft
        r2 = max(4, r - R.choice([3, 4, 5]))
        h2 = R.randrange(15, 60, 5)
        sol.append(cyl(0, 0, h, r2, h2))
        prompt = pick(
            f"A stepped shaft: {2*r} mm diameter for {h} mm, then {2*r2} mm "
            f"diameter for {h2} mm.",
            f"Model a two-diameter shaft, {2*r}mm then {2*r2}mm, total {h+h2}mm long.",
        )
    else:
        prompt = pick(
            f"A cylindrical shaft, {2*r} mm diameter, {h} mm long.",
            f"Model a {2*r}mm diameter rod {h}mm tall.",
        )
    return prompt, {"units": "mm", "solids": sol}


def gen_plate3d():
    w = R.randrange(80, 260, 10)
    d = R.randrange(60, 200, 10)
    t = R.choice([4, 5, 6, 8, 10])
    inset = R.choice([10, 12, 15])
    hr = R.choice([2.5, 3, 4, 5])
    sol = [box(0, 0, 0, w, d, t)]
    for cx, cy in [(inset, inset), (w - inset, inset),
                   (inset, d - inset), (w - inset, d - inset)]:
        sol.append(hole3(cx, cy, 0, hr, t))
    prompt = pick(
        f"A flat mounting plate {w} x {d} mm, {t} mm thick, with 4 corner holes "
        f"({hr} mm radius).",
        f"Model a {w}x{d}x{t}mm plate with four through-holes near the corners.",
    )
    return prompt, {"units": "mm", "solids": sol}


def gen_lbracket():
    a = R.randrange(40, 120, 5)   # base length
    b = R.randrange(40, 120, 5)   # upright height
    w = R.randrange(30, 80, 5)    # width
    t = R.choice([4, 5, 6, 8])
    sol = [box(0, 0, 0, a, w, t), box(0, 0, 0, t, w, b)]
    hr = R.choice([3, 4, 5])
    sol.append(hole3(a - 12, w / 2, 0, hr, t))       # base hole
    prompt = pick(
        f"An L-bracket: {a} mm base and {b} mm upright, {w} mm wide, {t} mm "
        f"thick, with a mounting hole in the base.",
        f"Model a right-angle bracket, base {a}mm, upright {b}mm, {t}mm plate.",
    )
    return prompt, {"units": "mm", "solids": sol}


def gen_tube():
    od = R.choice([20, 30, 40, 50, 60, 80])
    wall = R.choice([2, 3, 4, 5])
    h = R.randrange(20, 140, 5)
    sol = [cyl(0, 0, 0, od / 2, h), hole3(0, 0, 0, od / 2 - wall, h)]
    prompt = pick(
        f"A round tube, {od} mm outer diameter, {wall} mm wall, {h} mm long.",
        f"Model a hollow pipe: {od}mm OD, {od-2*wall}mm ID, {h}mm length.",
    )
    return prompt, {"units": "mm", "solids": sol}


def gen_washer():
    od = R.choice([16, 20, 24, 30, 40])
    idd = R.choice([6, 8, 10, 12])
    t = R.choice([2, 3, 4, 5])
    sol = [cyl(0, 0, 0, od / 2, t), hole3(0, 0, 0, idd / 2, t)]
    prompt = pick(
        f"A washer / spacer, {od} mm outer diameter, {idd} mm hole, {t} mm thick.",
        f"Model a {t}mm-thick ring, {od}mm OD and {idd}mm bore.",
    )
    return prompt, {"units": "mm", "solids": sol}


# --------------------------------------------------------------------------
SKETCH_GENS = [(gen_floorplan, 0.40), (gen_plate, 0.15), (gen_flange, 0.15),
               (gen_bracket2d, 0.12), (gen_gasket, 0.10), (gen_disc, 0.08)]
SOLID_GENS = [(gen_block, 0.20), (gen_shaft, 0.18), (gen_plate3d, 0.18),
              (gen_lbracket, 0.16), (gen_tube, 0.16), (gen_washer, 0.12)]


def weighted(gens):
    r, acc = R.random(), 0.0
    for fn, wgt in gens:
        acc += wgt
        if r <= acc:
            return fn
    return gens[-1][0]


def make_row(mode):
    fn = weighted(SKETCH_GENS if mode == "sketch" else SOLID_GENS)
    prompt, data = fn()
    # round-trip through the validator so training data == servable output
    data = clean_sketch(data, "mm") if mode == "sketch" else clean_solid(data, "mm")
    assistant = json.dumps(data, separators=(",", ":"))
    return {
        "subject": "cad",
        "mode": mode,
        "messages": [
            {"role": "system", "content": sys_for(mode, "mm")},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2400, help="total examples")
    ap.add_argument("--sketch-frac", type=float, default=0.6,
                    help="fraction that are 2D sketches (rest are 3D solids)")
    ap.add_argument("--out", default="data/cad.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    R.seed(args.seed)
    n_sketch = int(args.n * args.sketch_frac)
    modes = ["sketch"] * n_sketch + ["3d"] * (args.n - n_sketch)
    R.shuffle(modes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"sketch": 0, "3d": 0}
    with out.open("w") as f:
        for mode in modes:
            row = make_row(mode)
            f.write(json.dumps(row) + "\n")
            counts[mode] += 1

    print(f"[✓] Wrote {args.n} CAD examples -> {out}")
    print(f"    2D sketch: {counts['sketch']}   3D solid: {counts['3d']}")


if __name__ == "__main__":
    main()
