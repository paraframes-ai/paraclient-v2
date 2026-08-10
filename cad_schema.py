#!/usr/bin/env python3
"""
cad_schema.py — the single source of truth for ParaClient's CAD output formats.

Two representations, one per route:
  * SKETCH (2D)  -> /v1/sketch  -> the Grapher renders it + exports DXF.
  * SOLID  (3D)  -> /v1/3d      -> a positioned-primitive / CSG description.

Both the training-data synthesizer (build_cad_dataset.py) and the serving
gateway (auth_gateway.py) import the system prompts and validators from here so
the CAD adapter is trained on EXACTLY the instruction it is served with. UNITS
is replaced with the request's units (mm/cm/in/...) at call time.
"""

# --------------------------------------------------------------------------
# 2D SKETCH — geometry the app renders in 2D and exports to DXF.
# --------------------------------------------------------------------------
SKETCH_ENTITIES = {"line", "circle", "arc", "polyline", "rect", "text"}

SKETCH_SYS = (
    "You are a 2D CAD sketch generator. Output STRICT JSON only (no prose, no "
    "markdown fences) describing a 2D sketch in UNITS units, matching:\n"
    '{"units":"UNITS","entities":[...]}\n'
    "Allowed entities (numbers are coordinates/sizes in UNITS):\n"
    '  {"type":"line","x1":0,"y1":0,"x2":0,"y2":0}\n'
    '  {"type":"circle","cx":0,"cy":0,"r":0}\n'
    '  {"type":"arc","cx":0,"cy":0,"r":0,"start_deg":0,"end_deg":90}\n'
    '  {"type":"polyline","closed":false,"points":[[0,0]]}\n'
    '  {"type":"rect","x":0,"y":0,"w":0,"h":0}\n'
    '  {"type":"text","x":0,"y":0,"h":2.5,"value":"label"}\n'
    "Draw ALL furniture, fixtures, and appliances as real geometry placed "
    "inside the correct room (a sofa = outline with seat/back divisions; a bed "
    "= rectangle with a pillow; toilet/sink/stove/table/chairs = their real "
    "shapes from lines/rects/arcs/circles). Use realistic real-world "
    "dimensions, place items against sensible walls, keep everything inside "
    "room boundaries, and avoid overlaps. Do NOT write a text label on or in "
    "place of any furniture or fixture (never emit text like \"SOFA\" or "
    "\"COFFEE TABLE\"); use the `text` entity ONLY for room names, kept small, "
    "or omit labels entirely if the user did not ask for them. Output ONLY the "
    "JSON object."
)

# --------------------------------------------------------------------------
# 3D SOLID — constructive description: positioned primitives are UNIONED; any
# `hole` is SUBTRACTED. Renderable/exportable later via a CAD kernel
# (build123d/CadQuery -> STEP/STL/glTF). Kept deliberately small + parametric.
# --------------------------------------------------------------------------
SOLID_TYPES = {"box", "cylinder", "sphere", "extrude", "hole"}

SOLID_SYS = (
    "You are a 3D CAD model generator. Output STRICT JSON only (no prose, no "
    "markdown fences) describing a solid model in UNITS units, matching:\n"
    '{"units":"UNITS","solids":[...]}\n'
    "The listed solids are UNIONED; every `hole` is SUBTRACTED from them. "
    "Allowed solids (numbers are coordinates/sizes in UNITS; Z is up):\n"
    '  {"type":"box","x":0,"y":0,"z":0,"w":0,"d":0,"h":0}   (min corner at x,y,z)\n'
    '  {"type":"cylinder","cx":0,"cy":0,"z":0,"r":0,"h":0}  (base center; axis +Z)\n'
    '  {"type":"sphere","cx":0,"cy":0,"cz":0,"r":0}\n'
    '  {"type":"extrude","profile":[[0,0]],"z":0,"h":0}     (closed 2D profile up +Z)\n'
    '  {"type":"hole","cx":0,"cy":0,"z":0,"r":0,"depth":0}  (cylindrical cut, +Z)\n'
    "Use realistic real-world dimensions, keep features inside the part, align "
    "holes to sensible bolt patterns, and avoid degenerate (zero-size) solids. "
    "Output ONLY the JSON object."
)


def sys_for(mode: str, units: str) -> str:
    """Return the production system prompt for a CAD mode with UNITS filled in."""
    base = SOLID_SYS if mode == "3d" else SKETCH_SYS
    return base.replace("UNITS", units)


def guided_schema(mode: str) -> dict:
    """JSON Schema for guided decoding (`response_format: json_schema`) of a CAD
    response. Constrains the envelope and each element's `type` to the allowed
    set — so output always parses and every element is a known shape — while
    leaving coordinate fields open (additionalProperties). Dimensional
    correctness is still the job of floorplan_issues + repair_sketch, exactly as
    circuit erc() owns electrical validity."""
    if mode == "3d":
        key, types = "solids", sorted(SOLID_TYPES)
    else:
        key, types = "entities", sorted(SKETCH_ENTITIES)
    return {
        "type": "object",
        "properties": {
            "units": {"type": "string"},
            key: {"type": "array", "items": {
                "type": "object",
                "properties": {"type": {"enum": types}},
                "required": ["type"], "additionalProperties": True,
            }},
        },
        "required": ["units", key], "additionalProperties": True,
    }


def clean_sketch(data: dict, units: str) -> dict:
    """Keep only well-formed 2D entities; return {units, entities}."""
    ents = data.get("entities", []) if isinstance(data, dict) else []
    clean = [e for e in ents
             if isinstance(e, dict) and e.get("type") in SKETCH_ENTITIES]
    out_units = data.get("units", units) if isinstance(data, dict) else units
    return {"units": out_units, "entities": clean}


def clean_solid(data: dict, units: str) -> dict:
    """Keep only well-formed 3D solids; return {units, solids}."""
    sols = data.get("solids", []) if isinstance(data, dict) else []
    clean = [s for s in sols
             if isinstance(s, dict) and s.get("type") in SOLID_TYPES]
    out_units = data.get("units", units) if isinstance(data, dict) else units
    return {"units": out_units, "solids": clean}


# --------------------------------------------------------------------------
# Floor-plan layout engine + geometric validator.
#
# A floor plan must be a valid partition: rooms tile the footprint with no
# overlaps and stay inside the exterior wall. The trained model states the
# *program* (which rooms, roughly how big) but can emit overlapping rectangles.
# `floorplan_issues` detects that; `repair_sketch` re-derives a guaranteed-valid
# guillotine layout from the model's own room program — so the served sketch
# never needs manual editing. This is the sketch analogue of the circuit ERC gate.
# --------------------------------------------------------------------------
import random as _random

ROOM_KINDS = ["bedroom", "living room", "kitchen", "bathroom", "dining room",
              "office", "closet", "hallway"]


def _r0(v):
    return round(v, 1) if isinstance(v, (int, float)) else v


def _rect(x, y, w, h):
    return {"type": "rect", "x": _r0(x), "y": _r0(y), "w": _r0(w), "h": _r0(h)}


def _arc(cx, cy, r, s, e):
    return {"type": "arc", "cx": _r0(cx), "cy": _r0(cy), "r": _r0(r),
            "start_deg": s, "end_deg": e}


def _circle(cx, cy, r):
    return {"type": "circle", "cx": _r0(cx), "cy": _r0(cy), "r": _r0(r)}


def _text(x, y, h, v):
    return {"type": "text", "x": _r0(x), "y": _r0(y), "h": h, "value": v}


def _split(rooms, min_side, rng):
    """Guillotine-split the largest room once; True if a split happened."""
    rooms.sort(key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = rooms[0]
    if w >= h and w > 2 * min_side:
        cut = w * rng.uniform(0.4, 0.6)
        rooms[0] = (x, y, cut, h)
        rooms.append((x + cut, y, w - cut, h))
        return True
    if h > 2 * min_side:
        cut = h * rng.uniform(0.4, 0.6)
        rooms[0] = (x, y, w, cut)
        rooms.append((x, y + cut, w, h - cut))
        return True
    return False


def _furnish(x, y, w, h, kind, ents):
    """One clean furniture piece against a wall, inside the room."""
    m = 250
    iw, ih = w - 2 * m, h - 2 * m
    if iw < 400 or ih < 400:
        return
    cx, cy = x + w / 2, y + h / 2
    k = kind.lower()
    if "bed" in k:
        bw, bh = min(1500, iw), min(2000, ih)
        bx, by = x + m, y + h - m - bh
        ents.append(_rect(bx, by, bw, bh))
        ents.append(_rect(bx + bw * 0.1, by + bh - 350, bw * 0.8, 300))
    elif "living" in k:
        sw = min(2200, iw)
        ents.append(_rect(cx - sw / 2, y + m, sw, 850))
        ents.append(_rect(cx - sw / 2, y + m, sw, 180))
    elif "kitchen" in k:
        ents.append(_rect(x + m, y + h - m - 600, iw, 600))
        ents.append(_circle(x + m + iw * 0.3, y + h - m - 300, 200))
    elif "bath" in k:
        ents.append(_rect(x + m, y + m, min(1700, iw), 750))
        ents.append(_circle(x + w - m - 250, y + m + 250, 220))
    elif "dining" in k:
        tw, th = min(1400, iw * 0.7), min(900, ih * 0.7)
        tx, ty = cx - tw / 2, cy - th / 2
        ents.append(_rect(tx, ty, tw, th))
        for dx, dy in [(tw * .25, -260), (tw * .75, -260),
                       (tw * .25, th + 260), (tw * .75, th + 260)]:
            ents.append(_circle(tx + dx, ty + dy, 200))
    elif "office" in k:
        ents.append(_rect(x + m, y + m, min(1400, iw), 700))
        ents.append(_circle(x + m + 500, y + m + 900, 220))


def layout_floorplan(W, D, kinds, seed=0):
    """Deterministic, guaranteed-valid floor plan: guillotine partition of a
    W x D footprint into len(kinds) non-overlapping rooms, with a door, one
    furniture piece, and a label per room. Returns {units, entities}."""
    W, D = float(W), float(D)
    kinds = [str(k).strip().lower() for k in kinds if str(k).strip()] or ["room"]
    n = max(1, len(kinds))
    # Adapt the minimum room dimension down until every requested room fits the
    # footprint (guillotine can't always hit N rooms at a fixed 2200 mm min).
    rooms = [(0.0, 0.0, W, D)]
    for min_side in (2200, 1900, 1600, 1400, 1200, 1000):
        rng = _random.Random(seed)
        rooms = [(0.0, 0.0, W, D)]
        while len(rooms) < n:
            if not _split(rooms, min_side, rng):
                break
        if len(rooms) >= n:
            break
    rng = _random.Random(seed + 7)
    rooms.sort(key=lambda r: r[2] * r[3], reverse=True)
    kinds = kinds[:len(rooms)]
    while len(kinds) < len(rooms):
        kinds.append("room")
    ents = [_rect(0, 0, W, D)]                    # exterior wall
    for (x, y, w, h), kind in zip(rooms, kinds):
        ents.append(_rect(x, y, w, h))            # room partition
        if w >= h:
            dx = x + w * rng.uniform(0.3, 0.6)
            ents.append(_arc(dx, y + h, min(800, h * 0.4), 180, 270))
        else:
            dy = y + h * rng.uniform(0.3, 0.6)
            ents.append(_arc(x + w, dy, min(800, w * 0.4), 90, 180))
        _furnish(x, y, w, h, kind, ents)
        ents.append(_text(x + 200, y + h - 220, 240, kind.upper()))
    return {"units": "mm", "entities": ents}


def _corners(r):
    x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 0), r.get("h", 0)
    return x, y, x + w, y + h


def _area(r):
    return max(0.0, r.get("w", 0)) * max(0.0, r.get("h", 0))


def _overlap_area(a, b):
    ax0, ay0, ax1, ay1 = _corners(a)
    bx0, by0, bx1, by1 = _corners(b)
    return (max(0.0, min(ax1, bx1) - max(ax0, bx0)) *
            max(0.0, min(ay1, by1) - max(ay0, by0)))


def _label_inside(rect, texts):
    x0, y0, x1, y1 = _corners(rect)
    return any(x0 <= t.get("x", 0) <= x1 and y0 <= t.get("y", 0) <= y1
               for t in texts)


def _rooms_and_exterior(data):
    ents = data.get("entities", []) if isinstance(data, dict) else []
    rects = [e for e in ents if e.get("type") == "rect"]
    texts = [e for e in ents if e.get("type") == "text"]
    if not rects:
        return None, [], texts
    ext = max(rects, key=_area)                   # exterior = largest rectangle
    rooms = [r for r in rects if r is not ext and _label_inside(r, texts)]
    return ext, rooms, texts


def is_floorplan(data) -> bool:
    """A sketch we should hold to floor-plan rules (>= 2 labeled rooms)."""
    _ext, rooms, _texts = _rooms_and_exterior(data)
    return len(rooms) >= 2


def floorplan_issues(data):
    """Return (ok, [errors]) for a floor-plan sketch: no overlapping rooms, all
    rooms inside the exterior wall, no duplicate labels stacked on one spot."""
    ext, rooms, texts = _rooms_and_exterior(data)
    if not ext or len(rooms) < 2:
        return True, []                           # not a floor plan we judge
    errs = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if _overlap_area(a, b) > 0.02 * min(_area(a), _area(b)):
                errs.append("overlapping rooms")
                break
        else:
            continue
        break
    ex0, ey0, ex1, ey1 = _corners(ext)
    for r in rooms:
        rx0, ry0, rx1, ry1 = _corners(r)
        if rx0 < ex0 - 1 or ry0 < ey0 - 1 or rx1 > ex1 + 1 or ry1 > ey1 + 1:
            errs.append("room outside exterior wall")
            break
    seen = set()
    for t in texts:
        key = (round(t.get("x", 0)), round(t.get("y", 0)))
        if key in seen:
            errs.append("duplicate labels on one spot")
            break
        seen.add(key)
    return (not errs), errs


def repair_sketch(data: dict, units: str) -> dict:
    """Structurally clean the sketch; if it is a floor plan with geometric
    issues (overlaps / out-of-bounds / stacked labels), re-derive a valid
    guillotine layout from its OWN room program so it needs no manual editing.
    Non-floor-plan sketches (plates, brackets, ...) pass through untouched."""
    data = clean_sketch(data, units)
    ok, _errs = floorplan_issues(data)
    if ok:
        return data
    ext, rooms, texts = _rooms_and_exterior(data)
    W = ext.get("w") or 8000
    D = ext.get("h") or 6000
    kinds = [t.get("value", "room") for t in texts if str(t.get("value", "")).strip()]
    if not kinds:
        kinds = ["room"] * max(2, len(rooms))
    fixed = layout_floorplan(W, D, kinds, seed=len(kinds))
    fixed["units"] = data.get("units", units)
    return fixed
