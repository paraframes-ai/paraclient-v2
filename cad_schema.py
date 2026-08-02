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
