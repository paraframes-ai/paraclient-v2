#!/usr/bin/env python3
"""
circuit_schema.py — the contract for ParaFrames' circuit model (Tinkercad-style).

This mirrors cad_schema.py, but for the CPU-only circuit specialist (a fine-tuned
Qwen2.5-3B served on llama.cpp, NOT on the L4). The model emits a NETLIST — a set
of components plus the nets (wires) joining their pins — which the app renders on a
breadboard/schematic and a deterministic solver (ngspice / avr8js) actually
simulates. The model proposes; the solver decides.

Three quality layers keep a small CPU model reliable:
  1. GBNF grammar (grammar()) — llama.cpp constrains decoding so output is always
     valid JSON with known component types + well-formed pin refs.
  2. This schema's ERC (erc()) — referential + electrical sanity (no shorts, LEDs
     current-limited, everything powered/connected).
  3. Fast rejection sampling in the gateway — sample a few, keep the first that
     passes ERC.

Shared by the dataset synthesizer (build_circuit_dataset.py) and the gateway, so
the model is trained on exactly the schema + prompt it is served with.
"""

# --------------------------------------------------------------------------
# Component library — type -> ordered pin names. Beginner electronics set.
# --------------------------------------------------------------------------
ARDUINO_PINS = (["5V", "3V3", "VIN", "GND", "GND2", "AREF"]
                + [f"D{i}" for i in range(14)]
                + [f"A{i}" for i in range(6)])

COMPONENT_PINS = {
    # power
    "battery":        ["+", "-"],
    "coin_cell":      ["+", "-"],
    "power_supply":   ["+", "-"],
    # passives
    "resistor":       ["a", "b"],
    "potentiometer":  ["1", "w", "2"],      # w = wiper
    "capacitor":      ["+", "-"],
    "capacitor_np":   ["a", "b"],           # non-polar (ceramic)
    "inductor":       ["a", "b"],
    # inputs
    "pushbutton":     ["1", "2"],
    "switch":         ["1", "2"],
    "photoresistor":  ["a", "b"],
    # semiconductors
    "led":            ["anode", "cathode"],
    "diode":          ["anode", "cathode"],
    "transistor_npn": ["b", "c", "e"],
    # outputs
    "buzzer":         ["+", "-"],
    "dc_motor":       ["1", "2"],
    "servo":          ["vcc", "gnd", "sig"],
    # ICs
    "arduino_uno":    ARDUINO_PINS,
    "ne555":          ["GND", "TRIG", "OUT", "RESET", "CTRL", "THRES", "DISCH", "VCC"],
}

# Allowed parameter keys per component (validated in ERC; grammar allows the union).
PARAM_KEYS = {
    "battery": ["voltage"], "coin_cell": ["voltage"], "power_supply": ["voltage"],
    "resistor": ["ohms"], "potentiometer": ["ohms"],
    "capacitor": ["farads"], "capacitor_np": ["farads"], "inductor": ["henries"],
    "led": ["color"], "buzzer": [], "dc_motor": [], "servo": [],
    "transistor_npn": [], "diode": [], "pushbutton": [], "switch": [],
    "photoresistor": [], "arduino_uno": [], "ne555": [],
}

POWER_TYPES = {"battery", "coin_cell", "power_supply", "arduino_uno"}

CIRCUIT_SYS = (
    "You are a circuit design generator for a beginner electronics app. Output "
    "STRICT JSON only (no prose, no markdown fences) describing a netlist, "
    "matching:\n"
    '{"components":[{"id":"R1","type":"resistor","ohms":330}],'
    '"nets":[{"id":"n1","nodes":["U1.D13","R1.a"]}],"code":"<arduino sketch or empty>"}\n'
    "Rules:\n"
    "- Each component has a unique id (letter prefix + number, e.g. R1, LED1, U1) "
    "and a `type` from the allowed set.\n"
    "- A net's `nodes` are pin references \"COMPID.PIN\" using that component's real "
    "pin names.\n"
    "- Every LED (and motor/buzzer driven from a logic pin) must be current-limited "
    "by a series resistor. Never short a power source (+ to - on one net).\n"
    "- Connect grounds; leave no component pin dangling.\n"
    "- Include Arduino `code` only when an arduino_uno is present; otherwise use an "
    "empty string.\n"
    "Allowed types and pins:\n"
    + "\n".join(f'  {t}: {"/".join(p)}' for t, p in COMPONENT_PINS.items())
    + "\nOutput ONLY the JSON object."
)


# --------------------------------------------------------------------------
# GBNF grammar (llama.cpp): guarantees valid JSON + known types + pin-ref shape.
# Params/ids are shape-constrained; referential integrity is ERC's job.
# --------------------------------------------------------------------------
def grammar() -> str:
    types = " | ".join(f'"\\"{t}\\""' for t in COMPONENT_PINS)
    return r'''
root        ::= "{" ws "\"components\"" ws ":" ws comps ws "," ws
                    "\"nets\"" ws ":" ws nets ws
                    ("," ws "\"code\"" ws ":" ws string ws)? "}"
comps       ::= "[" ws (comp (ws "," ws comp)*)? ws "]"
comp        ::= "{" ws "\"id\"" ws ":" ws string ws "," ws
                    "\"type\"" ws ":" ws ctype
                    (ws "," ws param)* ws "}"
ctype       ::= ''' + types + r'''
param       ::= "\"" pkey "\"" ws ":" ws (number | string)
pkey        ::= "ohms" | "voltage" | "farads" | "henries" | "color"
nets        ::= "[" ws (net (ws "," ws net)*)? ws "]"
net         ::= "{" ws "\"id\"" ws ":" ws string ws "," ws
                    "\"nodes\"" ws ":" ws "[" ws string (ws "," ws string)* ws "]" ws "}"
string      ::= "\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""
number      ::= "-"? [0-9]+ ("." [0-9]+)?
ws          ::= [ \t\n]*
'''


# --------------------------------------------------------------------------
# ERC — electrical rule check + referential integrity.
# --------------------------------------------------------------------------
def _pin_ref(node):
    if not isinstance(node, str) or "." not in node:
        return None, None
    cid, pin = node.split(".", 1)
    return cid, pin


def erc(netlist: dict):
    """Return (ok: bool, errors: list[str]). Empty errors == clean netlist."""
    errs = []
    comps = netlist.get("components", []) if isinstance(netlist, dict) else []
    nets = netlist.get("nets", []) if isinstance(netlist, dict) else []
    by_id = {}
    for c in comps:
        cid, ctype = c.get("id"), c.get("type")
        if not cid or ctype not in COMPONENT_PINS:
            errs.append(f"bad component {c!r}")
            continue
        if cid in by_id:
            errs.append(f"duplicate id {cid}")
        by_id[cid] = ctype

    seen_pins = set()               # pins that appear in some net
    for n in nets:
        nodes = n.get("nodes", []) if isinstance(n, dict) else []
        if len(nodes) < 2:
            errs.append(f"net {n.get('id')} joins < 2 pins")
        node_types = []
        for node in nodes:
            cid, pin = _pin_ref(node)
            if cid not in by_id:
                errs.append(f"net references unknown component: {node}")
                continue
            if pin not in COMPONENT_PINS[by_id[cid]]:
                errs.append(f"invalid pin for {by_id[cid]}: {node}")
                continue
            seen_pins.add(node)
            node_types.append((cid, pin, by_id[cid]))
        # dead short: a single net carrying both terminals of one power source
        for cid, ctype in by_id.items():
            if ctype in ("battery", "coin_cell", "power_supply"):
                pins_here = {p for c2, p, _ in node_types if c2 == cid}
                if {"+", "-"} <= pins_here:
                    errs.append(f"dead short across {cid} (+ and - on one net)")

    # everything powered
    if not any(t in POWER_TYPES for t in by_id.values()):
        errs.append("no power source (battery/arduino) in circuit")

    # LEDs must be current-limited by a resistor somewhere in the circuit
    has_led = any(t == "led" for t in by_id.values())
    has_res = any(t in ("resistor", "potentiometer") for t in by_id.values())
    if has_led and not has_res:
        errs.append("LED present but no current-limiting resistor in circuit")

    # dangling pins
    for cid, ctype in by_id.items():
        for pin in COMPONENT_PINS[ctype]:
            # arduino/ic pins are allowed to be unused; discrete parts are not
            if ctype in ("arduino_uno", "ne555"):
                continue
            if f"{cid}.{pin}" not in seen_pins:
                errs.append(f"dangling pin {cid}.{pin}")

    return (len(errs) == 0), errs


def guided_schema() -> dict:
    """JSON Schema for OpenAI-style guided decoding (`response_format:
    json_schema`) — the portable equivalent of grammar() for llama.cpp builds
    whose GBNF parser rejects our grammar, and for vLLM. Constrains the envelope
    and every component `type` to the known set so output always parses into the
    servable shape; erc() still owns referential + electrical validity, so the
    gateway's rejection sampling spends its tries on real ERC failures, not on
    malformed JSON. Coordinate/param fields stay open (additionalProperties)."""
    return {
        "type": "object",
        "properties": {
            "components": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"enum": list(COMPONENT_PINS)},
                    "ohms": {"type": "number"}, "voltage": {"type": "number"},
                    "farads": {"type": "number"}, "henries": {"type": "number"},
                    "color": {"type": "string"},
                },
                "required": ["id", "type"], "additionalProperties": True,
            }},
            "nets": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "nodes": {"type": "array", "items": {"type": "string"},
                              "minItems": 2},
                },
                "required": ["id", "nodes"], "additionalProperties": True,
            }},
            "code": {"type": "string"},
        },
        "required": ["components", "nets"], "additionalProperties": True,
    }


def clean(netlist: dict) -> dict:
    """Drop unknown components / malformed nets; keep the servable shape."""
    comps = [c for c in netlist.get("components", [])
             if isinstance(c, dict) and c.get("type") in COMPONENT_PINS and c.get("id")]
    ids = {c["id"] for c in comps}
    nets = []
    for n in netlist.get("nets", []):
        if not isinstance(n, dict):
            continue
        nodes = [nd for nd in n.get("nodes", [])
                 if _pin_ref(nd)[0] in ids]
        if len(nodes) >= 2:
            nets.append({"id": n.get("id", f"n{len(nets)+1}"), "nodes": nodes})
    out = {"components": comps, "nets": nets}
    if isinstance(netlist.get("code"), str) and netlist["code"].strip():
        out["code"] = netlist["code"]
    return out
