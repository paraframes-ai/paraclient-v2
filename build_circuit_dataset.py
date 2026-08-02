#!/usr/bin/env python3
"""
build_circuit_dataset.py — synthesize the circuit model's training set.

Procedural, like the CAD builder: generate real beginner-electronics circuits as
exact netlists (components + nets + optional Arduino code) paired with a natural
-language request. EVERY row is gated through circuit_schema.erc() so the ground
truth is electrically valid — the model learns from correct circuits, not guesses.

Rows use the tutor schema (subject/mode + messages) so train_adapter.py consumes
it unchanged (fine-tune Qwen2.5-3B, then quantize to GGUF for CPU/llama.cpp):

    python build_circuit_dataset.py --n 2500 --out data/circuit.jsonl
    python train_adapter.py --subject circuits --data data/circuit.jsonl \
        --base Qwen/Qwen2.5-3B-Instruct --no-flash-attn
"""
import argparse
import json
import random

import circuit_schema as cs

R = random.Random()


def pick(*o):
    return R.choice(o)


def res_val():
    return pick(150, 220, 330, 470, 680, 1000, 2200, 4700, 10000)


def dpin():
    return R.randint(2, 13)


def apin():
    return R.randint(0, 5)


def color():
    return pick("red", "green", "blue", "yellow", "white", "orange")


# --------------------------------------------------------------------------
# generators: each returns (prompt, netlist dict)
# --------------------------------------------------------------------------

def gen_simple_led():
    v = pick(3, 4.5, 5, 6, 9)
    r = res_val()
    c = color()
    nl = {"components": [
        {"id": "B1", "type": "battery", "voltage": v},
        {"id": "R1", "type": "resistor", "ohms": r},
        {"id": "LED1", "type": "led", "color": c}],
        "nets": [
        {"id": "n1", "nodes": ["B1.+", "R1.a"]},
        {"id": "n2", "nodes": ["R1.b", "LED1.anode"]},
        {"id": "n3", "nodes": ["LED1.cathode", "B1.-"]}], "code": ""}
    p = pick(
        f"Light up a {c} LED from a {v}V battery using a {r} ohm resistor.",
        f"Simple circuit: {v}V battery, {r}Ω resistor, and a {c} LED in series.",
        f"Wire a {c} LED to a {v} volt battery, current-limited with {r} ohms.")
    return p, nl


def gen_blink():
    d = dpin()
    r = res_val()
    c = color()
    ms = pick(200, 250, 500, 1000)
    nl = {"components": [
        {"id": "U1", "type": "arduino_uno"},
        {"id": "R1", "type": "resistor", "ohms": r},
        {"id": "LED1", "type": "led", "color": c}],
        "nets": [
        {"id": "n1", "nodes": [f"U1.D{d}", "R1.a"]},
        {"id": "n2", "nodes": ["R1.b", "LED1.anode"]},
        {"id": "n3", "nodes": ["LED1.cathode", "U1.GND"]}],
        "code": (f"void setup(){{pinMode({d},OUTPUT);}}\n"
                 f"void loop(){{digitalWrite({d},HIGH);delay({ms});"
                 f"digitalWrite({d},LOW);delay({ms});}}")}
    p = pick(
        f"Blink a {c} LED on an Arduino pin D{d} every {ms} ms.",
        f"Arduino blink sketch: {c} LED on D{d} through a {r}Ω resistor.",
        f"Make an LED flash on and off from an Arduino Uno, {ms}ms interval.")
    return p, nl


def gen_traffic():
    pins = R.sample(range(2, 14), 3)
    comps = [{"id": "U1", "type": "arduino_uno"}]
    nets = []
    for i, (col, pn) in enumerate(zip(["red", "yellow", "green"], pins), 1):
        comps += [{"id": f"R{i}", "type": "resistor", "ohms": 330},
                  {"id": f"LED{i}", "type": "led", "color": col}]
        nets += [{"id": f"a{i}", "nodes": [f"U1.D{pn}", f"R{i}.a"]},
                 {"id": f"b{i}", "nodes": [f"R{i}.b", f"LED{i}.anode"]},
                 {"id": f"c{i}", "nodes": [f"LED{i}.cathode", "U1.GND"]}]
    code = ("int p[]={%d,%d,%d};\nvoid setup(){for(int i=0;i<3;i++)pinMode(p[i],"
            "OUTPUT);}\nvoid loop(){for(int i=0;i<3;i++){digitalWrite(p[i],HIGH);"
            "delay(700);digitalWrite(p[i],LOW);}}" % tuple(pins))
    p = pick(
        f"A traffic-light circuit: red, yellow, green LEDs on Arduino pins "
        f"{', '.join('D'+str(x) for x in pins)}, cycling.",
        "Three LEDs (red/yellow/green) on an Arduino that light in sequence.")
    return p, {"components": comps, "nets": nets, "code": code}


def gen_button():
    d = dpin()
    nl = {"components": [
        {"id": "U1", "type": "arduino_uno"},
        {"id": "BTN1", "type": "pushbutton"},
        {"id": "R1", "type": "resistor", "ohms": 10000}],
        "nets": [
        {"id": "n1", "nodes": ["BTN1.1", "U1.5V"]},
        {"id": "n2", "nodes": ["BTN1.2", f"U1.D{d}", "R1.a"]},
        {"id": "n3", "nodes": ["R1.b", "U1.GND"]}],
        "code": (f"void setup(){{pinMode({d},INPUT);pinMode(13,OUTPUT);}}\n"
                 f"void loop(){{digitalWrite(13,digitalRead({d}));}}")}
    p = pick(
        f"Read a pushbutton on Arduino pin D{d} with a 10k pull-down resistor.",
        f"Button input on D{d}: light the onboard LED while pressed.")
    return p, nl


def gen_pot():
    a = apin()
    nl = {"components": [
        {"id": "U1", "type": "arduino_uno"},
        {"id": "POT1", "type": "potentiometer", "ohms": 10000}],
        "nets": [
        {"id": "n1", "nodes": ["POT1.1", "U1.5V"]},
        {"id": "n2", "nodes": ["POT1.2", "U1.GND"]},
        {"id": "n3", "nodes": ["POT1.w", f"U1.A{a}"]}],
        "code": (f"void setup(){{Serial.begin(9600);}}\n"
                 f"void loop(){{Serial.println(analogRead(A{a}));delay(100);}}")}
    p = pick(
        f"Connect a 10k potentiometer to Arduino analog pin A{a} and print the value.",
        f"Read a potentiometer on A{a} (wiper to the analog input, ends to 5V/GND).")
    return p, nl


def gen_divider():
    v = pick(5, 9, 12)
    r1, r2 = res_val(), res_val()
    vout = round(v * r2 / (r1 + r2), 2)
    nl = {"components": [
        {"id": "B1", "type": "battery", "voltage": v},
        {"id": "R1", "type": "resistor", "ohms": r1},
        {"id": "R2", "type": "resistor", "ohms": r2}],
        "nets": [
        {"id": "n1", "nodes": ["B1.+", "R1.a"]},
        {"id": "n2", "nodes": ["R1.b", "R2.a"]},
        {"id": "n3", "nodes": ["R2.b", "B1.-"]}], "code": ""}
    p = pick(
        f"A voltage divider dropping {v}V to about {vout}V using {r1} and {r2} ohm "
        f"resistors.",
        f"Two-resistor divider from {v} volts: R1={r1}Ω, R2={r2}Ω, tap between them.")
    return p, nl


def gen_photo():
    a = apin()
    nl = {"components": [
        {"id": "U1", "type": "arduino_uno"},
        {"id": "PR1", "type": "photoresistor"},
        {"id": "R1", "type": "resistor", "ohms": 10000}],
        "nets": [
        {"id": "n1", "nodes": ["U1.5V", "PR1.a"]},
        {"id": "n2", "nodes": ["PR1.b", "R1.a", f"U1.A{a}"]},
        {"id": "n3", "nodes": ["R1.b", "U1.GND"]}],
        "code": (f"void setup(){{Serial.begin(9600);}}\n"
                 f"void loop(){{Serial.println(analogRead(A{a}));delay(200);}}")}
    p = pick(
        f"A light sensor: photoresistor and 10k resistor as a divider into A{a}.",
        f"Read ambient light with an LDR on analog pin A{a}.")
    return p, nl


def gen_motor():
    d = dpin()
    nl = {"components": [
        {"id": "U1", "type": "arduino_uno"},
        {"id": "R1", "type": "resistor", "ohms": 1000},
        {"id": "Q1", "type": "transistor_npn"},
        {"id": "M1", "type": "dc_motor"},
        {"id": "D1", "type": "diode"}],
        "nets": [
        {"id": "n1", "nodes": [f"U1.D{d}", "R1.a"]},
        {"id": "n2", "nodes": ["R1.b", "Q1.b"]},
        {"id": "n3", "nodes": ["Q1.c", "M1.2", "D1.anode"]},
        {"id": "n4", "nodes": ["M1.1", "U1.VIN", "D1.cathode"]},
        {"id": "n5", "nodes": ["Q1.e", "U1.GND"]}],
        "code": (f"void setup(){{pinMode({d},OUTPUT);}}\n"
                 f"void loop(){{digitalWrite({d},HIGH);delay(2000);"
                 f"digitalWrite({d},LOW);delay(2000);}}")}
    p = pick(
        f"Drive a DC motor from Arduino pin D{d} with an NPN transistor and a "
        f"flyback diode.",
        f"Switch a DC motor with a transistor controlled by D{d} (with a "
        f"protection diode).")
    return p, nl


def gen_servo():
    d = dpin()
    nl = {"components": [
        {"id": "U1", "type": "arduino_uno"},
        {"id": "S1", "type": "servo"}],
        "nets": [
        {"id": "n1", "nodes": ["S1.vcc", "U1.5V"]},
        {"id": "n2", "nodes": ["S1.gnd", "U1.GND"]},
        {"id": "n3", "nodes": ["S1.sig", f"U1.D{d}"]}],
        "code": (f"#include <Servo.h>\nServo s;\nvoid setup(){{s.attach({d});}}\n"
                 f"void loop(){{s.write(0);delay(800);s.write(180);delay(800);}}")}
    p = pick(
        f"Sweep a servo back and forth on Arduino pin D{d}.",
        f"Control a servo motor from D{d} (signal to the pin, power to 5V/GND).")
    return p, nl


def gen_555():
    r1, r2 = pick(1000, 4700, 10000), pick(10000, 47000, 100000)
    nl = {"components": [
        {"id": "B1", "type": "battery", "voltage": 9},
        {"id": "U1", "type": "ne555"},
        {"id": "R1", "type": "resistor", "ohms": r1},
        {"id": "R2", "type": "resistor", "ohms": r2},
        {"id": "C1", "type": "capacitor", "farads": pick(1e-5, 4.7e-5, 1e-4)},
        {"id": "R3", "type": "resistor", "ohms": 330},
        {"id": "LED1", "type": "led", "color": color()}],
        "nets": [
        {"id": "n1", "nodes": ["B1.+", "U1.VCC", "U1.RESET", "R1.a"]},
        {"id": "n2", "nodes": ["R1.b", "U1.DISCH", "R2.a"]},
        {"id": "n3", "nodes": ["R2.b", "U1.THRES", "U1.TRIG", "C1.+"]},
        {"id": "n4", "nodes": ["U1.OUT", "R3.a"]},
        {"id": "n5", "nodes": ["R3.b", "LED1.anode"]},
        {"id": "n6", "nodes": ["LED1.cathode", "C1.-", "U1.GND", "B1.-"]}],
        "code": ""}
    p = pick(
        "A 555 timer astable blinking an LED (no microcontroller).",
        "Blink an LED using a 555 timer in astable mode from a 9V battery.")
    return p, nl


GENS = [(gen_blink, 0.20), (gen_simple_led, 0.14), (gen_traffic, 0.10),
        (gen_button, 0.12), (gen_pot, 0.10), (gen_divider, 0.08),
        (gen_photo, 0.08), (gen_motor, 0.08), (gen_servo, 0.06), (gen_555, 0.04)]


def weighted():
    r, acc = R.random(), 0.0
    for fn, w in GENS:
        acc += w
        if r <= acc:
            return fn
    return GENS[-1][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", default="data/circuit.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    R.seed(args.seed)

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = rejected = 0
    with out.open("w") as f:
        while written < args.n:
            prompt, nl = weighted()()
            nl = cs.clean(nl)
            ok, errs = cs.erc(nl)
            if not ok:                       # ground-truth gate
                rejected += 1
                continue
            row = {"subject": "circuits", "mode": "generate", "messages": [
                {"role": "system", "content": cs.CIRCUIT_SYS},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(nl, separators=(",", ":"))}]}
            f.write(json.dumps(row) + "\n")
            written += 1
    print(f"[✓] Wrote {written} circuit examples -> {out}  (ERC-rejected: {rejected})")


if __name__ == "__main__":
    main()
