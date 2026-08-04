#!/usr/bin/env python3
"""
build_spreadsheet_dataset.py — procedural, correct-by-construction spreadsheet
data for the ParaClient spreadsheet adapter.

Two modes per row:
  * generate — prompt -> a {title, cells} JSON with real formulas. Every sheet
    is run through spreadsheet_schema.spreadsheet_issues (the same evaluator the
    server uses) and only emitted if all formulas compute. So the model trains
    on sheets whose SUM/AVERAGE/etc. are actually correct.
  * tutor — explain a spreadsheet concept/formula simply.

Row: {subject:"spreadsheet", mode, messages:[system,user,assistant]}
"""
import argparse
import json
import os
import random

import spreadsheet_schema as ss

R = random.Random(7)


def C(ref, value=None, formula=None):
    return {"ref": ref, "formula": formula} if formula is not None else {"ref": ref, "value": value}


CATEGORIES = ["Rent", "Groceries", "Utilities", "Transport", "Entertainment",
              "Savings", "Insurance", "Phone", "Internet", "Dining", "Gym",
              "Books", "Supplies", "Clothing", "Fuel", "Snacks"]
NAMES = ["Ava", "Liam", "Noah", "Emma", "Olivia", "Mateo", "Sophia", "Ethan",
         "Isabella", "Lucas", "Mia", "Aiden", "Zoe", "Kai", "Nia", "Omar",
         "Priya", "Leo", "Ruby", "Sam", "Dev", "Aria", "Yuki", "Hana"]
ITEMS = ["notebook", "pencil pack", "marker set", "ruler", "binder",
         "calculator", "glue stick", "scissors", "folder", "eraser",
         "highlighter", "stapler", "tape roll", "backpack", "sketchpad"]
BRANDS = ["Widget", "Gadget", "Gizmo", "Sprocket", "Cog", "Bolt", "Panel",
          "Sensor", "Module", "Cable", "Acme", "Nova"]


def gen_budget():
    n = R.randint(4, 8)
    cats = R.sample(CATEGORIES, n)
    cells = [C("A1", "Category"), C("B1", "Amount ($)"), C("C1", "% of Total")]
    for i, cat in enumerate(cats):
        row = i + 2
        cells += [C(f"A{row}", cat), C(f"B{row}", R.randrange(20, 2000, 5))]
    tot = n + 2
    cells += [C(f"A{tot}", "Total"), C(f"B{tot}", formula=f"=SUM(B2:B{n + 1})")]
    for i in range(n):
        row = i + 2
        cells.append(C(f"C{row}", formula=f"=ROUND(B{row}/B{tot}*100,1)"))
    title = R.choice(["Monthly Budget", "Household Budget", "Trip Budget", "Club Budget"])
    prompt = R.choice([
        f"Make a {title.lower()} with {n} categories, an amount column, a SUM total, and a % of total column.",
        f"Build a budget for: {', '.join(cats)}. Include a total and each category's percent of the total.",
        f"I need a {title.lower()} with a total row and a percent-of-total column.",
    ])
    return prompt, {"title": title, "cells": cells}


def gen_gradebook():
    ns, na = R.randint(4, 7), R.randint(3, 5)
    studs = R.sample(NAMES, ns)
    cells = [C("A1", "Student")]
    for a in range(na):
        cells.append(C(f"{ss.num_to_col(2 + a)}1", f"HW{a + 1}"))
    avg_col = ss.num_to_col(2 + na)
    cells.append(C(f"{avg_col}1", "Average"))
    first, last = ss.num_to_col(2), ss.num_to_col(1 + na)
    for i, st in enumerate(studs):
        row = i + 2
        cells.append(C(f"A{row}", st))
        for a in range(na):
            cells.append(C(f"{ss.num_to_col(2 + a)}{row}", R.randrange(50, 101)))
        cells.append(C(f"{avg_col}{row}", formula=f"=ROUND(AVERAGE({first}{row}:{last}{row}),1)"))
    crow = ns + 2
    cells.append(C(f"A{crow}", "Class Avg"))
    for a in range(na + 1):
        col = ss.num_to_col(2 + a)
        cells.append(C(f"{col}{crow}", formula=f"=ROUND(AVERAGE({col}2:{col}{ns + 1}),1)"))
    prompt = R.choice([
        f"Create a gradebook for {ns} students with {na} homework scores, an average per student, and a class-average row.",
        f"Build a gradebook: students {', '.join(studs)}, {na} assignments, with per-student and per-assignment averages.",
    ])
    return prompt, {"title": "Gradebook", "cells": cells}


def gen_invoice():
    n = R.randint(3, 7)
    cells = [C("A1", "Item"), C("B1", "Qty"), C("C1", "Unit Price"), C("D1", "Line Total")]
    for i in range(n):
        row = i + 2
        cells += [C(f"A{row}", f"{R.choice(BRANDS)} {R.choice(ITEMS)}"),
                  C(f"B{row}", R.randint(1, 12)),
                  C(f"C{row}", round(R.uniform(1.5, 49.99), 2)),
                  C(f"D{row}", formula=f"=B{row}*C{row}")]
    sub, tax, tot = n + 2, n + 3, n + 4
    rate = R.choice([0.05, 0.06, 0.07, 0.08, 0.0825])
    tlabel = f"Tax ({int(rate * 100)}%)" if rate * 100 == int(rate * 100) else f"Tax ({rate * 100:.2f}%)"
    cells += [C(f"C{sub}", "Subtotal"), C(f"D{sub}", formula=f"=SUM(D2:D{n + 1})"),
              C(f"C{tax}", tlabel), C(f"D{tax}", formula=f"=ROUND(D{sub}*{rate},2)"),
              C(f"C{tot}", "Total"), C(f"D{tot}", formula=f"=D{sub}+D{tax}")]
    prompt = R.choice([
        f"Make an invoice with {n} line items (qty x unit price), a subtotal, {int(rate * 100)}% tax, and a grand total.",
        "Build an invoice: each row has quantity and unit price; compute line totals, subtotal, tax, and total.",
    ])
    return prompt, {"title": "Invoice", "cells": cells}


def gen_stats():
    n = R.randint(6, 12)
    label = R.choice(["Temperature", "Score", "Sales", "Steps", "Rainfall", "Height (cm)", "Points"])
    cells = [C("A1", label)] + [C(f"A{i + 2}", R.randrange(1, 100)) for i in range(n)]
    base, rng = n + 2, f"A2:A{n + 1}"
    summ = [("Sum", f"=SUM({rng})"), ("Average", f"=ROUND(AVERAGE({rng}),2)"),
            ("Min", f"=MIN({rng})"), ("Max", f"=MAX({rng})"), ("Count", f"=COUNT({rng})")]
    for i, (lbl, f) in enumerate(summ):
        cells += [C(f"C{base + i}", lbl), C(f"D{base + i}", formula=f)]
    prompt = R.choice([
        f"Given a column of {n} {label.lower()} values, compute the sum, average, min, max, and count.",
        f"Make a data table of {label.lower()} with summary stats (SUM, AVERAGE, MIN, MAX, COUNT).",
    ])
    return prompt, {"title": f"{label} Stats", "cells": cells}


def gen_tutor():
    col = R.choice("BCD")
    a, b = R.randint(2, 5), R.randint(6, 12)
    rng = f"{col}{a}:{col}{b}"
    x, y = R.choice("BCDE"), R.randint(2, 9)
    tmpls = [
        (f"How do I add up the numbers in {rng}?",
         f"Use the SUM function: put =SUM({rng}) in the total cell. It adds every value from {col}{a} down to {col}{b}, and updates automatically if those numbers change."),
        (f"How do I average the values in {rng}?",
         f"Use =AVERAGE({rng}) — it adds the cells and divides by how many there are. To round to 1 decimal, wrap it: =ROUND(AVERAGE({rng}),1)."),
        ("What's the difference between A1 and $A$1 in a formula?",
         "A1 is *relative*: copy the formula elsewhere and it shifts (A1 → B1 one column right). $A$1 is *absolute*: the $ signs lock it so it always points to A1. Use absolute refs for a fixed value like a tax rate."),
        (f"My formula ={x}{y}*C{y} shows an error and {x}{y} is empty. Why?",
         f"The formula multiplies by whatever is in {x}{y}, but that cell is blank, so there's no number to multiply. Put a value in {x}{y}, or double-check you pointed at the right cell."),
        ("How do I add 8% tax to a subtotal in D10?",
         "Give the tax its own cell: =D10*0.08 is the tax amount, and =D10*1.08 gives subtotal-plus-tax in one step. Keeping tax on its own line makes the sheet easier to check."),
        (f"How do I find the highest and lowest value in {rng}?",
         f"Use =MAX({rng}) for the highest and =MIN({rng}) for the lowest."),
        ("What does every formula need to start with?",
         "An equals sign (=). That tells the spreadsheet to calculate. =2+2 shows 4, but 2+2 with no = just shows the text \"2+2\"."),
        (f"How can I count how many entries are in {rng}?",
         f"Use =COUNT({rng}) to count the cells that hold numbers. To count non-empty cells including text, use COUNTA instead."),
        (f"Each row's total should be quantity times price, with quantity in B{y} and price in C{y}. What formula?",
         f"In that row's total cell use =B{y}*C{y}, then copy it down — each row multiplies its own quantity and price because the references shift automatically."),
        (f"How do I turn {col}{a} into a percentage of a total in {col}{b}?",
         f"Divide and multiply by 100: =ROUND({col}{a}/${col}${b}*100,1). Lock the total with $ signs so it stays fixed when you copy the formula down."),
    ]
    return R.choice(tmpls)


TUTOR_SYS = ("You are a friendly, clear spreadsheet tutor for students. Explain "
             "concepts and formulas simply, with a concrete example, and keep it short.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2200)
    ap.add_argument("--out", default="data/spreadsheet.jsonl")
    args = ap.parse_args()
    gens = [gen_budget, gen_gradebook, gen_invoice, gen_stats]
    n_gen = int(args.n * 0.7)
    rows, bad = [], 0
    made = 0
    while made < n_gen:
        prompt, sheet = R.choice(gens)()
        ok, _errs = ss.spreadsheet_issues(sheet)
        if not ok:
            bad += 1
            continue
        rows.append({"subject": "spreadsheet", "mode": "generate", "messages": [
            {"role": "system", "content": ss.SPREADSHEET_SYS},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(sheet, separators=(",", ":"))}]})
        made += 1
    for _ in range(args.n - n_gen):
        q, a = gen_tutor()
        rows.append({"subject": "spreadsheet", "mode": "tutor", "messages": [
            {"role": "system", "content": TUTOR_SYS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}]})
    R.shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows ({n_gen} generate + {args.n - n_gen} tutor) "
          f"-> {args.out}; rejected {bad} sheets that failed the formula gate")


if __name__ == "__main__":
    main()
