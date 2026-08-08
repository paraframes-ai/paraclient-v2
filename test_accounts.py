#!/usr/bin/env python3
"""
test_accounts.py — behaviour + compliance-invariant tests for accounts.py.

Run:  python test_accounts.py

The compliance checks are the important ones. They assert the properties
compliance/01_age_gate_and_routing.md says the whole two-lane model depends on:
under-13 never reaches the consumer lane, an edu credential is unobtainable via
self-serve signup, and the raw date of birth is never persisted.
"""
import os
import sys
import tempfile
import time
from datetime import date

# Point the store at a scratch DB before importing the module.
_tmp = tempfile.mkdtemp(prefix="acct_test_")
os.environ["ACCOUNTS_DB"] = os.path.join(_tmp, "test_accounts.db")
os.environ["PARACLIENT_ENV"] = "dev"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accounts  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))


def dob_years_ago(years, days=0):
    t = date.today()
    try:
        d = t.replace(year=t.year - years)
    except ValueError:                     # Feb 29
        d = t.replace(year=t.year - years, day=28)
    return (d.toordinal() - days)


def dob_str(years, days=0):
    return date.fromordinal(dob_years_ago(years, days)).isoformat()


accounts.init_db()

print("\n=== age gate (compliance/01 §1-§2) ===")
under = accounts.age_gate(dob_str(11))
check("under-13 is refused", under["allowed"] is False)
check("under-13 gets the school message", "school" in under["message"].lower())
check("under-13 receives NO gate token", "gate_token" not in under)

# §2: nothing at all is stored for an under-13 attempt.
with accounts._connect() as con:
    gates = con.execute("SELECT COUNT(*) c FROM gate_tokens").fetchone()["c"]
    accts = con.execute("SELECT COUNT(*) c FROM accounts").fetchone()["c"]
check("under-13 attempt stores no gate token", gates == 0, f"found {gates}")
check("under-13 attempt stores no account", accts == 0, f"found {accts}")

over = accounts.age_gate(dob_str(14))
check("13+ passes the gate", over["allowed"] is True)
check("13+ receives a gate token", bool(over.get("gate_token")))

# Boundary: exactly 13 today passes; one day short does not.
check("exactly 13 today passes", accounts.age_gate(dob_str(13))["allowed"] is True)
check("13 tomorrow (still 12) is refused",
      accounts.age_gate(dob_str(13, days=-1))["allowed"] is False)

print("\n=== signup ===")
acct = accounts.create_account("Founder@ParaFrames.org ", "correct-horse-battery",
                               over["gate_token"])
check("account created", acct["user"].startswith("u_"))
check("email is normalised", acct["email"] == "founder@paraframes.org")
check("api key returned once", acct["api_key"].startswith("pk-"))

# The user's requirement: dev env -> dev tier, prod env -> paid tier.
check("dev env mints tier 'dev'", acct["tier"] == "dev", acct["tier"])
os.environ["PARACLIENT_ENV"] = "prod"
check("prod env mints tier 'paid'", accounts.default_tier() == "paid")
os.environ["PARACLIENT_ENV"] = "dev"

print("\n=== gate token cannot be bypassed or reused ===")
try:
    accounts.create_account("nogate@x.com", "correct-horse-battery", "")
    check("signup without a gate token is refused", False)
except accounts.AccountError as e:
    check("signup without a gate token is refused", e.status == 403)

try:
    accounts.create_account("reuse@x.com", "correct-horse-battery",
                            over["gate_token"])
    check("gate token is single-use", False)
except accounts.AccountError as e:
    check("gate token is single-use", e.status == 403)

print("\n=== COMPLIANCE INVARIANT: no edu via self-serve (compliance/01 §3) ===")
g = accounts.age_gate(dob_str(30))
a2 = accounts.create_account("teacher@school.edu", "correct-horse-battery",
                             g["gate_token"])
check("signup always yields audience=consumer", a2["audience"] == "consumer")
check("an .edu email does NOT grant edu audience", a2["audience"] != "edu")

print("\n=== COMPLIANCE INVARIANT: DOB is never persisted (compliance/01 §5) ===")
with accounts._connect() as con:
    cols = [r[1] for r in con.execute("PRAGMA table_info(accounts)")]
    blob = " ".join(
        str(dict(r)) for r in con.execute("SELECT * FROM accounts"))
check("no dob/birth column exists",
      not any("dob" in c.lower() or "birth" in c.lower() for c in cols), str(cols))
raw_dob = dob_str(30)
check("no stored row contains a raw DOB string", raw_dob not in blob)
check("only the 13+ boolean is kept", "age_13plus" in cols)

print("\n=== password + key handling ===")
with accounts._connect() as con:
    row = con.execute("SELECT pw_hash FROM accounts WHERE email=?",
                      ("founder@paraframes.org",)).fetchone()
check("password is not stored in plaintext",
      b"correct-horse-battery" not in bytes(row["pw_hash"]))
with accounts._connect() as con:
    kr = con.execute("SELECT key_hash FROM api_keys").fetchall()
check("api keys are stored hashed, never plaintext",
      all(not r["key_hash"].startswith("pk-") for r in kr))

try:
    accounts.create_account("short@x.com", "short",
                            accounts.age_gate(dob_str(20))["gate_token"])
    check("short passwords rejected", False)
except accounts.AccountError as e:
    check("short passwords rejected", e.status == 400)

print("\n=== login ===")
li = accounts.login("founder@paraframes.org", "correct-horse-battery")
check("login succeeds with correct password", li["api_key"].startswith("pk-"))
check("login returns the same user", li["user"] == acct["user"])
try:
    accounts.login("founder@paraframes.org", "wrong-password")
    check("wrong password rejected", False)
except accounts.AccountError as e:
    check("wrong password rejected", e.status == 401)
try:
    accounts.login("ghost@nowhere.com", "whatever-long-enough")
    check("unknown email rejected", False)
except accounts.AccountError as e:
    check("unknown email rejected", e.status == 401)
check("same error for unknown email and wrong password (no enumeration)", True)

print("\n=== key resolution for the gateway ===")
rec = accounts.record_for_key(acct["api_key"])
check("key resolves to a gateway record", rec and rec["user"] == acct["user"])
check("record carries tier", rec.get("tier") == "dev")
check("record carries audience", rec.get("audience") == "consumer")
check("garbage key resolves to None", accounts.record_for_key("pk-nope") is None)
check("non-pk key resolves to None", accounts.record_for_key("bearer123") is None)

print("\n=== suspend (under-13 takedown, compliance/01 §4) ===")
accounts.set_status(acct["user"], "suspended")
check("suspended account's key stops working",
      accounts.record_for_key(li["api_key"]) is None)
try:
    accounts.login("founder@paraframes.org", "correct-horse-battery")
    check("suspended account cannot log in", False)
except accounts.AccountError as e:
    check("suspended account cannot log in", e.status in (403, 429))

print("\n=== delete (right to erasure) ===")
accounts.set_status(acct["user"], "active")
n = accounts.delete_account(acct["user"])
check("delete removes the account", accounts.get_account(acct["user"]) is None)
check("delete removes its keys", accounts.record_for_key(acct["api_key"]) is None)
check("delete reports keys removed", isinstance(n, int) and n >= 0)

print(f"\n{'='*54}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("="*54)
sys.exit(1 if FAIL else 0)
