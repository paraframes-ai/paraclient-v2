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
check("prod env mints a paid tier", accounts.default_tier() == "plus")
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

print("\n=== tiers: model access, storage quotas, rate limiting ===")
import auth_gateway as gw  # noqa: E402

check("tiers are free/plus/premiere/dev",
      gw.TIERS == ("free", "plus", "premiere", "dev"), str(gw.TIERS))

# Storage allowances exactly as specified.
GB, TB = 1000 ** 3, 1000 ** 4
for tier, want, label in [("free", 128 * GB, "128 GB"), ("plus", 1 * TB, "1 TB"),
                          ("premiere", 2 * TB, "2 TB"), ("dev", 4 * TB, "4 TB")]:
    got = gw.storage_bytes_for({"tier": tier})
    check(f"{tier} library storage = {label}", got == want,
          f"got {gw.human_bytes(got)}")

# Rate limiting: everything IS capped except dev.
for tier, want in [("free", 20), ("plus", 60), ("premiere", 120), ("paid", 60)]:
    got = gw.rpm_for({"tier": tier})
    check(f"{tier} IS rate limited at {want}/min", got == want, f"got {got}")
check("dev is EXEMPT from rate limiting", gw.rpm_for({"tier": "dev"}) is None)
check("an explicit per-key rpm overrides the tier default",
      gw.rpm_for({"tier": "free", "rpm": 999}) == 999)
check("rpm=0 means 'use the tier default'",
      gw.rpm_for({"tier": "premiere", "rpm": 0}) == 120)
check("an unknown tier still gets capped",
      gw.rpm_for({"tier": "bogus"}) == gw.DEFAULT_RPM)

# The limiter itself must actually block past the cap for a non-dev tier.
_rl = gw.RateLimiter()
_now = time.time()
_blocked = sum(0 if _rl.allow("t", gw.rpm_for({"tier": "free"}), _now) else 1
               for _ in range(50))
check("free key is blocked past its cap in a burst", _blocked == 30, f"{_blocked}/50")

# Model access: free is CPU-only; every paid tier reaches v4 (GPU).
check("free cannot use v4", "v4" not in gw.versions_for_tier("free"))
for tier in ("plus", "premiere", "dev"):
    check(f"{tier} can use v4", "v4" in gw.versions_for_tier(tier))
check("free defaults to v2", gw.DEFAULT_VERSION_BY_TIER["free"] == "v2")

# Back-compat: keys already issued as "paid" keep Plus-level access.
check("legacy 'paid' keeps v4 access", "v4" in gw.versions_for_tier("paid"))
check("legacy 'paid' maps to the Plus quota",
      gw.storage_bytes_for({"tier": "paid"}) == 1 * TB)
check("legacy 'paid' displays as Plus", gw.TIER_LABEL["paid"] == "Plus")
check("edu audience still resolves to a paid tier",
      gw.tier_of({"audience": "edu"}) in gw.PAID_TIERS)
check("unknown tier falls back to free",
      gw.storage_bytes_for({"tier": "bogus"}) == 128 * GB)

os.environ["PARACLIENT_ENV"] = "prod"
check("prod signups get the Plus tier", accounts.default_tier() == "plus")
os.environ["PARACLIENT_ENV"] = "dev"

print("\n=== product naming: Kalvi (edu) vs ParaClient (consumer) ===")
edu = {"audience": "edu", "tier": "paid"}
con = {"audience": "consumer", "tier": "plus"}
check("edu product is Kalvi", gw.product_for(edu) == "Kalvi")
check("consumer product is ParaClient", gw.product_for(con) == "ParaClient")
for v, want in [("v2", "Kalvi 2"), ("v3", "Kalvi 3"), ("v4", "Kalvi 4")]:
    got = gw.display_version(edu, v)
    check(f"edu {v} displays as {want}", got == want, f"got {got}")
check("consumer v4 displays as ParaClient 4",
      gw.display_version(con, "v4") == "ParaClient 4")
check("internal/dev keeps the ParaClient name",
      gw.product_for({"audience": "internal"}) == "ParaClient")
check("edu version list maps in order",
      gw.display_versions(edu, ["v2", "v3", "v4"]) == ["Kalvi 2", "Kalvi 3", "Kalvi 4"])
# The serving contract must NOT be renamed by the branding layer.
check("served model ids are untouched by branding",
      gw.model_for("socratic", "math") == "ParaFrames/ParaClient-math-v2.2")

print("\n=== monthly usage quotas (free 1x, plus 2x, premiere 20x) ===")
qf = gw.monthly_quota_for({"tier": "free"})
qp = gw.monthly_quota_for({"tier": "plus"})
qpr = gw.monthly_quota_for({"tier": "premiere"})
check("plus is exactly 2x free", qp == 2 * qf, f"{qp} vs {qf}")
check("premiere is exactly 10x plus", qpr == 10 * qp, f"{qpr} vs {qp}")
check("premiere is exactly 20x free", qpr == 20 * qf, f"{qpr} vs {qf}")
check("dev is exempt from the usage quota",
      gw.monthly_quota_for({"tier": "dev"}) is None)
check("legacy 'paid' gets the Plus quota",
      gw.monthly_quota_for({"tier": "paid"}) == qp)
check("an unknown tier gets the free quota",
      gw.monthly_quota_for({"tier": "bogus"}) == qf)

# The meter itself.
g2 = accounts.age_gate(dob_str(22))
m = accounts.create_account("meter@example.com", "a-long-enough-password",
                            g2["gate_token"])
check("new account starts at zero usage",
      accounts.usage_for(m["user"])["requests"] == 0)
accounts.record_usage(m["user"], requests=3, tokens=250)
u = accounts.usage_for(m["user"])
check("usage accumulates requests", u["requests"] == 3, str(u))
check("usage accumulates tokens", u["tokens"] == 250, str(u))
accounts.record_usage(m["user"], requests=2)
check("usage adds across calls",
      accounts.usage_for(m["user"])["requests"] == 5)
check("usage period is the current month",
      accounts.usage_for(m["user"])["period"] == accounts.current_period())
accounts.reset_usage(m["user"])
check("usage can be reset", accounts.usage_for(m["user"])["requests"] == 0)

# Erasure must not leave usage rows behind.
accounts.record_usage(m["user"], requests=7)
accounts.delete_account(m["user"])
with accounts._connect() as con:
    left = con.execute("SELECT COUNT(*) c FROM usage WHERE user_id=?",
                       (m["user"],)).fetchone()["c"]
check("deleting an account erases its usage rows", left == 0, f"{left} left")

print("\n=== premiere perks: sessions, context, queue priority ===")
for tier, want in [("free", 50), ("plus", 100), ("premiere", 1000)]:
    got = gw.max_sessions_for({"tier": tier})
    check(f"{tier} allows {want} concurrent sessions", got == want, f"got {got}")
check("dev has unlimited sessions",
      gw.max_sessions_for({"tier": "dev"}) is None)
check("legacy 'paid' gets the Plus session cap",
      gw.max_sessions_for({"tier": "paid"}) == 100)

# Longer context for premiere.
cf, cp, cpr = (gw.max_context_for({"tier": t})
               for t in ("free", "plus", "premiere"))
check("context grows free < plus < premiere", cf < cp < cpr, f"{cf}/{cp}/{cpr}")
check("premiere gets the server's full 16k window", cpr == 16384, str(cpr))

# Priority: LOWER is handled earlier.
pf, pp, ppr = (gw.priority_for({"tier": t})
               for t in ("free", "plus", "premiere"))
check("premiere outranks plus outranks free in the queue",
      ppr < pp < pf, f"premiere={ppr} plus={pp} free={pf}")
check("dev shares premiere's top priority",
      gw.priority_for({"tier": "dev"}) == ppr)

# Session slots are really enforced and really released.
g3 = accounts.age_gate(dob_str(25))
s = accounts.create_account("sess@example.com", "a-long-enough-password",
                            g3["gate_token"])
uid = s["user"]
opened = [accounts.open_session(uid, 3)["session_id"] for _ in range(3)]
check("sessions open up to the cap", accounts.active_sessions(uid) == 3)
try:
    accounts.open_session(uid, 3)
    check("opening past the cap is refused", False)
except accounts.AccountError as e:
    check("opening past the cap is refused", e.status == 429)
accounts.close_session(opened[0])
check("closing frees a slot", accounts.active_sessions(uid) == 2)
check("a slot freed by close can be reused",
      bool(accounts.open_session(uid, 3)["session_id"]))
check("heartbeat keeps a session alive", accounts.touch_session(opened[1]))
check("heartbeat on an unknown session fails",
      accounts.touch_session("s_nonexistent") is False)

# Idle expiry must reclaim slots, or a crashed client leaks one forever.
_ttl = accounts.SESSION_IDLE_TTL
accounts.SESSION_IDLE_TTL = -1          # everything is instantly idle
check("idle sessions are reclaimed", accounts.active_sessions(uid) == 0)
accounts.SESSION_IDLE_TTL = _ttl

accounts.open_session(uid, 3)
accounts.delete_account(uid)
with accounts._connect() as con:
    left = con.execute("SELECT COUNT(*) c FROM sessions WHERE user_id=?",
                       (uid,)).fetchone()["c"]
check("deleting an account erases its sessions", left == 0, f"{left} left")

print(f"\n{'='*54}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("="*54)
sys.exit(1 if FAIL else 0)
