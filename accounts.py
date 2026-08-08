#!/usr/bin/env python3
"""
accounts.py — real account store for the ParaFrames gateway.

Replaces the hand-edited `gateway_keys.json` scaffold (plaintext keys, no
signup, no login) with a durable SQLite store that supports the actual consumer
flow: age gate -> sign up -> log in -> per-user API keys.

WHY SQLITE: the gateway is a single process on one box; SQLite gives durability,
atomic writes and concurrent-reader safety with no new dependency and no server
to run. (Same choice as the civics review console.) If ParaFrames ever runs
multiple gateway processes, this is the seam to swap for Postgres — every access
goes through the functions below, nothing reaches into the tables directly.

WHAT IS STORED, AND WHAT DELIBERATELY IS NOT
  * Passwords -> scrypt hash + per-user salt. Never the password.
  * API keys  -> SHA-256 hash ONLY. The plaintext key is shown once, at
    creation, and is unrecoverable afterwards. A dump of this DB yields no
    usable credentials (unlike gateway_keys.json, which held live keys).
  * Age       -> a single `age_13plus` boolean + the timestamp it was checked.
    The DATE OF BIRTH IS NEVER PERSISTED and never logged. This is
    compliance/01_age_gate_and_routing.md §5 (data minimization at the gate),
    and it is why the gate is its own endpoint: the DOB is used to make one
    routing decision, in memory, and then discarded.

THE AGE GATE IS THE HINGE (compliance/01 §1-§2)
  The whole two-lane compliance model rests on no under-13 ever reaching the
  consumer lane. So:
    - The gate runs BEFORE any other data is collected. `age_gate()` takes a DOB
      and nothing else -- no email, no password, no name.
    - Under 13 is a HARD STOP, not a downgrade to a lesser tier. Nothing is
      stored for that person: no row, no log line, no counter. The caller is
      told to go through their school.
    - Passing the gate mints a short-lived, single-use gate token. `signup()`
      refuses to create an account without one. There is no code path that
      creates an account without a passed gate.

  `audience` is HARD-CODED to "consumer" for every account created here. An
  `edu` credential may only be minted by the school-authenticated + DPA-on-file
  path (compliance/01 §3); it must be impossible to obtain one through
  self-serve signup. `test_accounts.py` asserts this as an invariant.

TIERS (interim, pre-billing)
  There is no billing integration yet, so tier is decided by DEPLOYMENT
  ENVIRONMENT rather than by a plan the user bought:
      PARACLIENT_ENV=dev  (default) -> new accounts get tier "dev"
      PARACLIENT_ENV=prod           -> new accounts get tier "paid"
  This keeps production users on full access while payments do not exist, and
  keeps dev boxes clearly labelled. When billing lands, `default_tier()` is the
  single function that changes: real accounts start "free" and are promoted on
  successful payment.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path(os.environ.get(
    "ACCOUNTS_DB", str(Path(__file__).with_name("accounts.db"))))

# scrypt cost parameters. n=2**15 keeps a single hash around ~100ms on this
# box -- slow enough to make offline cracking expensive, fast enough that a
# login is not noticeable. r/p are the standard values.
_SCRYPT = dict(n=2 ** 15, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=32)

# A gate token is only a short-lived receipt that "someone passed the age gate";
# it carries no identity. 30 minutes is long enough to finish a signup form and
# short enough that a leaked token is worthless.
GATE_TTL_SECONDS = 30 * 60

MIN_PASSWORD_LEN = 10          # NIST 800-63B favours length over composition
LOGIN_MAX_ATTEMPTS = 8         # per email, per window
LOGIN_WINDOW_SECONDS = 15 * 60

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


class AccountError(Exception):
    """Raised for any caller-correctable problem (bad input, duplicate email,
    wrong password). Carries an HTTP status so the gateway can map it directly
    without a second layer of error translation."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")      # concurrent readers during writes
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    """Create the schema if absent. Safe to call on every boot."""
    with _connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            user_id       TEXT PRIMARY KEY,
            email         TEXT NOT NULL UNIQUE,
            pw_hash       BLOB NOT NULL,
            pw_salt       BLOB NOT NULL,
            audience      TEXT NOT NULL,
            tier          TEXT NOT NULL,
            rpm           INTEGER NOT NULL DEFAULT 30,
            -- Age gate result ONLY. The DOB itself is never written here.
            age_13plus    INTEGER NOT NULL,
            age_checked_at REAL NOT NULL,
            status        TEXT NOT NULL DEFAULT 'active',
            created_at    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash   TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,
            label      TEXT,
            created_at REAL NOT NULL,
            last_used  REAL
        );
        CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
        CREATE TABLE IF NOT EXISTS gate_tokens (
            token      TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            used       INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS login_attempts (
            email      TEXT NOT NULL,
            at         REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_login_email ON login_attempts(email);
        """)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def default_tier() -> str:
    """Interim tier policy: dev boxes mint dev accounts, production mints paid.

    Pre-billing there is no way to *buy* anything, so production users get full
    access rather than being gated behind a plan that cannot be purchased. This
    is the one function to change when payments land."""
    env = os.environ.get("PARACLIENT_ENV", "dev").strip().lower()
    return "paid" if env in ("prod", "production") else "dev"


def _hash_password(password: str, salt: bytes | None = None):
    salt = salt or secrets.token_bytes(16)
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT), salt


def _hash_key(key: str) -> str:
    """API keys are high-entropy random strings, so a plain SHA-256 is the right
    tool (no salt/stretching needed -- there is nothing to guess)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _normalize_email(email: str) -> str:
    e = (email or "").strip().lower()
    if not _EMAIL_RE.match(e) or len(e) > 254:
        raise AccountError(400, "enter a valid email address")
    return e


def age_from_dob(dob: str) -> int:
    """Whole years between `dob` (YYYY-MM-DD) and today.

    The value is used for one comparison and thrown away -- never stored, never
    logged (compliance/01 §5)."""
    try:
        d = datetime.strptime(dob.strip(), "%Y-%m-%d").date()
    except Exception:
        raise AccountError(400, "enter your date of birth as YYYY-MM-DD")
    today = date.today()
    if d > today:
        raise AccountError(400, "date of birth cannot be in the future")
    if d.year < 1900:
        raise AccountError(400, "enter a valid date of birth")
    # Subtract one year if this year's birthday has not happened yet.
    return today.year - d.year - ((today.month, today.day) < (d.month, d.day))


# --------------------------------------------------------------------------
# Age gate  (compliance/01 §1-§2)
# --------------------------------------------------------------------------

UNDER_13_MESSAGE = (
    "ParaFrames for under-13 is available through your school. Ask your "
    "teacher, or have your school contact us.")


def age_gate(dob: str) -> dict:
    """Neutral date-of-birth gate. Runs BEFORE any other data is collected.

    13+  -> mint a single-use gate token; signup() will require it.
    <13  -> HARD STOP. No account, no token, and deliberately NOTHING written:
            no row, no audit line, no attempt counter. Storing "a child tried to
            sign up" would itself be collecting data from a child, which is the
            opposite of what §2 asks for.
    """
    age = age_from_dob(dob)          # local only; never persisted or logged
    if age < 13:
        return {"allowed": False, "message": UNDER_13_MESSAGE}
    token = secrets.token_urlsafe(24)
    with _connect() as con:
        con.execute("INSERT INTO gate_tokens(token, created_at) VALUES (?,?)",
                    (token, time.time()))
    return {"allowed": True, "gate_token": token,
            "expires_in": GATE_TTL_SECONDS}


def _consume_gate_token(token: str) -> None:
    """Single-use + TTL. Raises unless the token is a live, unused receipt."""
    if not token:
        raise AccountError(403, "complete the age check before creating an account")
    now = time.time()
    with _connect() as con:
        row = con.execute(
            "SELECT token, created_at, used FROM gate_tokens WHERE token=?",
            (token,)).fetchone()
        if not row or row["used"]:
            raise AccountError(403, "age check is invalid or already used")
        if now - row["created_at"] > GATE_TTL_SECONDS:
            con.execute("DELETE FROM gate_tokens WHERE token=?", (token,))
            raise AccountError(403, "age check expired — please start again")
        con.execute("UPDATE gate_tokens SET used=1 WHERE token=?", (token,))
        # Opportunistic cleanup of expired receipts.
        con.execute("DELETE FROM gate_tokens WHERE created_at < ?",
                    (now - GATE_TTL_SECONDS * 4,))


# --------------------------------------------------------------------------
# Signup / login
# --------------------------------------------------------------------------

def create_account(email: str, password: str, gate_token: str,
                   label: str = "signup") -> dict:
    """Create a consumer account after a passed age gate. Returns the account
    plus a ONE-TIME plaintext API key (never recoverable again).

    `audience` is hard-coded to "consumer": an edu credential must come from the
    school-authenticated + DPA-on-file path only (compliance/01 §3)."""
    email = _normalize_email(email)
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AccountError(
            400, f"password must be at least {MIN_PASSWORD_LEN} characters")
    _consume_gate_token(gate_token)

    user_id = "u_" + secrets.token_hex(8)
    pw_hash, salt = _hash_password(password)
    now = time.time()
    tier = default_tier()
    try:
        with _connect() as con:
            con.execute(
                "INSERT INTO accounts(user_id,email,pw_hash,pw_salt,audience,"
                "tier,rpm,age_13plus,age_checked_at,status,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (user_id, email, pw_hash, salt, "consumer", tier, 30,
                 1, now, "active", now))
    except sqlite3.IntegrityError:
        # Same message whether or not the email exists, so signup cannot be
        # used to enumerate registered addresses.
        raise AccountError(409, "could not create an account with that email")
    key = _issue_key(user_id, label)
    return {"user": user_id, "email": email, "audience": "consumer",
            "tier": tier, "api_key": key}


def _issue_key(user_id: str, label: str = "") -> str:
    """Mint an API key, store only its hash, return the plaintext once."""
    key = "pk-" + secrets.token_urlsafe(32)
    with _connect() as con:
        con.execute(
            "INSERT INTO api_keys(key_hash,user_id,label,created_at) "
            "VALUES (?,?,?,?)", (_hash_key(key), user_id, label, time.time()))
    return key


def _login_throttled(email: str) -> bool:
    """Simple per-email attempt cap. Without this, the login endpoint is an
    offline-speed password oracle."""
    cutoff = time.time() - LOGIN_WINDOW_SECONDS
    with _connect() as con:
        con.execute("DELETE FROM login_attempts WHERE at < ?", (cutoff,))
        n = con.execute("SELECT COUNT(*) c FROM login_attempts WHERE email=?",
                        (email,)).fetchone()["c"]
    return n >= LOGIN_MAX_ATTEMPTS


def _record_login_attempt(email: str) -> None:
    with _connect() as con:
        con.execute("INSERT INTO login_attempts(email, at) VALUES (?,?)",
                    (email, time.time()))


def login(email: str, password: str, label: str = "login") -> dict:
    """Verify credentials and issue a fresh API key.

    Every login mints a new key rather than returning the old one: keys are
    stored hashed, so the previous plaintext genuinely cannot be produced
    again. Callers who want fewer live keys should call revoke_key on logout."""
    email = _normalize_email(email)
    if _login_throttled(email):
        raise AccountError(429, "too many sign-in attempts — try again later")
    _record_login_attempt(email)

    with _connect() as con:
        row = con.execute(
            "SELECT user_id,pw_hash,pw_salt,audience,tier,status "
            "FROM accounts WHERE email=?", (email,)).fetchone()

    # Hash even when the account is missing, so a wrong email and a wrong
    # password take the same time (no user-enumeration by response latency).
    salt = row["pw_salt"] if row else b"\x00" * 16
    expect = row["pw_hash"] if row else b"\x00" * 32
    candidate, _ = _hash_password(password or "", salt)
    ok = hmac.compare_digest(candidate, expect) and row is not None
    if not ok:
        raise AccountError(401, "incorrect email or password")
    if row["status"] != "active":
        raise AccountError(403, "this account is suspended")

    key = _issue_key(row["user_id"], label)
    with _connect() as con:      # a clean login clears the throttle
        con.execute("DELETE FROM login_attempts WHERE email=?", (email,))
    return {"user": row["user_id"], "email": email,
            "audience": row["audience"], "tier": row["tier"], "api_key": key}


# --------------------------------------------------------------------------
# Lookup / administration
# --------------------------------------------------------------------------

def record_for_key(key: str) -> dict | None:
    """Resolve a bearer key to a gateway-shaped record, or None.

    The returned dict matches what the JSON keystore yielded ({user, audience,
    tier, rpm}) so the gateway's existing tier/mode logic works unchanged."""
    if not key or not key.startswith("pk-"):
        return None
    kh = _hash_key(key)
    with _connect() as con:
        row = con.execute(
            "SELECT a.user_id,a.audience,a.tier,a.rpm,a.status FROM api_keys k "
            "JOIN accounts a ON a.user_id=k.user_id WHERE k.key_hash=?",
            (kh,)).fetchone()
        if not row:
            return None
        if row["status"] != "active":
            return None          # suspended accounts authenticate as nobody
        con.execute("UPDATE api_keys SET last_used=? WHERE key_hash=?",
                    (time.time(), kh))
    return {"user": row["user_id"], "audience": row["audience"],
            "tier": row["tier"], "rpm": row["rpm"], "source": "accounts_db"}


def get_account(user_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT user_id,email,audience,tier,rpm,status,created_at,"
            "age_13plus FROM accounts WHERE user_id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def revoke_key(key: str) -> bool:
    with _connect() as con:
        cur = con.execute("DELETE FROM api_keys WHERE key_hash=?",
                          (_hash_key(key),))
    return cur.rowcount > 0


def set_status(user_id: str, status: str) -> bool:
    """Suspend/reactivate. Suspension is step 1 of the discovered-under-13
    takedown (compliance/01 §4) — it stops access immediately, before the
    slower data deletion runs."""
    if status not in ("active", "suspended"):
        raise AccountError(400, "status must be 'active' or 'suspended'")
    with _connect() as con:
        cur = con.execute("UPDATE accounts SET status=? WHERE user_id=?",
                          (status, user_id))
        if status == "suspended":     # revoke live credentials immediately
            con.execute("DELETE FROM api_keys WHERE user_id=?", (user_id,))
    return cur.rowcount > 0


def delete_account(user_id: str) -> int:
    """Erase the account and its keys. Returns the number of keys removed.
    Used by /v1/account/delete (right-to-erasure + under-13 takedown)."""
    with _connect() as con:
        n = con.execute("SELECT COUNT(*) c FROM api_keys WHERE user_id=?",
                        (user_id,)).fetchone()["c"]
        con.execute("DELETE FROM api_keys WHERE user_id=?", (user_id,))
        con.execute("DELETE FROM accounts WHERE user_id=?", (user_id,))
    return n


def user_id_for_email(email: str) -> str | None:
    with _connect() as con:
        row = con.execute("SELECT user_id FROM accounts WHERE email=?",
                          (_normalize_email(email),)).fetchone()
    return row["user_id"] if row else None


def count_accounts() -> int:
    with _connect() as con:
        return con.execute("SELECT COUNT(*) c FROM accounts").fetchone()["c"]
