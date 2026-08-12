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
  Plans are free -> plus -> premier, plus an internal dev tier; the gateway
  owns the per-tier model access and Knowledge Library quotas. There is no
  billing integration yet, so tier is decided by DEPLOYMENT ENVIRONMENT rather
  than by a plan the user bought:
      PARACLIENT_ENV=dev  (default) -> new accounts get tier "dev"
      PARACLIENT_ENV=prod           -> new accounts get tier "free"
  Everyone starts on Free and is promoted by a payment once billing exists, so
  no early cohort silently holds a paid tier it never bought (and nothing has
  to be taken away from them later). Free is also capped at v2/v3 on CPU, so a
  self-serve signup cannot queue work on the single GPU.
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
            created_at    REAL NOT NULL,
            -- Dev/staff accounts are grouped into a team derived from their
            -- paraframes.org subdomain (research.paraframes.org -> "research").
            -- NULL for ordinary consumer accounts.
            team          TEXT,
            -- 'password' (email+password signup) or 'google' (OAuth). OAuth
            -- accounts store an unusable random pw_hash so login() can't match.
            auth_provider TEXT NOT NULL DEFAULT 'password'
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
        -- Monthly usage meter. Deliberately NOT foreign-keyed to accounts:
        -- legacy keystore users (demo/internal keys) have no accounts row but
        -- still need metering. `period` is 'YYYY-MM' so a new month starts a
        -- new row and the quota resets with no cron job to run.
        CREATE TABLE IF NOT EXISTS usage (
            user_id  TEXT NOT NULL,
            period   TEXT NOT NULL,
            requests INTEGER NOT NULL DEFAULT 0,
            tokens   INTEGER NOT NULL DEFAULT 0,
            -- Quota-weighted cost: turbo requests bill 2x, eco 0.5x, normal 1x.
            -- `requests` stays the raw count (for display); `credits` is what the
            -- monthly quota is spent against.
            credits  REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, period)
        );
        -- Concurrent active sessions, capped per tier. Sessions expire on idle
        -- (SESSION_IDLE_TTL) so a crashed client cannot leak one of a user's
        -- slots forever; there is no reaper process to keep alive.
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            opened_at  REAL NOT NULL,
            last_seen  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        """)
        # Migrate older DBs created before the OAuth/team columns existed.
        cols = {r["name"] for r in con.execute("PRAGMA table_info(accounts)")}
        if "team" not in cols:
            con.execute("ALTER TABLE accounts ADD COLUMN team TEXT")
        if "auth_provider" not in cols:
            con.execute("ALTER TABLE accounts ADD COLUMN auth_provider "
                        "TEXT NOT NULL DEFAULT 'password'")
        ucols = {r["name"] for r in con.execute("PRAGMA table_info(usage)")}
        if "credits" not in ucols:
            con.execute("ALTER TABLE usage ADD COLUMN credits REAL NOT NULL "
                        "DEFAULT 0")
            # backfill: existing rows billed 1 credit per request
            con.execute("UPDATE usage SET credits = requests WHERE credits = 0")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def default_tier() -> str:
    """Tier a brand-new account starts on: free in production, dev on a dev box.

    Everyone signs up on Free and is promoted by a successful payment, so the
    plan ladder means the same thing before and after billing exists -- no
    cohort of early users silently holds a paid tier they never bought, and
    nothing has to be taken away from them later.

    It also keeps free signups off the GPU: Free is capped at v2/v3 (CPU), so a
    self-serve signup cannot queue work on the single L4.

    When billing lands, this function stays as it is; the promotion path is a
    separate call that sets tier to plus/premier on a successful charge."""
    env = os.environ.get("PARACLIENT_ENV", "dev").strip().lower()
    return "free" if env in ("prod", "production") else "dev"


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


def paraframes_team(email: str) -> str | None:
    """Team for a paraframes.org staff email, or None for anyone else.

    The team is the subdomain in front of paraframes.org:
      alice@research.paraframes.org -> "research"
      bob@paraframes.org            -> "paraframes"   (bare domain = core team)
      carol@x.y.paraframes.org      -> "x.y"
      dave@gmail.com                -> None           (ordinary consumer)
    A paraframes email is what marks an account as dev/staff (see provision_oauth).
    """
    dom = (email or "").strip().lower().rsplit("@", 1)[-1]
    if dom == "paraframes.org":
        return "paraframes"
    if dom.endswith(".paraframes.org"):
        return dom[: -len(".paraframes.org")]
    return None


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
                # rpm 0 = "no per-key override": the gateway applies the tier's
                # default cap (TIER_RPM). Only the legacy keystore pins per-key
                # limits explicitly.
                (user_id, email, pw_hash, salt, "consumer", tier, 0,
                 1, now, "active", now))
    except sqlite3.IntegrityError:
        # Same message whether or not the email exists, so signup cannot be
        # used to enumerate registered addresses.
        raise AccountError(409, "could not create an account with that email")
    key = _issue_key(user_id, label)
    return {"user": user_id, "email": email, "audience": "consumer",
            "tier": tier, "api_key": key}


def provision_oauth(email: str, *, provider: str = "google", tier: str,
                    team: str | None = None, audience: str = "consumer",
                    gate_token: str | None = None,
                    label: str = "oauth") -> dict:
    """Create-or-fetch a PASSWORDLESS account for a verified OAuth email and mint
    a fresh one-time API key. Idempotent per email.

    Two provisioning shapes, chosen by the caller from the email domain:
      * Dev/staff (team set, tier 'dev'): upsert tier+team+audience every time, so
        a paraframes.org staffer always resolves to a dev account in their team,
        even if a plain account already existed for that address.
      * Consumer (team None, tier 'free'): on FIRST creation this requires a
        passed age gate (gate_token), exactly like password signup (COPPA); an
        existing consumer's tier is left untouched (never silently changed).

    The stored pw_hash is a hash of a random secret, so the account can never be
    logged into with a password — OAuth is its only credential path."""
    email = _normalize_email(email)
    now = time.time()
    with _connect() as con:
        row = con.execute(
            "SELECT user_id,tier,team,audience,status FROM accounts WHERE email=?",
            (email,)).fetchone()
        if row:
            if row["status"] != "active":
                raise AccountError(403, "this account is suspended")
            user_id = row["user_id"]
            if team is not None:            # staff: keep them dev + in their team
                con.execute(
                    "UPDATE accounts SET tier=?,team=?,audience=?,auth_provider=? "
                    "WHERE user_id=?", (tier, team, audience, provider, user_id))
                final = (tier, team, audience)
            else:                           # existing consumer: don't clobber tier
                final = (row["tier"], row["team"], row["audience"])
        else:
            if team is None:                # new consumer must clear the age gate
                _consume_gate_token(gate_token)
            user_id = "u_" + secrets.token_hex(8)
            pw_hash, salt = _hash_password(secrets.token_urlsafe(32))  # unusable
            try:
                con.execute(
                    "INSERT INTO accounts(user_id,email,pw_hash,pw_salt,audience,"
                    "tier,rpm,age_13plus,age_checked_at,status,created_at,team,"
                    "auth_provider) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (user_id, email, pw_hash, salt, audience, tier, 0, 1, now,
                     "active", now, team, provider))
            except sqlite3.IntegrityError:
                raise AccountError(409, "could not provision that account")
            final = (tier, team, audience)
    key = _issue_key(user_id, label)
    return {"user": user_id, "email": email, "audience": final[2],
            "tier": final[0], "team": final[1], "api_key": key}


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
            "SELECT a.user_id,a.audience,a.tier,a.rpm,a.status,a.team FROM api_keys k "
            "JOIN accounts a ON a.user_id=k.user_id WHERE k.key_hash=?",
            (kh,)).fetchone()
        if not row:
            return None
        if row["status"] != "active":
            return None          # suspended accounts authenticate as nobody
        con.execute("UPDATE api_keys SET last_used=? WHERE key_hash=?",
                    (time.time(), kh))
    return {"user": row["user_id"], "audience": row["audience"],
            "tier": row["tier"], "rpm": row["rpm"], "team": row["team"],
            "source": "accounts_db"}


# --------------------------------------------------------------------------
# Usage metering (monthly quota)
# --------------------------------------------------------------------------

def current_period(now: float | None = None) -> str:
    """Billing period key, 'YYYY-MM'. Using the calendar month as the row key
    means the quota resets on its own when the month rolls over — there is no
    reset job that can fail to run."""
    t = time.gmtime(now if now is not None else time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}"


def record_usage(user_id: str, requests: int = 1, tokens: int = 0,
                 credits: float | None = None) -> None:
    """Add to this user's meter for the current period.

    `credits` is the quota-weighted cost of the request (turbo = 2x, eco = 0.5x);
    it defaults to the raw request count, so ordinary usage bills 1 credit each
    and existing behaviour is unchanged."""
    if not user_id:
        return
    if credits is None:
        credits = float(requests)
    with _connect() as con:
        con.execute(
            "INSERT INTO usage(user_id, period, requests, tokens, credits) "
            "VALUES (?,?,?,?,?) ON CONFLICT(user_id, period) DO UPDATE SET "
            "requests = requests + excluded.requests, "
            "tokens   = tokens   + excluded.tokens, "
            "credits  = credits  + excluded.credits",
            (user_id, current_period(), requests, tokens, credits))


def usage_for(user_id: str) -> dict:
    """This period's usage. Missing row = nothing used yet. `credits` is the
    quota-weighted total; `requests` is the raw count."""
    with _connect() as con:
        row = con.execute(
            "SELECT requests, tokens, credits FROM usage WHERE user_id=? AND period=?",
            (user_id, current_period())).fetchone()
    return {"period": current_period(),
            "requests": row["requests"] if row else 0,
            "tokens": row["tokens"] if row else 0,
            "credits": round(row["credits"], 3) if row else 0.0}


def reset_usage(user_id: str) -> None:
    """Clear a user's meter for this period (support / testing escape hatch)."""
    with _connect() as con:
        con.execute("DELETE FROM usage WHERE user_id=? AND period=?",
                    (user_id, current_period()))


# --------------------------------------------------------------------------
# Concurrent sessions
# --------------------------------------------------------------------------

SESSION_IDLE_TTL = float(os.environ.get("SESSION_IDLE_TTL", 30 * 60))


def _prune_sessions(con, user_id: str | None = None) -> None:
    """Drop idle-expired sessions. Called before any count so an abandoned
    client never permanently occupies one of a user's slots."""
    cutoff = time.time() - SESSION_IDLE_TTL
    if user_id:
        con.execute("DELETE FROM sessions WHERE last_seen < ? AND user_id = ?",
                    (cutoff, user_id))
    else:
        con.execute("DELETE FROM sessions WHERE last_seen < ?", (cutoff,))


def active_sessions(user_id: str) -> int:
    with _connect() as con:
        _prune_sessions(con, user_id)
        return con.execute("SELECT COUNT(*) c FROM sessions WHERE user_id=?",
                           (user_id,)).fetchone()["c"]


def open_session(user_id: str, max_active: int | None) -> dict:
    """Claim a session slot. `max_active=None` means unlimited (dev).

    Raises 429 when the user is already at their tier's cap — the caller should
    close an old session or wait for one to idle out."""
    now = time.time()
    with _connect() as con:
        _prune_sessions(con, user_id)
        n = con.execute("SELECT COUNT(*) c FROM sessions WHERE user_id=?",
                        (user_id,)).fetchone()["c"]
        if max_active is not None and n >= max_active:
            raise AccountError(
                429, f"all {max_active} concurrent sessions are in use — close "
                     f"one or wait for an idle session to expire")
        sid = "s_" + secrets.token_urlsafe(18)
        con.execute("INSERT INTO sessions(session_id,user_id,opened_at,last_seen)"
                    " VALUES (?,?,?,?)", (sid, user_id, now, now))
    return {"session_id": sid, "active": n + 1, "limit": max_active,
            "idle_ttl": SESSION_IDLE_TTL}


def touch_session(session_id: str) -> bool:
    """Heartbeat — keeps a session from idling out."""
    with _connect() as con:
        cur = con.execute("UPDATE sessions SET last_seen=? WHERE session_id=?",
                          (time.time(), session_id))
    return cur.rowcount > 0


def close_session(session_id: str) -> bool:
    with _connect() as con:
        cur = con.execute("DELETE FROM sessions WHERE session_id=?",
                          (session_id,))
    return cur.rowcount > 0


def get_account(user_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT user_id,email,audience,tier,rpm,status,created_at,"
            "age_13plus,team,auth_provider FROM accounts WHERE user_id=?",
            (user_id,)).fetchone()
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
        # Usage and session rows are keyed by user_id, not FK-cascaded — erase
        # them too, or deletion would leave per-user records behind.
        con.execute("DELETE FROM usage WHERE user_id=?", (user_id,))
        con.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    return n


def user_id_for_email(email: str) -> str | None:
    with _connect() as con:
        row = con.execute("SELECT user_id FROM accounts WHERE email=?",
                          (_normalize_email(email),)).fetchone()
    return row["user_id"] if row else None


def count_accounts() -> int:
    with _connect() as con:
        return con.execute("SELECT COUNT(*) c FROM accounts").fetchone()["c"]
