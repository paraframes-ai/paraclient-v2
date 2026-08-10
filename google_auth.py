#!/usr/bin/env python3
"""
google_auth.py — verify a Google Sign-In ID token, SERVER-SIDE.

The app (iOS/web) obtains a Google ID token ("credential") from Google Identity
Services and posts it to /v1/auth/google. We verify it HERE — never in the
browser — against Google's public keys (offline signature check via google-auth)
and require the token's audience to be OUR OAuth client id (GOOGLE_CLIENT_ID), so
a token minted for a different app cannot be replayed against us. Returns the
verified email address.

Fail closed: if Google sign-in isn't configured, or the token is missing,
malformed, unverified, or for the wrong audience/issuer, we raise — never return
an unverified identity.
"""
from __future__ import annotations

import os
import re

_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}


class GoogleAuthError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


def _allowed_client_ids() -> set[str]:
    """The OAuth client id(s) we accept as the token audience. GOOGLE_CLIENT_ID
    may hold several (comma/space separated) — e.g. the iOS app's client id and
    a web client id — because tokens from different app types carry different
    `aud` values. Read at call time so config changes need no code reload."""
    raw = os.environ.get("GOOGLE_CLIENT_ID", "")
    return {c for c in re.split(r"[,\s]+", raw) if c}


def verify_email(credential: str) -> str:
    """Return the verified email from a Google ID token, or raise GoogleAuthError.

    Verifies the token's signature/issuer/expiry against Google's public keys,
    then requires its `aud` to be one of our configured client ids (so a token
    minted for another app can't be replayed here)."""
    allowed = _allowed_client_ids()
    if not allowed:
        raise GoogleAuthError(503, "Google sign-in is not configured on the server")
    if not credential:
        raise GoogleAuthError(400, "missing Google credential")
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as g_requests
        # audience=None: verify signature/issuer/expiry now, check aud ourselves
        # below (so we can accept more than one client id).
        claims = id_token.verify_oauth2_token(credential, g_requests.Request())
    except GoogleAuthError:
        raise
    except Exception:  # noqa: BLE001 — bad signature/expiry all land here
        raise GoogleAuthError(401, "invalid Google credential")
    if claims.get("iss") not in _ISSUERS:
        raise GoogleAuthError(401, "unrecognized Google token issuer")
    if claims.get("aud") not in allowed:
        raise GoogleAuthError(401, "Google token was issued for a different app")
    email = claims.get("email")
    if not email or not claims.get("email_verified"):
        raise GoogleAuthError(401, "Google account email is not verified")
    return email
