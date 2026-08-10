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

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}


class GoogleAuthError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


def verify_email(credential: str) -> str:
    """Return the verified email from a Google ID token, or raise GoogleAuthError."""
    if not GOOGLE_CLIENT_ID:
        raise GoogleAuthError(503, "Google sign-in is not configured on the server")
    if not credential:
        raise GoogleAuthError(400, "missing Google credential")
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as g_requests
        # verify_oauth2_token checks the signature, expiry, and that aud == our
        # client id; it raises on any failure.
        claims = id_token.verify_oauth2_token(
            credential, g_requests.Request(), GOOGLE_CLIENT_ID)
    except GoogleAuthError:
        raise
    except Exception:  # noqa: BLE001 — bad signature/aud/expiry all land here
        raise GoogleAuthError(401, "invalid Google credential")
    if claims.get("iss") not in _ISSUERS:
        raise GoogleAuthError(401, "unrecognized Google token issuer")
    email = claims.get("email")
    if not email or not claims.get("email_verified"):
        raise GoogleAuthError(401, "Google account email is not verified")
    return email
