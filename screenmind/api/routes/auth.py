"""Auth routes — mandatory PIN lock, session management.

Security model (fail-closed):
  * A PIN is MANDATORY. With no PIN configured the dashboard is locked and no
    request unlocks it — an empty PIN is never a valid credential.
  * The PIN is stored only as a salted PBKDF2-SHA256 hash ("salt$hash").
    argon2-cffi is intentionally NOT a project dependency, so PBKDF2 (a
    stdlib KDF) is used instead of adding a heavy crypto dependency.
  * Comparisons are constant-time (hmac.compare_digest) to avoid timing leaks.
  * Failed attempts are rate-limited per client IP; the lockout is fail-closed.
"""

import hashlib
import hmac
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from screenmind.config import settings
from screenmind.api.dependencies import (
    SESSION_COOKIE,
    create_session,
    delete_session,
    pin_rate_limit_status,
    record_pin_failure,
    reset_pin_failures,
    set_session_cookie,
    verify_session,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _hash_pin(pin: str, salt: str = None) -> str:
    """Hash PIN with PBKDF2-SHA256 (salted, 100k iterations).
    Returns 'salt$hash' format. If salt provided, uses it (for verification)."""
    if salt is None:
        salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt.encode(), 100_000)
    return f"{salt}${dk.hex()}"


def _verify_pin(pin: str, stored_hash: str) -> bool:
    """Verify PIN against stored 'salt$hash'. Also supports legacy plain SHA-256.

    Fail-closed: an empty PIN or an empty stored hash never verifies.
    """
    if not pin or not stored_hash:
        return False
    # New format: salt$hash
    if "$" in stored_hash:
        salt = stored_hash.split("$")[0]
        return hmac.compare_digest(_hash_pin(pin, salt), stored_hash)
    # Legacy: plain SHA-256 (migrate on next set-pin)
    return hmac.compare_digest(hashlib.sha256(pin.encode()).hexdigest(), stored_hash)


def _client_id(request: Request) -> str:
    """Stable per-client key for rate limiting (IP, or 'unknown')."""
    return request.client.host if request.client else "unknown"


@router.get("/status")
async def auth_status(request: Request):
    """Check if PIN is set and if current session is valid."""
    has_pin = bool(settings.dashboard_pin_hash)
    token = request.cookies.get(SESSION_COOKIE)
    # Fail-closed: a session is only valid when a PIN exists AND the cookie
    # matches. With no PIN configured the client is NOT authenticated.
    authenticated = has_pin and verify_session(token)
    # First-run detection: setup_complete flag in settings.json
    # If settings.json exists with any data, user already ran setup (even without the flag)
    import json
    setup_complete = False
    try:
        if settings.settings_json_path.exists():
            data = json.loads(settings.settings_json_path.read_text())
            setup_complete = data.get("setup_complete", bool(data))
    except Exception:
        pass
    return {"has_pin": has_pin, "authenticated": authenticated, "first_run": not setup_complete}


@router.post("/setup-complete")
async def mark_setup_complete(request: Request):
    """First-run setup: set the mandatory PIN and mark setup complete.

    A PIN is REQUIRED — setup cannot be completed without one, so an empty
    PIN never unlocks the dashboard (fail-closed).
    """
    body = await request.json()
    pin = str(body.get("pin", "")).strip()
    if not pin:
        return JSONResponse(
            {"ok": False, "error": "PIN required"},
            status_code=400,
        )
    settings.save_runtime_overrides(
        {"setup_complete": True, "dashboard_pin_hash": _hash_pin(pin)}
    )
    token = create_session()
    response = JSONResponse({"ok": True, "pin_set": True})
    return set_session_cookie(response, token)


@router.post("/verify")
async def verify_pin(request: Request):
    """Verify PIN and create a session. Rate-limited per client (fail-closed)."""
    client_id = _client_id(request)
    allowed, retry_after = pin_rate_limit_status(client_id)
    if not allowed:
        response = JSONResponse(
            {"ok": False, "error": "Too many attempts", "retry_after": retry_after},
            status_code=429,
        )
        response.headers["Retry-After"] = str(retry_after)
        return response

    body = await request.json()
    pin = str(body.get("pin", ""))

    if _verify_pin(pin, settings.dashboard_pin_hash):
        reset_pin_failures(client_id)
        token = create_session()
        response = JSONResponse({"ok": True})
        return set_session_cookie(response, token)

    record_pin_failure(client_id)
    return JSONResponse({"ok": False, "error": "Invalid PIN"}, status_code=401)


@router.post("/set-pin")
async def set_pin(request: Request):
    """Set or change the dashboard PIN. The PIN is mandatory (never cleared)."""
    body = await request.json()
    pin = str(body.get("pin", "")).strip()
    current = str(body.get("current_pin", ""))

    if not pin:
        return JSONResponse(
            {"ok": False, "error": "PIN required"},
            status_code=400,
        )

    # If a PIN is already set, the current PIN must be verified first.
    if settings.dashboard_pin_hash:
        if not _verify_pin(current, settings.dashboard_pin_hash):
            return JSONResponse({"ok": False, "error": "Current PIN incorrect"}, status_code=401)

    new_hash = _hash_pin(pin)
    settings.save_runtime_overrides({"dashboard_pin_hash": new_hash})
    settings.dashboard_pin_hash = new_hash

    # Create a session for the user who just set the PIN.
    token = create_session()
    response = JSONResponse({"ok": True, "pin_set": True})
    return set_session_cookie(response, token)


@router.post("/logout")
async def logout(request: Request):
    """Clear session."""
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        delete_session(token)
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
