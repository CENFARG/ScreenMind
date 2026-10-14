"""
Shared state and helpers for API route modules.
All route files import from here instead of accessing globals.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from screenmind.config import settings
from screenmind.engine.embedder import Embedder
from screenmind.storage.database import Database


# ── Session store (in-memory) ────────────────────────────────────────
_sessions = {}  # token -> expiry datetime


def verify_session(token: str) -> bool:
    """Check if a session token is valid and not expired."""
    if not token or token not in _sessions:
        return False
    if datetime.utcnow() > _sessions[token]:
        del _sessions[token]
        return False
    return True


def create_session() -> str:
    """Create a new session token."""
    token = secrets.token_hex(32)
    timeout = settings.dashboard_lock_timeout or 30
    _sessions[token] = datetime.utcnow() + timedelta(minutes=timeout)
    return token


def delete_session(token: str):
    """Remove a session token."""
    _sessions.pop(token, None)


# ── Session cookie ───────────────────────────────────────────────────
SESSION_COOKIE = "screenmind_session"


def set_session_cookie(response, token: str):
    """Attach the session cookie with hardened flags.

    HttpOnly  — not readable from JavaScript (blocks XSS token theft).
    Secure    — only sent over TLS; browsers still treat http://localhost and
                http://127.0.0.1 as secure contexts, so local use is unaffected.
    SameSite  — "lax" allows top-level navigation but blocks cross-site POSTs.
    """
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=(settings.dashboard_lock_timeout or 30) * 60,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return response


# ── PIN brute-force rate limiter (in-memory, per client) ─────────────
# Sliding-window counter of recent PIN attempts, keyed by client IP.
# Fail-closed: an attempt RESERVES its slot before it is evaluated, so a burst
# of concurrent requests cannot slip past the check, and once the window is
# full every further attempt (even a correct PIN) is rejected until it ages out.
_pin_failures = {}  # client_id -> list[datetime utc] of recent attempts
_LIMITER_SWEEP_THRESHOLD = 256


def _pruned(client_id: str, now: datetime, window: int):
    return [
        t
        for t in _pin_failures.get(client_id, [])
        if (now - t).total_seconds() < window
    ]


def reserve_pin_attempt(client_id: str):
    """Atomically check the window and reserve a slot for one PIN attempt.

    Returns (allowed, retry_after_seconds). Fail-closed: the slot is recorded
    up-front, so the caller MUST call reset_pin_failures() when the PIN is
    correct. There is no await between the check and the reservation, so
    concurrent attempts from one client cannot over-spend the budget.
    """
    max_attempts = settings.dashboard_pin_max_attempts or 5
    window = settings.dashboard_pin_lockout_seconds or 60
    now = datetime.utcnow()

    # Opportunistic cleanup: drop clients whose window has fully aged out.
    if len(_pin_failures) > _LIMITER_SWEEP_THRESHOLD:
        for cid in list(_pin_failures):
            if cid != client_id and not _pruned(cid, now, window):
                _pin_failures.pop(cid, None)

    failures = _pruned(client_id, now, window)
    if len(failures) >= max_attempts:
        _pin_failures[client_id] = failures
        oldest = min(failures)
        retry_after = int(window - (now - oldest).total_seconds()) + 1
        return False, max(retry_after, 1)

    failures.append(now)
    _pin_failures[client_id] = failures
    return True, 0


def reset_pin_failures(client_id: str):
    """Clear the failure window after a successful PIN entry."""
    _pin_failures.pop(client_id, None)


# ── Shared instances (set by create_app) ─────────────────────────────
db: Optional[Database] = None
embedder: Optional[Embedder] = None
capture_worker = None
analysis_worker = None
audio_worker = None




def init(database: Database, emb: Optional[Embedder], cap_worker, ana_worker, aud_worker):
    """Initialize shared state. Called once from create_app."""
    global db, embedder, capture_worker, analysis_worker, audio_worker
    db = database
    embedder = emb
    capture_worker = cap_worker
    analysis_worker = ana_worker
    audio_worker = aud_worker
