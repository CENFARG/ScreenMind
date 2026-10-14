"""Ola2 auth tests (CT-06.1–06.4).

Covers the mandatory fail-closed PIN contract:
  * an empty / unset PIN locks protected routes and never unlocks them;
  * the PIN is stored as a salted PBKDF2-SHA256 hash and verified;
  * the session cookie is HttpOnly + Secure + SameSite;
  * PIN attempts are rate-limited (fail-closed lockout).
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import screenmind.api.dependencies as deps
from screenmind.config import settings
from screenmind.api.routes.auth import _hash_pin, _verify_pin
from screenmind.api.server import create_app


def _make_app():
    """Build an app with a mocked database (no disk writes)."""
    db = MagicMock()
    db.get_stats.return_value = {"total_activities": 0}
    app = create_app(
        database=db,
        capture_worker=MagicMock(),
        analysis_worker=MagicMock(),
        audio_worker=MagicMock(),
    )
    # Route modules snapshot `db` at import time; rebind for this app.
    import screenmind.api.routes.stats as stats_mod

    stats_mod.db = db
    return app


@pytest.fixture(autouse=True)
def _isolate_auth_state(monkeypatch, tmp_path):
    """Hermetic auth state: no disk writes, clean sessions and rate limiter."""

    def _in_memory_save(self, updates):
        # Apply runtime overrides in memory only — no settings.json writes.
        for key, value in updates.items():
            setattr(self, key, value)

    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(type(settings), "save_runtime_overrides", _in_memory_save)

    deps._sessions.clear()
    deps._pin_failures.clear()
    original_pin = settings.dashboard_pin_hash
    settings.dashboard_pin_hash = ""
    try:
        yield
    finally:
        settings.dashboard_pin_hash = original_pin
        deps._sessions.clear()
        deps._pin_failures.clear()


# ── CT-06.1: mandatory PIN, empty PIN is fail-closed ─────────────────

def test_empty_pin_locks_protected_routes():
    client = TestClient(_make_app())
    resp = client.get("/api/stats")
    assert resp.status_code == 401
    assert resp.json()["locked"] is True


def test_empty_pin_does_not_unlock_even_with_token():
    token = deps.create_session()
    client = TestClient(_make_app())
    client.headers["Cookie"] = f"{deps.SESSION_COOKIE}={token}"
    resp = client.get("/api/stats")
    assert resp.status_code == 401


def test_auth_status_fail_closed_without_pin():
    client = TestClient(_make_app())
    resp = client.get("/api/auth/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_pin"] is False
    assert data["authenticated"] is False


def test_verify_with_empty_pin_is_rejected():
    client = TestClient(_make_app())
    resp = client.post("/api/auth/verify", json={"pin": ""})
    assert resp.status_code == 401
    assert deps.SESSION_COOKIE not in resp.cookies


def test_setup_complete_requires_a_non_empty_pin():
    client = TestClient(_make_app())
    resp = client.post("/api/auth/setup-complete", json={"pin": ""})
    assert resp.status_code == 400
    assert settings.dashboard_pin_hash == ""


def test_set_pin_requires_a_non_empty_pin():
    client = TestClient(_make_app())
    resp = client.post("/api/auth/set-pin", json={"pin": ""})
    assert resp.status_code == 400


# ── CT-06.2: PIN stored as a salted PBKDF2-SHA256 hash ───────────────

def test_hash_is_salted_pbkdf2_and_verifies():
    stored = _hash_pin("1234")
    assert "$" in stored
    salt, digest = stored.split("$")
    assert len(salt) == 32  # 16 random bytes, hex-encoded
    assert len(digest) == 64  # PBKDF2-SHA256 output, hex-encoded
    assert _verify_pin("1234", stored) is True
    assert _verify_pin("12345", stored) is False
    # Empty PIN / empty hash never verify (fail-closed).
    assert _verify_pin("", stored) is False
    assert _verify_pin("1234", "") is False
    # Salting: the same PIN never produces the same hash twice.
    assert _hash_pin("1234") != stored


# ── CT-06.3: hardened session cookie ─────────────────────────────────

def test_correct_pin_sets_hardened_cookie_and_valid_session():
    settings.dashboard_pin_hash = _hash_pin("1234")
    app = _make_app()
    client = TestClient(app)
    resp = client.post("/api/auth/verify", json={"pin": "1234"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    set_cookie = resp.headers.get("set-cookie", "").lower()
    assert f"{deps.SESSION_COOKIE}=" in set_cookie
    assert "httponly" in set_cookie
    assert "secure" in set_cookie
    assert "samesite=lax" in set_cookie
    assert "path=/" in set_cookie

    # The minted token is a real, usable session.
    token = resp.headers["set-cookie"].split(f"{deps.SESSION_COOKIE}=", 1)[1].split(";", 1)[0]
    assert deps.verify_session(token) is True
    authed = TestClient(app)
    authed.headers["Cookie"] = f"{deps.SESSION_COOKIE}={token}"
    assert authed.get("/api/stats").status_code == 200


def test_wrong_pin_is_rejected_without_cookie():
    settings.dashboard_pin_hash = _hash_pin("1234")
    client = TestClient(_make_app())
    resp = client.post("/api/auth/verify", json={"pin": "0000"})
    assert resp.status_code == 401
    assert deps.SESSION_COOKIE not in resp.cookies


# ── CT-06.4: PIN brute-force rate limiting ───────────────────────────

def test_rate_limit_blocks_correct_pin_while_locked(monkeypatch):
    settings.dashboard_pin_hash = _hash_pin("1234")
    monkeypatch.setattr(settings, "dashboard_pin_max_attempts", 3)
    monkeypatch.setattr(settings, "dashboard_pin_lockout_seconds", 300)

    client = TestClient(_make_app())
    for _ in range(3):
        resp = client.post("/api/auth/verify", json={"pin": "0000"})
        assert resp.status_code == 401

    # Fail-closed: even the CORRECT PIN is throttled once the window is full.
    resp = client.post("/api/auth/verify", json={"pin": "1234"})
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_rate_limit_resets_after_successful_verify(monkeypatch):
    settings.dashboard_pin_hash = _hash_pin("1234")
    monkeypatch.setattr(settings, "dashboard_pin_max_attempts", 3)

    client = TestClient(_make_app())
    for _ in range(2):
        client.post("/api/auth/verify", json={"pin": "0000"})

    resp = client.post("/api/auth/verify", json={"pin": "1234"})
    assert resp.status_code == 200
    # A successful entry clears the failure window for this client.
    assert not deps._pin_failures.get("testclient")
