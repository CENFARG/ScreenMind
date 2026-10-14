"""Test API endpoints using FastAPI TestClient."""
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def client():
    """Create an authenticated test client with mocked dependencies.

    The PIN is mandatory, so we configure a real PIN and mint a valid session
    instead of disabling auth (no production bypass).
    """
    import screenmind.api.dependencies as deps
    from screenmind.api.routes.auth import _hash_pin
    from screenmind.api.server import create_app
    from screenmind.storage.database import Database
    from screenmind.config import settings
    from fastapi.testclient import TestClient

    db = MagicMock(spec=Database)
    db.get_activities_by_date.return_value = []
    db.get_bookmarks.return_value = []
    db.get_stats.return_value = {"total_activities": 0, "category_breakdown": {}, "top_apps": {}, "top_repos": {}, "meetings_count": 0, "meetings_minutes": 0}
    db.get_daily_summary.return_value = None
    db.get_rewind_data.return_value = []

    original_pin = settings.dashboard_pin_hash
    settings.dashboard_pin_hash = _hash_pin("test-pin")
    session_token = deps.create_session()

    app = create_app(database=db, capture_worker=MagicMock(), analysis_worker=MagicMock(), audio_worker=MagicMock())
    test_client = TestClient(app)
    test_client.headers["Cookie"] = f"{deps.SESSION_COOKIE}={session_token}"
    try:
        yield test_client
    finally:
        deps.delete_session(session_token)
        settings.dashboard_pin_hash = original_pin


def test_root_returns_html(client):
    """Dashboard HTML loads."""
    r = client.get("/")
    assert r.status_code == 200
    assert "ScreenMind" in r.text


def test_status_endpoint(client):
    """Status endpoint returns valid JSON."""
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


def test_auth_status(client):
    """Auth status returns first_run and has_pin fields."""
    r = client.get("/api/auth/status")
    assert r.status_code == 200
    data = r.json()
    assert "has_pin" in data
    assert "authenticated" in data


def test_timeline_returns_list(client):
    """Timeline endpoint returns activities list."""
    r = client.get("/api/timeline?date=2026-01-01")
    assert r.status_code == 200
    data = r.json()
    assert "activities" in data


def test_search_requires_query(client):
    """Search endpoint requires q parameter."""
    r = client.get("/api/search")
    assert r.status_code == 422  # validation error
