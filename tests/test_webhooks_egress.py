"""Ola2-T2.5 — fail-closed webhook egress lockdown.

Covers:
  * deny-by-default: an empty/absent allowlist blocks every destination;
  * allowlist: only explicitly listed URLs may ever be emitted to;
  * TLS mandatory: ``http://`` is rejected except to loopback;
  * a non-listed destination is rejected WITHOUT emitting any bytes;
  * settings.json cannot widen the allowlist nor enable an off-list target.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from screenmind.config import Settings, _ALLOWED_OVERRIDES, settings
from screenmind.integrations import webhooks
from screenmind.integrations.webhooks import (
    _is_tls_ok,
    _normalize_url,
    _parse_allowlist,
    _send,
    fire_all,
    is_egress_allowed,
    send_to_hook,
    test_webhook as webhook_ping,
)

ALLOWED_HTTPS = "https://hooks.example.com/ingest"
OTHER_HTTPS = "https://attacker.example.net/exfil"
LOOPBACK_HTTP = "http://127.0.0.1:7777/hook"


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def no_bytes():
    """Spy on the socket layer — any call would mean bytes left the process."""
    with patch("screenmind.integrations.webhooks.urllib.request.urlopen") as mock_urlopen:
        yield mock_urlopen


@pytest.fixture
def silent_log(monkeypatch):
    """Capture delivery-log writes so tests stay hermetic and can assert blocks."""
    spy = MagicMock()
    monkeypatch.setattr(webhooks, "_log_delivery", spy)
    return spy


@pytest.fixture
def allowlist(monkeypatch):
    """Setter for the env-only allowlist on the live settings singleton."""
    def _set(value):
        monkeypatch.setattr(settings, "webhook_egress_allowlist", value)
    return _set


def _ok_response(mock_urlopen, status: int = 200):
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = resp
    return resp


# ── Defaults & pure helpers ──────────────────────────────────────────

def test_defaults_deny_webhooks_and_egress():
    """Webhooks off, allowlist empty — the shipped defaults deny everything."""
    assert Settings.model_fields["webhook_enabled"].default is False
    assert Settings.model_fields["webhook_egress_allowlist"].default == ""


def test_allowlist_is_not_runtime_overridable():
    """The lockdown must be operator-only (env), never settings.json."""
    assert "webhook_egress_allowlist" not in _ALLOWED_OVERRIDES


@pytest.mark.parametrize("raw,expected", [
    ("", []),
    ("   ", []),
    (None, []),
    (f"{ALLOWED_HTTPS}, {OTHER_HTTPS}", [ALLOWED_HTTPS, OTHER_HTTPS]),
    (f"{ALLOWED_HTTPS}/,{OTHER_HTTPS},, ", [ALLOWED_HTTPS, OTHER_HTTPS]),
])
def test_parse_allowlist(raw, expected):
    assert _parse_allowlist(raw) == expected


def test_parse_allowlist_ignores_non_string():
    """A non-string allowlist (e.g. an unconfigured mock) must deny, not crash."""
    assert _parse_allowlist(MagicMock()) == []


def test_normalize_url_trims_single_trailing_slash():
    assert _normalize_url("  https://a.example.com/x/  ") == "https://a.example.com/x"
    assert _normalize_url("https://a.example.com/x//") == "https://a.example.com/x/"


@pytest.mark.parametrize("url", [
    "https://hooks.example.com/ingest",
    "http://127.0.0.1:7777/hook",
    "http://localhost:1234/x",
    "http://[::1]:80/x",
])
def test_tls_ok_accepts_https_and_loopback(url):
    assert _is_tls_ok(url) is True


@pytest.mark.parametrize("url", [
    "http://hooks.example.com/ingest",   # plaintext to a remote host
    "http://10.0.0.5/x",                 # private but not loopback
    "http://169.254.169.254/latest",     # cloud metadata
    "ftp://example.com/x",               # non-HTTP scheme
    "not-a-url",
    "",
])
def test_tls_ok_rejects_non_tls(url):
    assert _is_tls_ok(url) is False


# ── The gate ─────────────────────────────────────────────────────────

def test_empty_allowlist_denies_everything(allowlist):
    allowlist("")
    assert is_egress_allowed(ALLOWED_HTTPS) is False
    assert is_egress_allowed(LOOPBACK_HTTP) is False


def test_non_string_allowlist_denies(monkeypatch):
    monkeypatch.setattr(webhooks, "settings", MagicMock())
    assert is_egress_allowed(ALLOWED_HTTPS) is False


def test_empty_url_denied(allowlist):
    allowlist(ALLOWED_HTTPS)
    assert is_egress_allowed("") is False
    assert is_egress_allowed("   ") is False


def test_listed_https_is_allowed(allowlist):
    allowlist(ALLOWED_HTTPS)
    assert is_egress_allowed(ALLOWED_HTTPS) is True


def test_trailing_slash_is_normalized(allowlist):
    allowlist(f"{ALLOWED_HTTPS}/")
    assert is_egress_allowed(ALLOWED_HTTPS) is True


# ── No bytes leave for a non-listed destination (core guarantee) ─────

def test_unlisted_destination_rejected_without_emitting_bytes(allowlist, no_bytes, silent_log):
    allowlist(ALLOWED_HTTPS)  # OTHER_HTTPS is deliberately absent
    ok = _send(OTHER_HTTPS, {"event": "daily_summary"}, "", {}, "daily_summary")
    assert ok is False
    no_bytes.assert_not_called()        # nothing was written to the socket
    silent_log.assert_called()          # the block was recorded


def test_unlisted_destination_blocked_even_when_webhooks_enabled(
    allowlist, no_bytes, silent_log
):
    """Enabling webhooks is necessary but never sufficient."""
    allowlist(ALLOWED_HTTPS)
    assert _send(OTHER_HTTPS, {"event": "x"}, "", {}, "x") is False
    no_bytes.assert_not_called()


def test_plaintext_http_non_loopback_rejected_even_if_listed(allowlist, no_bytes, silent_log):
    url = "http://hooks.example.com/ingest"
    allowlist(url)  # listed, but the transport is not TLS
    assert is_egress_allowed(url) is False
    assert _send(url, {}, "", {}, "ev") is False
    no_bytes.assert_not_called()


# ── Permitted destinations do emit ───────────────────────────────────

def test_listed_https_destination_emits(allowlist, no_bytes, silent_log):
    allowlist(ALLOWED_HTTPS)
    _ok_response(no_bytes)
    assert _send(ALLOWED_HTTPS, {"event": "x"}, "", {}, "x") is True
    no_bytes.assert_called_once()


def test_loopback_http_destination_emits_when_listed(allowlist, no_bytes, silent_log):
    allowlist(LOOPBACK_HTTP)
    _ok_response(no_bytes)
    assert is_egress_allowed(LOOPBACK_HTTP) is True
    assert _send(LOOPBACK_HTTP, {}, "", {}, "ev") is True
    no_bytes.assert_called_once()


def test_send_via_retry_wrapper_blocks_without_bytes(allowlist, no_bytes, silent_log):
    """The retry wrapper must also refuse, without a pointless 5s retry loop."""
    allowlist(ALLOWED_HTTPS)
    webhooks._send_with_retry_named(OTHER_HTTPS, {"event": "x"}, "", {}, "x", "default")
    no_bytes.assert_not_called()
    silent_log.assert_called()


# ── test_webhook honours the gate ────────────────────────────────────

def test_test_webhook_blocked_without_bytes(allowlist, no_bytes, silent_log):
    allowlist("")
    result = webhook_ping(OTHER_HTTPS)
    assert result["ok"] is False
    no_bytes.assert_not_called()


def test_test_webhook_allowed_when_listed(allowlist, no_bytes, silent_log):
    allowlist(ALLOWED_HTTPS)
    _ok_response(no_bytes)
    result = webhook_ping(ALLOWED_HTTPS)
    assert result["ok"] is True
    no_bytes.assert_called_once()


# ── settings.json cannot widen the lockdown ──────────────────────────

def test_settings_json_cannot_set_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "webhook_egress_allowlist", "")

    settings.save_runtime_overrides({"webhook_egress_allowlist": ALLOWED_HTTPS})
    assert settings.webhook_egress_allowlist == ""

    (tmp_path / "settings.json").write_text(
        json.dumps({"webhook_egress_allowlist": ALLOWED_HTTPS})
    )
    settings.load_runtime_overrides()
    assert settings.webhook_egress_allowlist == ""
    assert is_egress_allowed(ALLOWED_HTTPS) is False


def test_settings_json_cannot_enable_offlist_target(tmp_path, monkeypatch, no_bytes, silent_log):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "webhook_egress_allowlist", "")
    # Snapshot runtime-overridable fields so teardown restores them.
    for name in ("webhook_enabled", "webhook_url", "webhook_events"):
        monkeypatch.setattr(settings, name, getattr(settings, name))

    # A hostile settings.json tries to switch webhooks on AND widen the allowlist.
    (tmp_path / "settings.json").write_text(json.dumps({
        "webhook_enabled": True,
        "webhook_url": OTHER_HTTPS,
        "webhook_events": "",
        "webhook_egress_allowlist": OTHER_HTTPS,
    }))
    settings.load_runtime_overrides()

    # The enable flags applied, but the allowlist did NOT (not overridable)...
    assert settings.webhook_enabled is True
    assert settings.webhook_url == OTHER_HTTPS
    assert settings.webhook_egress_allowlist == ""

    # ...so an event fires with zero bytes reaching the off-list target.
    with patch.object(webhooks, "_pool") as pool:
        pool.submit.side_effect = lambda fn, *a, **k: fn(*a, **k)  # run inline
        fire_all("daily_summary", {"summary": "secret"})

    no_bytes.assert_not_called()
    silent_log.assert_called()


def test_send_to_hook_blocks_offlist_without_bytes(tmp_path, monkeypatch, no_bytes, silent_log):
    monkeypatch.setattr(settings, "webhook_egress_allowlist", "")
    monkeypatch.setattr(settings, "webhook_enabled", True)
    monkeypatch.setattr(settings, "webhook_url", OTHER_HTTPS)
    monkeypatch.setattr(settings, "webhook_events", "")

    with patch.object(webhooks, "_pool") as pool:
        pool.submit.side_effect = lambda fn, *a, **k: fn(*a, **k)
        queued = send_to_hook("default", "agent", "output")

    assert queued is True          # queued for delivery...
    no_bytes.assert_not_called()   # ...but the gate dropped it before any byte
    silent_log.assert_called()
