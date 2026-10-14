"""Tests for config.py — settings parsing and properties."""

import logging

import pytest
from pydantic import ValidationError

from screenmind.config import Settings


def test_default_settings():
    s = Settings(data_dir="/tmp/screenmind_test")
    assert s.capture_interval == 40
    assert s.screenshot_quality == 70
    assert s.ollama_model == "gemma4:e2b"
    assert s.api_port == 7777


def test_blocked_apps_list_empty():
    s = Settings(data_dir="/tmp/test", blocked_apps="")
    assert s.blocked_apps_list == []


def test_blocked_apps_list_parsing():
    s = Settings(data_dir="/tmp/test", blocked_apps="1password, banking, keychain")
    assert s.blocked_apps_list == ["1password", "banking", "keychain"]


def test_workspace_dirs_list():
    s = Settings(data_dir="/tmp/test", workspace_dirs="~/Projects, ~/Code")
    dirs = s.workspace_dirs_list
    assert len(dirs) == 2


def test_heavy_apps_list():
    s = Settings(data_dir="/tmp/test", heavy_apps="game,valorant,blender")
    assert "game" in s.heavy_apps_list
    assert "valorant" in s.heavy_apps_list
    assert len(s.heavy_apps_list) == 3


def test_meeting_apps_list():
    s = Settings(data_dir="/tmp/test", meeting_apps="zoom,teams,meet")
    assert "zoom" in s.meeting_apps_list
    assert len(s.meeting_apps_list) == 3


def test_num_gpu_layers():
    s = Settings(data_dir="/tmp/test", performance_mode="minimal")
    assert s.num_gpu_layers == 0

    s = Settings(data_dir="/tmp/test", performance_mode="balanced")
    assert s.num_gpu_layers == 15

    s = Settings(data_dir="/tmp/test", performance_mode="maximum")
    assert s.num_gpu_layers == 99


def test_data_path_resolution():
    s = Settings(data_dir="~/.screenmind")
    assert s.data_path.is_absolute()
    assert "~" not in str(s.data_path)


# ── Vestigial config cleanup (PR-14): local-only, fail-closed ────────────────


def test_gemma_mode_defaults_to_local():
    assert Settings(data_dir="/tmp/test").gemma_mode == "local"


def test_gemma_mode_api_kwarg_is_blocked(caplog):
    """Constructing with 'api' collapses to 'local' and warns (no egress path)."""
    with caplog.at_level(logging.WARNING, logger="screenmind.config"):
        s = Settings(data_dir="/tmp/test", gemma_mode="api")
    assert s.gemma_mode == "local"
    assert any("local-only" in r.getMessage() for r in caplog.records)


def test_gemma_mode_api_env_is_blocked(monkeypatch, caplog):
    """A GEMMA_MODE=api in the environment is ignored at load time."""
    monkeypatch.setenv("GEMMA_MODE", "api")
    with caplog.at_level(logging.WARNING, logger="screenmind.config"):
        s = Settings(data_dir="/tmp/test")
    assert s.gemma_mode == "local"
    assert any("local-only" in r.getMessage() for r in caplog.records)


def test_gemma_mode_assignment_is_locked():
    """validate_assignment stays fail-closed: later writes can't enable api."""
    s = Settings(data_dir="/tmp/test")
    s.gemma_mode = "api"
    assert s.gemma_mode == "local"


def test_google_api_key_field_removed():
    """The vestigial Google API key field no longer exists (no secret sink)."""
    assert "google_api_key" not in Settings.model_fields
    with pytest.raises(ValidationError):
        Settings(data_dir="/tmp/test", google_api_key="leak")


# ── Encryption mandatory (Ola2 / CT-05): fail-closed, no runtime disable ─────


def test_encryption_enabled_default_true():
    assert Settings(data_dir="/tmp/test").encryption_enabled is True


def test_encryption_enabled_not_runtime_overridable():
    """settings.json / dashboard cannot turn encryption off."""
    from screenmind.config import _ALLOWED_OVERRIDES

    assert "encryption_enabled" not in _ALLOWED_OVERRIDES


def test_encryption_enabled_assignment_is_locked(caplog):
    """validate_assignment stays fail-closed: later writes can't disable it."""
    s = Settings(data_dir="/tmp/test")
    with caplog.at_level(logging.WARNING, logger="screenmind.config"):
        s.encryption_enabled = False
    assert s.encryption_enabled is True
    assert any("mandatory" in r.getMessage() for r in caplog.records)


def test_encryption_enabled_env_cannot_disable(monkeypatch, caplog):
    """ENCRYPTION_ENABLED=false in the environment is ignored (fail-closed)."""
    monkeypatch.setenv("ENCRYPTION_ENABLED", "false")
    with caplog.at_level(logging.WARNING, logger="screenmind.config"):
        s = Settings(data_dir="/tmp/test")
    assert s.encryption_enabled is True
    assert any("mandatory" in r.getMessage() for r in caplog.records)
