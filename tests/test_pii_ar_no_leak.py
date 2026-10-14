"""Ola2-T2.1 — Argentine PII (Ley 25.326) detectors and no-leak guarantees.

Covers:
  * valid AR identifiers are redacted (CUIT/CUIL mod-11, CBU/CVU mod-10,
    bank alias, contextual DNI);
  * invalid identifiers (bad check digit) are left intact;
  * end-to-end: no valid PII reaches SQLite, the FTS5 index, or the logs;
  * fail-closed: when the filter cannot guarantee redaction the capture is
    quarantined locally — never persisted, never sent to inference.
"""

import asyncio
import logging
from datetime import datetime

import pytest

from screenmind.config import settings
from screenmind.privacy import data_filter as df
from screenmind.storage.database import Database
from screenmind.storage.models import ActivityRecord, ScreenshotEntry
from screenmind.workers.capture_worker import CaptureResult


# ── Sample identifiers ───────────────────────────────────────────────

VALID_CUIT = "20111111112"                   # prefix 20, mod-11 check digit 2
VALID_CUIT_HYPHEN = "20-11111111-2"          # same value, hyphenated
INVALID_CUIT = "20111111113"                 # last digit tampered -> bad DV
VALID_CUIL = "23111111111"                   # prefix 23, mod-11 check digit 1
INVALID_CUIL = "23111111112"

VALID_CBU = "0170099220000067190001"         # BCRA mod-10, block1 + block2
INVALID_CBU_DV2 = "0170099220000067190002"   # block2 check digit tampered
INVALID_CBU_DV1 = "1170099220000067190001"   # block1 check digit tampered

VALID_EMAIL = "ana.gomez@example.com.ar"
VALID_DNI = "12345678"
VALID_DNI_DOTTED = "12.345.678"
VALID_ALIAS = "mi.alias.banco"

# Padding long enough (>100 chars) to take the worker's a11y fast-path, and
# free of the chrome markers and filter keywords that would alter the flow.
_PADDING = (
    "This is a benign line of application text used only to lengthen the "
    "sample so the accessibility fast-path is exercised. "
)


def _make_valid_cbu(payload20: str) -> str:
    """Build a CBU/CVU from 20 payload digits (7 block1 + 13 block2)."""
    digits = [int(c) for c in payload20]
    b1, b2 = digits[:7], digits[7:20]
    dv1 = (10 - sum(d * w for d, w in zip(b1, df._CBU_W1)) % 10) % 10
    dv2 = (10 - sum(d * w for d, w in zip(b2, df._CBU_W2)) % 10) % 10
    return f"{payload20[:7]}{dv1}{payload20[7:]}{dv2}"


# ── Valid identifiers are redacted ───────────────────────────────────

def test_cuit_valid_redacted():
    for value in (VALID_CUIT, VALID_CUIT_HYPHEN):
        result = df.filter_sensitive_text(f"CUIT {value} end", ["cuit"])
        assert "[REDACTED:cuit]" in result["clean_text"]
        assert result["redacted_count"] == 1


def test_cuil_valid_redacted():
    result = df.filter_sensitive_text(f"CUIL: {VALID_CUIL}", ["cuit"])
    assert "[REDACTED:cuit]" in result["clean_text"]


def test_cbu_valid_redacted():
    result = df.filter_sensitive_text(f"CBU {VALID_CBU} ok", ["cbu"])
    assert "[REDACTED:cbu]" in result["clean_text"]
    assert VALID_CBU not in result["clean_text"]


def test_cbu_helper_matches_known_sample():
    """Guard against the helper drifting from the real mod-10 rule."""
    assert _make_valid_cbu("01700992000006719000") == VALID_CBU


def test_cvu_valid_redacted():
    """A CVU shares the CBU 22-digit mod-10 layout."""
    cvu = _make_valid_cbu("00000031000100000000")
    assert len(cvu) == 22
    result = df.filter_sensitive_text(f"CVU {cvu}", ["cbu"])
    assert "[REDACTED:cbu]" in result["clean_text"]


def test_dni_with_context_redacted():
    for value in (VALID_DNI, VALID_DNI_DOTTED):
        result = df.filter_sensitive_text(f"DNI {value}", ["dni"])
        assert "[REDACTED:dni]" in result["clean_text"]


def test_alias_redacted():
    result = df.filter_sensitive_text(f"alias {VALID_ALIAS}", ["alias"])
    assert "[REDACTED:alias]" in result["clean_text"]
    assert VALID_ALIAS not in result["clean_text"]


# ── Invalid identifiers are left intact ──────────────────────────────

def test_invalid_cuit_intact():
    result = df.filter_sensitive_text(f"CUIT {INVALID_CUIT}", ["cuit"])
    assert result["redacted_count"] == 0
    assert INVALID_CUIT in result["clean_text"]


def test_invalid_cuil_intact():
    result = df.filter_sensitive_text(f"CUIL {INVALID_CUIL}", ["cuit"])
    assert result["redacted_count"] == 0


def test_invalid_cbu_check_digits_intact():
    for value in (INVALID_CBU_DV2, INVALID_CBU_DV1):
        result = df.filter_sensitive_text(f"CBU {value}", ["cbu"])
        assert result["redacted_count"] == 0
        assert value in result["clean_text"]


def test_dni_without_context_intact():
    text = "order 12345678 shipped today"
    result = df.filter_sensitive_text(text, ["dni"])
    assert result["redacted_count"] == 0
    assert result["clean_text"] == text


def test_valid_and_invalid_neighbours_are_distinguished():
    text = f"CUIT {VALID_CUIT} and CUIT {INVALID_CUIT}"
    result = df.filter_sensitive_text(text, ["cuit"])
    assert VALID_CUIT not in result["clean_text"]
    assert INVALID_CUIT in result["clean_text"]


def test_us_patterns_still_work():
    text = (
        "card 4111-1111-1111-1111 ssn 123-45-6789 "
        "key sk-abcdefghijklmnopqrstuvwxyz1234567890"
    )
    result = df.filter_sensitive_text(text, ["credit_card", "ssn", "api_key"])
    assert result["redacted_count"] == 3


def test_defaults_enable_email_and_ar_types():
    for key in ("email", "cuit", "cbu", "alias", "dni"):
        assert key in df.DEFAULT_ENABLED_TYPES


# ── Fail-closed wrapper ──────────────────────────────────────────────

def test_redact_or_raise_passes_clean_text():
    result = df.redact_or_raise("nothing sensitive here")
    assert result["redacted_count"] == 0


def test_redact_or_raise_fails_closed_when_redactor_is_noop(monkeypatch):
    monkeypatch.setattr(
        df,
        "filter_sensitive_text",
        lambda text, types=None: {
            "clean_text": text,
            "redacted_count": 0,
            "types_found": [],
            "details": [],
        },
    )
    with pytest.raises(df.FilterFailure) as exc:
        df.redact_or_raise(f"DNI {VALID_DNI}", ["dni"])
    # The failure message must never carry the matched value.
    assert VALID_DNI not in str(exc.value)


def test_redact_or_raise_wraps_redactor_exception(monkeypatch):
    def _boom(text, types=None):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(df, "filter_sensitive_text", _boom)
    with pytest.raises(df.FilterFailure):
        df.redact_or_raise("anything", ["dni"])


# ── End-to-end: nothing leaks to SQLite / FTS5 / logs ────────────────

class _FakeAnalyzer:
    """Stand-in for GemmaAnalyzer — returns a fixed, benign analysis."""

    def __init__(self):
        self.calls = 0

    def _result(self, **kwargs):
        self.calls += 1
        return (
            ActivityRecord(
                app_name="TestApp",
                activity_category="browsing",
                activity_summary="Benign summary",
                detailed_context="Benign details",
                scene_description="A window on screen",
                confidence=0.9,
            ),
            [],
        )

    def analyze_screenshot_fast(self, **kwargs):
        return self._result(**kwargs)

    def analyze_screenshot_balanced(self, **kwargs):
        return self._result(**kwargs)

    def analyze_screenshot(self, **kwargs):
        return self._result(**kwargs)


class _FakeOCR:
    is_available = False


class _FakeDevContext:
    def is_coding_activity(self, **kwargs):
        return False


def _make_worker(db):
    from screenmind.workers.analysis_worker import AnalysisWorker

    worker = AnalysisWorker(asyncio.Queue(), db)
    analyzer = _FakeAnalyzer()
    worker._analyzer = analyzer
    worker._ocr = _FakeOCR()
    worker._dev_context = _FakeDevContext()
    return worker, analyzer


def _make_capture(db, tmp_path, text):
    activity_id = db.insert_activity(
        ScreenshotEntry(
            timestamp=datetime.now(),
            screenshot_path=str(tmp_path / "shot.png"),
            window_title="Benign Window",
            detected_app_name="TestApp",
            bookmarked=False,
            analyzed=False,
        )
    )
    capture = CaptureResult(
        filepath=None,
        timestamp=datetime.now(),
        window_title="Benign Window",
        app_name="TestApp",
        bookmarked=False,
        image=None,
        activity_id=activity_id,
        a11y_text=text,
        phash=None,
    )
    return capture, activity_id


def _all_text_for(db, activity_id) -> str:
    """Every text field for a row, from both the table and the FTS5 index."""
    conn = db._get_conn()
    cols = "ocr_text, summary, details, organized_text"
    row = conn.execute(
        f"SELECT {cols} FROM activities WHERE id = ?", (activity_id,)
    ).fetchone()
    fts = conn.execute(
        f"SELECT {cols} FROM activities_fts WHERE rowid = ?", (activity_id,)
    ).fetchone()
    values = list(row) if row else []
    values += list(fts) if fts else []
    return " ".join(str(v) for v in values if v)


@pytest.mark.asyncio
async def test_e2e_valid_pii_never_reaches_db_fts_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "sensitive_filter_enabled", True)
    monkeypatch.setattr(
        settings,
        "sensitive_filter_types",
        "credit_card,ssn,api_key,jwt,password,email,cuit,cbu,alias,dni",
    )
    db = Database(db_path=tmp_path / "test.db")
    try:
        worker, _analyzer = _make_worker(db)
        pii_text = (
            f"{_PADDING}CUIT {VALID_CUIT}, CBU {VALID_CBU}, "
            f"email {VALID_EMAIL}, DNI {VALID_DNI}."
        )
        capture, activity_id = _make_capture(db, tmp_path, pii_text)

        with caplog.at_level(logging.DEBUG, logger="screenmind"):
            await worker._process(capture)

        stored = _all_text_for(db, activity_id)
        for needle in (VALID_CUIT, VALID_CBU, VALID_EMAIL, VALID_DNI):
            assert needle not in stored
            assert needle not in caplog.text
        assert "[REDACTED:" in stored

        status = db._get_conn().execute(
            "SELECT status FROM activities WHERE id = ?", (activity_id,)
        ).fetchone()[0]
        assert status == "ok"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_e2e_filter_failure_quarantines_no_inference_no_persist(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "sensitive_filter_enabled", True)
    monkeypatch.setattr(settings, "sensitive_filter_types", "dni,cuit")
    db = Database(db_path=tmp_path / "test.db")
    try:
        worker, analyzer = _make_worker(db)
        pii_text = f"{_PADDING}CUIT {VALID_CUIT} and DNI {VALID_DNI}."

        def _boom(text, types=None):
            raise df.FilterFailure("simulated redactor failure")

        monkeypatch.setattr(df, "redact_or_raise", _boom)

        capture, activity_id = _make_capture(db, tmp_path, pii_text)
        with caplog.at_level(logging.DEBUG, logger="screenmind"):
            await worker._process(capture)

        # Fail-closed: inference never ran.
        assert analyzer.calls == 0

        row = db._get_conn().execute(
            "SELECT ocr_text, status FROM activities WHERE id = ?", (activity_id,)
        ).fetchone()
        assert row["ocr_text"] is None
        assert row["status"] == "quarantined"
        assert worker.stats["quarantined"] == 1

        # Local quarantine artifact exists for human review.
        assert (tmp_path / "quarantine" / f"activity_{activity_id}.json").exists()

        # And no PII anywhere, in storage or logs.
        for needle in (VALID_CUIT, VALID_DNI):
            assert needle not in _all_text_for(db, activity_id)
            assert needle not in caplog.text
    finally:
        db.close()
