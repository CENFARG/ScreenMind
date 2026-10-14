"""
Encryption at Rest — MANDATORY and fail-closed (Ola2 / CT-05).

Screenshots are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) using a key
that lives ONLY in the OS keyring (Windows Credential Manager / macOS Keychain
/ Secret Service). The key is never written to disk in clear and there is no
runtime switch to disable encryption.

Fail-closed contract:
    * Keyring unavailable OR crypto failure -> EncryptionUnavailable is raised.
    * The plaintext screenshot is then QUARANTINED: it is removed from disk
      (best-effort shred + unlink) and only a metadata-only audit entry is
      recorded. A capture is NEVER persisted in clear.

Transparent read layer: encrypted files carry a magic header; legacy plaintext
fixtures still pass through on read.

Usage:
    from screenmind.privacy.encryption import encrypt_image, open_image, serve_image

    # Write: save screenshot then encrypt in-place (raises if it cannot encrypt)
    img.save(filepath, "JPEG", quality=80)
    encrypt_image(filepath)

    # Read: transparent — handles both encrypted and unencrypted
    pil_img = open_image(filepath)

    # Serve: returns Response (handles decrypt + streaming)
    return serve_image(filepath)
"""

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from screenmind.config import settings

logger = logging.getLogger("screenmind.privacy.encryption")

# Magic header to identify encrypted files (16 bytes)
MAGIC = b"OPENRECALL_ENC\x01\x00"
MAGIC_LEN = len(MAGIC)

# OS keyring coordinates. The key is stored ONLY here — never on disk.
KEYRING_SERVICE = "screenmind"
KEYRING_USERNAME = "encryption_key"

# Metadata-only audit log for quarantined captures (no image bytes).
QUARANTINE_LOG_NAME = "quarantine.jsonl"


class EncryptionUnavailable(RuntimeError):
    """Raised when at-rest encryption cannot be established (fail-closed).

    Callers MUST drop the capture when this is raised: the plaintext has
    already been quarantined by :func:`encrypt_image`, so nothing is left
    readable in clear.
    """


# ── Key Management (keyring-only, no plaintext fallback) ─────────────

_fernet_instance = None


def _get_keyring():
    """Return the keyring module.

    Raises EncryptionUnavailable when keyring is not importable: there is no
    weak fallback (e.g. a plaintext key file) by design.
    """
    try:
        import keyring
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise EncryptionUnavailable(
            "keyring is not installed; encryption is mandatory and has no "
            "plaintext fallback"
        ) from exc
    return keyring


def _get_or_create_key() -> bytes:
    """Return the Fernet key from the OS keyring, creating it on first use.

    The key never touches disk in clear. Any keyring failure (read, write or
    verification) is fail-closed and raises EncryptionUnavailable.
    """
    keyring = _get_keyring()

    try:
        stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception as exc:
        raise EncryptionUnavailable(f"OS keyring read failed: {exc}") from exc

    if stored:
        return stored.encode()

    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, key)
    except Exception as exc:
        raise EncryptionUnavailable(f"OS keyring write failed: {exc}") from exc

    # Verify the round-trip so we never proceed with an unstored key.
    try:
        if keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) != key:
            raise EncryptionUnavailable("OS keyring verification failed")
    except EncryptionUnavailable:
        raise
    except Exception as exc:
        raise EncryptionUnavailable(f"OS keyring verification failed: {exc}") from exc

    logger.info("Generated and stored encryption key in the OS keyring")
    return key.encode()


def _get_fernet():
    """Return a cached Fernet instance.

    Fail-closed: raises EncryptionUnavailable instead of ever returning None.
    """
    global _fernet_instance
    if _fernet_instance is None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise EncryptionUnavailable(
                "'cryptography' is not installed; encryption is mandatory"
            ) from exc
        _fernet_instance = Fernet(_get_or_create_key())
    return _fernet_instance


# ── Quarantine (fail-closed removal of unencryptable plaintext) ───────

def _best_effort_shred(filepath: Path) -> None:
    """Overwrite the file contents with zeros (best effort, then unlink)."""
    try:
        size = filepath.stat().st_size
        if size <= 0:
            return
        with open(filepath, "r+b") as fh:
            fh.write(b"\x00" * size)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass


def _record_quarantine(filepath: Path, reason: str, size: int) -> None:
    """Append a metadata-only audit entry (never the image bytes)."""
    try:
        log_path = Path(settings.data_path) / QUARANTINE_LOG_NAME
    except Exception:
        return
    record = {
        "ts": time.time(),
        "file": filepath.name,
        "dir": filepath.parent.name,
        "bytes": size,
        "reason": reason,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.error("Could not write quarantine record: %s", exc)


def quarantine_file(filepath: Path, reason: str) -> dict:
    """Destroy an unencryptable plaintext screenshot (fail-closed).

    Removes the capture from disk so it is never persisted in clear, and
    records a metadata-only audit entry describing the event.
    """
    filepath = Path(filepath)
    try:
        size = filepath.stat().st_size
    except OSError:
        size = 0

    _best_effort_shred(filepath)
    try:
        filepath.unlink()
    except OSError as exc:
        logger.error("Quarantine could not remove %s: %s", filepath.name, exc)

    _record_quarantine(filepath, reason, size)
    logger.warning("Quarantined unencryptable capture %s (%s)", filepath.name, reason)
    return {"file": filepath.name, "reason": reason, "bytes": size}


# ── Core Functions ───────────────────────────────────────────────────

def is_encrypted(filepath: Path) -> bool:
    """Check if a file has the encryption magic header."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(MAGIC_LEN)
        return header == MAGIC
    except Exception:
        return False


def encrypt_image(filepath: Path) -> bool:
    """Encrypt a screenshot file in place (MANDATORY, fail-closed).

    Returns True when the file is encrypted (or was already encrypted). On ANY
    failure the plaintext is quarantined and EncryptionUnavailable is raised so
    the caller drops the capture instead of keeping it in clear.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return False

    # Don't double-encrypt
    if is_encrypted(filepath):
        return True

    try:
        fernet = _get_fernet()
    except EncryptionUnavailable:
        quarantine_file(filepath, "keyring_unavailable")
        raise
    except Exception as exc:
        quarantine_file(filepath, f"key_init_failed:{type(exc).__name__}")
        raise EncryptionUnavailable(f"Key initialisation failed: {exc}") from exc

    tmp = filepath.parent / (filepath.name + ".enc.tmp")
    try:
        plaintext = filepath.read_bytes()
        with open(tmp, "wb") as fh:
            fh.write(MAGIC)
            fh.write(fernet.encrypt(plaintext))
        # Atomic replace: the plaintext is gone the moment this succeeds.
        os.replace(str(tmp), str(filepath))
        return True
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        quarantine_file(filepath, f"encrypt_failed:{type(exc).__name__}")
        raise EncryptionUnavailable(f"Failed to encrypt {filepath.name}: {exc}") from exc


def decrypt_image_bytes(filepath: Path) -> Optional[bytes]:
    """Read and decrypt a screenshot file.

    Transparent: if the file is not encrypted, returns raw bytes. Returns None
    when the file is missing or the key is unavailable.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        raw = filepath.read_bytes()
    except OSError as exc:
        logger.error("Failed to read %s: %s", filepath.name, exc)
        return None

    # Not encrypted (legacy plaintext) — return as-is
    if raw[:MAGIC_LEN] != MAGIC:
        return raw

    try:
        fernet = _get_fernet()
    except EncryptionUnavailable as exc:
        logger.error("Cannot decrypt %s: %s", filepath.name, exc)
        return None
    if fernet is None:  # defensive: reached only with a patched/None key
        logger.error("Cannot decrypt %s — no key available", filepath.name)
        return None

    try:
        return fernet.decrypt(raw[MAGIC_LEN:])
    except Exception as exc:
        logger.error("Failed to decrypt %s: %s", filepath.name, exc)
        return None


def open_image(filepath: Path):
    """
    Open an image file as PIL Image. Handles encrypted + unencrypted transparently.
    Drop-in replacement for PIL.Image.open(filepath).
    """
    from PIL import Image

    filepath = Path(filepath)

    if is_encrypted(filepath):
        data = decrypt_image_bytes(filepath)
        if data is None:
            raise IOError(f"Cannot decrypt {filepath}")
        return Image.open(io.BytesIO(data))
    else:
        # Not encrypted (legacy plaintext fixture) — normal PIL open
        return Image.open(filepath)


def serve_image(filepath: Path):
    """
    Create a FastAPI Response for an image file. Handles encrypted + unencrypted.
    Drop-in replacement for FileResponse(filepath).
    """
    from fastapi.responses import FileResponse, Response

    filepath = Path(filepath)

    if is_encrypted(filepath):
        data = decrypt_image_bytes(filepath)
        if data is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Cannot decrypt screenshot")
        return Response(content=data, media_type="image/jpeg")
    else:
        # Not encrypted — use efficient FileResponse (zero-copy sendfile)
        return FileResponse(str(filepath), media_type="image/jpeg")
