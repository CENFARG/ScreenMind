"""Comprehensive tests for encryption at rest (Ola2 / CT-05, fail-closed)."""
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

import screenmind.privacy.encryption as enc
from screenmind.privacy.encryption import (
    is_encrypted, encrypt_image, decrypt_image_bytes,
    open_image, serve_image, MAGIC, MAGIC_LEN,
    EncryptionUnavailable, KEYRING_SERVICE, KEYRING_USERNAME,
)


@pytest.fixture(autouse=True)
def _reset_fernet_cache():
    """Isolate the module-level Fernet cache between tests."""
    enc._fernet_instance = None
    yield
    enc._fernet_instance = None


@pytest.fixture
def fake_keyring(monkeypatch):
    """In-memory keyring stand-in; returns the backing store dict."""
    store = {}

    class _FakeKeyring:
        def get_password(self, service, username):
            return store.get((service, username))

        def set_password(self, service, username, value):
            store[(service, username)] = value

    monkeypatch.setattr(enc, "_get_keyring", lambda: _FakeKeyring())
    return store


@pytest.fixture
def tmp_data(monkeypatch, tmp_path):
    """Point encryption's settings.data_path at a temp dir (real filesystem)."""
    monkeypatch.setattr(enc, "settings", SimpleNamespace(data_path=tmp_path))
    return tmp_path


class TestIsEncrypted:
    """Tests for encrypted file detection."""

    def test_normal_jpeg_not_encrypted(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        assert is_encrypted(f) is False

    def test_magic_header_detected(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(MAGIC + b"\x00" * 100)
        assert is_encrypted(f) is True

    def test_nonexistent_file(self):
        assert is_encrypted(Path("/nonexistent/file.jpg")) is False

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jpg"
        f.write_bytes(b"")
        assert is_encrypted(f) is False

    def test_short_file(self, tmp_path):
        """File shorter than magic header length."""
        f = tmp_path / "short.jpg"
        f.write_bytes(b"short")
        assert is_encrypted(f) is False

    def test_partial_magic_not_detected(self, tmp_path):
        """Partial magic header is not a match."""
        f = tmp_path / "partial.jpg"
        f.write_bytes(MAGIC[:8] + b"\x00" * 100)
        assert is_encrypted(f) is False


class TestEncryptDecrypt:
    """Tests for encrypt/decrypt roundtrip."""

    def test_encrypt_active_by_default(self, tmp_path, fake_keyring, tmp_data):
        """CT-05.1: encryption is active by default (no opt-in required)."""
        f = tmp_path / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"image data here" * 20)
        assert encrypt_image(f) is True
        assert is_encrypted(f) is True

    @patch("screenmind.privacy.encryption._get_fernet")
    @patch("screenmind.privacy.encryption.settings")
    def test_encrypt_then_decrypt_roundtrip(self, mock_settings, mock_fernet, tmp_path):
        """Encrypted file decrypts back to original."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        real_fernet = Fernet(key)
        mock_fernet.return_value = real_fernet
        mock_settings.encryption_enabled = True

        f = tmp_path / "test.jpg"
        original = b"\xff\xd8\xff\xe0" + b"image data here" * 20
        f.write_bytes(original)

        result = encrypt_image(f)
        assert result is True
        assert is_encrypted(f) is True
        assert f.read_bytes() != original  # file changed

        decrypted = decrypt_image_bytes(f)
        assert decrypted == original

    @patch("screenmind.privacy.encryption._get_fernet")
    @patch("screenmind.privacy.encryption.settings")
    def test_double_encrypt_is_noop(self, mock_settings, mock_fernet, tmp_path):
        """Encrypting an already-encrypted file doesn't double-encrypt."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        mock_fernet.return_value = Fernet(key)
        mock_settings.encryption_enabled = True

        f = tmp_path / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"data" * 10)

        encrypt_image(f)
        size_after_first = f.stat().st_size
        encrypt_image(f)
        size_after_second = f.stat().st_size
        assert size_after_first == size_after_second

    def test_decrypt_unencrypted_returns_raw(self, tmp_path):
        """Decrypting an unencrypted file returns raw bytes (passthrough)."""
        f = tmp_path / "test.jpg"
        data = b"\xff\xd8\xff\xe0plaintext"
        f.write_bytes(data)
        assert decrypt_image_bytes(f) == data

    def test_decrypt_nonexistent_returns_none(self):
        """Decrypting non-existent file returns None."""
        assert decrypt_image_bytes(Path("/no/such/file.jpg")) is None

    @patch("screenmind.privacy.encryption._get_fernet")
    def test_decrypt_with_no_key_returns_none(self, mock_fernet, tmp_path):
        """Encrypted file with unavailable key returns None."""
        mock_fernet.return_value = None
        f = tmp_path / "test.jpg"
        f.write_bytes(MAGIC + b"encrypted stuff")
        assert decrypt_image_bytes(f) is None


class TestOpenImage:
    """Tests for open_image() PIL wrapper."""

    def test_open_unencrypted_jpeg(self, tmp_path):
        """Opens a normal JPEG file."""
        f = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 50), color="red")
        img.save(str(f), "JPEG")

        result = open_image(f)
        assert result.size == (100, 50)

    @patch("screenmind.privacy.encryption._get_fernet")
    @patch("screenmind.privacy.encryption.settings")
    def test_open_encrypted_jpeg(self, mock_settings, mock_fernet, tmp_path):
        """Opens an encrypted JPEG transparently."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        real_fernet = Fernet(key)
        mock_fernet.return_value = real_fernet
        mock_settings.encryption_enabled = True

        f = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 50), color="blue")
        img.save(str(f), "JPEG")
        encrypt_image(f)

        result = open_image(f)
        assert result.size == (100, 50)


class TestServeImage:
    """Tests for serve_image() FastAPI response."""

    def test_serve_unencrypted_returns_file_response(self, tmp_path):
        """Unencrypted file returns FileResponse."""
        from fastapi.responses import FileResponse
        f = tmp_path / "test.jpg"
        img = Image.new("RGB", (50, 50), color="green")
        img.save(str(f), "JPEG")

        response = serve_image(f)
        assert isinstance(response, FileResponse)

    def test_serve_encrypted_returns_response(self, tmp_path):
        """Encrypted file returns Response with decrypted bytes."""
        from fastapi.responses import Response
        from cryptography.fernet import Fernet

        f = tmp_path / "test.jpg"
        img = Image.new("RGB", (50, 50), color="green")
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        original_bytes = buf.getvalue()

        # Manually encrypt
        key = Fernet.generate_key()
        fernet = Fernet(key)
        encrypted = fernet.encrypt(original_bytes)
        f.write_bytes(MAGIC + encrypted)

        with patch("screenmind.privacy.encryption._get_fernet", return_value=fernet):
            response = serve_image(f)
        assert isinstance(response, Response)
        assert response.body == original_bytes


class TestFailClosedQuarantine:
    """CT-05.2: keyring/crypto failure -> quarantine, never persist in clear."""

    def _bad_keyring(self, monkeypatch):
        class _Broken:
            def get_password(self, service, username):
                raise RuntimeError("keyring backend unavailable")

            def set_password(self, service, username, value):
                raise RuntimeError("keyring backend unavailable")

        monkeypatch.setattr(enc, "_get_keyring", lambda: _Broken())

    def test_keyring_failure_quarantines_plaintext(self, monkeypatch, tmp_data):
        """Keyring down -> file removed, audit record written, no plaintext left."""
        self._bad_keyring(monkeypatch)
        shots = tmp_data / "screenshots" / "2026-10-14"
        shots.mkdir(parents=True)
        f = shots / "12-00-00_000.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"TOP SECRET PII" * 50)

        with pytest.raises(EncryptionUnavailable):
            encrypt_image(f)

        assert not f.exists()  # not persisted in clear
        assert not list(tmp_data.rglob("*.jpg"))  # nothing readable left behind

        records = (tmp_data / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(records) == 1
        assert "12-00-00_000.jpg" in records[0]
        assert "keyring_unavailable" in records[0]

    def test_encrypt_failure_quarantines_plaintext(self, monkeypatch, tmp_data):
        """Crypto failure mid-encrypt -> quarantine + raise (fail-closed)."""

        class _BoomFernet:
            def encrypt(self, data):
                raise ValueError("cipher blew up")

        monkeypatch.setattr(enc, "_get_fernet", lambda: _BoomFernet())
        f = tmp_data / "shot.jpg"
        f.write_bytes(b"\xff\xd8" + b"plain" * 100)

        with pytest.raises(EncryptionUnavailable):
            encrypt_image(f)

        assert not f.exists()
        assert not list(tmp_data.rglob("*.jpg"))
        recs = (tmp_data / "quarantine.jsonl").read_text(encoding="utf-8")
        assert "encrypt_failed" in recs

    def test_keyring_import_failure_is_fail_closed(self, monkeypatch):
        """No keyring package -> EncryptionUnavailable (no weak fallback)."""
        monkeypatch.setattr(enc, "_get_keyring", lambda: (_ for _ in ()).throw(
            EncryptionUnavailable("keyring missing")
        ))
        with pytest.raises(EncryptionUnavailable):
            enc._get_fernet()


class TestKeyHygiene:
    """CT-05.3: the key lives only in the OS keyring, never on disk in clear."""

    def test_key_never_written_to_disk(self, fake_keyring, tmp_data, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"data" * 50)
        encrypt_image(f)

        assert is_encrypted(f) is True
        # No plaintext key file is ever created.
        assert not (tmp_path / ".encryption_key").exists()
        assert not list(tmp_path.rglob("*encryption_key*"))

        key_bytes = fake_keyring[(KEYRING_SERVICE, KEYRING_USERNAME)].encode()
        assert key_bytes  # a key was generated
        # Key material must not leak into any file under the data dir.
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert key_bytes not in path.read_bytes()

    def test_key_is_reused_from_keyring(self, fake_keyring, tmp_path):
        first = enc._get_or_create_key()
        enc._fernet_instance = None  # force re-read
        second = enc._get_or_create_key()
        assert first == second
        assert fake_keyring[(KEYRING_SERVICE, KEYRING_USERNAME)].encode() == first

    def test_keyring_write_failure_raises(self, monkeypatch):
        class _NoWrite:
            def get_password(self, service, username):
                return None

            def set_password(self, service, username, value):
                raise RuntimeError("cannot persist to keyring")

        monkeypatch.setattr(enc, "_get_keyring", lambda: _NoWrite())
        with pytest.raises(EncryptionUnavailable):
            enc._get_or_create_key()
