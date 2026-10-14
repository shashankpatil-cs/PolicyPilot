from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.core.config import settings
from app.core.security import (
    PasswordTooLongError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_is_bcrypt_and_not_plaintext():
    h = hash_password("admin282005")
    assert h.startswith("$2") and "admin282005" not in h
    assert verify_password("admin282005", h)
    assert not verify_password("wrong", h)


def test_hashes_are_salted():
    assert hash_password("same-password") != hash_password("same-password")


def test_over_72_bytes_rejected_not_truncated():
    with pytest.raises(PasswordTooLongError):
        hash_password("x" * 73)
    assert not verify_password("x" * 73, hash_password("x" * 72))


def test_verify_handles_garbage_hash():
    assert verify_password("pw", "not-a-hash") is False


def test_jwt_roundtrip():
    payload = decode_access_token(create_access_token(7, "admin"))
    assert payload["sub"] == "7" and payload["role"] == "admin"


def test_jwt_expired_and_tampered_rejected():
    expired = jwt.encode(
        {"sub": "1", "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
        settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(expired)
    forged = jwt.encode(
        {"sub": "1", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        "some-other-secret-that-is-long-enough-123456", algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(forged)


def test_jwt_without_exp_rejected():
    no_exp = jwt.encode({"sub": "1"}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(no_exp)
