"""
backend/services/auth_service.py — Password hashing + JWT token management.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt

from backend.config import get_backend_config


def _password_bytes(password: str) -> bytes:
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_password_bytes(plain), hashed.encode("utf-8"))


def create_access_token(user_id: str, username: str) -> str:
    cfg = get_backend_config()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "username": username,
        "exp": now + timedelta(minutes=cfg.access_token_expire_minutes),
        "iat": now,
        "type": "access",
    }
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def create_refresh_token(user_id: str, username: str) -> str:
    cfg = get_backend_config()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "username": username,
        "exp": now + timedelta(days=cfg.refresh_token_expire_days),
        "iat": now,
        "type": "refresh",
    }
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def verify_token(token: str) -> dict:
    cfg = get_backend_config()
    return jwt.decode(token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm])
