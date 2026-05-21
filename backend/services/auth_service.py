"""
backend/services/auth_service.py — Password hashing + JWT token management.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from backend.config import get_backend_config

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


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
