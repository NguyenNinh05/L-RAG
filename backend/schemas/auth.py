"""
backend/schemas/auth.py — Authentication request/response schemas.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class AuthUserResponse(BaseModel):
    """Brief user payload embedded in auth responses (matches frontend `User`)."""

    id: uuid.UUID
    username: str
    email: str
    role: str = "user"

    model_config = ConfigDict(from_attributes=True)


class AuthResponse(BaseModel):
    """
    Full auth response returned by /login and /register.

    Includes the access token AND the user object — the frontend reads both
    (`data.access_token` and `data.user`) to establish a session.
    """

    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int = 3600
    user: AuthUserResponse


class RegisterRequest(BaseModel):
    email: str = Field(
        ...,
        min_length=5,
        max_length=255,
        pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
    )
    username: str = Field(
        ..., min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_]+$"
    )
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    expires_in: int = 3600
