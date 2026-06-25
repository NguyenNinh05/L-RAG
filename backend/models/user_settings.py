"""
backend/models/user_settings.py — Per-user LLM settings ORM model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    from backend.models.user import User


class UserSettings(Base, TimestampMixin):
    """One row per user holding their saved LLM model + hyperparameter config.

    `llm_config` is a JSONB blob whose keys mirror `PipelineRunConfig` fields
    (e.g. llm_model_name, llm_temperature_acu, max_concurrency, ...) so it can
    be applied directly as `config_overrides` when a job is created. Queried by
    `user_id` directly — no ORM relationship needed.
    """

    __tablename__ = "user_settings"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid_pk)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    llm_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
