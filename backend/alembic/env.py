"""Alembic environment configuration for LegalDiff Backend."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

# Load backend/.env before anything else
from pathlib import Path
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure project root is importable (for "backend" package)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.models.base import Base
from backend.models import User, Document, ComparisonJob, ComparisonReportModel  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url from env
db_url = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:hieu1205@localhost:5432/legaldiff",
)
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
