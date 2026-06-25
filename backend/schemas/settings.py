"""
backend/schemas/settings.py — Per-user LLM settings schemas.

Field names mirror `PipelineRunConfig` attrs so a saved config can be applied
directly as `config_overrides` when a job is created.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LLMConfigSchema(BaseModel):
    llm_provider: Literal["local", "deepseek"] = "local"
    llm_model_name: str = "qwen2.5:14b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "not-needed"
    llm_temperature_acu: float = Field(0.05, ge=0.0, le=2.0)
    llm_temperature_summary: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens_acu: int = Field(8192, ge=1)
    max_tokens_summary: int = Field(1024, ge=1)
    max_concurrency: int = Field(1, ge=1, le=32)
    llm_timeout_seconds: float = Field(300.0, ge=1.0)
    llm_max_retries: int = Field(3, ge=0, le=10)
    match_threshold: float = Field(0.65, ge=0.0, le=1.0)
    skip_phase3: bool = False

    # Phase 3 — speed / throughput knobs. Field names mirror PipelineRunConfig /
    # GenPipelineCfg so they apply through config_overrides unchanged.
    enable_second_pass: bool = True
    enable_number_enumeration: bool = True
    min_confidence_to_include: float = Field(0.2, ge=0.0, le=1.0)
    # None (and the UI's 0) means unlimited — _run_async only caps pairs when
    # this is a positive int.
    max_comparison_pairs: int | None = Field(default=None, ge=0)

    @field_validator("max_comparison_pairs", mode="before")
    @classmethod
    def _zero_means_unlimited(cls, v):
        """Normalize the UI convention 0/""/null -> None (unlimited)."""
        return None if v in (0, "", None) else v


class LLMSettingsResponse(BaseModel):
    """Saved config + the provider/model options for the settings UI."""

    config: LLMConfigSchema
    providers: list[str]
    models: list[str]


class AvailableModelsResponse(BaseModel):
    provider: str
    models: list[str]
    # Resolved default base_url for this provider (does NOT require an API key).
    # The settings UI uses it to repopulate base_url when switching providers so
    # a stale value from the previous provider (e.g. a local Ollama URL under
    # "deepseek") doesn't persist.
    base_url: str | None = None
