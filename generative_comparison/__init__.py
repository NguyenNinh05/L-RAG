"""
generative_comparison/__init__.py
===================================
Public API cho Phase 3 — Generative Comparison.

Usage nhanh:
    from generative_comparison import (
        GenerativeComparisonPipeline,
        PipelineConfig,
        ComparisonRequest,
        ComparisonReport,
    )

    config = PipelineConfig(llm_base_url="http://localhost:8000/v1")
    pipeline = GenerativeComparisonPipeline(config=config)

    # Async — xử lý nhiều cặp song song
    import asyncio
    requests = [ComparisonRequest(...), ...]
    reports = asyncio.run(pipeline.run_batch(requests))
"""

# Models & VerificationEngine không có external deps nặng → import trực tiếp
from generative_comparison.models import (
    ACUOutput,
    ChangeType,
    ComparisonReport,
    ComparisonRequest,
    ExecutiveSummary,
    VerificationResult,
    VerificationStatus,
)
from generative_comparison.verification_engine import (
    VerificationEngine,
    VerificationConfig,
    extract_numbers,
)

__all__ = [
    # Models
    "ACUOutput",
    "ChangeType",
    "ComparisonReport",
    "ComparisonRequest",
    "ExecutiveSummary",
    "VerificationResult",
    "VerificationStatus",
    # Engine (no LLM dep)
    "VerificationEngine",
    "VerificationConfig",
    "extract_numbers",
    # LLM-dependent — imported lazily
    "LocalLLMClient",
    "LLMConfig",
    "GenerativeComparisonPipeline",
    "PipelineConfig",
]


def __getattr__(name: str):
    """Lazy import cho các symbol phụ thuộc openai (chỉ load khi cần)."""
    if name in ("LocalLLMClient", "LLMConfig"):
        from generative_comparison.llm_manager import LocalLLMClient, LLMConfig
        return {"LocalLLMClient": LocalLLMClient, "LLMConfig": LLMConfig}[name]
    if name in ("GenerativeComparisonPipeline", "PipelineConfig"):
        from generative_comparison.comparison_pipeline import (
            GenerativeComparisonPipeline,
            PipelineConfig,
        )
        return {"GenerativeComparisonPipeline": GenerativeComparisonPipeline, "PipelineConfig": PipelineConfig}[name]
    raise AttributeError(f"module 'generative_comparison' has no attribute {name!r}")
