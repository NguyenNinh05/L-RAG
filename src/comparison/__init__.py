"""
src/comparison/__init__.py
===========================
Public API của comparison package — Phase 3: Generative Comparison.

Usage:
    from src.comparison import (
        GenerativeComparisonPipeline,
        PipelineConfig,
        ComparisonRequest,
        ComparisonReport,
    )
"""

# Models & VerificationEngine — không có external deps nặng
from .models import (
    ACUOutput,
    ChangeType,
    ComparisonReport,
    ComparisonRequest,
    ExecutiveSummary,
    VerificationResult,
    VerificationStatus,
)
from .verifier import (
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
    # Engine
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
        from .llm_client import LocalLLMClient, LLMConfig
        return {"LocalLLMClient": LocalLLMClient, "LLMConfig": LLMConfig}[name]
    if name in ("GenerativeComparisonPipeline", "PipelineConfig"):
        from .report_generator import GenerativeComparisonPipeline, PipelineConfig
        return {
            "GenerativeComparisonPipeline": GenerativeComparisonPipeline,
            "PipelineConfig": PipelineConfig,
        }[name]
    raise AttributeError(f"module 'src.comparison' has no attribute {name!r}")
