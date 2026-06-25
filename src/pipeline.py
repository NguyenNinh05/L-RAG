"""
src/pipeline.py
===============
Orchestrator toàn bộ 3 Phase của hệ thống đối chiếu văn bản pháp lý.

Luồng xử lý:
┌──────────────────────────────────────────────────────────────────────┐
│  Input: 2 file PDF/DOCX (V1, V2)                                     │
│                                                                       │
│  Phase 1 — Ingestion (src.ingestion):                                │
│    LegalDocumentParser → LsuChunker → HybridGraphBuilder             │
│    → LegalDocument, list[LsuChunk], Kuzu Graph, ChromaDB             │
│                                                                       │
│  Phase 2 — Alignment (src.alignment):                                │
│    BGEM3Manager → LegalAlignmentEngine → DiffPairCatalog             │
│    → N cặp (matched/added/deleted/split/merged)                      │
│                                                                       │
│  Phase 3 — Generative Comparison (src.comparison):                   │
│    GenerativeComparisonPipeline → ComparisonReport[]                 │
│    → Biên bản so sánh (JSON + Markdown)                              │
└──────────────────────────────────────────────────────────────────────┘

Usage:
    from src.pipeline import LegalDiffPipeline, PipelineRunConfig

    pipeline = LegalDiffPipeline.from_config()
    result = pipeline.run(
        file_v1="data/raw/contract_v1.docx",
        file_v2="data/raw/contract_v2.docx",
    )
    print(result["markdown"])
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Run Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineRunConfig:
    """Cấu hình cho một lần chạy pipeline đầy đủ."""

    # Paths
    file_v1: str = ""
    file_v2: str = ""
    output_dir: str = "./data/reports"

    # Phase 1
    kuzu_db_path: str = "./data/processed/graph_db"
    chroma_db_path: str = "./data/processed/chroma_db"
    confidence_threshold: float = 0.75
    max_chunk_chars: int = 2000

    # Phase 2
    qdrant_path: str | None = "./data/processed/qdrant_db"
    collection_name: str | None = None  # auto-generated if None
    match_threshold: float = 0.65

    # Phase 3
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_api_key: str = "not-needed"
    max_concurrency: int = 4
    max_tokens_acu: int = 4096
    max_tokens_summary: int = 1024
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 3
    llm_temperature_acu: float = 0.05
    llm_temperature_summary: float = 0.3
    max_comparison_pairs: int | None = None

    # Phase 3 — speed / throughput knobs (mirror GenPipelineCfg so they flow
    # through config_overrides; defaults match the comparison engine defaults).
    enable_second_pass: bool = True
    enable_number_enumeration: bool = True
    min_confidence_to_include: float = 0.2

    # Callback — gọi tại ranh giới phase: cb(pct, phase, message)
    # None → không gọi (chạy như cũ), dùng cho CLI/testing
    progress_callback: Callable[[int, str, str], None] | None = None

    # LLM provider identification
    llm_provider: str = "local"

    # Flags
    skip_phase3: bool = False  # True để chỉ chạy Phase 1+2

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LegalDiffPipeline — Main Orchestrator
# ---------------------------------------------------------------------------


class LegalDiffPipeline:
    """
    Orchestrator điều phối toàn bộ 3 Phase.

    Usage:
        pipeline = LegalDiffPipeline.from_config()
        result = pipeline.run(
            file_v1="path/to/v1.docx",
            file_v2="path/to/v2.docx",
        )
    """

    def __init__(self, run_config: PipelineRunConfig | None = None) -> None:
        self._cfg = run_config or PipelineRunConfig()

    @classmethod
    def from_config(cls, config_path: str | None = None, provider: str | None = None) -> "LegalDiffPipeline":
        """
        Khởi tạo từ YAML config.

        Args:
            config_path: Đường dẫn đến pipeline_config.yaml.
                         None → dùng configs/ trong project root.
            provider:    "local" (Qwen), "deepseek" (DeepSeek API), hoặc None (auto).
        """
        from src.config import get_config, get_llm_config
        cfg = get_config()
        llm_cfg = get_llm_config(provider=provider)

        run_cfg = PipelineRunConfig(
            kuzu_db_path=cfg["ingestion"]["kuzu_db_path"],
            chroma_db_path=cfg["ingestion"]["chroma_db_path"],
            confidence_threshold=cfg["ingestion"]["confidence_threshold"],
            max_chunk_chars=cfg["ingestion"]["max_chunk_chars"],
            match_threshold=cfg["alignment"]["match_threshold"],
            llm_base_url=llm_cfg["base_url"],
            llm_model_name=llm_cfg["model_name"],
            llm_api_key=llm_cfg.get("api_key", "not-needed"),
            max_concurrency=cfg["comparison"]["max_concurrency"],
            max_tokens_acu=llm_cfg["max_tokens_acu"],
            max_tokens_summary=llm_cfg["max_tokens_summary"],
            llm_timeout_seconds=llm_cfg.get("timeout_seconds", 120.0),
            llm_max_retries=llm_cfg.get("max_retries", 3),
            llm_temperature_acu=llm_cfg.get("temperature_acu", 0.05),
            llm_temperature_summary=llm_cfg.get("temperature_summary", 0.3),
            llm_provider=provider or "local",
        )
        return cls(run_config=run_cfg)

    def run(
        self,
        file_v1: str | None = None,
        file_v2: str | None = None,
        skip_phase3: bool | None = None,
    ) -> dict[str, Any]:
        """
        Chạy toàn bộ pipeline đồng bộ.

        Args:
            file_v1:     Đường dẫn file PDF/DOCX phiên bản V1.
            file_v2:     Đường dẫn file PDF/DOCX phiên bản V2.
            skip_phase3: True để bỏ qua Generative Comparison (Phase 3).

        Returns:
            dict chứa: doc_v1, doc_v2, catalog, reports, markdown, json_report.
        """
        if file_v1:
            self._cfg.file_v1 = file_v1
        if file_v2:
            self._cfg.file_v2 = file_v2
        if skip_phase3 is not None:
            self._cfg.skip_phase3 = skip_phase3

        return asyncio.run(self._run_async())

    async def _run_async(self) -> dict[str, Any]:
        """Async implementation của pipeline."""
        cfg = self._cfg
        cb = cfg.progress_callback
        result: dict[str, Any] = {}

        def _notify(pct: int, phase: str, msg: str) -> None:
            if cb:
                try:
                    cb(pct, phase, msg)
                except Exception:
                    pass  # callback is advisory only

        # ── Phase 1: Ingestion ──────────────────────────────────────────
        logger.info("[Phase 1] Ingestion bắt đầu: %s | %s", cfg.file_v1, cfg.file_v2)
        _notify(10, "ingestion", "Đang phân tích tài liệu...")

        from src.ingestion import LegalDocumentParser, LsuChunker

        parser = LegalDocumentParser(confidence_threshold=cfg.confidence_threshold)
        chunker = LsuChunker(max_chunk_chars=cfg.max_chunk_chars)

        doc_v1 = parser.parse(cfg.file_v1)
        doc_v2 = parser.parse(cfg.file_v2)

        chunks_v1 = chunker.chunk(doc_v1)
        chunks_v2 = chunker.chunk(doc_v2)

        result["doc_v1"] = doc_v1
        result["doc_v2"] = doc_v2
        result["chunks_v1"] = chunks_v1
        result["chunks_v2"] = chunks_v2

        logger.info(
            "[Phase 1] Done. V1: %d articles, %d chunks | V2: %d articles, %d chunks",
            len(doc_v1.iter_all_articles()),
            len(chunks_v1),
            len(doc_v2.iter_all_articles()),
            len(chunks_v2),
        )
        _notify(30, "alignment", "Đang nhúng vector BGE-M3...")

        # ── Phase 2: Alignment ──────────────────────────────────────────
        logger.info("[Phase 2] Alignment bắt đầu...")

        from src.alignment.alignment_engine import LegalAlignmentEngine
        from src.alignment.similarity_matrix import AlignmentConfig
        from src.alignment.embedder import BGEM3Manager
        from src.alignment.qdrant_indexer import QdrantManager

        from src.config import get_config
        acfg = get_config()["alignment"]

        embed_manager = BGEM3Manager()
        alignment_config = AlignmentConfig(
            w_semantic=acfg["w_semantic"],
            w_jaro_winkler=acfg["w_jaro_winkler"],
            w_ordinal=acfg["w_ordinal"],
            w_sparse=acfg.get("w_sparse", 0.0),
            match_threshold=acfg["match_threshold"],
            split_merge_threshold=acfg["split_merge_threshold"],
        )
        qdrant = QdrantManager(path=cfg.qdrant_path) if cfg.qdrant_path else None

        engine = LegalAlignmentEngine(
            embed_manager=embed_manager,
            config=alignment_config,
            qdrant_manager=qdrant,
        )

        collection_name = cfg.collection_name or (
            f"legal_{doc_v1.doc_id[:8]}_{doc_v2.doc_id[:8]}"
        )
        catalog = engine.align_documents(doc_v1, doc_v2, collection_name=collection_name)
        result["catalog"] = catalog

        logger.info("[Phase 2] Done. %s", catalog.summary())
        _notify(55, "comparison", "Đang sinh báo cáo so sánh...")

        if cfg.skip_phase3:
            logger.info("[Phase 3] Skipped (skip_phase3=True).")
            _notify(100, "done", "Hoàn thành!")
            return result

        # ── Phase 3: Generative Comparison ─────────────────────────────
        logger.info("[Phase 3] Generative Comparison bắt đầu...")

        from src.comparison import GenerativeComparisonPipeline, ComparisonRequest
        from src.comparison import PipelineConfig as GenPipelineCfg

        pipeline_cfg = GenPipelineCfg(
            llm_base_url=cfg.llm_base_url,
            llm_model_name=cfg.llm_model_name,
            llm_api_key=cfg.llm_api_key,
            acu_temperature=cfg.llm_temperature_acu,
            summary_temperature=cfg.llm_temperature_summary,
            max_concurrency=cfg.max_concurrency,
            max_tokens_acu=cfg.max_tokens_acu,
            max_tokens_summary=cfg.max_tokens_summary,
            timeout_seconds=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
            provider=cfg.llm_provider,
            min_confidence_to_include=cfg.min_confidence_to_include,
            enable_second_pass=cfg.enable_second_pass,
            enable_number_enumeration=cfg.enable_number_enumeration,
        )
        gen_pipeline = GenerativeComparisonPipeline(config=pipeline_cfg)

        # Xử lý MỌI match type cho generative comparison — không chỉ matched.
        # Added/deleted/split/merged cũng được LLM phân tích (guidance_map trong
        # report_generator đã xử lý từng type). Concatenate multi-text cho split/merged.
        all_pairs = catalog.pairs
        if cfg.max_comparison_pairs is not None:
            all_pairs = all_pairs[: cfg.max_comparison_pairs]
        requests = [
            ComparisonRequest(
                pair_id=pair.pair_id,
                match_type=pair.match_type.value,
                raw_text_v1="\n\n".join(pair.v1_texts),
                raw_text_v2="\n\n".join(pair.v2_texts),
            )
            for pair in all_pairs
        ]

        reports = await gen_pipeline.run_batch(requests, max_concurrency=cfg.max_concurrency)
        result["reports"] = reports

        logger.info("[Phase 3] Done. %d reports generated.", len(reports))
        _notify(95, "done", "Đang lưu kết quả...")
        return result
