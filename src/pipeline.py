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
    max_concurrency: int = 4

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
    def from_config(cls, config_path: str | None = None) -> "LegalDiffPipeline":
        """
        Khởi tạo từ YAML config.

        Args:
            config_path: Đường dẫn đến pipeline_config.yaml.
                         None → dùng configs/ trong project root.
        """
        from src.config import get_config
        cfg = get_config()

        run_cfg = PipelineRunConfig(
            kuzu_db_path=cfg["ingestion"]["kuzu_db_path"],
            chroma_db_path=cfg["ingestion"]["chroma_db_path"],
            confidence_threshold=cfg["ingestion"]["confidence_threshold"],
            max_chunk_chars=cfg["ingestion"]["max_chunk_chars"],
            match_threshold=cfg["alignment"]["match_threshold"],
            llm_base_url=cfg["llm"]["base_url"],
            llm_model_name=cfg["llm"]["model_name"],
            max_concurrency=cfg["comparison"]["max_concurrency"],
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
        result: dict[str, Any] = {}

        # ── Phase 1: Ingestion ──────────────────────────────────────────
        logger.info("[Phase 1] Ingestion bắt đầu: %s | %s", cfg.file_v1, cfg.file_v2)

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

        # ── Phase 2: Alignment ──────────────────────────────────────────
        logger.info("[Phase 2] Alignment bắt đầu...")

        from comparison.alignment_engine import LegalAlignmentEngine, AlignmentConfig
        from src.alignment.embedder import BGEM3Manager
        from src.alignment.qdrant_indexer import QdrantManager

        from src.config import get_config
        acfg = get_config()["alignment"]

        embed_manager = BGEM3Manager()
        alignment_config = AlignmentConfig(
            w_semantic=acfg["w_semantic"],
            w_jaro_winkler=acfg["w_jaro_winkler"],
            w_ordinal=acfg["w_ordinal"],
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

        if cfg.skip_phase3:
            logger.info("[Phase 3] Skipped (skip_phase3=True).")
            return result

        # ── Phase 3: Generative Comparison ─────────────────────────────
        logger.info("[Phase 3] Generative Comparison bắt đầu...")

        from src.comparison import GenerativeComparisonPipeline, ComparisonRequest
        from src.comparison import PipelineConfig as GenPipelineCfg

        pipeline_cfg = GenPipelineCfg(
            llm_base_url=cfg.llm_base_url,
            llm_model_name=cfg.llm_model_name,
            max_concurrency=cfg.max_concurrency,
        )
        gen_pipeline = GenerativeComparisonPipeline(config=pipeline_cfg)

        # Chỉ xử lý matched pairs cho generative comparison
        matched = catalog.matched_pairs
        requests = [
            ComparisonRequest(
                pair_id=pair.pair_id,
                match_type=pair.match_type.value,
                raw_text_v1=pair.v1_texts[0] if pair.v1_texts else "",
                raw_text_v2=pair.v2_texts[0] if pair.v2_texts else "",
            )
            for pair in matched
        ]

        reports = await gen_pipeline.run_batch(requests, max_concurrency=cfg.max_concurrency)
        result["reports"] = reports

        logger.info("[Phase 3] Done. %d reports generated.", len(reports))
        return result
