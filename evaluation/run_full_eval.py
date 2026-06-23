"""
evaluation/run_full_eval.py
============================
Full pipeline evaluation on all document pairs in data_test/.

Runs Phase 1 (Ingestion) + Phase 2 (Alignment) + Phase 3 (Comparison)
on every (v1.docx, v2.docx) pair, compares pipeline output against
ground truth JSON, and computes metrics.

Metrics:
  - Alignment: #articles matched/added/deleted/split/merged per pair
  - Change Detection: recall, precision, F1 at the change level
  - Hallucination: rate of rejected ACUs
  - Per-category breakdown (A-F)

Usage:
    python evaluation/run_full_eval.py
    python evaluation/run_full_eval.py --skip-phase3  # fast mode, alignment only
    python evaluation/run_full_eval.py --pair 01-tand_signed  # single pair
    python evaluation/run_full_eval.py --output-dir ./eval_results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path (needed when run from anywhere)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ["httpx", "httpcore", "urllib3", "PIL", "torch", "sentence_transformers",
               "transformers", "qdrant_client", "chromadb", "docling"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("eval")

DIVIDER = "=" * 80

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data_test"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "eval_results"

# Map ground truth category letters to descriptions
CATEGORY_MAP = {
    "A": "Thay đổi con số",
    "B": "Diễn đạt lại ngữ nghĩa",
    "C": "Thay đổi cấu trúc",
    "D": "Thay đổi logic pháp lý",
    "E": "Xử lý bảng biểu",
    "F": "Nhiễu định dạng",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PairEvalResult:
    """Evaluation result for a single document pair."""
    pair_name: str = ""
    v1_file: str = ""
    v2_file: str = ""
    gt_file: str = ""

    # Phase 1 stats
    v1_articles: int = 0
    v2_articles: int = 0
    v1_chunks: int = 0
    v2_chunks: int = 0

    # Phase 2 stats
    aligned_matched: int = 0
    aligned_added: int = 0
    aligned_deleted: int = 0
    aligned_split: int = 0
    aligned_merged: int = 0

    # Phase 3 stats
    total_acus_generated: int = 0
    total_acus_passed: int = 0
    total_acus_rejected: int = 0
    hallucination_rate: float = 0.0

    # Ground truth stats
    gt_total_changes: int = 0
    gt_categories: dict[str, int] = field(default_factory=dict)

    # Change detection metrics (at article level)
    gt_changes_detected: int = 0  # GT changes found in pipeline output
    pipeline_changes_total: int = 0  # Total ACUs from pipeline
    pipeline_changes_matched: int = 0  # Pipeline ACUs matching a GT change

    # Per-category metrics
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)

    # Timing
    phase1_time: float = 0.0
    phase2_time: float = 0.0
    phase3_time: float = 0.0
    total_time: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)
    skipped_phase3: bool = False


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------


def discover_pairs(data_dir: Path) -> list[dict[str, str]]:
    """Find all (v1.docx, v2.docx, ground_truth.json) triples."""
    pairs = []
    v1_files = sorted(data_dir.glob("*_v1.docx"))

    for v1_path in v1_files:
        stem = v1_path.stem  # e.g., "01-tand_signed_v1"
        base = stem[:-3]  # remove "_v1" -> "01-tand_signed"
        v2_path = data_dir / f"{base}_v2.docx"
        json_path = data_dir / f"{base}.json"

        if v2_path.exists() and json_path.exists():
            pairs.append({
                "name": base,
                "v1": str(v1_path),
                "v2": str(v2_path),
                "gt": str(json_path),
            })
        else:
            logger.warning("Incomplete pair for %s: v2=%s, json=%s",
                           base, v2_path.exists(), json_path.exists())

    return pairs


# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------


def load_ground_truth(gt_path: str) -> list[dict[str, Any]]:
    """Load and validate ground truth JSON."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "changes" in data:
        return data["changes"]
    logger.warning("Unexpected GT format in %s: %s", gt_path, type(data).__name__)
    return []


def extract_article_number(location: str) -> str | None:
    """Extract article number from a location string like 'Điều 4, Khoản 2'."""
    # Try "Điều X" or "Điều X.Y" or "Mục X" patterns
    m = re.search(r'Điều\s+(\d+[a-z]?)', location, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'Mục\s+([IVXLCDM]+|\d+)', location, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def group_gt_by_article(gt_changes: list[dict]) -> dict[str, list[dict]]:
    """Group ground truth changes by article number."""
    by_article: dict[str, list[dict]] = defaultdict(list)
    unplaced: list[dict] = []

    for change in gt_changes:
        loc = change.get("location", "")
        art_num = extract_article_number(loc)
        if art_num:
            by_article[art_num].append(change)
        else:
            unplaced.append(change)

    if unplaced:
        by_article["_unplaced"] = unplaced

    return dict(by_article)


# ---------------------------------------------------------------------------
# Change detection helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, remove punctuation."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Remove common punctuation for fuzzy matching
    text = re.sub(r'[,\.;:\(\)\[\]"\'«»『』「」]', '', text)
    return text


def text_overlap_ratio(text_a: str, text_b: str) -> float:
    """Simple word-level overlap ratio between two text segments."""
    words_a = set(normalize_text(text_a).split())
    words_b = set(normalize_text(text_b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))


def fuzzy_find(text: str, query: str, threshold: float = 0.3) -> bool:
    """
    Check if query text can be found in text (fuzzy match).
    Uses sliding window of word n-grams.
    """
    if not query or not text:
        return False

    norm_text = normalize_text(text)
    norm_query = normalize_text(query)

    query_words = norm_query.split()
    text_words = norm_text.split()

    if len(query_words) == 0:
        return False

    # Short queries: exact substring
    if len(query_words) <= 3:
        return norm_query in norm_text

    # Longer queries: sliding window overlap
    window_size = len(query_words)
    for i in range(len(text_words) - window_size + 1):
        window = text_words[i:i + window_size]
        overlap = len(set(query_words) & set(window)) / len(query_words)
        if overlap >= threshold:
            return True

    return False


def match_gt_to_acus(
    gt_changes: list[dict],
    acus: list[dict],
    article_text_v1: str,
    article_text_v2: str,
) -> tuple[int, int]:
    """
    Match ground truth changes to pipeline ACUs.

    Returns:
        (num_gt_detected, num_pipeline_matched)
    """
    gt_detected = 0
    pipeline_matched = set()

    for gt_change in gt_changes:
        gt_orig = normalize_text(gt_change.get("original_text", ""))
        gt_mod = normalize_text(gt_change.get("modified_text", ""))

        # For additions (empty original), just check modified_text
        is_addition = gt_change.get("change_type") == "addition" or not gt_orig
        is_deletion = gt_change.get("change_type") == "deletion" or not gt_mod

        found_match = False
        for i, acu in enumerate(acus):
            if i in pipeline_matched:
                continue

            acu_orig = normalize_text(acu.get("original_value", ""))
            acu_new = normalize_text(acu.get("new_value", ""))
            acu_ev1 = normalize_text(acu.get("verbatim_evidence_v1", ""))
            acu_ev2 = normalize_text(acu.get("verbatim_evidence_v2", ""))

            # Check: does the ACU capture this ground truth change?
            if is_addition and gt_mod:
                # GT says something was added → ACU should have new_value or evidence_v2
                if gt_mod in acu_new or gt_mod in acu_ev2:
                    found_match = True
            elif is_deletion and gt_orig:
                # GT says something was deleted → ACU should have original_value or evidence_v1
                if gt_orig in acu_orig or gt_orig in acu_ev1:
                    found_match = True
            else:
                # Normal change: both original and modified
                orig_match = (gt_orig in acu_orig or gt_orig in acu_ev1 or
                              text_overlap_ratio(gt_orig, acu_orig) > 0.6)
                mod_match = (gt_mod in acu_new or gt_mod in acu_ev2 or
                             text_overlap_ratio(gt_mod, acu_new) > 0.6)
                if orig_match and mod_match:
                    found_match = True
                elif orig_match or mod_match:
                    # Partial match
                    found_match = True

            if found_match:
                pipeline_matched.add(i)
                break

        if found_match:
            gt_detected += 1

    return gt_detected, len(pipeline_matched)


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------


async def evaluate_pair(
    pair_info: dict[str, str],
    output_dir: Path,
    parser: "LegalDocumentParser",
    chunker: "LsuChunker",
    embed_manager: "BGEM3Manager",
    alignment_config: "AlignmentConfig",
    cfg: dict[str, Any],
    skip_phase3: bool = False,
) -> PairEvalResult:
    """Run full pipeline on one document pair and evaluate against ground truth.

    Args:
        pair_info: Dict with 'name', 'v1', 'v2', 'gt' paths.
        output_dir: Where to save per-pair results.
        parser: Shared LegalDocumentParser (reused across pairs).
        chunker: Shared LsuChunker (reused across pairs).
        embed_manager: Shared BGEM3Manager — created ONCE to avoid
                       re-loading BGE-M3 model per pair.
        alignment_config: Shared AlignmentConfig.
        cfg: Pipeline config dict (shared across pairs).
        skip_phase3: If True, skip generative comparison.
    """
    result = PairEvalResult(
        pair_name=pair_info["name"],
        v1_file=pair_info["v1"],
        v2_file=pair_info["v2"],
        gt_file=pair_info["gt"],
        skipped_phase3=skip_phase3,
    )

    t_total_start = time.perf_counter()

    # ── Load ground truth ──────────────────────────────────────────
    gt_changes = load_ground_truth(pair_info["gt"])
    result.gt_total_changes = len(gt_changes)
    for c in gt_changes:
        cat = c.get("category", "?")
        result.gt_categories[cat] = result.gt_categories.get(cat, 0) + 1

    gt_by_article = group_gt_by_article(gt_changes)

    # ── Phase 1: Ingestion ─────────────────────────────────────────
    logger.info("[%s] Phase 1 — Ingestion", pair_info["name"])
    t0 = time.perf_counter()

    try:
        doc_v1 = parser.parse(pair_info["v1"])
        doc_v2 = parser.parse(pair_info["v2"])

        chunks_v1 = chunker.chunk(doc_v1)
        chunks_v2 = chunker.chunk(doc_v2)

        result.v1_articles = len(doc_v1.iter_all_articles())
        result.v2_articles = len(doc_v2.iter_all_articles())
        result.v1_chunks = len(chunks_v1)
        result.v2_chunks = len(chunks_v2)
        result.phase1_time = time.perf_counter() - t0

        logger.info("[%s] Phase 1 done: V1=%d articles/%d chunks, V2=%d articles/%d chunks (%.1fs)",
                    pair_info["name"], result.v1_articles, result.v1_chunks,
                    result.v2_articles, result.v2_chunks, result.phase1_time)

    except Exception as e:
        result.errors.append(f"Phase 1 failed: {e}")
        logger.exception("[%s] Phase 1 error", pair_info["name"])
        result.total_time = time.perf_counter() - t_total_start
        return result

    # ── Phase 2: Alignment ─────────────────────────────────────────
    logger.info("[%s] Phase 2 — Alignment", pair_info["name"])
    t0 = time.perf_counter()

    try:
        from src.alignment.alignment_engine import LegalAlignmentEngine

        # No Qdrant needed for alignment — the engine works without it.
        # Qdrant is only useful for post-alignment vector search queries,
        # which the eval harness doesn't use.
        engine = LegalAlignmentEngine(
            embed_manager=embed_manager,
            config=alignment_config,
            qdrant_manager=None,
        )

        collection_name = f"eval_{doc_v1.doc_id[:8]}_{doc_v2.doc_id[:8]}"
        catalog = engine.align_documents(doc_v1, doc_v2, collection_name=collection_name)

        summary = catalog.summary()
        result.aligned_matched = summary["matched"]
        result.aligned_added = summary["added"]
        result.aligned_deleted = summary["deleted"]
        result.aligned_split = summary["split"]
        result.aligned_merged = summary["merged"]
        result.phase2_time = time.perf_counter() - t0

        logger.info("[%s] Phase 2 done: %s (%.1fs)",
                    pair_info["name"], summary, result.phase2_time)

    except Exception as e:
        result.errors.append(f"Phase 2 failed: {e}")
        logger.exception("[%s] Phase 2 error", pair_info["name"])
        result.total_time = time.perf_counter() - t_total_start
        return result

    # ── Phase 3: Generative Comparison ─────────────────────────────
    if skip_phase3:
        logger.info("[%s] Phase 3 — Skipped", pair_info["name"])
        result.total_time = time.perf_counter() - t_total_start
        return result

    logger.info("[%s] Phase 3 — Generative Comparison (%d matched pairs)",
                pair_info["name"], result.aligned_matched)
    t0 = time.perf_counter()

    try:
        from src.comparison import GenerativeComparisonPipeline, ComparisonRequest
        from src.comparison.report_generator import PipelineConfig as GenPipelineCfg

        llm_cfg = cfg["llm"]
        pipeline_cfg = GenPipelineCfg(
            llm_base_url=llm_cfg["base_url"],
            llm_model_name=llm_cfg["model_name"],
            max_concurrency=cfg.get("comparison", {}).get("max_concurrency", 4),
            max_tokens_acu=llm_cfg["max_tokens_acu"],
            max_tokens_summary=llm_cfg["max_tokens_summary"],
            timeout_seconds=llm_cfg.get("timeout_seconds", 120.0),
            max_retries=llm_cfg.get("max_retries", 3),
        )
        gen_pipeline = GenerativeComparisonPipeline(config=pipeline_cfg)

        matched_pairs = catalog.matched_pairs
        requests = [
            ComparisonRequest(
                pair_id=pair.pair_id,
                match_type=pair.match_type.value,
                raw_text_v1=pair.v1_texts[0] if pair.v1_texts else "",
                raw_text_v2=pair.v2_texts[0] if pair.v2_texts else "",
                breadcrumb_v1=getattr(pair, "breadcrumb_v1", "") or "",
                breadcrumb_v2=getattr(pair, "breadcrumb_v2", "") or "",
            )
            for pair in matched_pairs
        ]

        reports = await gen_pipeline.run_batch(requests)

        # ── Compare with ground truth ────────────────────────────
        # Build article number → text mapping from doc_v1 and doc_v2
        def _get_article_text(article) -> str:
            """Build full text from an ArticleNode for matching."""
            parts = [article.full_title]
            if article.intro:
                parts.append(article.intro)
            for clause in article.clauses:
                parts.append(f"{clause.number}. {clause.content}")
                for point in clause.points:
                    parts.append(f"  {point.label}) {point.content}")
            return "\n".join(parts)

        v1_articles = {str(a.number): a for a in doc_v1.iter_all_articles()}
        v2_articles = {str(a.number): a for a in doc_v2.iter_all_articles()}

        # Collect all ACU dicts across all reports
        all_acu_dicts: list[dict] = []
        total_acus_passed = 0
        total_acus_rejected = 0

        for report in reports:
            total_acus_passed += report.total_acus_passed
            total_acus_rejected += report.total_acus_rejected

            acu_dicts = [
                {
                    "change_type": acu.change_type.value,
                    "original_value": acu.original_value,
                    "new_value": acu.new_value,
                    "verbatim_evidence_v1": acu.verbatim_evidence_v1,
                    "verbatim_evidence_v2": acu.verbatim_evidence_v2,
                    "location_v1": acu.location_v1,
                    "location_v2": acu.location_v2,
                    "confidence": acu.confidence,
                }
                for acu in report.verified_acus
            ]
            all_acu_dicts.extend(acu_dicts)
            result.pipeline_changes_total += len(acu_dicts)

        # Match all ACUs against all GT changes at the pair level
        # Track which GT changes and which pipeline ACUs are matched
        matched_gt_ids: set[int] = set()
        matched_acu_indices: set[int] = set()

        for gt_idx, gt_change in enumerate(gt_changes):
            gt_orig = normalize_text(gt_change.get("original_text", ""))
            gt_mod = normalize_text(gt_change.get("modified_text", ""))
            is_addition = gt_change.get("change_type") == "addition" or not gt_orig
            is_deletion = gt_change.get("change_type") == "deletion" or not gt_mod

            for acu_idx, acu_d in enumerate(all_acu_dicts):
                if acu_idx in matched_acu_indices:
                    continue

                acu_orig = normalize_text(acu_d.get("original_value", ""))
                acu_new = normalize_text(acu_d.get("new_value", ""))
                acu_ev1 = normalize_text(acu_d.get("verbatim_evidence_v1", ""))
                acu_ev2 = normalize_text(acu_d.get("verbatim_evidence_v2", ""))

                found = False
                if is_addition and gt_mod:
                    if gt_mod in acu_new or gt_mod in acu_ev2:
                        found = True
                elif is_deletion and gt_orig:
                    if gt_orig in acu_orig or gt_orig in acu_ev1:
                        found = True
                else:
                    orig_match = (gt_orig and acu_orig and
                                  (gt_orig in acu_orig or acu_orig in gt_orig or
                                   text_overlap_ratio(gt_orig, acu_orig) > 0.5))
                    mod_match = (gt_mod and acu_new and
                                 (gt_mod in acu_new or acu_new in gt_mod or
                                  text_overlap_ratio(gt_mod, acu_new) > 0.5))
                    ev1_match = (gt_orig and acu_ev1 and
                                 (gt_orig in acu_ev1 or text_overlap_ratio(gt_orig, acu_ev1) > 0.5))
                    ev2_match = (gt_mod and acu_ev2 and
                                 (gt_mod in acu_ev2 or text_overlap_ratio(gt_mod, acu_ev2) > 0.5))
                    if orig_match or mod_match or ev1_match or ev2_match:
                        found = True

                if found:
                    matched_gt_ids.add(gt_idx)
                    matched_acu_indices.add(acu_idx)
                    break

        result.gt_changes_detected = len(matched_gt_ids)
        result.pipeline_changes_matched = len(matched_acu_indices)

        # Per-category breakdown (done once at pair level)
        for gt_idx, gt_change in enumerate(gt_changes):
            cat = gt_change.get("category", "?")
            if cat not in result.per_category:
                result.per_category[cat] = {"gt_total": 0, "detected": 0}
            result.per_category[cat]["gt_total"] += 1
            if gt_idx in matched_gt_ids:
                result.per_category[cat]["detected"] += 1

        result.total_acus_generated = total_acus_passed + total_acus_rejected
        result.total_acus_passed = total_acus_passed
        result.total_acus_rejected = total_acus_rejected
        if result.total_acus_generated > 0:
            result.hallucination_rate = total_acus_rejected / result.total_acus_generated

        result.phase3_time = time.perf_counter() - t0
        logger.info("[%s] Phase 3 done: %d ACUs (%d passed, %d rejected) (%.1fs)",
                    pair_info["name"], result.total_acus_generated,
                    result.total_acus_passed, result.total_acus_rejected,
                    result.phase3_time)

    except Exception as e:
        result.errors.append(f"Phase 3 failed: {e}")
        logger.exception("[%s] Phase 3 error", pair_info["name"])
        result.phase3_time = time.perf_counter() - t0

    result.total_time = time.perf_counter() - t_total_start
    return result


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_aggregate_metrics(results: list[PairEvalResult]) -> dict[str, Any]:
    """Compute aggregate metrics across all pairs."""
    n = len(results)
    if n == 0:
        return {"error": "No results"}

    # Summations
    total_v1_articles = sum(r.v1_articles for r in results)
    total_v2_articles = sum(r.v2_articles for r in results)
    total_gt_changes = sum(r.gt_total_changes for r in results)
    total_gt_detected = sum(r.gt_changes_detected for r in results)
    total_pipe_changes = sum(r.pipeline_changes_total for r in results)
    total_pipe_matched = sum(r.pipeline_changes_matched for r in results)
    total_acus_gen = sum(r.total_acus_generated for r in results)
    total_acus_passed = sum(r.total_acus_passed for r in results)
    total_acus_rejected = sum(r.total_acus_rejected for r in results)

    # Alignment stats
    total_matched = sum(r.aligned_matched for r in results)
    total_added = sum(r.aligned_added for r in results)
    total_deleted = sum(r.aligned_deleted for r in results)
    total_split = sum(r.aligned_split for r in results)
    total_merged = sum(r.aligned_merged for r in results)

    # Change detection recall/precision/F1
    recall = total_gt_detected / total_gt_changes if total_gt_changes > 0 else 0.0
    precision = total_pipe_matched / total_pipe_changes if total_pipe_changes > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Hallucination rate
    hal_rate = total_acus_rejected / total_acus_gen if total_acus_gen > 0 else 0.0

    # Per-category aggregation
    per_cat_agg: dict[str, dict] = {}
    for r in results:
        for cat, stats in r.per_category.items():
            if cat not in per_cat_agg:
                per_cat_agg[cat] = {"gt_total": 0, "detected": 0}
            per_cat_agg[cat]["gt_total"] += stats["gt_total"]
            per_cat_agg[cat]["detected"] += stats["detected"]

    for cat, stats in per_cat_agg.items():
        stats["recall"] = (stats["detected"] / stats["gt_total"]
                           if stats["gt_total"] > 0 else 0.0)

    # Timing
    total_time = sum(r.total_time for r in results)
    avg_time = total_time / n if n > 0 else 0.0

    # Pairs with errors
    error_pairs = [r.pair_name for r in results if r.errors]
    # A pair actually ran Phase 3 only if phase3_time > 0 (not just "didn't error").
    # Pairs failing at Phase 2 return early and never reach Phase 3.
    phase3_pairs = sum(1 for r in results if r.phase3_time > 0)

    return {
        "num_pairs": n,
        "num_pairs_with_phase3": phase3_pairs,
        "num_pairs_with_errors": len(error_pairs),
        "error_pairs": error_pairs,

        # Ingestion
        "total_v1_articles": total_v1_articles,
        "total_v2_articles": total_v2_articles,
        "avg_v1_articles": total_v1_articles / n,
        "avg_v2_articles": total_v2_articles / n,

        # Alignment
        "total_aligned_matched": total_matched,
        "total_aligned_added": total_added,
        "total_aligned_deleted": total_deleted,
        "total_aligned_split": total_split,
        "total_aligned_merged": total_merged,
        "avg_matched_per_pair": total_matched / n,

        # Ground truth
        "total_gt_changes": total_gt_changes,
        "avg_gt_changes_per_pair": total_gt_changes / n,

        # Change detection
        "change_recall": round(recall, 4),
        "change_precision": round(precision, 4),
        "change_f1": round(f1, 4),
        "gt_changes_detected": total_gt_detected,
        "pipeline_changes_total": total_pipe_changes,
        "pipeline_changes_matched": total_pipe_matched,

        # ACU quality
        "total_acus_generated": total_acus_gen,
        "total_acus_passed": total_acus_passed,
        "total_acus_rejected": total_acus_rejected,
        "hallucination_rate": round(hal_rate, 4),

        # Per-category
        "per_category": per_cat_agg,

        # Timing
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_pair_seconds": round(avg_time, 1),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def print_aggregate_report(metrics: dict[str, Any]) -> None:
    """Print a formatted aggregate report."""
    print(f"\n{DIVIDER}")
    print("📊  L-RAG FULL PIPELINE EVALUATION — AGGREGATE REPORT")
    print(DIVIDER)

    print(f"\n{'─' * 50}")
    print("📋  OVERVIEW")
    print(f"{'─' * 50}")
    print(f"  Pairs evaluated        : {metrics['num_pairs']}")
    print(f"  Pairs with Phase 3     : {metrics['num_pairs_with_phase3']}")
    print(f"  Pairs with errors      : {metrics['num_pairs_with_errors']}")
    if metrics["error_pairs"]:
        for ep in metrics["error_pairs"]:
            print(f"    ❌ {ep}")

    print(f"\n{'─' * 50}")
    print("📄  PHASE 1 — INGESTION")
    print(f"{'─' * 50}")
    print(f"  Total V1 articles      : {metrics['total_v1_articles']}")
    print(f"  Total V2 articles      : {metrics['total_v2_articles']}")
    print(f"  Avg V1 articles/pair   : {metrics['avg_v1_articles']:.1f}")
    print(f"  Avg V2 articles/pair   : {metrics['avg_v2_articles']:.1f}")

    print(f"\n{'─' * 50}")
    print("🔗  PHASE 2 — ALIGNMENT")
    print(f"{'─' * 50}")
    print(f"  Matched pairs          : {metrics['total_aligned_matched']}")
    print(f"  Added (V2 only)        : {metrics['total_aligned_added']}")
    print(f"  Deleted (V1 only)      : {metrics['total_aligned_deleted']}")
    print(f"  Split cases            : {metrics['total_aligned_split']}")
    print(f"  Merged cases           : {metrics['total_aligned_merged']}")
    print(f"  Avg matched/pair       : {metrics['avg_matched_per_pair']:.1f}")

    print(f"\n{'─' * 50}")
    print("🎯  CHANGE DETECTION vs GROUND TRUTH")
    print(f"{'─' * 50}")
    print(f"  Total GT changes       : {metrics['total_gt_changes']}")
    print(f"  GT changes detected    : {metrics['gt_changes_detected']}")
    print(f"  Pipeline ACUs total    : {metrics['pipeline_changes_total']}")
    print(f"  Pipeline ACUs matched  : {metrics['pipeline_changes_matched']}")
    print(f"")
    print(f"  ⭐ RECALL              : {metrics['change_recall']:.2%}")
    print(f"  ⭐ PRECISION            : {metrics['change_precision']:.2%}")
    print(f"  ⭐ F1 SCORE             : {metrics['change_f1']:.2%}")

    print(f"\n{'─' * 50}")
    print("✅  ACU QUALITY (Phase 3)")
    print(f"{'─' * 50}")
    print(f"  Total ACUs generated   : {metrics['total_acus_generated']}")
    print(f"  ACUs passed            : {metrics['total_acus_passed']}")
    print(f"  ACUs rejected          : {metrics['total_acus_rejected']}")
    print(f"  Hallucination rate     : {metrics['hallucination_rate']:.2%}")

    if metrics.get("per_category"):
        print(f"\n{'─' * 50}")
        print("🏷️   PER-CATEGORY BREAKDOWN")
        print(f"{'─' * 50}")
        print(f"  {'Category':<8s} {'Description':<30s} {'GT':>5s} {'Detected':>8s} {'Recall':>8s}")
        print(f"  {'─' * 8} {'─' * 30} {'─' * 5} {'─' * 8} {'─' * 8}")
        for cat in sorted(metrics["per_category"].keys()):
            stats = metrics["per_category"][cat]
            desc = CATEGORY_MAP.get(cat, "Unknown")
            print(f"  {cat:<8s} {desc:<30s} {stats['gt_total']:>5d} {stats['detected']:>8d} {stats['recall']:>7.1%}")

    print(f"\n{'─' * 50}")
    print("⏱️   TIMING")
    print(f"{'─' * 50}")
    print(f"  Total wall time        : {metrics['total_time_seconds']:.1f}s")
    print(f"  Avg time per pair      : {metrics['avg_time_per_pair_seconds']:.1f}s")

    print(f"\n{DIVIDER}")
    print("🏁  EVALUATION COMPLETE")
    print(DIVIDER)


def print_per_pair_summary(results: list[PairEvalResult]) -> None:
    """Print per-pair summary table."""
    print(f"\n{'─' * 80}")
    print("📋  PER-PAIR SUMMARY")
    print(f"{'─' * 80}")
    header = (f"  {'Pair':<30s} {'V1Art':>5s} {'V2Art':>5s} {'Matched':>7s} "
              f"{'GTChg':>5s} {'Recall':>7s} {'Prec':>6s} {'F1':>6s} {'ACUs':>5s} {'Hal%':>6s} {'Time':>7s}")
    print(header)
    print(f"  {'─' * 30} {'─' * 5} {'─' * 5} {'─' * 7} {'─' * 5} {'─' * 7} {'─' * 6} {'─' * 6} {'─' * 5} {'─' * 6} {'─' * 7}")

    for r in results:
        gt_total = r.gt_total_changes
        recall = r.gt_changes_detected / gt_total if gt_total > 0 else 0.0
        prec = (r.pipeline_changes_matched / r.pipeline_changes_total
                if r.pipeline_changes_total > 0 else 0.0)
        f1 = 2 * recall * prec / (recall + prec) if (recall + prec) > 0 else 0.0
        hal = r.hallucination_rate * 100

        name = r.pair_name[:28]
        print(f"  {name:<30s} {r.v1_articles:>5d} {r.v2_articles:>5d} {r.aligned_matched:>7d} "
              f"{gt_total:>5d} {recall:>6.1%} {prec:>5.1%} {f1:>5.2f} "
              f"{r.total_acus_passed:>5d} {hal:>5.1f}% {r.total_time:>6.1f}s")

        if r.errors:
            for e in r.errors:
                print(f"    ⚠️  {e}")

    print(f"  {'─' * 30} {'─' * 5} {'─' * 5} {'─' * 7} {'─' * 5} {'─' * 7} {'─' * 6} {'─' * 6} {'─' * 5} {'─' * 6} {'─' * 7}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    """Async main entry point."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover pairs
    if args.pair:
        # Single pair mode
        v1_path = data_dir / f"{args.pair}_v1.docx"
        v2_path = data_dir / f"{args.pair}_v2.docx"
        gt_path = data_dir / f"{args.pair}.json"
        if not all(p.exists() for p in [v1_path, v2_path, gt_path]):
            print(f"❌ Pair '{args.pair}' not found in {data_dir}")
            sys.exit(1)
        pairs = [{"name": args.pair, "v1": str(v1_path), "v2": str(v2_path), "gt": str(gt_path)}]
    else:
        pairs = discover_pairs(data_dir)

    if not pairs:
        print(f"❌ No document pairs found in {data_dir}")
        sys.exit(1)

    print(f"\n{DIVIDER}")
    print(f"🚀  L-RAG FULL PIPELINE EVALUATION")
    print(f"{DIVIDER}")
    print(f"  Data directory : {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Pairs to eval  : {len(pairs)}")
    print(f"  Skip Phase 3   : {args.skip_phase3}")
    print(f"  Max pairs      : {args.max_pairs or 'all'}")
    print(DIVIDER)

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    # ── Initialize shared components (created ONCE) ──────────────────
    # Phase 1: Parser + Chunker are stateless — safe to reuse.
    # Phase 2: BGEM3Manager loads the BGE-M3 model (~2 GB VRAM).
    # Creating it per pair would reload the model N times, wasting
    # time and VRAM. AlignmentConfig is also shared.
    from src.ingestion.docling_parser import LegalDocumentParser
    from src.ingestion.lsu_chunker import LsuChunker
    from src.alignment.embedder import BGEM3Manager
    from src.alignment.similarity_matrix import AlignmentConfig
    from src.config import get_config

    cfg = get_config()
    parser = LegalDocumentParser(
        confidence_threshold=cfg.get("ingestion", {}).get("confidence_threshold", 0.75)
    )
    chunker = LsuChunker(
        max_chunk_chars=cfg.get("ingestion", {}).get("max_chunk_chars", 2000),
        overlap_chars=cfg.get("ingestion", {}).get("overlap_chars", 200),
    )
    embed_manager = BGEM3Manager()
    acfg = cfg["alignment"]
    alignment_config = AlignmentConfig(
        w_semantic=acfg["w_semantic"],
        w_jaro_winkler=acfg["w_jaro_winkler"],
        w_ordinal=acfg["w_ordinal"],
        match_threshold=acfg["match_threshold"],
        split_merge_threshold=acfg["split_merge_threshold"],
    )

    # ── Run evaluation ──────────────────────────────────────────────
    results: list[PairEvalResult] = []
    per_pair_outputs: dict[str, Any] = {}

    for i, pair_info in enumerate(pairs, 1):
        print(f"\n{'▼' * 60}")
        print(f"  [{i}/{len(pairs)}] {pair_info['name']}")
        print(f"{'▼' * 60}")

        result = await evaluate_pair(
            pair_info=pair_info,
            output_dir=output_dir,
            parser=parser,
            chunker=chunker,
            embed_manager=embed_manager,
            alignment_config=alignment_config,
            cfg=cfg,
            skip_phase3=args.skip_phase3,
        )
        results.append(result)

        # Save per-pair detailed output
        pair_output = {
            "pair_name": result.pair_name,
            "v1_file": result.v1_file,
            "v2_file": result.v2_file,
            "gt_file": result.gt_file,
            "phase1": {
                "v1_articles": result.v1_articles,
                "v2_articles": result.v2_articles,
                "v1_chunks": result.v1_chunks,
                "v2_chunks": result.v2_chunks,
                "time": result.phase1_time,
            },
            "phase2": {
                "matched": result.aligned_matched,
                "added": result.aligned_added,
                "deleted": result.aligned_deleted,
                "split": result.aligned_split,
                "merged": result.aligned_merged,
                "time": result.phase2_time,
            },
            "phase3": {
                "skipped": result.skipped_phase3,
                "acus_generated": result.total_acus_generated,
                "acus_passed": result.total_acus_passed,
                "acus_rejected": result.total_acus_rejected,
                "hallucination_rate": result.hallucination_rate,
                "time": result.phase3_time,
            },
            "ground_truth": {
                "total_changes": result.gt_total_changes,
                "categories": result.gt_categories,
            },
            "change_detection": {
                "gt_detected": result.gt_changes_detected,
                "pipeline_total": result.pipeline_changes_total,
                "pipeline_matched": result.pipeline_changes_matched,
                "recall": (result.gt_changes_detected / result.gt_total_changes
                           if result.gt_total_changes > 0 else 0.0),
                "precision": (result.pipeline_changes_matched / result.pipeline_changes_total
                              if result.pipeline_changes_total > 0 else 0.0),
            },
            "per_category": result.per_category,
            "total_time": result.total_time,
            "errors": result.errors,
        }
        per_pair_outputs[result.pair_name] = pair_output

    # ── Compute aggregate metrics ──────────────────────────────────
    metrics = compute_aggregate_metrics(results)

    # Add per-category aggregated recall
    for cat, stats in metrics.get("per_category", {}).items():
        if "recall" not in pair_output.get("change_detection", {}):
            pass
    metrics["per_pair"] = {
        r.pair_name: {
            "recall": (r.gt_changes_detected / r.gt_total_changes
                       if r.gt_total_changes > 0 else 0.0),
            "precision": (r.pipeline_changes_matched / r.pipeline_changes_total
                          if r.pipeline_changes_total > 0 else 0.0),
            "f1": (lambda rec, prec: 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0)(
                r.gt_changes_detected / r.gt_total_changes if r.gt_total_changes > 0 else 0.0,
                r.pipeline_changes_matched / r.pipeline_changes_total if r.pipeline_changes_total > 0 else 0.0
            ),
            "hallucination_rate": r.hallucination_rate,
            "errors": r.errors,
        }
        for r in results
    }

    # ── Print reports ─────────────────────────────────────────────
    print_per_pair_summary(results)
    print_aggregate_report(metrics)

    # ── Save results ──────────────────────────────────────────────
    # Per-pair details
    per_pair_path = output_dir / "per_pair_results.json"
    with open(per_pair_path, "w", encoding="utf-8") as f:
        json.dump(per_pair_outputs, f, ensure_ascii=False, indent=2)
    print(f"\n📁 Per-pair results saved to: {per_pair_path}")

    # Aggregate metrics
    metrics_path = output_dir / "aggregate_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"📁 Aggregate metrics saved to: {metrics_path}")

    # Markdown summary
    md_path = output_dir / "evaluation_report.md"
    _write_markdown_report(metrics, results, md_path)
    print(f"📁 Markdown report saved to: {md_path}")


def _write_markdown_report(
    metrics: dict[str, Any],
    results: list[PairEvalResult],
    output_path: Path,
) -> None:
    """Write a Markdown evaluation report."""
    from datetime import datetime

    lines = []
    lines.append("# L-RAG Pipeline Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Pairs evaluated:** {metrics['num_pairs']}")
    lines.append("")

    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total document pairs | {metrics['num_pairs']} |")
    lines.append(f"| Pairs with Phase 3 (LLM) | {metrics['num_pairs_with_phase3']} |")
    lines.append(f"| Pairs with errors | {metrics['num_pairs_with_errors']} |")
    lines.append(f"| Total wall time | {metrics['total_time_seconds']:.1f}s |")
    lines.append(f"| Avg time per pair | {metrics['avg_time_per_pair_seconds']:.1f}s |")
    lines.append("")

    lines.append("## 2. Phase 1 — Ingestion")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total V1 articles | {metrics['total_v1_articles']} |")
    lines.append(f"| Total V2 articles | {metrics['total_v2_articles']} |")
    lines.append(f"| Avg V1 articles/pair | {metrics['avg_v1_articles']:.1f} |")
    lines.append(f"| Avg V2 articles/pair | {metrics['avg_v2_articles']:.1f} |")
    lines.append("")

    lines.append("## 3. Phase 2 — Alignment")
    lines.append("")
    lines.append("| Match Type | Count | Avg/Pair |")
    lines.append("|------------|-------|----------|")
    n = metrics['num_pairs']
    lines.append(f"| Matched | {metrics['total_aligned_matched']} | {metrics['total_aligned_matched'] / n:.1f} |")
    lines.append(f"| Added (V2 only) | {metrics['total_aligned_added']} | {metrics['total_aligned_added'] / n:.1f} |")
    lines.append(f"| Deleted (V1 only) | {metrics['total_aligned_deleted']} | {metrics['total_aligned_deleted'] / n:.1f} |")
    lines.append(f"| Split | {metrics['total_aligned_split']} | {metrics['total_aligned_split'] / n:.1f} |")
    lines.append(f"| Merged | {metrics['total_aligned_merged']} | {metrics['total_aligned_merged'] / n:.1f} |")
    lines.append("")

    lines.append("## 4. Change Detection vs Ground Truth")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total GT changes | {metrics['total_gt_changes']} |")
    lines.append(f"| GT changes detected | {metrics['gt_changes_detected']} |")
    lines.append(f"| Pipeline ACUs (matched) | {metrics['pipeline_changes_matched']} |")
    lines.append(f"| Pipeline ACUs (total passed) | {metrics['pipeline_changes_total']} |")
    lines.append(f"| **Recall** | **{metrics['change_recall']:.2%}** |")
    lines.append(f"| **Precision** | **{metrics['change_precision']:.2%}** |")
    lines.append(f"| **F1 Score** | **{metrics['change_f1']:.2%}** |")
    lines.append("")

    lines.append("## 5. ACU Quality")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total ACUs generated | {metrics['total_acus_generated']} |")
    lines.append(f"| ACUs passed verification | {metrics['total_acus_passed']} |")
    lines.append(f"| ACUs rejected (hallucination) | {metrics['total_acus_rejected']} |")
    lines.append(f"| **Hallucination Rate** | **{metrics['hallucination_rate']:.2%}** |")
    lines.append("")

    if metrics.get("per_category"):
        lines.append("## 6. Per-Category Breakdown")
        lines.append("")
        lines.append("| Category | Description | GT Count | Detected | Recall |")
        lines.append("|----------|-------------|----------|----------|--------|")
        for cat in sorted(metrics["per_category"].keys()):
            stats = metrics["per_category"][cat]
            desc = CATEGORY_MAP.get(cat, "Unknown")
            lines.append(f"| {cat} | {desc} | {stats['gt_total']} | {stats['detected']} | {stats['recall']:.1%} |")
        lines.append("")

    lines.append("## 7. Per-Pair Details")
    lines.append("")
    lines.append("| Pair | V1 Art | V2 Art | Matched | GT Chg | Recall | Prec | F1 | ACUs | Hal% | Time |")
    lines.append("|------|--------|--------|---------|--------|--------|------|----|------|------|------|")
    for r in results:
        gt_total = r.gt_total_changes
        recall = r.gt_changes_detected / gt_total if gt_total > 0 else 0.0
        prec = (r.pipeline_changes_matched / r.pipeline_changes_total
                if r.pipeline_changes_total > 0 else 0.0)
        f1 = 2 * recall * prec / (recall + prec) if (recall + prec) > 0 else 0.0
        hal = r.hallucination_rate * 100
        name = r.pair_name[:30]
        lines.append(f"| {name} | {r.v1_articles} | {r.v2_articles} | {r.aligned_matched} | "
                     f"{gt_total} | {recall:.1%} | {prec:.1%} | {f1:.2f} | "
                     f"{r.total_acus_passed} | {hal:.1f}% | {r.total_time:.0f}s |")
        if r.errors:
            for e in r.errors:
                lines.append(f"| | ⚠️ {e} | | | | | | | | | |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Report generated by L-RAG evaluation pipeline._")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L-RAG Full Pipeline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/run_full_eval.py                          # All pairs, full pipeline
  python evaluation/run_full_eval.py --skip-phase3             # Alignment only (fast)
  python evaluation/run_full_eval.py --pair 01-tand_signed     # Single pair
  python evaluation/run_full_eval.py --max-pairs 5             # First 5 pairs only
  python evaluation/run_full_eval.py --output-dir ./my_results
        """,
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help=f"Directory containing docx pairs and ground truth JSONs (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Directory to save evaluation results (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip Phase 3 (Generative Comparison). Only run Phase 1+2.",
    )
    parser.add_argument(
        "--pair",
        default=None,
        help="Evaluate a single pair by base name (e.g., '01-tand_signed')",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit to first N pairs (for quick testing)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
