#!/usr/bin/env python3
"""
evaluate_phase2.py
==================
Script đánh giá từng module của Phase 2 — Indexing & Alignment Strategy.

Cách chạy:
    # Test models (không cần GPU)
    python evaluate_phase2.py --module models

    # Test Qdrant indexer (local in-memory)
    python evaluate_phase2.py --module qdrant

    # Test embedding (cần GPU/CPU + model BAAI/bge-m3)
    python evaluate_phase2.py --module embed --model BAAI/bge-m3

    # Full alignment pipeline (end-to-end với 2 file docx)
    python evaluate_phase2.py --module align \\
        --v1 data_test/01-tand_signed_v1.docx \\
        --v2 docs_test/v1.docx

    # Chỉ test alignment với mock data (không cần GPU)
    python evaluate_phase2.py --module align --mock

    # Xuất kết quả ra JSON
    python evaluate_phase2.py --module align --mock --output eval_output/phase2_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Setup logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_phase2")


# ---------------------------------------------------------------------------
# Helper: print section header
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def fail(msg: str) -> None:
    print(f"  ❌ {msg}")


# ---------------------------------------------------------------------------
# Module 1: Test comparison/models.py
# ---------------------------------------------------------------------------


def test_models() -> bool:
    header("Module test: comparison/models.py")

    try:
        from comparison.models import (
            DiffPair,
            DiffPairCatalog,
            MatchType,
            NodeEmbeddings,
            NodeVersion,
            QdrantPayload,
        )

        ok("Import thành công")

        # --- Test QdrantPayload ---
        payload = QdrantPayload(
            node_id="article_abc123",
            doc_id="doc_v1_001",
            version=NodeVersion.V1,
            node_type="article",
            ordinal=0,
            raw_text="Bên A có quyền yêu cầu thanh toán theo hợp đồng.",
            title="Điều 5. Quyền của Bên A",
        )
        d = payload.to_dict()
        assert d["version"] == "v1", f"Expected 'v1', got {d['version']}"
        assert "ordinal" in d
        assert "raw_text" in d
        ok(f"QdrantPayload.to_dict() OK → version='{d['version']}', ordinal={d['ordinal']}")

        # --- Test DiffPair: matched ---
        p_matched = DiffPair(
            v1_ids=["article_001"],
            v2_ids=["article_002"],
            match_type=MatchType.MATCHED,
            confidence_score=0.92,
            semantic_score=0.95,
            jaro_winkler_score=0.88,
            ordinal_proximity_score=0.90,
        )
        ok(f"DiffPair MATCHED: confidence={p_matched.confidence_score}")

        # --- Test DiffPair: split ---
        p_split = DiffPair(
            v1_ids=["article_003"],
            v2_ids=["article_004", "article_005"],
            match_type=MatchType.SPLIT,
            confidence_score=0.81,
        )
        ok(f"DiffPair SPLIT: 1→{len(p_split.v2_ids)} nodes")

        # --- Test DiffPair: merged ---
        p_merged = DiffPair(
            v1_ids=["article_006", "article_007"],
            v2_ids=["article_008"],
            match_type=MatchType.MERGED,
            confidence_score=0.85,
        )
        ok(f"DiffPair MERGED: {len(p_merged.v1_ids)}→1 nodes")

        # --- Test validation failure ---
        try:
            DiffPair(
                v1_ids=["a", "b"],  # match không thể có 2 v1
                v2_ids=["c"],
                match_type=MatchType.MATCHED,
                confidence_score=0.9,
            )
            fail("Phải raise ValidationError nhưng không raise!")
            return False
        except Exception:
            ok("Validation đúng: MATCHED với 2 v1_ids bị reject")

        # --- Test DiffPairCatalog ---
        catalog = DiffPairCatalog(
            v1_doc_id="doc_v1_001",
            v2_doc_id="doc_v2_001",
        )
        catalog.pairs.extend([p_matched, p_split, p_merged])
        catalog.pairs.append(
            DiffPair(v1_ids=["del_001"], match_type=MatchType.DELETED, confidence_score=0.0)
        )
        catalog.pairs.append(
            DiffPair(v2_ids=["add_001"], match_type=MatchType.ADDED, confidence_score=0.0)
        )

        summary = catalog.summary()
        assert summary["matched"] == 1, f"Expected 1 matched, got {summary['matched']}"
        assert summary["split"] == 1
        assert summary["merged"] == 1
        assert summary["deleted"] == 1
        assert summary["added"] == 1
        ok(f"DiffPairCatalog.summary() → {summary}")

        print()
        ok("✅ TẤT CẢ MODELS TEST PASSED")
        return True

    except Exception as e:
        fail(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Module 2: Test Qdrant Indexer (in-memory)
# ---------------------------------------------------------------------------


def test_qdrant() -> bool:
    header("Module test: comparison/qdrant_indexer.py (in-memory mode)")

    try:
        import numpy as np
        from comparison.qdrant_indexer import QdrantCollectionConfig, QdrantManager
        from comparison.models import NodeEmbeddings, NodeVersion, QdrantPayload

        manager = QdrantManager()  # in-memory
        ok("QdrantManager khởi tạo (in-memory) OK")

        # --- Tạo collection ---
        config = QdrantCollectionConfig(collection_name="test_legal_phase2")
        manager.create_collection(config, recreate_if_exists=True)
        ok("create_collection() OK")

        assert manager.collection_exists("test_legal_phase2")
        ok("collection_exists() OK")

        # --- Tạo mock embeddings ---
        rng = np.random.default_rng(42)
        dim = 1024

        def make_emb(node_id: str, version: NodeVersion, ordinal: int) -> NodeEmbeddings:
            struct_vec = rng.random(dim).astype(np.float32)
            sem_vec = rng.random(dim).astype(np.float32)
            # L2 normalize
            struct_vec /= np.linalg.norm(struct_vec)
            sem_vec /= np.linalg.norm(sem_vec)

            return NodeEmbeddings(
                node_id=node_id,
                structural_dense=struct_vec.tolist(),
                semantic_dense=sem_vec.tolist(),
                structural_sparse={100: 0.5, 200: 0.3, 300: 0.8},
                semantic_sparse={150: 0.6, 250: 0.4},
                payload=QdrantPayload(
                    node_id=node_id,
                    doc_id=f"doc_{version.value}_001",
                    version=version,
                    node_type="article",
                    ordinal=ordinal,
                    raw_text=f"Nội dung điều {ordinal + 1} phiên bản {version.value}",
                    title=f"Điều {ordinal + 1}",
                ),
            )

        embs_v1 = [make_emb(f"article_v1_{i:03d}", NodeVersion.V1, i) for i in range(5)]
        embs_v2 = [make_emb(f"article_v2_{i:03d}", NodeVersion.V2, i) for i in range(4)]

        # --- Upsert ---
        n = manager.upsert_embeddings("test_legal_phase2", embs_v1 + embs_v2)
        assert n == 9, f"Expected 9, got {n}"
        ok(f"upsert_embeddings() OK → {n} points")

        # --- Get info ---
        info = manager.get_collection_info("test_legal_phase2")
        ok(f"Collection info: points={info['points_count']}, vectors={info['vectors_count']}")

        # --- Search ---
        query_vec = rng.random(dim).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)
        results = manager.search_by_semantic(
            "test_legal_phase2",
            query_vector=query_vec.tolist(),
            version_filter="v1",
            top_k=3,
        )
        assert len(results) <= 3
        ok(f"search_by_semantic() (version=v1) → {len(results)} kết quả")

        # --- Scroll all V2 ---
        all_v2 = manager.get_all_points("test_legal_phase2", version_filter="v2", with_vectors=False)
        assert len(all_v2) == 4, f"Expected 4 V2 records, got {len(all_v2)}"
        ok(f"get_all_points(version=v2) → {len(all_v2)} records (sorted by ordinal)")

        # Verify ordinal sort
        ordinals = [r.payload.get("ordinal", -1) for r in all_v2 if r.payload]
        assert ordinals == sorted(ordinals), f"Ordinal không được sort: {ordinals}"
        ok(f"Ordinal sorting OK: {ordinals}")

        # --- Delete collection ---
        manager.delete_collection("test_legal_phase2")
        assert not manager.collection_exists("test_legal_phase2")
        ok("delete_collection() OK")

        print()
        ok("✅ TẤT CẢ QDRANT TESTS PASSED")
        return True

    except Exception as e:
        fail(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Module 3: Test Alignment Engine với mock data (không cần GPU)
# ---------------------------------------------------------------------------


def test_alignment_mock() -> bool:
    header("Module test: alignment_engine.py (MOCK mode, không cần GPU)")

    try:
        import numpy as np
        from unittest.mock import MagicMock, patch

        from comparison.alignment_engine import AlignmentConfig, LegalAlignmentEngine
        from comparison.models import MatchType

        # --- Test AlignmentConfig validation ---
        cfg = AlignmentConfig()
        ok(f"AlignmentConfig default: w_sem={cfg.w_semantic}, w_jaro={cfg.w_jaro_winkler}, w_ord={cfg.w_ordinal}")

        try:
            AlignmentConfig(w_semantic=0.5, w_jaro_winkler=0.5, w_ordinal=0.5)
            fail("Phải raise ValueError khi tổng trọng số != 1.0")
            return False
        except ValueError:
            ok("AlignmentConfig validation OK: bắt tổng trọng số != 1.0")

        # --- Mock BGEM3Manager ---
        mock_emb = MagicMock()
        rng = np.random.default_rng(99)
        dim = 1024

        def mock_embed_semantic(texts: list[str]) -> np.ndarray:
            """Mock embedding: mỗi text → random unit vector (reproducible by index)."""
            vecs = []
            for i, t in enumerate(texts):
                # Tạo vector dựa trên nội dung text để mô phỏng similarity
                seed = sum(ord(c) for c in t[:20]) % 1000
                local_rng = np.random.default_rng(seed)
                v = local_rng.random(dim).astype(np.float32)
                v /= np.linalg.norm(v)
                vecs.append(v)
            return np.stack(vecs, axis=0)

        mock_emb.embed_texts_semantic.side_effect = mock_embed_semantic

        engine = LegalAlignmentEngine(embed_manager=mock_emb, config=cfg)
        ok("LegalAlignmentEngine khởi tạo với mock embed OK")

        # --- Test Similarity Matrix ---
        from comparison.alignment_engine import NodeRecord

        v1_records = [
            NodeRecord(node_id=f"v1_art_{i}", title=f"Điều {i+1}. Tiêu đề {i+1}",
                       raw_text=f"Nội dung điều {i+1} từ phiên bản V1", ordinal=i)
            for i in range(4)
        ]
        v2_records = [
            NodeRecord(node_id=f"v2_art_{i}", title=f"Điều {i+1}. Tiêu đề {i+1} sửa đổi",
                       raw_text=f"Nội dung điều {i+1} từ phiên bản V2", ordinal=i)
            for i in range(3)
        ]

        # Inject mock vectors
        for r in v1_records:
            r.semantic_vec = mock_embed_semantic([r.raw_text])[0]
        for r in v2_records:
            r.semantic_vec = mock_embed_semantic([r.raw_text])[0]

        sim_matrix = engine.compute_similarity_matrix(v1_records, v2_records)
        assert sim_matrix.shape == (4, 3), f"Expected (4,3), got {sim_matrix.shape}"
        assert np.all(sim_matrix >= 0) and np.all(sim_matrix <= 1), "Giá trị ngoài [0,1]"
        ok(f"compute_similarity_matrix() → shape={sim_matrix.shape}, "
           f"range=[{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")

        # In matrix để debug
        print("\n  Similarity Matrix (4×3):")
        for i, row in enumerate(sim_matrix):
            vals = " | ".join(f"{v:.3f}" for v in row)
            print(f"    V1[{i}] → [{vals}]")

        # --- Test Hungarian Matching ---
        matched, v1_unm, v2_unm = engine.hungarian_match(sim_matrix)
        ok(f"hungarian_match() → {len(matched)} matched, "
           f"V1 unmatched={v1_unm}, V2 unmatched={v2_unm}")

        # --- Test _run_alignment (full pipeline, mock embed already injected) ---
        # Inject vectors vào records mới
        v1_new = [
            NodeRecord(node_id=f"v1_n_{i}", title=f"Điều {i+1}",
                       raw_text=f"Quyền của Bên A khoản {i+1}", ordinal=i)
            for i in range(3)
        ]
        v2_new = [
            NodeRecord(node_id=f"v2_n_{i}", title=f"Điều {i+1}",
                       raw_text=f"Quyền của Bên A khoản {i+1}", ordinal=i)
            for i in range(3)
        ]
        for r in v1_new + v2_new:
            r.semantic_vec = mock_embed_semantic([r.raw_text])[0]

        pairs, _, _ = engine._run_alignment(v1_new, v2_new)
        matched_pairs = [p for p in pairs if p.match_type == MatchType.MATCHED]
        print(f"\n  _run_alignment(): {len(pairs)} tổng pairs, {len(matched_pairs)} matched")
        for p in pairs:
            print(f"    [{p.match_type.value:8s}] v1={p.v1_ids} → v2={p.v2_ids}, score={p.confidence_score:.3f}")
        ok("_run_alignment() OK")

        # --- Test edge cases ---
        # Empty V1
        pairs_empty, _, _ = engine._run_alignment([], v2_new)
        added = [p for p in pairs_empty if p.match_type == MatchType.ADDED]
        assert len(added) == len(v2_new)
        ok(f"Edge case (V1 rỗng): tất cả {len(added)} V2 nodes được đánh dấu ADDED")

        # Empty V2
        pairs_empty2, _, _ = engine._run_alignment(v1_new, [])
        deleted = [p for p in pairs_empty2 if p.match_type == MatchType.DELETED]
        assert len(deleted) == len(v1_new)
        ok(f"Edge case (V2 rỗng): tất cả {len(deleted)} V1 nodes được đánh dấu DELETED")

        print()
        ok("✅ TẤT CẢ ALIGNMENT MOCK TESTS PASSED")
        return True

    except Exception as e:
        fail(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Module 4: Full end-to-end alignment với real documents + real model
# ---------------------------------------------------------------------------


def test_alignment_e2e(v1_path: str, v2_path: str, output_path: str | None = None) -> bool:
    header(f"End-to-End Alignment: '{Path(v1_path).name}' ↔ '{Path(v2_path).name}'")

    try:
        from ingestion import ingest_document
        from ingestion.parser import LegalDocumentParser

        from comparison.alignment_engine import AlignmentConfig, LegalAlignmentEngine
        from comparison.embedding_manager import BGEM3Manager

        # --- Parse V1 ---
        print(f"\n  Parsing V1: {v1_path} ...")
        t0 = time.time()
        parser = LegalDocumentParser()
        doc_v1 = parser.parse(v1_path)
        t_parse_v1 = time.time() - t0
        arts_v1 = doc_v1.iter_all_articles()
        ok(f"V1 parsed: {len(arts_v1)} articles, {sum(len(a.clauses) for a in arts_v1)} clauses [{t_parse_v1:.1f}s]")

        # --- Parse V2 ---
        print(f"\n  Parsing V2: {v2_path} ...")
        t0 = time.time()
        doc_v2 = parser.parse(v2_path)
        t_parse_v2 = time.time() - t0
        arts_v2 = doc_v2.iter_all_articles()
        ok(f"V2 parsed: {len(arts_v2)} articles, {sum(len(a.clauses) for a in arts_v2)} clauses [{t_parse_v2:.1f}s]")

        # --- Load model ---
        print("\n  Loading BAAI/bge-m3 (FP16)...")
        t0 = time.time()
        emb_manager = BGEM3Manager(use_fp16=True, batch_size=8)
        t_load = time.time() - t0
        ok(f"Model loaded [{t_load:.1f}s]")

        # --- Run alignment ---
        cfg = AlignmentConfig(match_threshold=0.65, split_merge_threshold=0.80)
        engine = LegalAlignmentEngine(embed_manager=emb_manager, config=cfg)

        print("\n  Running alignment...")
        t0 = time.time()
        catalog = engine.align_documents(doc_v1, doc_v2)
        t_align = time.time() - t0
        ok(f"Alignment hoàn tất [{t_align:.1f}s]")

        # --- Print results ---
        summary = catalog.summary()
        print(f"\n  {'─' * 40}")
        print(f"  📊 KẾT QUẢ ALIGNMENT")
        print(f"  {'─' * 40}")
        print(f"  Tổng cặp:   {summary['total_pairs']}")
        print(f"  ✅ Matched:  {summary['matched']}")
        print(f"  ➕ Added:    {summary['added']}")
        print(f"  ➖ Deleted:  {summary['deleted']}")
        print(f"  ✂️  Split:    {summary['split']}")
        print(f"  🔗 Merged:   {summary['merged']}")

        # Print matched pairs details
        if catalog.matched_pairs:
            print(f"\n  Matched pairs (top 5):")
            for pair in catalog.matched_pairs[:5]:
                print(f"    [{pair.confidence_score:.3f}] {pair.v1_ids[0][:20]} ↔ {pair.v2_ids[0][:20]}")

        # --- Save output ---
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(catalog.to_report_dict(), f, ensure_ascii=False, indent=2, default=str)
            ok(f"Kết quả lưu tại: {output_path}")

        print()
        ok("✅ END-TO-END ALIGNMENT PASSED")
        return True

    except Exception as e:
        fail(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá Phase 2 — Indexing & Alignment Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--module",
        choices=["models", "qdrant", "align", "all"],
        default="all",
        help="Module cần test",
    )
    parser.add_argument("--mock", action="store_true", help="Chạy alignment với mock data (không cần GPU)")
    parser.add_argument("--v1", default="data_test/01-tand_signed_v1.docx", help="File V1")
    parser.add_argument("--v2", default="docs_test/v1.docx", help="File V2")
    parser.add_argument("--output", default=None, help="Lưu kết quả JSON ra file")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Tên model embedding")
    args = parser.parse_args()

    results: dict[str, bool] = {}
    t_total = time.time()

    run_models = args.module in ("models", "all")
    run_qdrant = args.module in ("qdrant", "all")
    run_align  = args.module in ("align", "all")

    if run_models:
        results["models"] = test_models()

    if run_qdrant:
        results["qdrant"] = test_qdrant()

    if run_align:
        if args.mock or args.module == "all":
            results["align_mock"] = test_alignment_mock()
        else:
            results["align_e2e"] = test_alignment_e2e(args.v1, args.v2, args.output)

    # --- Summary ---
    header("TỔNG KẾT")
    all_pass = True
    for name, passed in results.items():
        status = "PASS ✅" if passed else "FAIL ❌"
        print(f"  {name:<20} {status}")
        if not passed:
            all_pass = False

    print(f"\n  Thời gian tổng: {time.time() - t_total:.1f}s")

    if all_pass:
        print("\n  🎉 TẤT CẢ TESTS PASSED — Phase 2 sẵn sàng!")
    else:
        print("\n  ⚠️  Một số tests FAILED — kiểm tra lỗi ở trên.")
        sys.exit(1)


if __name__ == "__main__":
    main()
