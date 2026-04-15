"""
evaluate_phase1.py
==================
Script đánh giá toàn bộ Phase 1 Ingestion Pipeline.

Chạy lần lượt:
    1. Module 1 — LegalDocumentParser  → xem cấu trúc DOM được parse ra
    2. Module 2 — LsuChunker           → xem breadcrumb và nội dung từng chunk
    3. Module 3 — HybridGraphBuilder   → xem nodes/edges Kuzu + vectors ChromaDB

Usage:
    # Chạy với file mặc định (docs_test/v1.docx):
    python evaluate_phase1.py

    # Chạy với file cụ thể:
    python evaluate_phase1.py --file docs_test/v1.pdf

    # Chỉ test Module 1:
    python evaluate_phase1.py --module 1

    # Chỉ test Module 2:
    python evaluate_phase1.py --module 2

    # Chạy toàn bộ pipeline + hỏi thử ChromaDB:
    python evaluate_phase1.py --query "quyền của bên A trong hợp đồng"

    # Không lưu graph/vector DB (dry-run):
    python evaluate_phase1.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Giảm noise từ thư viện bên thứ ba
for noisy_logger in ["httpx", "httpcore", "urllib3", "PIL", "torch"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger("evaluate")

DIVIDER = "=" * 70
SECTION = "-" * 50


# ===========================================================================
# Module 1 Evaluation
# ===========================================================================


def run_module1(file_path: Path) -> "ingestion.models.LegalDocument":  # type: ignore
    """Parse tài liệu và in ra cấu trúc DOM."""
    from ingestion.parser import LegalDocumentParser
    from ingestion.models import LegalDocument

    print(f"\n{DIVIDER}")
    print("📄  MODULE 1 — LegalDocumentParser")
    print(DIVIDER)
    print(f"   File: {file_path}")

    t0 = time.perf_counter()
    parser = LegalDocumentParser(confidence_threshold=0.75)
    doc: LegalDocument = parser.parse(str(file_path))
    elapsed = time.perf_counter() - t0

    # ── Thống kê cơ bản ───────────────────────────────────────────────
    all_articles = doc.iter_all_articles()
    all_clause_pairs = doc.iter_all_clauses()
    all_tables = sum(len(a.tables) for a in all_articles)

    print(f"\n{'[RESULT]':>10} doc_id      : {doc.doc_id}")
    print(f"{'':>10} file_name   : {doc.file_name}")
    print(f"{'':>10} doc_title   : {doc.doc_title or '(không trích được)'}")
    print(f"{'':>10} doc_number  : {doc.doc_number or '(không trích được)'}")
    print(f"{'':>10} signing_date: {doc.signing_date or '(không trích được)'}")
    print(f"{'':>10} sections    : {len(doc.sections)}")
    print(f"{'':>10} articles    : {len(all_articles)}")
    print(f"{'':>10} clauses     : {len(all_clause_pairs)}")
    print(f"{'':>10} tables      : {all_tables}")
    print(f"{'':>10} engine      : {doc.quality_metrics.engine_used.value if doc.quality_metrics else 'unknown'}")
    if doc.quality_metrics:
        conf = doc.quality_metrics.avg_confidence
        print(f"{'':>10} avg_conf    : {f'{conf:.3f}' if conf is not None else 'N/A'}")
        print(f"{'':>10} ocr_trigger : {doc.quality_metrics.ocr_triggered}")
    print(f"{'':>10} parse_time  : {elapsed:.2f}s")

    # ── In cây DOM (5 điều đầu) ───────────────────────────────────────
    print(f"\n{SECTION}")
    print("   DOM TREE PREVIEW (5 điều đầu tiên):")
    print(SECTION)

    preview_articles = all_articles[:5]
    for art in preview_articles:
        print(f"\n  📌 {art.full_title}")
        if art.intro:
            intro_preview = art.intro[:120].replace("\n", " ")
            print(f"     intro : {intro_preview}{'...' if len(art.intro) > 120 else ''}")
        for clause in art.clauses[:3]:
            content_preview = clause.content[:100].replace("\n", " ")
            print(f"     ├── Khoản {clause.number}: {content_preview}{'...' if len(clause.content) > 100 else ''}")
            for point in clause.points[:2]:
                print(f"     │   ├── Điểm {point.number} {point.content[:60]}...")
        if art.tables:
            for tbl in art.tables[:1]:
                print(f"     📊 Table: {tbl.num_rows} rows × {tbl.num_cols} cols | headers: {tbl.headers[:3]}")

    if len(all_articles) > 5:
        print(f"\n  ... (và {len(all_articles) - 5} điều khác)")

    # ── Preamble ──────────────────────────────────────────────────────
    if doc.preamble:
        print(f"\n{SECTION}")
        print("   PREAMBLE (200 ký tự đầu):")
        print(SECTION)
        print(f"   {doc.preamble[:200]}...")

    # ── Cảnh báo nếu parse ra 0 điều ────────────────────────────────
    if len(all_articles) == 0:
        print(f"\n  ⚠️  CẢNH BÁO: Parser không nhận diện được Điều nào!")
        print(f"      Kiểm tra xem preamble có chứa text hợp lệ không.")
        print(f"      Thử --verbose để xem docling export markdown thô.")
        if doc.preamble:
            # In thêm 10 dòng preamble để debug
            for i, ln in enumerate(doc.preamble.splitlines()[:10], 1):
                print(f"      preamble[{i:02d}]: {repr(ln[:80])}")

    print(f"\n✅ Module 1 DONE — {elapsed:.2f}s")
    return doc


# ===========================================================================
# Module 2 Evaluation
# ===========================================================================


def run_module2(doc: "ingestion.models.LegalDocument") -> "list[ingestion.models.LsuChunk]":  # type: ignore
    """Chunk tài liệu và in thống kê + preview."""
    from ingestion.chunker import LsuChunker
    from ingestion.models import NodeType, ContentType

    print(f"\n{DIVIDER}")
    print("📦  MODULE 2 — LsuChunker")
    print(DIVIDER)

    t0 = time.perf_counter()
    chunker = LsuChunker(max_chunk_chars=2000, overlap_chars=200)
    chunks = chunker.chunk(doc)
    elapsed = time.perf_counter() - t0

    stats = chunker.get_stats(chunks)

    # ── Thống kê (xử lý cả trường hợp 0 chunks) ─────────────────────
    print(f"\n{'[RESULT]':>10} total_chunks  : {stats['total']}")
    if stats['total'] == 0:
        print(f"\n  ⚠️  CẢNH BÁO: Không tạo được chunk nào!")
        print(f"      Nguyên nhân có thể:")
        print(f"      1. Module 1 parse ra 0 điều (articles=0)")
        print(f"      2. File không có cấu trúc Điều/Khoản chuẩn")
        print(f"      3. Docling export sai format (thử --verbose để debug)")
    else:
        print(f"{'':>10} article_level : {stats.get('article_level', 0)}")
        print(f"{'':>10} clause_level  : {stats.get('clause_level', 0)}")
        print(f"{'':>10} with_tables   : {stats.get('with_tables', 0)}")
        print(f"{'':>10} avg_chars     : {stats.get('avg_chars', 0)}")
        print(f"{'':>10} min_chars     : {stats.get('min_chars', 0)}")
        print(f"{'':>10} max_chars     : {stats.get('max_chars', 0)}")
        print(f"{'':>10} content_types : {stats.get('content_types', {})}")
    print(f"{'':>10} chunk_time    : {elapsed:.2f}s")

    # ── Preview 8 chunks đầu ──────────────────────────────────────────
    print(f"\n{SECTION}")
    print("   CHUNK PREVIEW (8 chunks đầu tiên):")
    print(SECTION)

    for i, chunk in enumerate(chunks[:8]):
        node_icon = "📌" if chunk.source_node_type == NodeType.ARTICLE else "  📎"
        table_flag = " [TABLE]" if chunk.tables_json else ""
        print(f"\n  [{i+1}] {node_icon} {chunk.breadcrumb}")
        print(f"       node_id : {chunk.source_node_id}")
        print(f"       chars   : {chunk.char_count}{table_flag}")
        # In content_with_prefix (120 ký tự đầu)
        preview = chunk.content_with_prefix.replace("\n", " ")[:120]
        print(f"       content : {preview}...")

    if len(chunks) > 8:
        print(f"\n  ... (và {len(chunks) - 8} chunks còn lại)")

    # ── Validate critical fields ──────────────────────────────────────
    print(f"\n{SECTION}")
    print("   VALIDATION:")
    print(SECTION)
    errors: list[str] = []

    for c in chunks:
        if not c.breadcrumb:
            errors.append(f"chunk {c.chunk_id} thiếu breadcrumb!")
        if not c.source_node_id:
            errors.append(f"chunk {c.chunk_id} thiếu source_node_id!")
        if not c.content_with_prefix:
            errors.append(f"chunk {c.chunk_id} thiếu content_with_prefix!")
        if c.breadcrumb and c.content_with_prefix:
            if not c.content_with_prefix.startswith(c.breadcrumb):
                errors.append(f"chunk {c.chunk_id}: content_with_prefix không bắt đầu bằng breadcrumb!")

    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print(f"  ✅ {len(chunks)} chunks đều có breadcrumb, node_id, content_with_prefix hợp lệ.")

    print(f"\n✅ Module 2 DONE — {elapsed:.2f}s")
    return chunks


# ===========================================================================
# Module 3 Evaluation
# ===========================================================================


def run_module3(
    doc: "ingestion.models.LegalDocument",
    chunks: "list[ingestion.models.LsuChunk]",
    kuzu_path: str,
    chroma_path: str,
    query_text: str | None = None,
) -> None:
    """Build Knowledge Graph và Vector DB, rồi in thống kê."""
    from ingestion.graph_builder import HybridGraphBuilder, placeholder_embedding_fn

    print(f"\n{DIVIDER}")
    print("🕸️   MODULE 3 — HybridGraphBuilder")
    print(DIVIDER)
    print(f"   Kuzu DB  : {kuzu_path}")
    print(f"   ChromaDB : {chroma_path}")
    print(f"   Embedding: placeholder (zero vectors — thay bằng model thực khi cần)")

    builder = HybridGraphBuilder(
        kuzu_db_path=kuzu_path,
        chroma_db_path=chroma_path,
        chroma_collection_name="legal_documents_eval",
        embedding_fn=placeholder_embedding_fn,
        embedding_batch_size=32,
    )

    try:
        t0 = time.perf_counter()
        result = builder.build(doc, chunks)
        elapsed = time.perf_counter() - t0

        print(f"\n{'[RESULT]':>10} nodes    : {result['nodes']}")
        print(f"{'':>10} ref_edges: {result['edges']}")
        print(f"{'':>10} vectors  : {result['vectors']}")
        print(f"{'':>10} time     : {elapsed:.2f}s")

        # ── Graph stats ───────────────────────────────────────────────
        print(f"\n{SECTION}")
        print("   GRAPH DB STATS:")
        print(SECTION)
        graph_stats = builder.get_graph_stats()
        for k, v in graph_stats.items():
            print(f"  {k:25s}: {v}")

        # ── Kiểm tra REFERENCES edges ─────────────────────────────────
        if result["edges"] > 0:
            # Lấy 1 article ngẫu nhiên để demo get_node_references
            first_article = doc.iter_all_articles()[0] if doc.iter_all_articles() else None
            if first_article:
                refs = builder.get_node_references(first_article.node_id)
                if refs:
                    print(f"\n{SECTION}")
                    print(f"   REFERENCES từ '{first_article.full_title}':")
                    print(SECTION)
                    for r in refs[:5]:
                        print(f"  → {r['target_breadcrumb']}")
                        print(f"    context: '{r['context'][:80]}...'")

        # ── Semantic search demo ──────────────────────────────────────
        if query_text:
            print(f"\n{SECTION}")
            print(f"   SEMANTIC SEARCH: '{query_text}'")
            print(SECTION)
            print("   ⚠️  Dùng placeholder embedding → kết quả không có nghĩa semantic.")
            print("      Thay embedding_fn bằng model thực để có kết quả đúng.\n")
            try:
                results = builder.query_similar_chunks(query_text, n_results=3)
                for i, r in enumerate(results, 1):
                    print(f"  [{i}] {r['breadcrumb']}")
                    print(f"      node_id : {r['node_id']}")
                    print(f"      distance: {r['distance']:.4f}")
                    print(f"      text    : {r['document'][:100].replace(chr(10), ' ')}...")
            except Exception as e:
                print(f"  ❌ Query thất bại: {e}")

        # ── Validate node_id linkage ──────────────────────────────────
        print(f"\n{SECTION}")
        print("   VALIDATION — Vector ↔ Graph Linkage:")
        print(SECTION)
        chroma_ok = result["vectors"] > 0
        kuzu_ok = result["nodes"] > 0
        print(f"  ChromaDB vectors stored : {'✅' if chroma_ok else '❌'} {result['vectors']}")
        print(f"  Kuzu nodes stored       : {'✅' if kuzu_ok else '❌'} {result['nodes']}")
        print(f"  REFERENCES edges        : {'✅' if result['edges'] >= 0 else '❌'} {result['edges']}")

        # Lấy 1 vector record từ ChromaDB để kiểm tra node_id
        if chroma_ok:
            sample = builder._chroma_collection.get(
                limit=1, include=["metadatas"]
            )
            if sample and sample["metadatas"]:
                meta = sample["metadatas"][0]
                node_id_in_meta = meta.get("node_id", "")
                print(f"\n  Sample ChromaDB metadata:")
                print(f"    node_id    : {node_id_in_meta}  ← cầu nối về Kuzu")
                print(f"    doc_id     : {meta.get('doc_id', '')}")
                print(f"    breadcrumb : {meta.get('breadcrumb', '')[:60]}")
                print(f"    node_type  : {meta.get('node_type', '')}")
                print(
                    f"  ✅ node_id trong Vector metadata tồn tại: {bool(node_id_in_meta)}"
                )

    finally:
        builder.close()

    print(f"\n✅ Module 3 DONE — {elapsed:.2f}s")


# ===========================================================================
# Main
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá Phase 1 Ingestion Pipeline (Module 1/2/3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python evaluate_phase1.py
  python evaluate_phase1.py --file docs_test/v1.pdf
  python evaluate_phase1.py --file data_test/01-tand_signed_v1.docx
  python evaluate_phase1.py --module 1
  python evaluate_phase1.py --module 2
  python evaluate_phase1.py --query "quyền của bên trong hợp đồng"
  python evaluate_phase1.py --dry-run
        """,
    )
    parser.add_argument(
        "--file",
        default="data_test/01-tand_signed_v1.docx",
        help="Đường dẫn file PDF/DOCX cần parse (default: docs_test/v1.docx)",
    )
    parser.add_argument(
        "--module",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Chỉ chạy module cụ thể (1, 2, hoặc 3). Mặc định: chạy cả 3.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Câu query để test semantic search trong ChromaDB (Module 3).",
    )
    parser.add_argument(
        "--kuzu-path",
        default="./eval_output/graph_db",
        help="Đường dẫn lưu Kuzu DB (default: ./eval_output/graph_db)",
    )
    parser.add_argument(
        "--chroma-path",
        default="./eval_output/chroma_db",
        help="Đường dẫn lưu ChromaDB (default: ./eval_output/chroma_db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ chạy Module 1 và Module 2, bỏ qua ghi DB (Module 3).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Bật DEBUG logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ File không tồn tại: {file_path}")
        print(f"   Files có sẵn để test:")
        for p in sorted(Path("docs_test").glob("*")):
            print(f"     --file {p}")
        for p in sorted(Path("data_test").glob("*.docx"))[:5]:
            print(f"     --file {p}")
        sys.exit(1)

    total_start = time.perf_counter()
    run_all = args.module is None
    doc = None
    chunks = None

    try:
        # ── Module 1 ──────────────────────────────────────────────────
        if run_all or args.module == 1:
            doc = run_module1(file_path)

        # ── Module 2 ──────────────────────────────────────────────────
        if run_all or args.module == 2:
            if doc is None:
                print("⚠️  Module 2 cần output từ Module 1. Chạy Module 1 trước...")
                doc = run_module1(file_path)
            chunks = run_module2(doc)

        # ── Module 3 ──────────────────────────────────────────────────
        if (run_all or args.module == 3) and not args.dry_run:
            if doc is None:
                doc = run_module1(file_path)
            if chunks is None:
                from ingestion.chunker import LsuChunker
                chunks = LsuChunker().chunk(doc)
            run_module3(
                doc=doc,
                chunks=chunks,
                kuzu_path=args.kuzu_path,
                chroma_path=args.chroma_path,
                query_text=args.query,
            )
        elif args.dry_run and args.module == 3:
            print("\n⚠️  --dry-run được bật, bỏ qua Module 3 (Graph/Vector DB).")

    except KeyboardInterrupt:
        print("\n\n⛔ Bị interrupt bởi người dùng.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"❌ Lỗi pipeline: {e}")
        sys.exit(1)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{DIVIDER}")
    print(f"🏁  TỔNG KẾT: Tổng thời gian = {total_elapsed:.2f}s")
    print(DIVIDER)


if __name__ == "__main__":
    main()
