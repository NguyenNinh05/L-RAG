"""
test.py — Demo & Benchmark cho Chương 4: Kết quả truy xuất và đánh giá hiệu năng
==================================================================================
Tập trung đánh giá hệ thống RETRIEVAL (query_similar, ChromaDB).
KHÔNG bao gồm thuật toán đối chiếu (Anchor + NW) — phần đó thuộc Chương 5.

Chạy:  python test.py
Yêu cầu: Ollama đang chạy với model qwen3-embedding:0.6b

Kết quả lưu tại: test_results.md
"""

import sys
import time
import os
import difflib
import logging
import psutil
from pathlib import Path
from collections import Counter

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")

# ── Import pipeline ────────────────────────────────────────────────────────────
from ingestion import process_two_documents
from ingestion.loader import load_document
from embedding import embed_and_store, embed_chunks, query_similar, MODEL_NAME
from embedding.embedder import INSTRUCTION_DOC, INSTRUCTION_QUERY, _get_batch_embeddings
from config import (
    UNCHANGED_THRESHOLD,
    MODIFIED_THRESHOLD,
    GAP_PENALTY,
    TEXT_UNCHANGED_RATIO,
    CHROMA_DIR,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
FILE_A = os.path.join("docs_test", "v1.pdf")
FILE_B = os.path.join("docs_test", "v2.pdf")
REPORT_FILE = "test_results.md"

# ── Output buffer ─────────────────────────────────────────────────────────────
_lines: list[str] = []

def log(msg: str = ""):
    print(msg)
    _lines.append(msg)

def section(title: str):
    log(f"\n{'='*70}")
    log(f"  {title}")
    log(f"{'='*70}")

def subsection(title: str):
    log(f"\n{'─'*70}")
    log(f"  {title}")
    log(f"{'─'*70}")


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH: Điều khoản kỳ vọng cho từng truy vấn
#  (Dựa trên nội dung thực tế của v1.pdf / v2.pdf — Hợp đồng CNTT)
# ══════════════════════════════════════════════════════════════════════════════
RETRIEVAL_GROUND_TRUTH = [
    {
        "query": "Mức phạt khi chậm tiến độ triển khai là bao nhiêu?",
        "description": "Kiểm tra khả năng tìm đúng điều khoản phạt dù dùng từ đồng nghĩa",
        "expected_article": "Điều 3",           # Phần III > Điều 3
        "expected_section": "Phần III",
        "doc_filter": "doc_A",
    },
    {
        "query": "Nghĩa vụ bảo mật và bảo vệ dữ liệu cá nhân",
        "description": "Tìm điều khoản bảo mật (khác nhau giữa v1 và v2)",
        "expected_article": "Điều 7",           # Phần II > Điều 7
        "expected_section": "Phần II",
        "doc_filter": "doc_B",
    },
    {
        "query": "Giá trị hợp đồng và lịch thanh toán theo đợt",
        "description": "Tìm điều khoản tài chính",
        "expected_article": "Điều 4",           # Phần II > Điều 4
        "expected_section": "Phần II",
        "doc_filter": "doc_A",
    },
    {
        "query": "Thời hạn hiệu lực và các giai đoạn triển khai",
        "description": "Tìm điều khoản thời hạn",
        "expected_article": "Điều 3",           # Phần I > Điều 3
        "expected_section": "Phần I",
        "doc_filter": "doc_A",
    },
    {
        "query": "Tiêu chí nghiệm thu phần mềm ERP",
        "description": "Tìm điều khoản nghiệm thu",
        "expected_article": "Điều 5",           # Phần II > Điều 5 hoặc Phụ lục B
        "expected_section": "Phần II",
        "doc_filter": "doc_A",
    },
    {
        "query": "Trường hợp nào được miễn trách nhiệm hợp đồng?",
        "description": "Hỏi gián tiếp về bất khả kháng — không dùng từ khóa chính xác",
        "expected_article": "Điều 8",           # Phần II > Điều 8 (Bất khả kháng)
        "expected_section": "Phần II",
        "doc_filter": "doc_A",
    },
    {
        "query": "Danh sách nhân sự chủ chốt thực hiện dự án",
        "description": "Tìm phụ lục nhân sự",
        "expected_article": "Phụ lục A",
        "expected_section": "Phụ lục",
        "doc_filter": "doc_A",
    },
    {
        "query": "Nơi giải quyết tranh chấp hợp đồng",
        "description": "Tìm điều khoản tranh chấp (Hà Nội vs TP.HCM)",
        "expected_article": "Điều 5",           # Phần III > Điều 5
        "expected_section": "Phần III",
        "doc_filter": "doc_A",
    },
    {
        "query": "Đền bù thiệt hại khi vi phạm thông tin mật",
        "description": "Dùng 'đền bù' thay vì 'phạt', 'thông tin mật' thay vì 'bảo mật'",
        "expected_article": "Điều 3",           # Phần III > Điều 3
        "expected_section": "Phần III",
        "doc_filter": "doc_A",
    },
    {
        "query": "Cam kết mức độ dịch vụ SLA và uptime",
        "description": "Tìm SLA — chỉ có trong v1, bị xóa ở v2",
        "expected_article": "Điều 7",           # Phần II > Điều 7 (v1 only)
        "expected_section": "Phần II",
        "doc_filter": "doc_A",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 1: INGESTION & CHUNKING
# ══════════════════════════════════════════════════════════════════════════════
def test_ingestion():
    section("4.3 | BENCHMARK: INGESTION & CHUNKING")

    t0 = time.time()
    paragraphs_a = load_document(FILE_A)
    t_load_a = time.time() - t0

    t0 = time.time()
    paragraphs_b = load_document(FILE_B)
    t_load_b = time.time() - t0

    log(f"\n  📄 File A: {FILE_A}  ({os.path.getsize(FILE_A) / 1024:.1f} KB)")
    log(f"     Đoạn thô: {len(paragraphs_a):4d} paragraphs")
    log(f"     Load time: {t_load_a:.3f}s")
    log(f"\n  📄 File B: {FILE_B}  ({os.path.getsize(FILE_B) / 1024:.1f} KB)")
    log(f"     Đoạn thô: {len(paragraphs_b):4d} paragraphs")
    log(f"     Load time: {t_load_b:.3f}s")

    t0 = time.time()
    chunks_a, chunks_b = process_two_documents(FILE_A, FILE_B)
    t_total = time.time() - t0

    subsection("Danh sách chunks")

    log(f"\n  Doc A ({len(chunks_a)} chunks):")
    for i, c in enumerate(chunks_a):
        log(f"    [{i:2d}] {c.article_number or 'N/A':40s} | {len(c.content):5d} chars | page={c.page}")

    log(f"\n  Doc B ({len(chunks_b)} chunks):")
    for i, c in enumerate(chunks_b):
        log(f"    [{i:2d}] {c.article_number or 'N/A':40s} | {len(c.content):5d} chars | page={c.page}")

    log(f"\n  Tổng thời gian ingestion+chunking: {t_total:.3f}s")

    return chunks_a, chunks_b, t_total


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 2: EMBEDDING & VECTOR DB
# ══════════════════════════════════════════════════════════════════════════════
def test_embedding(chunks_a, chunks_b):
    section("4.3 | BENCHMARK: EMBEDDING & VECTOR DB")

    total_chunks = len(chunks_a) + len(chunks_b)

    log(f"\n  Model:           {MODEL_NAME}")
    log(f"  Tổng chunks:     {total_chunks}")
    log(f"  INSTRUCTION_DOC: \"{INSTRUCTION_DOC}\"")

    # Ghi nhận RAM trước
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024 / 1024  # MB

    # === Cold-start test: embed 1 chunk để nạp model ===
    subsection("Cold-start: Nạp model lần đầu")
    t0 = time.time()
    _ = _get_batch_embeddings([f"{INSTRUCTION_DOC}: test cold start"])
    t_cold = time.time() - t0
    log(f"  Cold-start latency: {t_cold:.2f}s")

    # === Batch embedding ===
    subsection("Batch Embedding toàn bộ")
    t0 = time.time()
    collection, embeds_a, embeds_b = embed_and_store(chunks_a, chunks_b)
    t_embed = time.time() - t0

    vec_dim = len(embeds_a[0]) if embeds_a else "N/A"
    mem_after = proc.memory_info().rss / 1024 / 1024

    log(f"  Thời gian embedding + store: {t_embed:.2f}s")
    log(f"  Throughput:                  {total_chunks / t_embed:.1f} chunks/s")
    log(f"  Vector dimension:            {vec_dim}")
    log(f"  Avg time per chunk:          {t_embed / total_chunks * 1000:.1f} ms")
    log(f"  RAM trước:                   {mem_before:.0f} MB")
    log(f"  RAM sau:                     {mem_after:.0f} MB")
    log(f"  RAM tăng thêm:               {mem_after - mem_before:.0f} MB")

    # ChromaDB disk size
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        total_size = sum(f.stat().st_size for f in chroma_path.rglob("*") if f.is_file())
        log(f"  ChromaDB on disk:            {total_size / 1024:.1f} KB ({total_size / 1024 / 1024:.2f} MB)")

    return collection, embeds_a, embeds_b, t_embed, t_cold


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 3: DEMO TRUY VẤN (4.1)
# ══════════════════════════════════════════════════════════════════════════════
def test_retrieval_demo(collection):
    section("4.1 | DEMO TRUY XUẤT THEO ĐIỀU KHOẢN")

    N_RESULTS = 3
    query_latencies = []      # (total_ms, embed_ms, search_ms)
    hit_results = []

    for q_idx, gt in enumerate(RETRIEVAL_GROUND_TRUTH):
        query = gt["query"]
        expected_art = gt["expected_article"]
        doc_filter = gt["doc_filter"]
        desc = gt["description"]

        subsection(f"Query {q_idx+1}/{len(RETRIEVAL_GROUND_TRUTH)}")
        log(f"  🔍 \"{query}\"")
        log(f"     Mô tả:    {desc}")
        log(f"     Kỳ vọng:  {gt['expected_section']} > {expected_art}")
        log(f"     Bộ lọc:   {doc_filter}")

        # ── Bước 1: Nhúng câu hỏi qua Ollama ──
        query_with_inst = f"{INSTRUCTION_QUERY}: {query}"
        t0 = time.time()
        query_embedding_list = _get_batch_embeddings([query_with_inst])
        t_embed_q = time.time() - t0

        query_embedding = query_embedding_list[0] if query_embedding_list else None

        # ── Bước 2: Quét ChromaDB HNSW ──
        where_filter = {"doc_label": doc_filter} if doc_filter else None
        t0 = time.time()
        if query_embedding:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=N_RESULTS,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        t_search = time.time() - t0

        t_total = t_embed_q + t_search
        query_latencies.append((t_total, t_embed_q, t_search))

        # Check kết quả
        hit_rank = -1
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, doc, meta, dist) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                similarity = 1 - dist
                breadcrumb = meta.get("breadcrumb", "")

                is_match = (
                    expected_art.lower() in breadcrumb.lower()
                    and gt["expected_section"].lower() in breadcrumb.lower()
                )
                marker = " ✅ HIT" if is_match else ""
                if is_match and hit_rank == -1:
                    hit_rank = i + 1

                log(f"     [{i+1}] {breadcrumb:40s} | sim={similarity:.4f}{marker}")
                log(f"         ── FULL TEXT ({len(doc)} chars) ──")
                for text_line in doc.splitlines():
                    log(f"         │ {text_line}")
                log(f"         ── END ──")
        else:
            log(f"     ❌ Không tìm thấy kết quả.")

        log(f"     ⏱ Latency tổng: {t_total*1000:.1f} ms  (Embed: {t_embed_q*1000:.1f} ms | ChromaDB: {t_search*1000:.1f} ms)")

        hit_results.append({
            "query_idx": q_idx + 1,
            "query": query[:50],
            "expected": f"{gt['expected_section']} > {expected_art}",
            "hit_rank": hit_rank,
            "is_hit": hit_rank > 0,
        })

    return query_latencies, hit_results


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 4: ĐÁNH GIÁ CHẤT LƯỢNG TRUY XUẤT (4.2)
# ══════════════════════════════════════════════════════════════════════════════
def test_retrieval_quality(hit_results, query_latencies):
    section("4.2 | ĐÁNH GIÁ CHẤT LƯỢNG TRUY XUẤT")

    # ── 4.2.1: Bảng tổng hợp Hit Rate ──
    subsection("4.2.1 | Bảng tổng hợp kết quả truy xuất")
    log(f"  {'─'*95}")
    log(f"  {'STT':>4} | {'Truy vấn':<50} | {'Kỳ vọng':<22} | {'Hit@K':^6} | {'Kết quả':^8}")
    log(f"  {'─'*95}")

    total_hits = 0
    hit_at_1 = 0
    hit_at_3 = 0

    for r in hit_results:
        rank_str = f"Top-{r['hit_rank']}" if r['is_hit'] else "MISS"
        verdict = "✅" if r['is_hit'] else "❌"

        log(f"  {r['query_idx']:4d} | {r['query']:<50} | {r['expected']:<22} | {rank_str:^6} | {verdict:^8}")

        if r['is_hit']:
            total_hits += 1
            if r['hit_rank'] <= 3:
                hit_at_3 += 1
            if r['hit_rank'] == 1:
                hit_at_1 += 1

    log(f"  {'─'*95}")

    total = len(hit_results)
    log(f"\n  📊 Metrics tổng hợp:")
    log(f"     Hit Rate (Top-3):  {hit_at_3}/{total} = {hit_at_3/total*100:.1f}%")
    log(f"     Hit Rate (Top-1):  {hit_at_1}/{total} = {hit_at_1/total*100:.1f}%")

    # MRR (Mean Reciprocal Rank)
    mrr_sum = 0
    for r in hit_results:
        if r['is_hit']:
            mrr_sum += 1.0 / r['hit_rank']
    mrr = mrr_sum / total
    log(f"     MRR (Mean Reciprocal Rank): {mrr:.4f}")

    # ── 4.2.2: Phân tích ưu/nhược điểm ──
    subsection("4.2.2 | Phân tích rủi ro")

    miss_queries = [r for r in hit_results if not r['is_hit']]
    if miss_queries:
        log(f"\n  ⚠️ Các truy vấn MISS (không tìm thấy điều khoản kỳ vọng trong Top-3):")
        for r in miss_queries:
            log(f"     Query {r['query_idx']}: \"{r['query']}\"")
            log(f"     → Kỳ vọng: {r['expected']}")
            log(f"     → Nguyên nhân có thể: semantic gap hoặc chunking chia sai ranh giới")
    else:
        log(f"\n  ✅ Tất cả {total} truy vấn đều hit trong Top-3!")

    # ── Query latency stats (phân tách Embed vs ChromaDB) ──
    subsection("4.3.2 | Thống kê Query Latency (phân tách)")
    if query_latencies:
        totals_ms  = [t[0] * 1000 for t in query_latencies]
        embeds_ms  = [t[1] * 1000 for t in query_latencies]
        searches_ms = [t[2] * 1000 for t in query_latencies]

        avg_total  = sum(totals_ms) / len(totals_ms)
        avg_embed  = sum(embeds_ms) / len(embeds_ms)
        avg_search = sum(searches_ms) / len(searches_ms)

        log(f"  Số truy vấn:           {len(totals_ms)}")
        log(f"")
        log(f"  ┌─ TỔNG CỘNG ─────────────────────────────┐")
        log(f"  │  Min:       {min(totals_ms):>8.1f} ms               │")
        log(f"  │  Max:       {max(totals_ms):>8.1f} ms               │")
        log(f"  │  Trung bình:{avg_total:>8.1f} ms               │")
        log(f"  ├─ OLLAMA EMBEDDING ──────────────────────┤")
        log(f"  │  Min:       {min(embeds_ms):>8.1f} ms               │")
        log(f"  │  Max:       {max(embeds_ms):>8.1f} ms               │")
        log(f"  │  Trung bình:{avg_embed:>8.1f} ms               │")
        log(f"  ├─ CHROMADB HNSW SEARCH ───────────────────┤")
        log(f"  │  Min:       {min(searches_ms):>8.1f} ms               │")
        log(f"  │  Max:       {max(searches_ms):>8.1f} ms               │")
        log(f"  │  Trung bình:{avg_search:>8.1f} ms               │")
        log(f"  └──────────────────────────────────────────┘")
        log(f"")
        log(f"  📊 Phân bổ: Ollama chiếm {avg_embed/avg_total*100:.1f}% | ChromaDB chiếm {avg_search/avg_total*100:.1f}%")

    return hit_at_1, hit_at_3, total, mrr


# ══════════════════════════════════════════════════════════════════════════════
#  PHẦN 5: TRUY XUẤT CHÉO (4.1.2) — Lấy chunk doc_A query trên doc_B
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_retrieval(collection, chunks_a, chunks_b):
    section("4.1.2 | TRUY XUẤT CHÉO: Chunk Doc_A → Tìm trên Doc_B")
    log("  Mục tiêu: Lấy nội dung 1 điều khoản từ Bản A, tìm đoạn tương đồng ở Bản B.")
    log("  Mô phỏng bước đệm cho thuật toán căn chỉnh (Chương 5).\n")

    # Chọn vài chunk tiêu biểu từ doc_A để cross-query
    test_chunks_a = []
    for c in chunks_a:
        art = (c.article_number or "").lower()
        # Chọn các điều khoản quan trọng
        if any(kw in art for kw in ["điều 3", "điều 4", "điều 7", "điều 5"]):
            test_chunks_a.append(c)
        if len(test_chunks_a) >= 5:
            break

    if not test_chunks_a:
        # Fallback: lấy 5 chunk đầu tiên
        test_chunks_a = chunks_a[:5]

    for i, chunk in enumerate(test_chunks_a):
        subsection(f"Cross-query {i+1}: {chunk.article_number or 'N/A'}")
        log(f"  Nguồn: Doc_A | {chunk.article_number}")
        log(f"  Nội dung truy vấn (100 chars): {chunk.content[:100].replace(chr(10), ' ')}...")

        # Query trên doc_B
        t0 = time.time()
        results = query_similar(
            chunk.content[:500],  # Giới hạn độ dài query
            collection=collection,
            n_results=3,
            doc_label_filter="doc_B",
        )
        t_query = time.time() - t0

        if results["ids"] and results["ids"][0]:
            for j, (doc_id, doc, meta, dist) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                similarity = 1 - dist
                breadcrumb_b = meta.get("breadcrumb", "")
                log(f"  → [{j+1}] Doc_B | {breadcrumb_b:40s} | sim={similarity:.4f}")

        log(f"  ⏱ Latency: {t_query*1000:.1f} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  TỔNG HỢP
# ══════════════════════════════════════════════════════════════════════════════
def test_summary(t_ingest, t_embed, t_cold, query_latencies,
                 chunks_a, chunks_b, hit_at_1, hit_at_3, total_queries, mrr):
    section("TỔNG HỢP BENCHMARK — CHƯƠNG 4")

    total_chunks = len(chunks_a) + len(chunks_b)
    avg_query_ms = sum(t[0] * 1000 for t in query_latencies) / len(query_latencies) if query_latencies else 0
    avg_embed_ms = sum(t[1] * 1000 for t in query_latencies) / len(query_latencies) if query_latencies else 0
    avg_search_ms = sum(t[2] * 1000 for t in query_latencies) / len(query_latencies) if query_latencies else 0

    log(f"""
  ┌───────────────────────────────────────────────────────────────┐
  │              BẢNG TỔNG HỢP HIỆU NĂNG HỆ THỐNG              │
  ├─────────────────────────────────────┬─────────────────────────┤
  │  CẤU HÌNH                          │                         │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  File test                          │  v1.pdf ↔ v2.pdf        │
  │  Embedding model                    │  {MODEL_NAME:<23s} │
  │  Chunks (A + B)                     │  {len(chunks_a)} + {len(chunks_b)} = {total_chunks:<16d} │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  THỜI GIAN                          │                         │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  Ingestion + Chunking               │  {t_ingest:>10.3f}s             │
  │  Cold-start (nạp model)             │  {t_cold:>10.2f}s             │
  │  Embedding + ChromaDB store         │  {t_embed:>10.2f}s             │
  │  Avg query latency                  │  {avg_query_ms:>10.1f} ms           │
  │    ├ Ollama embedding                │  {avg_embed_ms:>10.1f} ms           │
  │    └ ChromaDB HNSW search            │  {avg_search_ms:>10.1f} ms           │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  CHẤT LƯỢNG TRUY XUẤT              │                         │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  Hit Rate (Top-1)                   │  {hit_at_1}/{total_queries} = {hit_at_1/total_queries*100:>5.1f}%           │
  │  Hit Rate (Top-3)                   │  {hit_at_3}/{total_queries} = {hit_at_3/total_queries*100:>5.1f}%           │
  │  MRR                                │  {mrr:>10.4f}              │
  └─────────────────────────────────────┴─────────────────────────┘""")

    log(f"\n  Cấu hình ngưỡng (tham khảo):")
    log(f"    UNCHANGED_THRESHOLD  = {UNCHANGED_THRESHOLD}")
    log(f"    MODIFIED_THRESHOLD   = {MODIFIED_THRESHOLD}")
    log(f"    TEXT_UNCHANGED_RATIO = {TEXT_UNCHANGED_RATIO}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log("╔══════════════════════════════════════════════════════════════════════╗")
    log("║  L-RAG — Chương 4: Demo truy xuất & Đánh giá hiệu năng            ║")
    log("║  Cặp tài liệu: docs_test/v1.pdf ↔ docs_test/v2.pdf               ║")
    log(f"║  Thời gian:     {time.strftime('%Y-%m-%d %H:%M:%S'):>49s} ║")
    log("╚══════════════════════════════════════════════════════════════════════╝")

    for f in [FILE_A, FILE_B]:
        if not os.path.exists(f):
            log(f"\n  ❌ Không tìm thấy: {f}")
            sys.exit(1)

    total_start = time.time()

    # ── 1) Ingestion ──
    chunks_a, chunks_b, t_ingest = test_ingestion()

    # ── 2) Embedding ──
    collection, embeds_a, embeds_b, t_embed, t_cold = test_embedding(chunks_a, chunks_b)

    # ── 3) Demo truy vấn đơn lẻ (4.1.1) ──
    query_latencies, hit_results = test_retrieval_demo(collection)

    # ── 4) Demo truy xuất chéo (4.1.2) ──
    test_cross_retrieval(collection, chunks_a, chunks_b)

    # ── 5) Đánh giá chất lượng (4.2) ──
    hit_at_1, hit_at_3, total_q, mrr = test_retrieval_quality(hit_results, query_latencies)

    # ── 6) Tổng hợp ──
    test_summary(t_ingest, t_embed, t_cold, query_latencies,
                 chunks_a, chunks_b, hit_at_1, hit_at_3, total_q, mrr)

    elapsed = time.time() - total_start
    section("HOÀN TẤT")
    log(f"\n  ⏱  Tổng thời gian test: {elapsed:.2f}s")
    log(f"  📄 Kết quả lưu tại:     {REPORT_FILE}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))

    log(f"  ✅ Done!")


if __name__ == "__main__":
    main()
