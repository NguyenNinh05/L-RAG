import sys
import time

from ingestion import process_two_documents
from embedding import embed_and_store, query_similar, print_query_results
from retrieval import build_comparison_pairs, print_diff_summary
from llm import generate_comparison_report


def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <fileA.docx/pdf> <fileB.docx/pdf>")
        sys.exit(1)

    file_a, file_b = sys.argv[1], sys.argv[2]
    print("\n" + "="*60)
    print("BƯỚC 1-2: INGESTION & CHUNKING")
    print("="*60)
    chunks_a, chunks_b = process_two_documents(file_a, file_b)

    start = time.time()
    collection, embeds_a, embeds_b = embed_and_store(chunks_a, chunks_b)
    print(f"\nHoàn thành embedding trong {time.time() - start:.2f} giây.")
    print("\n" + "="*60)
    print("BƯỚC 4: CROSS-MATCHING (Embedding + Needleman-Wunsch)")
    print("="*60)

    pairs = build_comparison_pairs(
        chunks_a, chunks_b,
        embeds_a=embeds_a,
        embeds_b=embeds_b,
    )

    print_diff_summary(pairs)

    modified = [p for p in pairs if p.match_type == "MODIFIED"]

    print("\n" + "="*60)
    print("BƯỚC 6: DEMO TRUY VẤN THEO ĐIỀU KHOẢN (RAG Retrieval)")
    print("="*60)

    demo_queries = [
        "Điều khoản về phạt vi phạm và bồi thường thiệt hại",
        "Nghĩa vụ bảo mật và bảo vệ dữ liệu",
        "Điều kiện và lịch thanh toán hợp đồng",
        "Quyền sở hữu trí tuệ phần mềm",
        "Điều kiện chấm dứt hợp đồng",
    ]

    for query in demo_queries:
        print(f"\n{'─'*60}")
        print(f"  🔍 Query: \"{query}\"")
        print(f"{'─'*60}")

        # Tim trong ca 2 tai lieu
        for doc_label in ["doc_A", "doc_B"]:
            doc_name = file_a if doc_label == "doc_A" else file_b
            print(f"\n  [{doc_label}] ({doc_name})")
            results = query_similar(query, collection=collection, n_results=2, doc_label_filter=doc_label)
            print_query_results(results)

    # ── BUOC 7: Sinh bao cao LLM + citation ──────────────────────────────────
    if modified:
        print("\n" + "="*60)
        print("BUOC 7: SINH BAO CAO SO SANH (LLM + Citation)")
        print("="*60)
        print("[!] Dang load LLM... (co the mat 1-2 phut lan dau)")

        report_md = generate_comparison_report(
            pairs,
            file_a=file_a,
            file_b=file_b,
        )

        # Luu ra file
        report_path = "comparison_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        print(f"\n[LLM] Bao cao da luu tai: {report_path}")
        print("\n" + "─"*60)
        print(report_md[:1000])  # in preview 1000 ky tu
        print("... (xem day du trong file comparison_report.md)")
    else:
        print("\n[Buoc 7] Khong co dieu khoan sua doi → Bo qua sinh bao cao LLM.")

    print("\n Tổng thời gian xử lý: ", time.time() - start)
if __name__ == "__main__":
    main()