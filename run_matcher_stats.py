"""Script chạy thực tế matcher để lấy số liệu cho Chương 5."""
import sys
import os
import logging

# Tắt log để output sạch
logging.basicConfig(level=logging.ERROR)

from ingestion import process_two_documents
from retrieval.matcher import build_comparison_pairs

FILE_A = os.path.join("docs_test", "v1.pdf")
FILE_B = os.path.join("docs_test", "v2.pdf")

# 1. Ingest & Chunk
chunks_a, chunks_b = process_two_documents(FILE_A, FILE_B)

# 2. Chạy so sánh (sẽ gọi embedding model thực tế)
pairs = build_comparison_pairs(chunks_a, chunks_b)

print("\n" + "="*80)
print(f"{'TYPE':<12} | {'ARTICLE':<30} | {'SIMILARITY':<10}")
print("-"*80)

for p in pairs:
    # Lấy label (Điều khoản)
    if p.chunk_a and p.chunk_b:
        label = p.chunk_a.article_number or "N/A"
        if p.chunk_a.article_number != p.chunk_b.article_number:
            label = f"{p.chunk_a.article_number} -> {p.chunk_b.article_number}"
    elif p.chunk_a:
        label = p.chunk_a.article_number or "N/A"
    else:
        label = p.chunk_b.article_number or "N/A"
        
    print(f"{p.match_type:<12} | {label:<30} | {p.similarity:.4f}")

print("="*80)
