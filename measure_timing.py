import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')

from ingestion import process_two_documents
from embedding import embed_and_store
from retrieval import build_comparison_pairs
from comparison import build_comparison_result
from config import LLM_ENABLE_REPORT

file_a = 'docs_test/v1.pdf'
file_b = 'docs_test/v2.pdf'
timings = {}

t0 = time.time()

t = time.time()
chunks_a, chunks_b = process_two_documents(file_a, file_b)
timings['Ingestion'] = time.time() - t
print(f'[1/5] INGESTION  {timings["Ingestion"]:6.2f}s  | A={len(chunks_a)} B={len(chunks_b)}', flush=True)

t = time.time()
collection, embeds_a, embeds_b = embed_and_store(chunks_a, chunks_b)
timings['Embedding'] = time.time() - t
print(f'[2/5] EMBEDDING  {timings["Embedding"]:6.2f}s  | {len(chunks_a)+len(chunks_b)} chunks', flush=True)

t = time.time()
pairs = build_comparison_pairs(chunks_a, chunks_b, embeds_a=embeds_a, embeds_b=embeds_b)
timings['Matching'] = time.time() - t
print(f'[3/5] MATCHING   {timings["Matching"]:6.2f}s  | {len(pairs)} pairs', flush=True)

t = time.time()
comparison_result = build_comparison_result(pairs, file_a=file_a, file_b=file_b)
timings['Comparison'] = time.time() - t
print(f'[4/5] COMPARISON {timings["Comparison"]:6.2f}s  | {comparison_result.stats.clauses_affected} affected', flush=True)

if LLM_ENABLE_REPORT and comparison_result.stats.clauses_affected > 0:
    from llm import generate_comparison_report
    t = time.time()
    report_md = generate_comparison_report(comparison_result, file_a=file_a, file_b=file_b, enable_llm=True)
    timings['LLM Report'] = time.time() - t
    print(f'[5/5] LLM REPORT {timings["LLM Report"]:6.2f}s  | {len(report_md)} chars', flush=True)
else:
    timings['LLM Report'] = 0

total = time.time() - t0
print()
for name, dur in timings.items():
    if dur > 0:
        pct = dur/total*100
        bar = '#' * int(pct/2)
        print(f'  {name:<15} {bar:<25} {dur:7.2f}s  ({pct:5.1f}%)')
print(f'  {"TOTAL":<15} {"":<25} {total:7.2f}s')
