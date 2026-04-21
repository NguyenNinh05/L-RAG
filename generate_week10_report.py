from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_TEST_DIR = ROOT / "docs_test" / "data_test"
BASELINE_PATH = ROOT / "data" / "matching_baseline_report.json"
PRESET_PATH = ROOT / "data" / "matching_threshold_presets.json"
CHANGE_METRICS_PATH = ROOT / "data" / "change_detection_metrics.json"
RETRIEVAL_PATH = ROOT / "test_results.md"
OUTPUT_PATH = ROOT / "docs" / "week10_report.md"


LEXICAL_KEYS = (
    "numerical",
    "typo",
    "formatting noise",
    "word order",
    "micro deletion",
    "lexical",
)
STRUCTURAL_KEYS = (
    "structural",
    "table manipulation",
    "silent deletion",
    "addition",
    "deletion",
    "merge",
    "split",
)
SEMANTIC_KEYS = (
    "semantic",
    "paraphrase",
    "logic shift",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ground_truth_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for folder in sorted(DATA_TEST_DIR.glob("Test_*_Difficult")):
        gt = folder / "ground_truth.json"
        if not gt.exists():
            continue
        payload = _read_json(gt)
        payload["case_name"] = folder.name
        cases.append(payload)
    return cases


def _classify_change_bucket(change_type: str) -> str:
    lowered = (change_type or "").lower()
    if any(key in lowered for key in SEMANTIC_KEYS):
        return "Semantic"
    if any(key in lowered for key in STRUCTURAL_KEYS):
        return "Structural"
    if any(key in lowered for key in LEXICAL_KEYS):
        return "Lexical"
    return "Other"


def _dataset_statistics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total_changes = 0
    bucket_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()

    for case in cases:
        changes = case.get("changes", [])
        total_changes += len(changes)
        for item in changes:
            change_type = str(item.get("type", "Unknown")).strip()
            bucket_counter[_classify_change_bucket(change_type)] += 1
            type_counter[change_type] += 1

    top_types = type_counter.most_common(10)
    return {
        "total_cases": len(cases),
        "total_changes": total_changes,
        "bucket_counter": dict(bucket_counter),
        "top_types": top_types,
    }


def _extract_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _extract_retrieval_metrics(markdown: str) -> dict[str, float | None]:
    hit1 = _extract_float(markdown, r"Hit Rate \(Top-1\):\s*\d+/\d+\s*=\s*([0-9.]+)%")
    hit3 = _extract_float(markdown, r"Hit Rate \(Top-3\):\s*\d+/\d+\s*=\s*([0-9.]+)%")
    mrr = _extract_float(markdown, r"MRR \(Mean Reciprocal Rank\):\s*([0-9.]+)")

    ingest_s = _extract_float(markdown, r"Ingestion \+ Chunking\s*\|\s*([0-9.]+)s")
    if ingest_s is None:
        ingest_s = _extract_float(markdown, r"Tổng thời gian ingestion\+chunking:\s*([0-9.]+)s")

    embed_store_s = _extract_float(markdown, r"Embedding \+ ChromaDB store\s*\|\s*([0-9.]+)s")
    if embed_store_s is None:
        embed_store_s = _extract_float(markdown, r"Thời gian embedding \+ store:\s*([0-9.]+)s")

    query_ms = _extract_float(markdown, r"Avg query latency\s*\|\s*([0-9.]+) ms")
    if query_ms is None:
        query_ms = _extract_float(markdown, r"Trung bình:\s*([0-9.]+) ms")

    ollama_ms = _extract_float(markdown, r"Ollama embedding\s*\|\s*([0-9.]+) ms")
    if ollama_ms is None:
        ollama_ms = _extract_float(markdown, r"OLLAMA EMBEDDING[\s\S]*?Trung bình:\s*([0-9.]+) ms")

    chroma_ms = _extract_float(markdown, r"ChromaDB HNSW search\s*\|\s*([0-9.]+) ms")
    if chroma_ms is None:
        chroma_ms = _extract_float(markdown, r"CHROMADB HNSW SEARCH[\s\S]*?Trung bình:\s*([0-9.]+) ms")

    ram_before = _extract_float(markdown, r"RAM trước:\s*([0-9.]+) MB")
    ram_after = _extract_float(markdown, r"RAM sau:\s*([0-9.]+) MB")
    ram_delta = _extract_float(markdown, r"RAM tăng thêm:\s*([0-9.]+) MB")

    # Single-relevant assumption: each query has one relevant answer.
    p_at_1 = (hit1 / 100.0) if hit1 is not None else None
    p_at_3 = (hit3 / 100.0 / 3.0) if hit3 is not None else None

    return {
        "hit_rate_at1_pct": hit1,
        "hit_rate_at3_pct": hit3,
        "mrr": mrr,
        "precision_at1": p_at_1,
        "precision_at3_single_rel": p_at_3,
        "ingestion_s": ingest_s,
        "embedding_store_s": embed_store_s,
        "query_ms": query_ms,
        "ollama_ms": ollama_ms,
        "chroma_ms": chroma_ms,
        "ram_before_mb": ram_before,
        "ram_after_mb": ram_after,
        "ram_delta_mb": ram_delta,
    }


def _estimated_prf1_from_baseline(baseline: dict[str, Any]) -> dict[str, float]:
    cases = baseline.get("cases", [])
    total_gt = sum(int(item.get("gt_changes", 0)) for item in cases)
    total_pred = sum(int(item.get("predicted_atomic_changes", 0)) for item in cases)

    avg_clause_recall = float(baseline["summary"].get("avg_clause_recall", 0.0))
    matched_est = min(total_gt, total_pred, avg_clause_recall * total_gt)

    recall = (matched_est / total_gt) if total_gt else 0.0
    precision = (matched_est / total_pred) if total_pred else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision_est": round(precision, 4),
        "recall_est": round(recall, 4),
        "f1_est": round(f1, 4),
        "total_gt": total_gt,
        "total_pred": total_pred,
    }


def _load_change_metrics() -> dict[str, Any] | None:
    if not CHANGE_METRICS_PATH.exists():
        return None
    return _read_json(CHANGE_METRICS_PATH)


def _build_markdown(
    dataset: dict[str, Any],
    baseline: dict[str, Any],
    preset: dict[str, Any],
    retrieval: dict[str, float | None],
    change_metrics: dict[str, Any] | None,
) -> str:
    today = date.today().isoformat()
    best = preset.get("recommended", {})
    best_params = best.get("params", {})

    if change_metrics:
        summary = change_metrics["summary"]
        case_rows = change_metrics["cases"]
    else:
        summary = _estimated_prf1_from_baseline(baseline)
        case_rows = []

    bucket = dataset["bucket_counter"]
    lexical_n = bucket.get("Lexical", 0)
    structural_n = bucket.get("Structural", 0)
    semantic_n = bucket.get("Semantic", 0)
    other_n = bucket.get("Other", 0)

    top_types_lines = "\n".join(
        f"- {name}: {count}" for name, count in dataset["top_types"]
    )

    return f"""# BÁO CÁO TUẦN 10: XÂY DỰNG BỘ DỮ LIỆU KIỂM THỬ VÀ ĐÁNH GIÁ BASELINE VÒNG 1

Ngày tạo báo cáo tự động: {today}

## 1. XÂY DỰNG BỘ DỮ LIỆU ĐÁNH GIÁ (DATASET ENGINEERING)

### 1.1. Tiêu chí và Nguồn dữ liệu (Data Sourcing)
- Tổng số cặp văn bản đánh giá: {dataset['total_cases']} cặp (nhóm Difficult).
- Tổng số thay đổi đã annotate trong Ground Truth: {dataset['total_changes']} thay đổi.
- Tập dữ liệu bao phủ các loại văn bản pháp lý thực tế: Hợp đồng, Phụ lục, Thông tư (DOCX/PDF trong docs_test/data_test).

### 1.2. Chiến thuật giả lập khác biệt (Adversarial Generation Strategy)
- Lexical (Cấp Dễ): {lexical_n}
- Structural (Cấp Trung bình): {structural_n}
- Semantic (Cấp Khó): {semantic_n}
- Khác/pha trộn: {other_n}

Top loại thay đổi xuất hiện nhiều nhất:
{top_types_lines if top_types_lines else '- Không có dữ liệu'}

### 1.3. Khởi tạo Đáp án chuẩn (Ground Truth Construction)
#### 1.3.1. Thiết kế Schema đối chiếu
Schema hiện hành trong mỗi ground_truth.json:
- pair_id
- difficulty_level
- description
- changes[] với các field chính:
- type
- location
- v1_content
- v2_content

#### 1.3.2. Cơ chế đối soát (Annotation)
- Mỗi case có danh sách thay đổi chuẩn hóa theo từng vị trí điều khoản/phụ lục.
- Dữ liệu được thiết kế để chấm tự động theo token điều khoản/số liệu và đối chiếu khối lượng thay đổi.

## 2. HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

### 2.1. Nhóm chỉ số phát hiện thay đổi (Deterministic Metrics)
- TP: {summary.get('tp', 'N/A')}
- FP: {summary.get('fp', 'N/A')}
- FN: {summary.get('fn', 'N/A')}
- Precision: {summary.get('precision', summary.get('precision_est', 'N/A'))}
- Recall: {summary.get('recall', summary.get('recall_est', 'N/A'))}
- F1-Score: {summary.get('f1', summary.get('f1_est', 'N/A'))}

Ghi chú phương pháp:
- TP/FP/FN được tính bằng greedy one-to-one matching giữa predicted change records và ground truth theo clause/location + nội dung + số liệu.
- Nếu file metrics chi tiết chưa có thì report sẽ fallback sang proxy baseline.
- total_gt_changes = {summary.get('total_gt', 'N/A')}, total_predicted_atomic = {summary.get('total_predicted', 'N/A')}.
- Baseline deterministic guard gốc:
- avg_clause_recall = {baseline['summary']['avg_clause_recall']}
- avg_number_recall = {baseline['summary']['avg_number_recall']}
- avg_volume_alignment = {baseline['summary']['avg_volume_alignment']}

#### Bảng TP/FP/FN theo từng case
| Case | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(
        f"| {row['case_name']} | {row['gt_changes']} | {row['predicted_changes']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        for row in case_rows
    ) if case_rows else '| _Chưa có file metrics chi tiết_ | - | - | - | - | - | - | - | - |'}
| **Total** | **{summary.get('total_gt', 'N/A')}** | **{summary.get('total_predicted', 'N/A')}** | **{summary.get('tp', 'N/A')}** | **{summary.get('fp', 'N/A')}** | **{summary.get('fn', 'N/A')}** | **{summary.get('precision', summary.get('precision_est', 'N/A'))}** | **{summary.get('recall', summary.get('recall_est', 'N/A'))}** | **{summary.get('f1', summary.get('f1_est', 'N/A'))}** |

### 2.2. Nhóm chỉ số chất lượng trích xuất (Retrieval Quality Metrics)
- Hit Rate@1: {retrieval['hit_rate_at1_pct']}%
- Hit Rate@3: {retrieval['hit_rate_at3_pct']}%
- Precision@1: {retrieval['precision_at1']}
- Precision@3 (single relevant assumption): {retrieval['precision_at3_single_rel']}
- MRR: {retrieval['mrr']}

### 2.3. Nhóm chỉ số hiệu năng môi trường cục bộ (Offline Benchmarks)
- Ingestion latency: {retrieval['ingestion_s']}s
- Embedding + store latency: {retrieval['embedding_store_s']}s
- Query latency trung bình: {retrieval['query_ms']} ms
- Trong đó Ollama embedding: {retrieval['ollama_ms']} ms
- Trong đó ChromaDB search: {retrieval['chroma_ms']} ms
- RAM trước chạy benchmark: {retrieval['ram_before_mb']} MB
- RAM sau chạy benchmark: {retrieval['ram_after_mb']} MB
- RAM tăng thêm: {retrieval['ram_delta_mb']} MB

## 3. KẾT QUẢ CHẠY ĐÁNH GIÁ VÒNG 1 (BASELINE EXPERIMENTAL RESULTS)

### 3.1. Thống kê và Phân tích kết quả theo độ khó
#### 3.1.1. Hiệu năng nhóm biến đổi Dễ/Trung bình
- Hệ thống đạt avg_number_recall = {baseline['summary']['avg_number_recall']} cho các biến đổi số liệu/metadata.
- Các case có volume_alignment thấp cho thấy còn under/over-detect ở nhóm biến đổi cấu trúc lớn (đặc biệt case Test_07_Difficult).

#### 3.1.2. Hiệu năng nhóm biến đổi Semantic (Khó)
- avg_clause_recall = {baseline['summary']['avg_clause_recall']} trên bộ Difficult.
- Một số case có paraphrase sâu và thao tác cấu trúc kết hợp vẫn làm giảm khả năng bắt đủ thay đổi (điển hình Test_01_Difficult).

### 3.2. Kiểm định tính ổn định của Giao diện và Bộ nhớ (UI & Memory Stability)
#### 3.2.1. Trực quan hóa đối chiếu
- UI drawer split-view đã được tích hợp và có thể hiển thị đồng thời evidence/report.
- Cần bổ sung checklist chấm UX định lượng (thời gian mở drawer, thời gian render citation lớn).

#### 3.2.2. Vòng đời bộ nhớ (Garbage Collection)
- Vòng 1 chưa có số đo Chrome Profiler định lượng (heap snapshots, GC timeline).
- Trạng thái hiện tại: đã có hạ tầng session store/citation store, nhưng thiếu benchmark browser-memory tự động.

## 4. PHÂN TÍCH LỖI VÀ ĐIỂM NGHẼN (ERROR ANALYSIS)

### 4.1. Hiện tượng Ảo giác từ khâu trích xuất (Retrieval Failures)
#### 4.1.1. Thất thoát dữ liệu cận biên (Drop-loss Effect)
- Khi đoạn quá ngắn hoặc bị chuẩn hóa mạnh, bằng chứng pháp lý ngắn có thể bị mất trọng số khi retrieval.
- Rủi ro: thiếu citation nhưng downstream vẫn cố suy luận.

#### 4.1.2. Lỗi phân ranh giới (Boundary Drift)
- Ranh giới đánh số lồng (b1, (i), 1.2.3) có thể gây split chưa tối ưu.
- Hệ quả: nhiễu ghép cặp và tăng false modified/false added.

### 4.2. Nguyên nhân từ khả năng kiểm soát Ràng buộc LLM
- Nếu bundle evidence không đủ mạnh, Local LLM có xu hướng suy diễn vượt citation.
- Cần siết refusal policy theo nguyên tắc: Không bằng chứng -> Không kết luận.

## 5. KẾ HOẠCH CẢI TIẾN CHẤT LƯỢNG KỲ SAU (TUẦN 11)

### 5.1. Nâng cấp Pipeline mảnh hóa (Advanced Chunking Tuning)
- Triển khai tiered-drop thay cho hard-drop.
- Mở rộng regex phân ranh văn bản pháp lý Việt cho numbering lồng.

### 5.2. Mở rộng chiến lược Trích xuất và Đối soát (Matching Strategy)
- Preset khuyến nghị sau calibration vòng 1: {best.get('preset')}.
- Tham số:
- unchanged_threshold = {best_params.get('unchanged_threshold')}
- modified_threshold = {best_params.get('modified_threshold')}
- near_unchanged_band = {best_params.get('near_unchanged_band')}
- clause_hint_alpha = {best_params.get('clause_hint_alpha')}
- clause_hint_min_gap = {best_params.get('clause_hint_min_gap')}

### 5.3. Tối ưu hóa Ràng buộc Prompt (Prompt Engineering Guardrails)
- Bổ sung refusal mechanism khi evidence bundle rỗng hoặc citation không hợp lệ.
- Ép schema output: answer_type=insufficient_evidence trong mọi trường hợp thiếu bằng chứng.

## Phụ lục: Dấu mốc dữ liệu nguồn
- Baseline report: data/matching_baseline_report.json
- Preset sweep: data/matching_threshold_presets.json
- Retrieval benchmark: test_results.md
"""


def main() -> None:
    cases = _load_ground_truth_cases()
    baseline = _read_json(BASELINE_PATH)
    preset = _read_json(PRESET_PATH)
    change_metrics = _load_change_metrics()
    retrieval_text = RETRIEVAL_PATH.read_text(encoding="utf-8", errors="ignore")

    dataset = _dataset_statistics(cases)
    retrieval = _extract_retrieval_metrics(retrieval_text)

    report = _build_markdown(dataset, baseline, preset, retrieval, change_metrics)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
