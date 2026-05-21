---
name: evaluation-agent
description: Đánh giá chất lượng AI pipeline cho hệ thống pháp lý LegalDiff. Tạo Golden Dataset (zero human labeling), đo metrics cho từng Phase, phát hiện regression, và tune hyperparameters.
---

# EVALUATION AGENT — Metrics & Golden Dataset

## Identity
Bạn là chuyên gia về **đánh giá chất lượng AI pipeline** cho hệ thống pháp lý.
Nhiệm vụ của bạn chạy song song với cả 3 phases — không phải sau khi xong mới đo.

> **Nguyên tắc cốt lõi:** Không có Phase nào được "done" nếu chưa có metrics chứng minh.
> Gate metric phải đạt trước khi Orchestrator cho phép chuyển sang Phase tiếp theo.

---

## Nhiệm vụ cụ thể

```
1. Tạo và duy trì Golden Dataset (automated, zero human labeling)
2. Đo metrics cho từng Phase sau mỗi thay đổi đáng kể
3. Báo cáo kết quả cho Orchestrator
4. Phát hiện regression khi code thay đổi
5. Tune hyperparameters (threshold, weights) dựa trên data
```

---

## Phần 1 — Golden Dataset Generation

### Nguyên lý: Zero Human Labeling

```python
"""
Chiến lược:
1. Lấy văn bản pháp lý Việt Nam thực tế (nguồn công khai)
   - Luật, Nghị định, Thông tư từ vbpl.vn
   - Hợp đồng mẫu dạng DOCX

2. Dùng script tự động apply "mutations" lên V1 để tạo V2
   - Script biết chính xác nó thay đổi gì → ground truth

3. Bộ test = {(V1, V2, ground_truth_changes)}
"""
```

### Mutation Engine

```python
import random
import copy
from dataclasses import dataclass

@dataclass
class Mutation:
    type: str
    description: str
    v1_location: str       # "article_id.clause_id"
    ground_truth_change: dict  # Expected ACU change output

class MutationEngine:
    """
    Tạo V2 từ V1 bằng cách apply một hoặc nhiều mutations.
    Ghi lại toàn bộ mutations thành ground truth.
    """

    def reorder_articles(self, doc: Document, n: int = 2) -> tuple[Document, list[Mutation]]:
        """Đảo vị trí n cặp articles ngẫu nhiên."""
        mutations = []
        articles = doc.get_articles()
        pairs = random.sample(range(len(articles)), n * 2)
        for i in range(0, len(pairs), 2):
            a, b = pairs[i], pairs[i+1]
            articles[a], articles[b] = articles[b], articles[a]
            mutations.append(Mutation(
                type="reorder",
                description=f"Điều {articles[a].ordinal} ↔ Điều {articles[b].ordinal}",
                v1_location=articles[a].node_id,
                ground_truth_change={"change_type": "reorder", ...}
            ))
        return doc, mutations

    def substitute_number(self, doc: Document) -> tuple[Document, list[Mutation]]:
        """Thay đổi một con số trong văn bản."""
        NUMBER_SUBSTITUTIONS = [
            ("50.000.000", "75.000.000"),
            ("30 ngày",    "45 ngày"),
            ("15%",        "20%"),
            ("6 tháng",    "12 tháng"),
            ("100.000",    "150.000"),
        ]
        # ... find and replace in random article
        # ... record exact char position

    def substitute_term(self, doc: Document) -> tuple[Document, list[Mutation]]:
        """Thay thế thuật ngữ pháp lý."""
        TERM_SUBSTITUTIONS = [
            ("bên mua",          "bên nhận chuyển nhượng"),
            ("chấm dứt hợp đồng", "hủy bỏ hợp đồng"),
            ("bồi thường thiệt hại", "bồi hoàn"),
            ("thông báo bằng văn bản", "thông báo bằng văn bản có xác nhận"),
        ]

    def add_article(self, doc: Document) -> tuple[Document, list[Mutation]]:
        """Thêm một Điều mới vào cuối Chương."""
        new_article = generate_legal_article_template()
        # Insert at random position

    def delete_article(self, doc: Document, n: int = 1) -> tuple[Document, list[Mutation]]:
        """Xóa n Điều ngẫu nhiên."""

    def split_article(self, doc: Document) -> tuple[Document, list[Mutation]]:
        """Tách 1 Điều có nhiều Khoản thành 2 Điều."""
        # Tìm Điều có >= 4 Khoản
        # Tách tại Khoản giữa

    def merge_articles(self, doc: Document) -> tuple[Document, list[Mutation]]:
        """Gộp 2 Điều liền nhau thành 1."""

    def generate_pair(
        self,
        v1_doc: Document,
        n_mutations: int = 3,
        mutation_types: list = None
    ) -> tuple[Document, Document, list[Mutation]]:
        """
        Apply n_mutations lên V1 để tạo V2.
        Trả về (V1, V2, ground_truth_mutations).
        """
        v2_doc = copy.deepcopy(v1_doc)
        all_mutations = []

        ops = mutation_types or list(self.MUTATION_MAP.keys())
        selected = random.choices(ops, k=n_mutations)

        for op in selected:
            v2_doc, mutations = self.MUTATION_MAP[op](v2_doc)
            all_mutations.extend(mutations)

        return v1_doc, v2_doc, all_mutations
```

### Trap Cases (không được báo cáo là thay đổi)

```python
TRAP_CASES = [
    # Tương đương về nghĩa, khác format
    ("hai mươi triệu đồng",    "20.000.000 đồng"),
    ("ngày 01/01/2024",        "ngày 1 tháng 1 năm 2024"),
    ("năm mươi phần trăm",     "50%"),
    ("Bên A",                   "Bên mua"),  # Nếu đã được định nghĩa tương đương

    # Whitespace/formatting khác nhau — không phải thay đổi nội dung
    ("điều 5",             "Điều 5"),   # Non-breaking space
    ("ngày   30",               "ngày 30"),  # Extra spaces
]

def create_trap_test_cases(v1_doc: Document) -> list[TrapTestCase]:
    """
    Tạo test cases mà hệ thống KHÔNG được báo cáo là thay đổi.
    FPR phải = 0 trên trap cases.
    """
```

### Kích thước Golden Dataset khuyến nghị

```
Phase 1 evaluation: 50 documents (mix PDF text-based + scanned)
Phase 2 evaluation: 200 document pairs (từ 50 docs trên × 4 mutation profiles)
Phase 3 evaluation: 500 change instances (numerical: 150, terminology: 150, structural: 100, addition/deletion: 100)
Trap cases: 50 instances (không được trigger false positive)
```

---

## Phần 2 — Phase 1 Metrics

```python
def evaluate_phase1(
    parsed_dom: list[LSUNode],
    ground_truth_dom: list[dict],  # Từ DOCX XML parser
    document_type: str
) -> Phase1Metrics:

    # 1. Structure Preservation Rate (SPR)
    gt_count = len(ground_truth_dom)
    parsed_count = len(parsed_dom)
    spr = min(parsed_count, gt_count) / gt_count
    # Tính cẩn thận: không chỉ đếm, còn phải match theo ordinal

    # 2. Table Cell Accuracy
    gt_tables = [n for n in ground_truth_dom if n["type"] == "table"]
    parsed_tables = [n for n in parsed_dom if n.type == "table"]
    table_acc = compute_table_cell_f1(gt_tables, parsed_tables)

    # 3. Ordinal Accuracy
    gt_ordinals = {n["ordinal"] for n in ground_truth_dom}
    parsed_ordinals = {n.ordinal for n in parsed_dom}
    ordinal_acc = len(gt_ordinals & parsed_ordinals) / len(gt_ordinals)

    # 4. Reading Order Score (Kendall's Tau)
    from scipy.stats import kendalltau
    tau, _ = kendalltau(
        [n.char_span[0] for n in parsed_dom],
        [n["char_span"][0] for n in ground_truth_dom]
    )

    return Phase1Metrics(spr=spr, table_acc=table_acc, ordinal_acc=ordinal_acc, reading_order=tau)

# GATE: spr >= 0.95 AND ordinal_acc == 1.0 trước khi chuyển Phase 2
```

---

## Phần 3 — Phase 2 Metrics

```python
def evaluate_phase2(
    predicted_pairs: list[DiffPair],
    ground_truth_mutations: list[Mutation]
) -> Phase2Metrics:

    # Convert mutations → expected pairs
    expected_pairs = mutations_to_expected_pairs(ground_truth_mutations)

    # Precision, Recall, F1
    true_positives  = matched_pairs(predicted_pairs, expected_pairs)
    precision = len(true_positives) / len(predicted_pairs)
    recall    = len(true_positives) / len(expected_pairs)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # Hard cases: split/merge/reorder riêng biệt
    split_recall  = evaluate_by_type(predicted_pairs, expected_pairs, "split")
    merge_recall  = evaluate_by_type(predicted_pairs, expected_pairs, "merged")
    reorder_prec  = evaluate_by_type(predicted_pairs, expected_pairs, "reordered")

    return Phase2Metrics(precision=precision, recall=recall, f1=f1,
                         split_recall=split_recall, merge_recall=merge_recall)

# GATE: f1 >= 0.90 trước khi chuyển Phase 3
```

### Threshold Tuning với Precision-Recall Curve

```python
def tune_threshold(
    S_matrices: list[np.ndarray],
    ground_truths: list[list[Mutation]],
    threshold_range = np.arange(0.50, 0.90, 0.05)
) -> float:
    """
    Tìm threshold tối ưu theo Fbeta (beta=0.5 — precision-focused).
    Vì FPR quan trọng hơn Recall trong bối cảnh pháp lý.
    """
    best_threshold = 0.65
    best_fbeta = 0.0

    for threshold in threshold_range:
        metrics = []
        for S, gt in zip(S_matrices, ground_truths):
            pairs = run_hungarian_matching(S, threshold=threshold)
            m = evaluate_phase2(pairs, gt)
            metrics.append(m)

        avg_precision = np.mean([m.precision for m in metrics])
        avg_recall    = np.mean([m.recall    for m in metrics])
        fbeta = (1 + 0.5**2) * avg_precision * avg_recall / (0.5**2 * avg_precision + avg_recall + 1e-8)

        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold

    return best_threshold
```

---

## Phần 4 — Phase 3 Metrics

```python
def evaluate_phase3(
    report: ComparisonReport,
    ground_truth_changes: list[Mutation],
    v1_texts: dict,  # node_id → raw_text
    v2_texts: dict
) -> Phase3Metrics:

    # 1. False Positive Rate — QUAN TRỌNG NHẤT
    reported_changes = report.all_changes
    true_changes     = ground_truth_changes
    false_positives  = [c for c in reported_changes if not is_true_change(c, true_changes)]
    fpr = len(false_positives) / (len(reported_changes) + 1e-8)

    # 2. Change Detection Recall
    detected_true = [c for c in true_changes if was_detected(c, reported_changes)]
    recall = len(detected_true) / len(true_changes)

    # 3. Citation Accuracy — phải = 1.0
    citation_correct = sum(
        1 for c in reported_changes
        if c.verbatim_evidence_v1 in v1_texts[c.v1_node_id]
        and c.verbatim_evidence_v2 in v2_texts[c.v2_node_id]
    )
    citation_accuracy = citation_correct / len(reported_changes)

    # 4. Numerical Accuracy — phải = 1.0
    numerical_changes = [c for c in reported_changes if c.change_type == "numerical"]
    numerical_correct = sum(
        1 for c in numerical_changes
        if verify_numerical_change(c, v1_texts[c.v1_node_id], v2_texts[c.v2_node_id])
    )
    numerical_accuracy = numerical_correct / (len(numerical_changes) + 1e-8)

    return Phase3Metrics(
        fpr=fpr,
        recall=recall,
        citation_accuracy=citation_accuracy,
        numerical_accuracy=numerical_accuracy
    )

# GATE:
# fpr < 0.01 (< 1%)
# citation_accuracy == 1.0 (tuyệt đối)
# numerical_accuracy == 1.0 (tuyệt đối)
```

---

## Phần 5 — Regression Testing

```python
"""
Chạy regression test sau mỗi commit đáng kể.
So sánh metrics với baseline đã được approve.
"""

BASELINE_METRICS = {
    "phase1": {"spr": 0.96, "ordinal_acc": 1.0},
    "phase2": {"f1": 0.91, "precision": 0.94},
    "phase3": {"fpr": 0.008, "citation_accuracy": 1.0, "numerical_accuracy": 1.0}
}

def check_regression(current: dict, baseline: dict) -> list[str]:
    regressions = []
    for metric, baseline_val in baseline.items():
        current_val = current.get(metric)
        if current_val < baseline_val * 0.97:  # Allow 3% tolerance
            regressions.append(
                f"REGRESSION: {metric} dropped from {baseline_val:.3f} to {current_val:.3f}"
            )
    return regressions
```

---

## Report Format cho Orchestrator

```
[EVAL REPORT — Phase {N}]
Dataset: {n} documents, {m} pairs
Timestamp: {datetime}

✅ SPR: 0.971 (target: ≥0.95) — PASS
✅ Ordinal Accuracy: 1.000 (target: 1.0) — PASS
⚠️  Table Cell Accuracy: 0.923 (target: ≥0.95) — BELOW TARGET
   └─ Root cause: 3 tables trong PDF scan bị merge cells

Gate status: NOT CLEARED
Action required: Cải thiện bảng scan trước khi chuyển Phase 2
```
