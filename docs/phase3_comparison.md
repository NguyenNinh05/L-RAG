# Phase 3 — Generative Comparison (End-to-End)

> Tài liệu mô tả chi tiết **Module Comparison** của hệ thống LegalDiff (L-RAG):
> từ một `DiffPair` (Phase 2) → **LLM trích xuất ACU** → **Verification 3 tầng
> chống ảo giác** → **Executive Summary + Markdown** → `ComparisonReport`.
> Toàn bộ nội dung được đối chiếu trực tiếp với mã nguồn tại `src/comparison/`
> và ví dụ thật từ `tests/tand_signed_comparison_output/15_report_06_*.json`.

---

## Mục lục

1. [Vai trò của Phase 3 trong pipeline](#1-vai-trò-của-phase-3-trong-pipeline)
2. [Bài toán Phase 3 giải quyết & triết lý Zero-Hallucination](#2-bài-toán-phase-3-giải-quyết--triết-lý-zero-hallucination)
3. [Sơ đồ luồng dữ liệu end-to-end (4 tầng)](#3-sơ-đồ-luồng-dữ-liệu-end-to-end-4-tầng)
4. [Cấu trúc source code & Public API](#4-cấu-trúc-source-code--public-api)
5. [Cấu hình (config)](#5-cấu-hình-config)
6. [Data Models — `models.py`](#6-data-models--modelspy)
7. [Đầu vào: từ `DiffPair` → `ComparisonRequest`](#7-đầu-vào-từ-diffpair--comparisonrequest)
8. [LLM Client — `llm_client.py`](#8-llm-client--llm_clientpy)
9. [Tier 1 — ACU Extraction (`report_generator.py`)](#9-tier-1--acu-extraction-report_generatorpy)
10. [Các chiến lược tăng recall (B / C1 / C2 / S4)](#10-các-chiến-lược-tăng-recall-b--c1--c2--s4)
11. [Verification Engine — `verifier.py` (Tier 2 & Tier 3)](#11-verification-engine--verifierpy-tier-2--tier-3)
12. [Orchestrator — `GenerativeComparisonPipeline`](#12-orchestrator--generativecomparisonpipeline)
13. [Tier 4 — Executive Summary & Markdown render](#13-tier-4--executive-summary--markdown-render)
14. [Worked example end-to-end (từ dữ liệu thật)](#14-worked-example-end-to-end-từ-dữ-liệu-thật)
15. [Tích hợp backend (Celery worker)](#15-tích-hợp-backend-celery-worker)
16. [Nguyên tắc bất biến (Design invariants)](#16-nguyên-tắc-bất-biến-design-invariants)
17. [Cách chạy & mở rộng](#17-cách-chạy--mở-rộng)

---

## 1. Vai trò của Phase 3 trong pipeline

| Phase | Tên | Module | Vai trò |
|-------|-----|--------|---------|
| 1 | Ingestion | `src.ingestion` | Parse PDF/DOCX → Legal DOM + LSU chunks |
| 2 | Alignment | `src.alignment` | Ghép cặp tối ưu các Điều V1↔V2 → `DiffPairCatalog` |
| **3** | **Generative Comparison** | `src.comparison` | **(tài liệu này)** Mỗi `DiffPair` → LLM sinh diff có evidence → verify → báo cáo |

**Phase 2 trả lời câu hỏi "Điều nào của V1 tương ứng Điều nào của V2?"** (bài toán
*alignment*). **Phase 3 trả lời câu hỏi "Giữa hai đoạn đã được ghép cặp, CHÍNH XÁC
những gì đã thay đổi?"** (bài toán *diff*). Phase 2 chỉ cần đủ giống để ghép cặp;
Phase 3 phải liệt kê **mọi** sự khác biệt — từ đổi một con số, đảo một chữ "không"
thành "có thể", đến thêm/xoá cả một bảng — kèm **bằng chứng trích dẫn nguyên văn**
từ cả hai phiên bản.

> **Vai trò của LLM trong Phase 3 là *transcribe* (phiên dịch diff), KHÔNG phải
> *synthesize*.** Điều này khác căn bản với RAG truyền thống (LLM tổng hợp câu
> trả lời). Ở đây LLM chỉ được phép mô tả thay đổi **có thật trong văn bản**; mọi
> phát ngôn đều phải *traceable* tới một chuỗi ký tự gốc. Đó là cốt lõi của ràng
> buộc **Zero-Hallucination** (xem [§2](#2-bài-toán-phase-3-giải-quyết--triết-lý-zero-hallucination)).

---

## 2. Bài toán Phase 3 giải quyết & triết lý Zero-Hallucination

### Bài toán

Cho một `DiffPair` (đã biết `match_type` ∈ {`matched`, `added`, `deleted`,
`split`, `merged`}) cùng `raw_text_v1`, `raw_text_v2`, sinh ra:

- **Lớp machine-readable:** danh sách **ACU** (Atomic Comparison Unit) — mỗi ACU
  là *một* thay đổi nguyên tử, có phân loại, giá trị trước/sau, và evidence nguyên
  văn.
- **Lớp human-readable:** Executive Summary tiếng Việt + báo cáo Markdown có cấu
  trúc (bảng thống kê, nhóm theo loại thay đổi, nhật ký ACU bị bác bỏ).

### Triết lý Zero-Hallucination (3 ràng buộc tuyệt đối của dự án)

| Ràng buộc | Phase 3 đảm bảo bằng cách nào |
|-----------|-------------------------------|
| **100% offline/local** | LLM gọi qua OpenAI-compatible server chạy local (vLLM / llama-cpp / Ollama). Provider duy nhất "ngoại tuyến logic" là DeepSeek (optional, opt-in) — không bắt buộc. |
| **Zero Hallucination** | (1) Prompt ép LLM chỉ trích nguyên văn; (2) **VerificationEngine 3 tầng** kiểm tra từng ACU bằng Python deterministic (string/regex/fuzzy) — không tin tưởng LLM; (3) ACU fail → **drop + log**, không bao giờ giữ im lặng. |
| **Bảo mật dữ liệu** | Toàn bộ text pháp lý xử lý trong máy nội bộ; không gửi đi đâu nếu dùng provider `local`. |

> **Quan trọng:** Việc xác minh (verification) là **deterministic** — không phải
> một LLM khác "chấm điểm" LLM thứ nhất. Tier 2 dùng `in` operator +
> `SequenceMatcher`; Tier 3 dùng regex số liệu. Điều này làm guardrail có thể tái
> lặp, không phụ thuộc may rủi của model.

---

## 3. Sơ đồ luồng dữ liệu end-to-end (4 tầng)

```
 DiffPair (Phase 2)                          pair_id, match_type
 list[str] v1_texts    ──"\n\n".join──▶      raw_text_v1
 list[str] v2_texts    ──"\n\n".join──▶      raw_text_v2
        │
        ▼
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║                   PHASE 3 — GENERATIVE COMPARISON                     ║
 ║                   (GenerativeComparisonPipeline)                       ║
 ╠═══════════════════════════════════════════════════════════════════════╣
 ║                                                                       ║
 ║  SHORT-CIRCUIT: nếu norm(v1) == norm(v2) → bỏ qua LLM, trả []         ║
 ║                                                                       ║
 ║  ┌─ TẦNG 1 — ACU Extraction (LLM, temp thấp 0.05) ──────────────────┐ ║
 ║  │  build_acu_user_prompt + logic_hint(S4) + number_hint(C1)         │ ║
 ║  │  → LocalLLMClient.chat_json → {"acus":[ {...}, ... ]}             │ ║
 ║  │  + Windowing (B) nếu text dài  + Two-pass (C2) để tăng recall     │ ║
 ║  │  + dedupe theo (change_type, original, new)                       │ ║
 ║  └──────────────────────────────┬────────────────────────────────────┘ ║
 ║                                 ▼ list[ACUOutput] (raw)                ║
 ║  ┌─ (DeepSeek only) SELF-VERIFICATION (LLM reasoning) ──────────────┐ ║
 ║  │  gửi lại ACU cho model tự flag hallucination TRƯỚC Verification   │ ║
 ║  └──────────────────────────────┬────────────────────────────────────┘ ║
 ║                                 ▼                                      ║
 ║  PRE-FILTER: drop ACU có confidence < min_confidence_to_include (0.2) ║
 ║                                 ▼                                      ║
 ║  ┌─ TẦNG 2 — Evidence Verification (deterministic) ─────────────────┐ ║
 ║  │  verbatim_evidence_v1 ∈ raw_text_v1 ?  (exact → ws-norm → fuzzy)  │ ║
 ║  │  verbatim_evidence_v2 ∈ raw_text_v2 ?  fuzzy ≥ 0.85               │ ║
 ║  │  → PASSED / FAILED_EVIDENCE                                        │ ║
 ║  └──────────────────────────────┬────────────────────────────────────┘ ║
 ║                  FAIL → drop      │ PASS                                ║
 ║                                 ▼                                      ║
 ║  ┌─ TẦNG 3 — Numerical Verification (deterministic, regex) ─────────┐ ║
 ║  │  Mọi số trong original_value/new_value phải có trong raw_text     │ ║
 ║  │  (chạy cho change_type=NUMERICAL VÀ mọi ACU có số — chiến lược S5)│ ║
 ║  │  strict mode: 100% số phải match → PASSED / FAILED_NUMERICAL      │ ║
 ║  └──────────────────────────────┬────────────────────────────────────┘ ║
 ║                  FAIL → drop      │ PASS                                ║
 ║                                 ▼ verified_acus                       ║
 ║  ┌─ TẦNG 4 — Executive Summary (LLM, temp 0.3) + Markdown render ───┐ ║
 ║  │  chat_json → ExecutiveSummary (overall_assessment, critical_…)     │ ║
 ║  │  render_markdown_report → markdown_report (header/summary/table/log)│ ║
 ║  └──────────────────────────────┬────────────────────────────────────┘ ║
 ║                                 ▼                                      ║
 ╚═════════════════════════════════╪═════════════════════════════════════╝
                                   ▼
                         ComparisonReport
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
 verified_acus / rejected_acus              executive_summary
 (machine-readable JSON)                    markdown_report (human)
```

> **Đếm tầng:** tài liệu này và code gọi là "4 tầng" (Tier 1–4). Tier 2 & 3 cùng
> sống trong `VerificationEngine` (xem [§11](#11-verification-engine--verifierpy-tier-2--tier-3));
> giữa chúng có short-circuit — Tier 3 chỉ chạy khi Tier 2 đã PASS.

---

## 4. Cấu trúc source code & Public API

```
src/comparison/
├── __init__.py          # Public API — lazy import cho symbol phụ thuộc openai
├── models.py            # Data models: ChangeType, VerificationStatus, ACUOutput,
│                        #   VerificationResult, ExecutiveSummary, ComparisonReport, ComparisonRequest
├── llm_client.py        # LocalLLMClient (AsyncOpenAI) + LLMConfig + truncation recovery
├── logic_detector.py    # detect_logic_signals / build_logic_hint — hint Category D (S4)
├── verifier.py          # VerificationEngine + VerificationConfig + extract_numbers (Tier 2 & 3)
├── report_generator.py  # GenerativeComparisonPipeline + PipelineConfig + prompts + Markdown render
└── acu_prompter.py      # ⚠️ Bản trích xuất prompt KHÔNG được import (xem ghi chú)
```

### Public API (`__init__.py`)

```python
from src.comparison import (
    # --- Models & Engine (import ngay, không cần openai) ---
    ACUOutput, ChangeType, ComparisonReport, ComparisonRequest,
    ExecutiveSummary, VerificationResult, VerificationStatus,
    VerificationEngine, VerificationConfig, extract_numbers,
    # --- LLM-dependent (lazy import) ---
    LocalLLMClient, LLMConfig,
    GenerativeComparisonPipeline, PipelineConfig,
)
```

`__init__.py` dùng `__getattr__` để **lazy-import** `LocalLLMClient`,
`LLMConfig`, `GenerativeComparisonPipeline`, `PipelineConfig` — những symbol phụ
thuộc thư viện `openai`. Nhờ đó `from src.comparison import VerificationEngine`
hoạt động ngay cả khi server `openai` chưa cài (ví dụ khi chỉ cần verify offline).

> **Ghi chú về `acu_prompter.py`:** file này chứa `ACU_SYSTEM_PROMPT`,
> `build_acu_user_prompt`, `SUMMARY_SYSTEM_PROMPT`, `build_summary_user_prompt` —
> trông giống prompt production, nhưng **không có file nào `import` nó** (đã kiểm
> tra toàn repo). Orchestrator thật (`report_generator.py`) **định nghĩa lại bản
> riêng, đầy đủ hơn** (8 rules, có trường `reasoning`, có biến thể DeepSeek — xem
> [§9](#9-tier-1--acu-extraction-report_generatorpy)). Tức `acu_prompter.py` hiện
> là bản refactor tách rời, chưa được wire vào pipeline. Khi đọc code Phase 3,
> nguồn sự thật về prompt là `report_generator.py`, **không** phải
> `acu_prompter.py`.

---

## 5. Cấu hình (config)

Phase 3 dùng **3 lớp config** + **YAML**:

### 5.1. `PipelineConfig` (`report_generator.py`) — cấu hình orchestrator

| Trường | Default | Ý nghĩa |
|--------|---------|---------|
| `llm_base_url` | `http://localhost:8000/v1` | Endpoint OpenAI-compatible |
| `llm_model_name` | `Qwen/Qwen2.5-7B-Instruct` | Tên model trên server |
| `llm_api_key` | `not-needed` | Local thường không cần; DeepSeek dùng key thật |
| `acu_temperature` | `0.05` | Nhiệt độ Tier 1 — **cực thấp** để JSON deterministic |
| `summary_temperature` | `0.3` | Nhiệt độ Tier 4 — hơi cao hơn cho văn phong |
| `max_tokens_acu` / `max_tokens_summary` | `4096` / `1024` | Giới hạn output token |
| `timeout_seconds` / `max_retries` | `120.0` / `3` | Retry với exponential backoff |
| `fuzzy_threshold` | `0.85` | Ngưỡng fuzzy match Tier 2 |
| `strict_numerical` | `True` | 100% số phải match (Tier 3) |
| `max_concurrency` | `4` | `asyncio.Semaphore` — số pair xử lý song song |
| `min_confidence_to_include` | `0.2` | Pre-filter confidence (TRƯỚC verification) |
| `provider` | `local` | `local` (Qwen/vLLM/Ollama) hoặc `deepseek` |
| `enable_self_verification` | `False` | DeepSeek-only: LLM tự verify trước Verification Engine |
| `windowing_threshold_chars` | `5000` | Nếu `max(len_v1,len_v2) > threshold` → windowing |
| `window_max_chars` / `window_overlap_chars` | `2500` / `400` | Kích thước cửa sổ |
| `enable_number_enumeration` | `True` | C1: đưa danh sách số vào prompt |
| `enable_second_pass` | `True` | C2: lượt 2 tìm thay đổi bổ sung |

### 5.2. `VerificationConfig` (`verifier.py`)

| Trường | Default | Ý nghĩa |
|--------|---------|---------|
| `fuzzy_match_threshold` | `0.85` | SequenceMatcher ratio tối thiểu để chấp nhận fuzzy |
| `min_evidence_length` | `5` | Evidence ngắn hơn → SKIPPED (= PASS) tránh false positive |
| `normalize_whitespace` | `True` | Chuẩn hoá khoảng trắng trước khi so |
| `strict_numerical` | `True` | `True` = 100% số phải có trong raw_text |
| `numerical_context_window` | `50` | Số ký tự quanh mỗi số (dự phòng ngữ cảnh) |

### 5.3. `LLMConfig` (`llm_client.py`)

Gói gọn endpoint + inference params + retry + `force_json_mode` (bật
`response_format={"type":"json_object"}`). Được tạo **2 instance** trong
orchestrator: một cho Tier 1 (low-temp), một cho Tier 4 (higher-temp).

### 5.4. YAML (`configs/`)

```yaml
# configs/pipeline_config.yaml — section comparison
comparison:
  fuzzy_match_threshold: 0.85
  min_evidence_length: 5
  normalize_whitespace: true
  strict_numerical: true
  numerical_context_window: 50
  min_confidence_to_include: 0.2
  max_concurrency: 1            # ← production đặt 1 (sequential, tránh overload GPU)

# configs/model_config.yaml — section llm + provider_presets
llm:
  provider: local
  model_name: qwen2.5:14b
  base_url: http://localhost:11434/v1     # Ollama
  temperature_acu: 0.05
  temperature_summary: 0.3
  max_tokens_acu: 8192
  timeout_seconds: 300.0
provider_presets:
  deepseek:
    model_name: deepseek-v4-flash
    base_url: https://api.deepseek.com/v1
    temperature_acu: 0.0
    max_tokens_acu: 8192
    timeout_seconds: 180.0
```

`src/config.py::get_llm_config(provider)` resolve theo provider: `local` đọc từ
`model_config.yaml`; `deepseek` merge preset với biến môi trường
(`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`). Provider auto-detect
từ `LLM_PROVIDER` env → YAML `llm.provider` → default `local`.

---

## 6. Data Models — `models.py`

Hierarchy (Pydantic v2):

```
ComparisonRequest        — Đầu vào pipeline (1 DiffPair + raw texts)
ACUOutput                — Atomic Comparison Unit (1 thay đổi nguyên tử)
VerificationResult       — ACU + kết quả sau Tier 2 & 3
ExecutiveSummary         — Tóm tắt tiếng Việt từ LLM (Tier 4)
ComparisonReport         — Đầu ra cuối: machine_readable + human_readable
```

### 6.1. Enums

```python
class ChangeType(str, Enum):       # Loại thay đổi nguyên tử
    NUMERICAL    = "numerical"     # đổi số/ngày/%/tiền/thời hạn
    TERMINOLOGY  = "terminology"   # đổi thuật ngữ pháp lý (không phải số)
    STRUCTURAL   = "structural"    # thêm/bỏ mệnh đề, đổi cấu trúc câu
    ADDITION     = "addition"      # nội dung hoàn toàn mới chỉ ở V2
    DELETION     = "deletion"      # nội dung V1 bị xoá khỏi V2
    REORDER      = "reorder"       # đổi thứ tự, nội dung không đổi

class VerificationStatus(str, Enum):
    PASSED           = "passed"
    FAILED_EVIDENCE  = "failed_evidence"   # evidence không có trong text gốc
    FAILED_NUMERICAL = "failed_numerical"  # số liệu không khớp
    SKIPPED          = "skipped"
```

### 6.2. `ACUOutput` — đơn vị cốt lõi

```python
class ACUOutput(BaseModel):
    acu_id: str               # "acu_" + 12 hex
    change_type: ChangeType
    location_v1: str          # "Điều X, Khoản Y, Điểm Z" (rỗng nếu addition)
    location_v2: str          # (rỗng nếu deletion)
    original_value: str       # giá trị/cụm từ GỐC ngắn gọn (rỗng nếu addition)
    new_value: str            # giá trị/cụm từ MỚI ngắn gọn (rỗng nếu deletion)
    verbatim_evidence_v1: str # ★ trích NGUYÊN VĂN từ V1 (rỗng ⇔ addition)
    verbatim_evidence_v2: str # ★ trích NGUYÊN VĂN từ V2 (rỗng ⇔ deletion)
    confidence: float         # [0,1] — validator clamp + default 0.5
    reasoning: str            # DeepSeek: Chain-of-Thought 1-2 câu
    pair_id: str              # điền bởi pipeline
    created_at: datetime      # UTC
```

★ Hai trường `verbatim_evidence_v1/v2` là **nền tảng Zero-Hallucination**: chúng
phải là **copy-paste chính xác** từ `raw_text`. Đây chính là cái Verification
Engine (Tier 2) sẽ kiểm tra. Validators:

- `_clamp_confidence`: clamp `confidence` về `[0,1]`; giá trị rác → `0.5` (để
  không reject mất ACU chỉ vì LLM quên điền).
- `_validate_evidence_consistency`: ghi nhận rule addition↔v1-rỗng /
  deletion↔v2-rỗng (hiện cho phép pass-through, vì VerificationEngine sẽ xử lý).

### 6.3. `VerificationResult`

Wrapper gói `ACUOutput` cùng kết quả verify:

```python
class VerificationResult(BaseModel):
    acu: ACUOutput
    status: VerificationStatus                 # PASSED / FAILED_*
    evidence_v1_found: bool | None             # evidence_v1 có trong raw_text_v1?
    evidence_v2_found: bool | None
    numerical_verified: bool | None            # số liệu đúng 100%?
    rejection_reason: str                      # lý do reject (nếu có)
    fuzzy_match_score_v1: float | None         # điểm fuzzy nếu exact fail
    fuzzy_match_score_v2: float | None

    @property
    def is_passed(self) -> bool: ...           # status == PASSED
```

### 6.4. `ComparisonReport` — đầu ra cuối

Có **2 lớp** output:

- **machine-readable**: `verified_acus: list[ACUOutput]` + `rejected_acus: list[VerificationResult]`.
- **human-readable**: `executive_summary: ExecutiveSummary | None` + `markdown_report: str`.

`_compute_stats` (model_validator) tự tính `total_acus_passed`,
`total_acus_rejected`, `total_acus_raw`, và `hallucination_rate` (=
rejected/raw). `to_machine_readable()` xuất dict JSON cho downstream; nguyên liệu
chính backend persist vào DB (xem [§15](#15-tích-hợp-backend-celery-worker)).

### 6.5. `ComparisonRequest` — đầu vào

```python
class ComparisonRequest(BaseModel):
    pair_id: str
    match_type: str           # "matched"/"added"/"deleted"/"split"/"merged"
    raw_text_v1: str          # rỗng nếu ADDED
    raw_text_v2: str          # rỗng nếu DELETED
    breadcrumb_v1: str        # "[Chương II > Điều 5]"
    breadcrumb_v2: str
    v1_doc_id: str
    v2_doc_id: str
```

---

## 7. Đầu vào: từ `DiffPair` → `ComparisonRequest`

Phase 3 **không** gọi Phase 2 — nó nhận `DiffPairCatalog` sẵn. Việc chuyển đổi
xảyy ở `src/pipeline.py` (Phase 3 section, ~dòng 292–308):

```python
all_pairs = catalog.pairs                       # MỌI match_type, không chỉ matched
requests = [
    ComparisonRequest(
        pair_id=pair.pair_id,
        match_type=pair.match_type.value,
        raw_text_v1="\n\n".join(pair.v1_texts),  # split/merged có nhiều text → nối
        raw_text_v2="\n\n".join(pair.v2_texts),
    )
    for pair in all_pairs
]
reports = await gen_pipeline.run_batch(requests, max_concurrency=cfg.max_concurrency)
```

Điểm đáng chú ý: Phase 3 xử lý **tất cả** loại pair — `added`/`deleted`/`split`/
`merged` cũng vào LLM (guidance_map trong prompt xử lý từng type, xem
[§9](#9-tier-1--acu-extraction-report_generatorpy)). Với `split`/`merged`, nhiều
`v1_texts`/`v2_texts` được concatenate bằng `"\n\n"`. Có thể giới hạn số pair qua
`max_comparison_pairs` (debug).

---

## 8. LLM Client — `llm_client.py`

`LocalLLMClient` bọc `openai.AsyncOpenAI` để gọi server OpenAI-compatible (vLLM /
llama-cpp-python / Ollama) theo **async**. Hai phương thức chính:

```python
result_dict = await client.chat_json(system_prompt, user_prompt)  # JSON mode bật
text        = await client.chat_text(system_prompt, user_prompt)  # plain text
ok          = await client.health_check()                          # models.list()
```

### Retry & backoff

`_chat_completion` lặp `max_retries + 1` lần:

- `APIConnectionError` / `APITimeoutError` → retry, delay = `base * 2^attempt`.
- `RateLimitError` → retry, delay gấp đôi (x2).
- Hết lượt → raise `RuntimeError`.

Ghi log `finish_reason`, `prompt_tokens`, `completion_tokens`. Nếu
`finish_reason == "length"` (output bị cắt) → cảnh báo tăng `max_tokens`.

### JSON mode & truncation recovery

Khi `force_json_mode=True`, client gửi `response_format={"type":"json_object"}`.
Parser `_parse_json_response` xử lý 3 trường hợp rủi ro của LLM local:

1. **Markdown fence**: ` ```json ... ``` ` → bóc phần trong.
2. **Preamble**: model thêm chữ rườm → lấy JSON object đầu tiên bằng regex
   `\{.*\}`.
3. **Truncation** (`finish_reason=length`): khi danh sách ACU dài vượt
   `max_tokens`, JSON bị cắt giữa chừng → `_extract_complete_json_objects`
   **salvage** các object `{...}` đã đóng ngoặc đầy đủ (dùng stack vị trí `{`,
   track string/escape để không bị nhầm ngoặc trong chuỗi). Chỉ giữ object có
   khóa `"change_type"` (= ACU), bỏ wrapper `{"acus": ...}`.

```python
# VD response bị cắt: {"acus": [{...ok}, {...ok}, {...CẮT
# → salvage 2 ACU đầu, bỏ object cuối đang dở.
```

Đây là cơ chế **graceful degradation**: thay vì drop toàn bộ response khi bị cắt,
ta giữ được phần lớn ACU đã hoàn chỉnh.

---

## 9. Tier 1 — ACU Extraction (`report_generator.py`)

Hàm vào cửa: `_tier1_extract_acus(request)`. Đây là tầng **LLM**, orchestrator
chạy **3 chiến lược** xen kẽ để tối đa recall (chi tiết [§10](#10-các-chiến-lược-tăng-recall-b--c1--c2--s4)):

```
_tier1_extract_acus(request)
   ├─ _build_number_hint(v1, v2)          # C1: liệt kê số liệu tự động
   ├─ _extract_windowed(...)              # lượt 1: single-call HOẶC windowing (B)
   │     └─ _llm_extract_acus()           #   build prompt → chat_json → parse ACU
   ├─ _dedupe_acus(first)                 # bỏ ACU trùng (cross-window)
   ├─ _second_pass(...)  nếu matched/split/merged  # C2: lượt 2 tìm thêm
   └─ _merge_dedupe(first, extra)         # gộp, bỏ trùng
```

### System prompt (`ACU_SYSTEM_PROMPT`)

Prompt cho Qwen local, **8 rules** bắt buộc (vi phạm = output bị bác bỏ):

| Rule | Nội dung |
|------|----------|
| 1 | **Chỉ trích nguyên văn** — `verbatim_evidence_v1/v2` phải copy-paste, NGHIÊM CẤM paraphrase |
| 2 | **Mỗi ACU = 1 thay đổi duy nhất** — tách mịn tối đa (đổi "30→45 ngày" + "500k→600k" là 2 ACU) |
| 3 | **Phân loại `change_type` chính xác** |
| 4 | **Evidence cho từng type** — addition→v1 rỗng, deletion→v2 rỗng, type khác→cả 2 có |
| 5 | **Không suy diễn** — chỉ mô tả thực tế, không luận hậu quả pháp lý |
| 6 | **Confidence** — bắt buộc điền cho mọi ACU |
| 7 | **Không bỏ qua thay đổi nhỏ** — mọi khác biệt đều báo cáo |
| 8 | **Thay đổi logic/nghĩa pháp lý** — Category D, đặt confidence cao |

Kèm **few-shot example** trong prompt chỉ cách tách ACU. Yêu cầu trả về **DUY
NHẤT** JSON `{"acus":[...]}`, rỗng thì `{"acus":[]}`.

### Biến thể DeepSeek (`ACU_SYSTEM_PROMPT_DEEPSEEK`)

Khi `provider == "deepseek"`, orchestrator chọn prompt này — khác biệt:

- Thêm **"THINK STEP-BY-STEP"** — tận dụng Chain-of-Thought của DeepSeek.
- Thêm trường **`reasoning`** bắt buộc vào schema (1-2 câu giải thích cơ chế).
- RULE 8 và RULE 9 nhấn mạnh logic-shift + điền reasoning.

### User prompt (`build_acu_user_prompt`)

- Bọc text trong thẻ XML `<v1_text>...</v1_text>` / `<v2_text>...</v2_text>`.
- `guidance_map` theo `match_type`:

  | match_type | Hướng dẫn đặc biệt |
  |------------|--------------------|
  | `matched`  | tập trung thay đổi tinh tế |
  | `added`    | tạo 1 ACU `addition` duy nhất |
  | `deleted`  | tạo 1 ACU `deletion` duy nhất |
  | `split`    | xác định nội dung giữ nguyên vs thay đổi/bổ sung |
  | `merged`   | xác định nội dung giữ nguyên vs bị xoá/thay đổi |

- Nối thêm `logic_hint` (S4) và `number_hint` (C1) / extra (C2) ở cuối.

### `_llm_extract_acus` — core call

1. **Short-circuit**: nếu `norm(v1) == norm(v2)` (sau collapse whitespace) → trả
   `[]`, **bỏ qua LLM call** (tiết kiệm chi phí + thời gian).
2. Build user prompt + logic hint + extra instructions.
3. `chat_json(system=self._acu_system_prompt, user=...)`.
4. Parse `raw_json["acus"]` (phải là list).
5. Với mỗi dict: `ACUOutput.model_validate(...)`. Nếu `ValidationError` → gọi
   `_attempt_acu_recovery` (sửa `change_type` rác → `structural`, `confidence`
   None → `0.5`) rồi validate lại; vẫn fail → drop ACU đó.

---

## 10. Các chiến lược tăng recall (B / C1 / C2 / S4)

LLM local thường **bỏ sót** thay đổi khi danh sách dài (truncation) hoặc khi thay
đổi tinh tế (đảo nghĩa). Phase 3 thêm 4 chiến lược **deterministic, rẻ** để ép
LLM kỹ hơn:

### S4 — Logic-signal hint (`logic_detector.py`)

Quét token-level hai văn bản, đếm sự chênh lệch của "logic lexicon" pháp lý Việt
Nam:

```python
LOGIC_LEXICON = [
  "không được phép", "không được", "không thể", "không",
  "được phép", "được", "phải", "có thể", "có quyền", "được quyền",
  "bắt buộc", "cấm", "cho phép", "không cấm",
  "trừ khi", "chỉ khi", "nếu không", "nếu", "ngoại trừ",
  "trong trường hợp", "khi", "điều kiện là",
  "chỉ", "duy nhất", "tất cả", "mọi", "toàn bộ", "một phần", "không quá",
]
```

`detect_logic_signals(v1, v2)` so số lượng mỗi token; nếu chênh → phát tín hiệu
dạng `"thêm 'có thể'"`, `"bỏ 'không'"`. Sắp xếp token dài hơn lên đầu (specific
hơn). `build_logic_hint` nối thành chuỗi **HINT** gắn vào Tier-1 prompt:

> ⚠️ Dấu hiệu thay đổi LOGIC/NGHĨA phát hiện (V1→V2): thêm 'có thể', bỏ 'không'.
> Hãy kiểm tra kỹ… kèm verbatim evidence.

**Đây chỉ là *hint*, không phải gate/reject** — ACU vẫn phải qua VerificationEngine
như thường. Rất quan trọng cho **Category D** (đảo nghĩa pháp lý) — loại thay đổi
mà LLM dễ bỏ sót nhất.

### B — Windowing (tránh truncation)

Nếu `max(len_v1, len_v2) > windowing_threshold_chars` (mặc định 5000; DeepSeek
nâng lên 20000 nhờ context 128K) → `_split_text_windows` chia text thành cửa sổ
~`window_max_chars` (2500), **cắt theo ranh giới câu** tiếng Việt (regex nhận
chữ hoa có dấu). Mỗi cặp cửa sổ (vị trí tương ứng) → 1 LLM call riêng. ACU từ mọi
cửa sổ được gom rồi **dedupe**.

### C1 — Number enumeration

`_build_number_hint` dùng `extract_numbers` (cùng regex Tier 3) để liệt kê sẵn
mọi số/ngày/tiền của V1 & V2, đưa vào prompt:

> ℹ️ DANH SÁCH SỐ LIỆU/NGÀY… V1: 30, 500.000 đồng, 01/06/2025. V2: 45, 600.000 đồng.
> → Mỗi sự KHÁC NHAU phải thành 1 ACU numerical riêng.

Buộc LLM đối chiếu từng số, giảm bỏ sót số liệu.

### C2 — Two-pass extraction

Sau lượt 1, `_second_pass` gửi lại danh sách ACU đã tìm + yêu cầu: "liệt kê CHỈ
các thay đổi BỔ SUNG chưa có, ưu tiên số/ngày, từ phủ định/cho phép, phạm vi &
điều kiện". ACU mới → `_merge_dedupe`. **Bỏ qua** với `added`/`deleted` (chỉ sinh
1 ACU toàn bài, lượt 2 không thêm gì). Tăng recall cho cặp phức tạp.

### Dedupe

`_acu_dedupe_key(acu)` = `(change_type, norm(original_value), norm(new_value))`
(chuẩn hoá: lowercase, collapse ws, bỏ dấu câu). ACU trùng (cross-window /
cross-pass) bị gộp, giữ lần đầu. ACU **không có giá trị** (addition/deletion thuần)
được **bỏ qua dedupe** (giữ nguyên) vì key không ổn định khi rỗng.

---

## 11. Verification Engine — `verifier.py` (Tier 2 & Tier 3)

Đây là **trái tim Zero-Hallucination** — xác minh **deterministic**, không tin
LLM. Điểm vào: `engine.verify(acu, raw_text_v1, raw_text_v2)`.

```
verify(acu)
  ├─ _verify_evidence(...)        ← TẦNG 2
  │     if FAIL → return FAILED_EVIDENCE     (short-circuit, không chạy Tier 3)
  └─ _verify_numerical(...)       ← TẦNG 3 (chỉ khi Tier 2 PASS)
        if FAIL → FAILED_NUMERICAL
        else    → PASSED
```

### Tier 2 — Evidence check (`_verify_evidence`)

Quyết định cần check evidence nào theo `change_type`:

- `ADDITION` → chỉ check `evidence_v2` (v1 rỗng).
- `DELETION` → chỉ check `evidence_v1` (v2 rỗng).
- Các type khác → check **cả 2**.

Mỗi evidence → `_evidence_in_text(evidence, raw_text)`, **3 bước** theo thứ tự ưu
tiên (nhanh → chậm):

1. **Exact match** (`evidence in text`) — O(n), thành công phần lớn trường hợp.
2. **Whitespace-normalised match** — collapse `\s+` rồi `in` (chống lỗi OCR/PDF).
3. **Fuzzy sliding-window** — `_fuzzy_evidence_match`: duyệt mọi cửa sổ dài
   80%–150% độ dài evidence (stride = `len//4`), tính `SequenceMatcher.ratio()`
   với mỗi cửa sổ, lấy max. Nếu ≥ `fuzzy_match_threshold` (0.85) → PASS.

Nếu evidence rỗng hoặc `< min_evidence_length` (5 ký tự) → **SKIPPED = PASS**
(tránh false positive trên evidence quá ngắn). Bất kỳ evidence nào `False` →
toàn ACU `FAILED_EVIDENCE`, ghi `rejection_reason` kèm score fuzzy.

> **Optimisation fuzzy:** text giới hạn 10 000 ký tự đầu; cửa sổ chỉ trong ±50%
> độ dài evidence; early-exout khi ratio = 1.0. Tránh O(n²) bùng nổ.

### Tier 3 — Numerical check (`_verify_numerical`)

Áp dụng **chiến lược S5**: chạy cho `change_type=NUMERICAL` **VÀ** mọi ACU nào có
số trong `original_value`/`new_value` (bắt cả số liệu bịa trong
terminology/structural/addition — vd bảng OCR misread bịa con số).

Quy trình:

1. `extract_numbers(acu.original_value)` → `numbers_from_original`.
2. `extract_numbers(acu.new_value)` → `numbers_from_new`.
3. Mỗi số V1 phải có trong `raw_text_v1`; mỗi số V2 phải có trong `raw_text_v2`
   (qua `_number_in_raw_text`).
4. `strict_numerical=True` (default): thiếu 1 số → `FAILED_NUMERICAL`.

`_number_in_raw_text` — 3 chiến lược khớp số:

1. **Exact** `raw_str in text` (vd `"30%"`).
2. **Normalised** — bỏ separator, chuẩn hoá `.,` rồi `in`.
3. **Value-based** — parse giá trị float, tìm cùng magnitude trong text (bỏ qua
   dấu `.,`).

### `extract_numbers` — regex multi-pattern

Lần lượt 7 pattern, **thứ tự quan trọng** (pattern dài/complex trước để không
overlap):

| Pattern | num_type | Ví dụ |
|---------|----------|-------|
| `\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}` | date | `01/06/2025` |
| `\d+([,\.]\d+)?\s*%` | percentage | `10,5%` |
| `\d{1,3}(\.\d{3})+(,\d+)?\s*(đồng\|VNĐ\|VND\|đ)?` | currency_vn | `500.000 đồng` |
| `(USD\|EUR\|$\|€...)\d{1,3}(,\.\d{3})+` | currency_intl | `$1,000.00` |
| `\d+[,\.]\d+` | float | `1,5` |
| `\d{2,}` | integer | `90` |
| `[1-9]` | integer_single | `5` |

Có `covered_spans` set chống duplicate từ overlapping patterns. Hàm này được dùng
cả ở Tier 3 (verify) **lẫn** C1 (build number hint) — **cùng regex**, đảm bảo nhất
quán.

### `verify_batch` — fail-soft

Verify từng ACU trong khối `try`; exception → đánh dấu `FAILED_EVIDENCE` (không
crash). Returns `list[VerificationResult]` tương ứng từng ACU vào.

---

## 12. Orchestrator — `GenerativeComparisonPipeline`

### Khởi tạo (`__init__`)

- **Provider auto-config**: nếu `provider == "deepseek"`:
  - Nâng `windowing_threshold_chars` → 20000, `window_max_chars` → 8000 (context
    128K).
  - Bật `enable_self_verification = True`.
  - Chọn `ACU_SYSTEM_PROMPT_DEEPSEEK`.
- Tạo **2 LLM client**: `_acu_llm` (temp 0.05, Tier 1) và `_summary_llm` (temp
  0.3, Tier 4) — cùng endpoint, nhiệt độ khác.
- Tạo `_verifier = VerificationEngine(fuzzy=0.85, strict_numerical=True)`.

### `run_single(request)` — luồng 1 pair qua 4 tầng

```python
raw_acus      = await _tier1_extract_acus(request)          # TẦNG 1 (LLM)

if provider == "deepseek" and enable_self_verification:
    raw_acus = await _deepseek_self_verify(...)              # (DeepSeek only)

filtered = [a for a in raw_acus if a.confidence >= min_confidence_to_include]  # PRE-FILTER
for a in filtered: a.pair_id = request.pair_id

results  = verifier.verify_batch(filtered, raw_text_v1, raw_text_v2)  # TẦNG 2 & 3
verified = [r.acu for r in results if r.is_passed]
rejected = [r for r in results if not r.is_passed]

summary  = await _tier4_generate_summary(verified, request) # TẦNG 4 (LLM)

report = ComparisonReport(verified_acus=verified, rejected_acus=rejected,
                          executive_summary=summary, ...)
report.markdown_report = render_markdown_report(report, summary)   # RENDER
return report
```

### DeepSeek Self-Verification (`_deepseek_self_verify`)

Chạy **trước** Verification Engine, chỉ DeepSeek. Gửi lại danh sách ACU (compact,
evidence cắt 200 ký tự, kèm reasoning) + raw text gốc (cắt 8000 ký tự) cho model
với vai trò "Quality Assurance". Model trả:
`{"verified":[{"acu_id":..., "confidence":...}], "flagged":[{"acu_id":..., "reason":...}]}`.

- ACU `flagged` → **drop** (hallucination do chính model xác nhận).
- `verified.confidence` → cập nhật lại (model có thể điều chỉnh).
- Khớp theo `acu_id` hậu tố (ID ngắn 8 ký tự).

Lý do: tận dụng khả năng self-reflection của DeepSeek — giảm hallucination rate
3–5% **trước** khi vào Verification deterministic. Nếu LLM call fail → giữ nguyên
tất cả ACU (best-effort, không làm hỏng pipeline).

> ⚠️ Lưu ý code: cuối `_deepseek_self_verify` có một dòng `return merged` **sau**
> `return acus` — đây là dead code (không bao giờ chạy). Lỗi nhỏ, không ảnh hưởng
> chức năng vì `merged` không tồn tại trong scope hàm này. Khi refactor nên xoá.

### `run_batch(requests)` — song song + fail-soft

```python
semaphore = asyncio.Semaphore(max_concurrency)
async def _bounded_run(req):
    async with semaphore:
        try:
            return await run_single(req)
        except Exception as exc:
            logger.exception(...)
            return ComparisonReport(pair_id=req.pair_id, markdown_report="# ❌ Lỗi xử lý...")
reports = await asyncio.gather(*[_bounded_run(r) for r in requests])
```

Một pair crash → trả report lỗi rỗng, **không** làm sập toàn batch. Giữ nguyên
thứ tự requests ↔ reports.

---

## 13. Tier 4 — Executive Summary & Markdown render

### `_tier4_generate_summary(verified_acus, request)`

- Nếu không có ACU nào → trả `ExecutiveSummary` tĩnh: "Không phát hiện thay đổi…".
- Không thì `build_summary_user_prompt` (JSON compact các ACU đã verified) +
  `SUMMARY_SYSTEM_PROMPT` (5 rules: chỉ dựa vào ACU, không suy diễn, tiếng Việt
  trang trọng, không có ACU thì nói rõ, trả JSON) → `_summary_llm.chat_json` →
  `ExecutiveSummary.model_validate`.
- **Fallback** (`_fallback_summary`): nếu LLM fail hoặc ValidationError → sinh
  summary **bằng Python** (Counter theo change_type, top-5 confidence) — không
  suy diễn, luôn có output.

`ExecutiveSummary`: `overall_assessment`, `critical_changes[]`,
`numerical_changes_summary`, `risk_flags[]`, `recommendation`.

### `render_markdown_report(report, exec_summary)`

Sinh Markdown hoàn chỉnh, layout:

```
# 📋 Báo cáo So sánh Văn bản Pháp lý
   (Report ID, V1/V2 doc_id, vị trí, thời điểm)
## 📊 Thống kê Tổng quan        ← bảng: tổng/passed/rejected/hallucination_rate
## 🔍 Tóm tắt Điều hành          ← ExecutiveSummary
## ✅ Các Thay đổi Đã Xác minh   ← nhóm theo change_type (icon), mỗi ACU:
                                    vị trí, confidence, Trước/Sau, bằng chứng V1/V2
## ❌ ACU Bị Bác bỏ              ← nhật ký hallucination (status + lý do)
_footer_
```

Nhóm ACU theo `change_type` với icon + label (🔢 numerical, 📝 terminology, 🏗️
structural, ➕ addition, ➖ deletion, 🔄 reorder). Evidence preview cắt 200 ký tự.
Section "❌ Bị bác bỏ" minh bạch: liệt kê **mọi** ACU bị drop kèm lý do — đây là
audit trail của Zero-Hallucination.

---

## 14. Worked example end-to-end (từ dữ liệu thật)

Nguồn: `tests/tand_signed_comparison_output/15_report_06_6ca5ff2a-…json`
(pair `pair_fa26698f8df0`, `match_type = matched`).

### Bối cảnh

Đây là một Điều về "Cấp đổi, cấp lại Giấy chứng minh Thẩm phán/Hội thẩm". V1 mở
đầu bằng câu dẫn `"... khi thuộc một trong các trường hợp sau đây:"`; V2 **đảo cấu
trúc** câu dẫn thành `"Các trường hợp được cấp đổi, cấp lại…"` **VÀ thêm mới một
bảng** liệt kê 5 trường hợp cụ thể + loại giấy tờ cần cung cấp.

### Kết quả Phase 3

| Chỉ số | Giá trị |
|--------|---------|
| Tổng ACU phát hiện | **6** |
| ACU đã xác minh (PASSED) | **6** |
| ACU bị bác bỏ | **0** |
| Tỷ lệ ảo giác | **0.0%** |

### 6 ACU đã xác minh

**1 ACU `reorder` (conf 1.0)** — phát hiện đảo cấu trúc câu dẫn:

```json
{
  "change_type": "reorder",
  "confidence": 1.0,
  "original_value": "Cấp đổi, cấp lại Giấy chứng minh Thẩm phán, Giấy chứng minh Hội thẩm khi thuộc một trong các trường hợp sau đây:",
  "new_value":      "Các trường hợp được cấp đổi, cấp lại Giấy chứng minh Thẩm phán, Giấy chứng minh Hội thẩm",
  "verbatim_evidence_v1": "Cấp đổi, cấp lại Giấy chứng minh Thẩm phán, Giấy chứng minh Hội thẩm khi thuộc một trong các trường hợp sau đây:",
  "verbatim_evidence_v2": "Các trường hợp được cấp đổi, cấp lại Giấy chứng minh Thẩm phán, Giấy chứng minh Hội thẩm"
}
```

**5 ACU `addition` (conf 1.0)** — 5 dòng bảng mới ở V2 (evidence_v1 = "" đúng
rule). Ví dụ 2 dòng:

```json
{ "change_type": "addition",
  "new_value": "| 1 | Thay đổi thông tin trong Giấy chứng minh Thẩm phán, Giấy chứng minh Hội thẩm | Quyết định bổ nhiệm, quyết định thay đổi thông tin |",
  "verbatim_evidence_v2": "| 1 | Thay đổi thông tin trong Giấy chứng minh Thẩm phán, Giấy chứng minh Hội thẩm | Quyết định bổ nhiệm, quyết định thay đổi thông tin |" }

{ "change_type": "addition",
  "new_value": "| 4 | Giấy chứng minihng hết hạn sử dụng theo quy định mới | Giấy chứng minh cũ và quyết định tiếp tục bổ nhiệm/bầu |",
  "verbatim_evidence_v2": "| 4 | Giấy chứng minihng hết hạn sử dụng theo quy định mới | ..." }
```

> Lưu ý: dòng 4 giữ nguyên lỗi chính tả `"minihng"` (có trong V2 gốc) — đúng
> nguyên tắc **trích nguyên văn, không sửa**.

### Tại sao tất cả PASS Tier 2?

Mỗi `verbatim_evidence_v1`/`v2` đều là **copy-paste chính xác** từ raw text →
exact match (Bước 1) thành công ngay, không cần fuzzy. ACU `addition` không cần
evidence_v1 (rule 4) → chỉ verify evidence_v2 → PASS.

### Executive Summary (Tier 4)

```json
{
  "overall_assessment": "Đánh giá tổng thể: Thay đổi cơ bản về cấu trúc và thêm các trường thông tin.",
  "critical_changes": [
    "Cấu trúc văn bản được sắp xếp lại, từ 'khi thuộc một trong các trường hợp sau đây:' sang 'Các trường hợp được cấp đổi, cấp lại Giấy chứng minh Thẩm phán, Giấy chứng minh Hội thẩm'.",
    "Thêm bảng liệt kê các trường hợp cụ thể và loại giấy tờ cần cung cấp bổ sung."
  ],
  "numerical_changes_summary": "",
  "risk_flags": [],
  "recommendation": ""
}
```

### Diễn giải luồng cho pair này

```
DiffPair (matched) → ComparisonRequest(raw_text_v1, raw_text_v2)
   │
   ▼ Tier 1 (LLM)
 build_acu_user_prompt(matched) + number_hint (C1: các số "1".."4" trong bảng)
   → chat_json → 6 ACU (1 reorder + 5 addition)
   → norm(v1) ≠ norm(v2) nên KHÔNG short-circuit
   → matched pair nên có two-pass (C2) — lượt 2 không thêm gì (đã đủ)
   → dedupe: 6 ACU distinct → giữ 6
   │
   ▼ PRE-FILTER  conf >= 0.2  → giữ 6 (conf=1.0)
   │
   ▼ Tier 2 (evidence)  mỗi evidence exact-in raw_text → 6 PASS
   ▼ Tier 3 (numerical) reorder không có số → skip; addition có số "1".."4"
                          nhưng đều có trong raw_text_v2 → 6 PASS
   │
   ▼ Tier 4 (LLM) → ExecutiveSummary (đảo cấu trúc + thêm bảng)
   ▼ render_markdown_report → markdown_report
   │
   ▼ ComparisonReport (6 verified, 0 rejected, hallucination_rate 0.0%)
```

---

## 15. Tích hợp backend (Celery worker)

Trong production, Phase 3 được chạy qua Celery task
`backend/workers/pipeline_task.py::run_pipeline`:

```
run_pipeline(job_id, file_v1_path, file_v2_path, config_overrides)
   ├─ GPULock.acquire(job_id)              — 1 job/GPU tại một thời điểm
   ├─ get_llm_config(provider)             — resolve local/deepseek
   ├─ PipelineRunConfig(... progress_callback=progress_cb ...)
   ├─ LegalDiffPipeline(cfg).run()         — Phase 1 → 2 → 3 (asyncio bên trong)
   ├─ _store_results(job_id, result)       — persist xuống PostgreSQL
   └─ GPULock.release(job_id)
```

`progress_cb(pct, phase, msg)` vừa update DB (`ComparisonJob.current_phase`,
`progress_pct`) vừa publish Redis pub/sub → WebSocket phát tới frontend (stepper
"Đang phân tích tài liệu… → Đang nhúng vector… → Đang sinh báo cáo…").

**Lưu ý hosted-provider** (DeepSeek): `_store_results` và `run_pipeline` bảo vệ
`llm_base_url`/`llm_api_key` không bị config override cũ (local Ollama) clobber —
tránh gửi model name DeepSeek tới Ollama → 404 "model not found".

### Persist kết quả (`_store_results`)

Mỗi `ComparisonReport` → 1 dòng `ComparisonReportModel`:

| Cột | Nguồn |
|-----|-------|
| `pair_id`, `match_type`, `location_context` | report |
| `total_acus` | `total_acus_passed + total_acus_rejected` |
| `passed_acus`, `rejected_acus`, `hallucination_rate` | report stats |
| `executive_summary` | `report.executive_summary.model_dump()` |
| `markdown_report` | `render_markdown_report` output |
| `verified_acus` | `[acu.to_dict() for acu in report.verified_acus]` |
| `rejected_acus_detail` | `[{acu, status, reason} for v in report.rejected_acus]` |

Catalog (`DiffPairCatalog`) + thống kê pairs được update lên `ComparisonJob`.

---

## 16. Nguyên tắc bất biến (Design invariants)

1. **LLM là *transcribe*, không phải *synthesize*** — LLM chỉ mô tả diff có thật;
   mọi suy luận pháp lý bị cấm (RULE 5). Verification là deterministic, không phải
   LLM.
2. **Verbatim evidence là bắt buộc & kiểm được** — mỗi ACU phải mang
   `verbatim_evidence_v1/v2` copy-paste; Tier 2 chứng minh nó có trong raw_text.
3. **ACU fail → drop + log, không giữ im lặng** — rejected ACU vào
   `rejected_acus` + section "❌ Bị bác bỏ" trong Markdown (audit trail).
4. **Tier 2 trước, Tier 3 sau, có short-circuit** — Tier 3 chỉ chạy khi Tier 2
   PASS; FAILED_EVIDENCE return ngay.
5. **Tier 3 cho mọi ACU có số (S5)** — không chỉ `change_type=NUMERICAL`; bắt cả
   số liệu bịa trong terminology/structural/addition.
6. **`strict_numerical=True` mặc định** — 100% số phải match raw_text; mới FAIL.
7. **`min_evidence_length=5`** — evidence quá ngắn → SKIPPED (= PASS) tránh false
   positive.
8. **`fuzzy_match_threshold=0.85`** — fuzzy chỉ là tiebreak khi exact + ws-norm
   fail; sliding-window SequenceMatcher trên ±50% độ dài.
9. **`min_confidence_to_include` pre-filter TRƯỚC verification** (default 0.2) —
   recall-friendly; Verification mới là guardrail thật.
10. **Dedupe theo `(change_type, original, new)` chuẩn hoá** — chống cross-window
    / cross-pass trùng; addition/deletion thuần bypass dedupe.
11. **Short-circuit text đồng nhất** — `norm(v1)==norm(v2)` → bỏ LLM, trả `[]`.
12. **Fail-soft mọi nơi** — `verify_batch` catch exception → FAILED_EVIDENCE;
    `run_batch` catch → report lỗi rỗng; Tier 4 fail → fallback summary Python;
    truncation → salvage ACU hoàn chỉnh. Pipeline không crash trên 1 bad input.
13. **2 LLM client, nhiệt độ khác** — Tier 1 (0.05, deterministic) vs Tier 4
    (0.3, văn phong).
14. **Lazy import** trong `__init__` cho symbol phụ thuộc `openai` — verifier/models
    dùng được ngay cả khi openai chưa cài.
15. **Cùng regex `extract_numbers`** cho Tier 3 (verify) và C1 (number hint) —
    nhất quán giữa "gợi ý" và "kiểm tra".

---

## 17. Cách chạy & mở rộng

### Chạy trong pipeline đầy đủ (cách production dùng)

```python
from src.pipeline import LegalDiffPipeline
pipeline = LegalDiffPipeline.from_config(provider="local")   # hoặc "deepseek"
result = pipeline.run(file_v1="data_test/01-tand_signed_v1.docx",
                      file_v2="data_test/01-tand_signed_v2.docx")
# result["reports"]  → list[ComparisonReport]
# result["markdown"] / result["catalog"]
```

`skip_phase3=True` để chỉ chạy Phase 1+2 (debug alignment).

### Chạy riêng Phase 3 (debug)

```python
import asyncio
from src.comparison import GenerativeComparisonPipeline, PipelineConfig, ComparisonRequest

pipeline = GenerativeComparisonPipeline(config=PipelineConfig(
    llm_base_url="http://localhost:11434/v1",
    llm_model_name="qwen2.5:14b",
    acu_temperature=0.05,
    provider="local",
))

requests = [ComparisonRequest(
    pair_id="pair_demo",
    match_type="matched",
    raw_text_v1="Bên A phải thanh toán 500.000.000 đồng trong vòng 30 ngày. Bên A không được chuyển nhượng.",
    raw_text_v2="Bên A phải thanh toán 600.000.000 đồng trong vòng 45 ngày. Bên A có thể chuyển nhượng nếu được bên B đồng ý.",
    breadcrumb_v1="[Hợp đồng > Điều 5]",
    breadcrumb_v2="[Hợp đồng > Điều 5]",
)]

reports = asyncio.run(pipeline.run_batch(requests, max_concurrency=2))
print(reports[0].markdown_report)
print(f"passed={reports[0].total_acus_passed} rejected={reports[0].total_acus_rejected}")
```

### Chạy riêng Verification Engine (không cần LLM)

```python
from src.comparison import VerificationEngine, ACUOutput, ChangeType

engine = VerificationEngine()  # fuzzy=0.85, strict_numerical=True
acu = ACUOutput(change_type=ChangeType.NUMERICAL,
                original_value="500.000.000 đồng", new_value="600.000.000 đồng",
                verbatim_evidence_v1="thanh toán 500.000.000 đồng",
                verbatim_evidence_v2="thanh toán 600.000.000 đồng",
                confidence=1.0)
r = engine.verify(acu, raw_text_v1="Bên A phải thanh toán 500.000.000 đồng...",
                  raw_text_v2="Bên A phải thanh toán 600.000.000 đồng...")
print(r.status, r.evidence_v1_found, r.numerical_verified)  # PASSED True True
```

### Dependencies

```bash
pip install openai pydantic
# + server LLM local một trong: vLLM / llama-cpp-python / Ollama
```

| Package | Mục đích |
|---------|----------|
| `openai` | `AsyncOpenAI` client (OpenAI-compatible API) |
| `pydantic` | Data models + validation (v2) |
| `difflib` (stdlib) | `SequenceMatcher` cho fuzzy match Tier 2 |
| `re` (stdlib) | Regex số liệu Tier 3 + `extract_numbers` |

### Hướng mở rộng

- **Thêm `change_type` mới** (vd `formatting`) — thêm enum `models.py`, cập nhật
  RULE 3 trong cả 2 prompt, thêm icon/label trong `render_markdown_report`.
- **Loose numerical mode** — đặt `strict_numerical=False` để chấp nhận heuristic
  linh hoạt (không khuyến nghị cho pháp lý, nhưng hữu ích khi OCR nhiễu).
- **Tier 3 cho logic-shift** — hiện Category D chỉ có *hint* (S4); có thể thêm
  verifier deterministic cho đảo nghĩa (phủ định↔cho phép) song song Tier 2.
- **Reuse `acu_prompter.py`** — hiện file này tách rời chưa được wire vào (xem
  [§4](#4-cấu-trúc-source-code--public-api) ghi chú). Có thể refactor
  `report_generator.py` import lại để tập trung prompt tại một chỗ.
- **Tăng context cho local LLM** — mở `windowing_threshold_chars` /
  `max_tokens_acu` khi dùng model có context lớn (vd Qwen 32K).

### Test artifacts tham khảo

| File | Nội dung |
|------|----------|
| `tests/test_comparison.py` | Unit test Phase 3: verifier, report generation |
| `tests/tand_signed_comparison_output/15_report_*.json` | `ComparisonReport` thật (DB row) — machine-readable layer |
| `tests/tand_signed_comparison_output/16_report_*.md` | Markdown report thật — human-readable layer |
| `tests/tand_signed_comparison_output/13_catalog.json` | `DiffPairCatalog` (đầu vào Phase 3) |
| `tests/e2e_backend/outputs/*/15_report_*.json` | Báo cáo từ các lần chạy e2e |
| `data_test/*_v1.docx` / `*_v2.docx` | Cặp văn bản đầu vào (V1/V2) |
| `configs/prompt_templates/*.txt` | Template prompt (tham khảo; runtime dùng prompt nhúng trong `report_generator.py`) |

---

*Phiên bản tài liệu này được viết để khớp với mã nguồn hiện hành tại
`src/comparison/`. Nguồn sự thật về prompt là `report_generator.py` (8 rules, biến
thể DeepSeek, có `reasoning`); `acu_prompter.py` là bản trích xuất tách rời chưa
được wire vào pipeline. Tier 3 áp dụng cho **mọi ACU có số** (chiến lược S5), không
chỉ `change_type=NUMERICAL`. Phase 3 xử lý **mọi** `match_type`, không chỉ `matched`.*
