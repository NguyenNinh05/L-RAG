# Hướng Dẫn Toàn Tập Dự Án L-RAG (LegalDiff)

Tài liệu này cung cấp cho bạn một cái nhìn toàn cảnh để dễ dàng tiếp cận, đọc hiểu mã nguồn, chạy thử hệ thống và bắt tay vào test dự án **L-RAG (LegalDiff - Hệ thống AI Đối chiếu Văn bản Pháp lý)**.

---

## 1. Tổng Quan Về Kiến Trúc

Đây **KHÔNG PHẢI** là một hệ thống RAG (Retrieval-Augmented Generation) thông thường yêu cầu User gõ câu hỏi để sinh câu trả lời. Đây là hệ thống **Pairwise Document Intelligence Pipeline (So sánh cặp tài liệu tự động)**. 

Toàn bộ quá trình chia làm 3 Phase hoạt động nối tiếp nhau, với đầu vào là 2 file (V1 - cũ, V2 - mới) và đầu ra là báo cáo những sự thay đổi giữa 2 văn bản. Do môi trường làm việc với văn bản pháp lý, tiêu chí **Zero Hallucination** (kiểm định chống bịa đặt thông tin) là tuyệt đối.

### Cấu trúc dự án
- `configs/`: File config YAML và Prompt Templates.
- `src/`: Cấu trúc source code chính.
  - `src/ingestion/`: Phase 1 - Đọc, bóc tách cấu trúc văn bản.
  - `src/alignment/`: Phase 2 - Tính toán ma trận độ tương đồng và thuật toán ghép cặp.
  - `src/comparison/`: Phase 3 - Gọi LLM để sinh báo cáo khác biệt.
- `tests/`: Chứa các kịch bản kiểm thử tĩnh chạy bằng Pytest tương ứng cho từng Phase.
- `run.py`: Script khởi chạy cho toàn hệ thống.
- `README.md`: Giải thích **rất kĩ** mọi triết lý xây dựng hệ thống. Bạn nên đọc tập trung vào phần `4.1` đến `4.7` trước khi debug code.

---

## 2. Hướng Dẫn Đọc Hiểu Code (Thứ Tự Khuyến Nghị)

Để không bị ngợp, hãy đọc code theo đúng thứ tự luồng dữ liệu chạy:

### Bước 0: Bắt đầu từ Orchestrator
Mở file `src/pipeline.py` (Class `LegalDiffPipeline`).
- File này là "bộ não" điều phối 3 Phase. Nhìn vào hàm `_run_async()` trong file này, bạn sẽ nhận ra cấu trúc dữ liệu `doc_v1`, `doc_v2` đi qua `Phase 1`, tạo thành `catalog` ở `Phase 2` và kết thúc chuyển đổi thành `reports` ở `Phase 3`.

### Bước 1: Khám phá Phase 1 - Ingestion (Bóc tách dữ liệu)
Đọc trong thư mục `src/ingestion/`:
1. Mở `models.py`: Đây là thiết kế Data Models. Đọc để hiểu `LegalDocument` (Văn bản chứa nhiều điều/khoản) và `LsuChunk` (Viên gạch ngữ nghĩa cấp thấp nhất để so sánh).
2. Mở `docling_parser.py`: Trình phân tích PDF/DOCX sang dạng JSON có cấu trúc cây (tree DOM).
3. Mở `lsu_chunker.py`: Đọc để hiểu cách phân đoạn (chunk) văn bản theo ngữ nghĩa pháp luật (Điều, Khoản) thay vì số lượng ký tự/token mặc định.
4. Mở `knowledge_store.py`: Xem cách lưu metadata vào GraphDB (Kuzu/Neo4j) và VectorDB (Qdrant).

### Bước 2: Khám phá Phase 2 - Alignment (Ghép Cặp Văn Bản)
Đọc trong thư mục `src/alignment/`:
1. Mở `similarity_matrix.py`: Nơi tính toán chỉ số tương đồng (Semantic Cosine + String Matching JaroWinkler + Ordinal).
2. Mở `hungarian_matcher.py`: Thuật toán trái tim của dự án! Chứa thuật toán Hungarian để giải quyết bài toán "Bipartite matching" một cách tối ưu nhất, xử lý và phân loại các trạng thái `matched`, `added`, `deleted`, `split`, `merged`.
3. Mở `diff_catalog.py`: Khảo sát cấu trúc object `DiffPairCatalog` trả về sau khi phân tích để đưa tiếp cho Phase 3.

### Bước 3: Khám phá Phase 3 - Comparison (LLM & Sinh Báo cáo)
Đọc trong thư mục `src/comparison/`:
1. Mở `models.py` (của riêng phần comparison) để thấy định dạng `ACUOutput` (Atomic Comparison Unit) quy định LLM phải trả về theo schema JSON chuẩn như thế nào.
2. Mở `verifier.py`: **Quan trọng nhất của toàn hệ thống.** Chứa logic `VerificationEngine` đảm trách việc đối phó với hiện tượng ảo giác AI (Hallucination). Đoạn logic này kiểm tra lại thông số phần trăm, ngày tháng hay `verbatim_evidence` có tồn tại trong text văn bản thực tế hay không.
3. Mở `llm_client.py` và `acu_prompter.py`: Hiểu cách đóng gói prompt từ hệ thống gọi xuống LLM (tùy biến trên API local của Llama.cpp Host).

---

## 3. Hướng Dẫn Chạy Thực Tế Dự Án (Run Pipeline)

Trước khi chạy, hãy đảm bảo bạn đã cài toàn bộ các thư viện trong danh sách `requirements.txt` và đã load model LLM vào môi trường local memory (Ví dụ chạy `llama.cpp server` ở port `8000` với model cấu hình `Qwen2.5`).

Sử dụng tiện ích pipeline đầu não trong tệp `run.py`.

**Cú pháp 1: Chạy Xuyên Suốt Toàn Hệ Thống (End-to-End Both 3 Phases)**
```bash
python run.py --v1 data_test/v1_filename.docx --v2 data_test/v2_filename.docx
```
Kết quả báo cáo (Markdown và metadata JSON) sẽ sinh ra tự động nằm ngay trong thư mục `./data/reports/`.

**Cú pháp 2: Chạy Dev & Debug Mode (Bỏ Qua LLM, Chỉ Chạy Phase 1 & 2)**
Nếu bạn đang tinh chỉnh tính logic của Thuật toán Toán học Phase 2 (Hungarian match), bạn không nên phí thời gian chạy mô hình LLM nặng nề. Sử dụng cờ sau để bypass qua Phase 3:
```bash
python run.py --v1 data_test/v1.docx --v2 data_test/v2.docx --skip-phase3
```

---

## 4. Hướng Dẫn Chạy Test Bằng Pytest (Kiểm Thử Theo Giai Đoạn)

Dự án này đã chia sẵn kịch bản unit tests đi theo 3 file tương ứng với 3 Phase trong thư mục `tests/`.

Đầu tiên, luôn đảm bảo bạn đã có Pytest:
```bash
pip install pytest
```

### Cách 1: Chạy toàn bộ kho tests hiện có trong dự án
```bash
pytest tests/ -v
```

### Cách 2: Test đơn giản cho Node/Chunk logic (Phase 1)
File `tests/test_ingestion.py` sẽ tự mock parser ảo ra nội dung docx giả sau đó test các file cấu trúc class Pydantic (Models) và Logic Chunker.
```bash
pytest tests/test_ingestion.py -v
```

### Cách 3: Test Thuật toán Thuần Túy Hungarian (Phase 2)
File `tests/test_alignment.py` chứa test coverage của việc tạo hàm khoảng cách toạ độ văn bản (Ordinal similarity), test ngưỡng Orphan/Match và Threshold. Nơi lý tưởng để test công thức Cosine vs JaroWinkler.
```bash
pytest tests/test_alignment.py -v
```

### Cách 4: Test chống Hallucination Engine của LLM (Phase 3)
File `tests/test_comparison.py` kiểm định strict mode rằng regex nhận dạng % hay ngày tháng, số tiền có khả năng extract text chuẩn xác ra hay không (Chặn việc AI trả lại báo cáo ảo có sai khác 1 số 0 ở hợp đồng kinh tế).
```bash
pytest tests/test_comparison.py -v
```
**Mẹo Dev:** Bạn có thể target chạy 1 Function lỗi duy nhất bằng flag `-k` trong Pytest:
```bash
pytest tests/test_comparison.py -v -k "test_evidence_hallucination_detected"
```
