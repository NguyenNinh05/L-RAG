# Legal RAG Comparison System

**He thong so sanh van ban phap ly su dung Retrieval-Augmented Generation (RAG) va Large Language Model (LLM)**

Hệ thống tự động phát hiện, phân loại và phân tích sự khác biệt giữa hai phiên bản văn bản pháp lý (hợp đồng, nghị định, thông tư...) bằng cách kết hợp Embedding ngữ nghĩa, thuật toán Needleman-Wunsch và mô hình ngôn ngữ lớn chạy hoàn toàn cục bộ (local LLM).

---

## Muc luc

1. [Tổng quan](#tong-quan)
2. [Kiến trúc hệ thống](#kien-truc-he-thong)
3. [Cấu trúc thư mục](#cau-truc-thu-muc)
4. [Yêu cầu hệ thống](#yeu-cau-he-thong)
5. [Cài đặt](#cai-dat)
6. [Cấu hình](#cau-hinh)
7. [Hướng dẫn sử dụng](#huong-dan-su-dung)
8. [Chi tiết Pipeline](#chi-tiet-pipeline)
9. [API Reference](#api-reference)
10. [Công nghệ sử dụng](#cong-nghe-su-dung)

---

## Tong quan

### Bài toán

Trong lĩnh vực pháp lý, việc so sánh hai phiên bản văn bản (ví dụ: hợp đồng gốc và hợp đồng sửa đổi) là một tác vụ tốn nhiều thời gian và dễ sai sót khi thực hiện thủ công. Các công cụ so sánh văn bản thông thường (diff) chỉ phát hiện sự khác biệt ở mức ký tự/từ, không hiểu được ngữ nghĩa pháp lý.

### Giải pháp

Hệ thống này giải quyết bài toán trên bằng pipeline RAG gồm nhiều bước:

- **Trích xuất thông minh**: Đọc file PDF/DOCX, nhận diện cấu trúc pháp lý Việt Nam (Phần, Chương, Mục, Điều, Khoản, Điểm).
- **So sánh ngữ nghĩa**: Sử dụng embedding vector để phát hiện các điều khoản tương đồng về nghĩa, ngay cả khi câu chữ thay đổi.
- **Alignment tối ưu**: Áp dụng thuật toán Needleman-Wunsch (sequence alignment) để ghép cặp chính xác các điều khoản giữa hai văn bản.
- **Phân tích bằng AI**: Mô hình LLM phân tích chi tiết từng điều khoản sửa đổi, đánh giá tác động pháp lý.
- **Chạy hoàn toàn cục bộ**: Toàn bộ pipeline chạy offline qua Ollama, phù hợp với yêu cầu bảo mật dữ liệu pháp lý.

---

## Kien truc he thong

Hệ thống được thiết kế theo kiến trúc module, mỗi module đảm nhận một giai đoạn trong pipeline:

```
                        +------------------+
                        |   PDF / DOCX     |
                        +--------+---------+
                                 |
                    +------------v------------+
                    |   INGESTION MODULE      |
                    |  (Loader + Chunker +    |
                    |   Normalizer)           |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |   EMBEDDING MODULE      |
                    |  (Ollama API + ChromaDB)|
                    +------------+------------+
                                 |
                    +------------v------------+
                    |   RETRIEVAL MODULE      |
                    |  (Anchor Detection +    |
                    |   Needleman-Wunsch)     |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |   LLM MODULE            |
                    |  (Qwen3 via Ollama)     |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |   OUTPUT                |
                    |  (Markdown Report +     |
                    |   Web UI)               |
                    +-------------------------+
```

### Pipeline 5 bước

| Bước | Module | Mô tả |
|:-----|:-------|:------|
| 1 | Ingestion | Đọc tài liệu PDF/DOCX, nhận diện cấu trúc pháp lý, tách thành các chunk theo Điều/Khoản |
| 2 | Embedding | Chuyển đổi các chunk thành vector embedding qua Ollama, lưu vào ChromaDB |
| 3 | Retrieval | So sánh ngữ nghĩa hai tài liệu bằng Anchor Detection + Needleman-Wunsch alignment |
| 4 | LLM Analysis | Phân tích chi tiết các điều khoản sửa đổi bằng Qwen3-4B |
| 5 | Report | Sinh báo cáo Markdown chuyên nghiệp với bảng so sánh và phân tích tác động |

---

## Cau truc thu muc

```
L-RAG/
|-- api.py                  # FastAPI backend (SSE streaming)
|-- main.py                 # CLI entry point
|-- config.py               # Cấu hình tập trung (model, threshold, path)
|-- requirements.txt        # Python dependencies
|-- README.md
|
|-- ingestion/              # Module 1: Đọc & phân tích cấu trúc tài liệu
|   |-- __init__.py         #   Orchestrator: process_two_documents()
|   |-- loader.py           #   Đọc PDF (pymupdf4llm) và DOCX (python-docx)
|   |-- chunker.py          #   Nhận diện cấu trúc pháp lý & tách chunk
|   |-- normalizer.py       #   Chuẩn hóa Unicode, lọc noise
|   |-- models.py           #   Data model: ArticleChunk
|
|-- embedding/              # Module 2: Embedding & Vector DB
|   |-- __init__.py
|   |-- embedder.py         #   Ollama batch embedding + ChromaDB storage
|
|-- retrieval/              # Module 3: So sánh ngữ nghĩa
|   |-- __init__.py
|   |-- matcher.py          #   Anchor + NW alignment + phân loại thay đổi
|
|-- llm/                    # Module 4: Phân tích bằng LLM
|   |-- __init__.py
|   |-- generator.py        #   Gọi Qwen3 via Ollama, sinh báo cáo Markdown
|
|-- ui/                     # Giao diện web
|   |-- index.html          #   Single-page chat UI
|
|-- chroma_db/              # ChromaDB persistent storage (auto-generated)
|-- comparison/             # Module so sánh mở rộng (reserved)
|-- docs_test/              # Tài liệu mẫu để test
```

---

## Yeu cau he thong

### Phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|:-----------|:----------|:------------|
| RAM | 8 GB | 16 GB trở lên |
| VRAM (GPU) | Không bắt buộc | 6 GB trở lên (NVIDIA) |
| Ổ cứng | 5 GB trống | 10 GB trống |

### Phần mềm

- **Python**: 3.10 trở lên
- **Ollama**: Phiên bản mới nhất ([ollama.com](https://ollama.com))
- Hệ điều hành: Windows 10/11, Linux, hoặc macOS

---

## Cai dat

### 1. Clone repository

```bash
git clone https://github.com/NguyenNinh05/L-RAG
cd L-RAG
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cài đặt và khởi động Ollama

Tải Ollama tại [ollama.com](https://ollama.com), sau đó pull các model cần thiết:

```bash
# Model embedding
ollama pull qwen3-embedding:0.6b

# Model LLM (phân tích văn bản)
ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

Đảm bảo Ollama đang chạy:

```bash
ollama serve
```

---

## Cau hinh

Toàn bộ cấu hình được quản lý tập trung trong file `config.py`:

### Embedding

| Tham số | Giá trị mặc định | Mô tả |
|:--------|:------------------|:------|
| `EMBEDDING_MODEL_NAME` | `qwen3-embedding:0.6b` | Model embedding qua Ollama |
| `OLLAMA_API_BASE` | `http://localhost:11434/api` | Địa chỉ Ollama API |

### LLM

| Tham số | Giá trị mặc định | Mô tả |
|:--------|:------------------|:------|
| `OLLAMA_LLM_MODEL` | `Qwen3-4B-Instruct (Q4_K_M)` | Model phân tích văn bản |
| `LLM_MAX_TOKENS` | `1500` | Số token tối đa cho mỗi response |
| `LLM_TEMPERATURE` | `0.5` | Mức độ sáng tạo của model |
| `LLM_NUM_CTX` | `4096` | Kích thước context window |

### Ngưỡng phân loại (Matcher)

| Tham số | Giá trị mặc định | Mô tả |
|:--------|:------------------|:------|
| `UNCHANGED_THRESHOLD` | `0.95` | Cosine similarity >= 0.95: không thay đổi |
| `MODIFIED_THRESHOLD` | `0.75` | 0.75 <= similarity < 0.95: đã sửa đổi |
| `GAP_PENALTY` | `0.40` | Chi phí gap trong thuật toán Needleman-Wunsch |
| `TEXT_UNCHANGED_RATIO` | `0.99` | Ngưỡng character-level ratio (difflib) |

---

## Huong dan su dung

### Cách 1: Giao diện dòng lệnh (CLI)

```bash
python main.py <file_v1.pdf> <file_v2.docx>
```

Ví dụ:

```bash
python main.py docs_test/hopdong_v1.pdf docs_test/hopdong_v2.pdf
```

Kết quả sẽ được lưu tại `comparison_report.md`.

### Cách 2: Giao diện Web (khuyến nghị)

Khởi động server:

```bash
python -m uvicorn api:app --reload --port 8000
```

Truy cập giao diện tại: `http://localhost:8000`

Giao diện web cung cấp:
- Upload hai file PDF/DOCX cần so sánh
- Hiển thị tiến trình xử lý real-time qua Server-Sent Events (SSE)
- Kết quả báo cáo được render dưới dạng Markdown

### Cách 3: Gọi API trực tiếp

```bash
curl -X POST http://localhost:8000/api/compare \
  -F "file_a=@hopdong_v1.pdf" \
  -F "file_b=@hopdong_v2.pdf"
```

---

## Chi tiet Pipeline

### Bước 1 — Ingestion & Chunking

Module `ingestion/` thực hiện:

1. **Đọc tài liệu**: Hỗ trợ PDF (qua `pymupdf4llm`) và DOCX (qua `python-docx`). Đọc đúng thứ tự xuất hiện, bao gồm cả bảng xen kẽ với văn bản.

2. **Chuẩn hóa**: Unicode NFC normalization, thay thế ký tự typographic (smart quotes, em-dash), lọc noise (số trang, dòng ký tên, đường kẻ).

3. **Nhận diện cấu trúc pháp lý**: Phân cấp 3 tầng theo chuẩn văn bản pháp lý Việt Nam:
   - **Cấp 1**: Phần, Phụ lục
   - **Cấp 2**: Chương, Mục, Tiểu mục, Căn cứ
   - **Cấp 3**: Điều, Mẫu số

4. **Tách chunk thông minh**: Chunk quá dài (> 800 ký tự) được tách theo ranh giới Khoản/Điểm (a, b, c...) hoặc theo dòng bảng, kèm context prefix.

### Bước 2 — Embedding & Vector DB

Module `embedding/` thực hiện:

- Gọi Ollama Embedding API theo batch mode (50 items/batch) với retry logic.
- Mỗi chunk được prepend instruction prefix (`"Represent this legal document for retrieval"`) trước khi embedding.
- Vector được lưu vào ChromaDB (persistent, cosine distance) với metadata đầy đủ (doc_label, article_number, page, breadcrumb).
- Mỗi phiên so sánh tạo collection riêng (timestamp suffix) để không xung đột dữ liệu.

### Bước 3 — Semantic Matching (2 tầng)

Module `retrieval/` áp dụng chiến lược so sánh 2 tầng để tối ưu hiệu suất:

**Tang 1 — Anchor Detection**:
- Dùng `difflib.SequenceMatcher` để tìm các chunk giống nhau 100% về text.
- Các chunk này được đánh dấu `UNCHANGED` ngay lập tức mà không cần đến embedding.
- Giảm 80–90% khối lượng tính toán cho tầng 2.

**Tang 2 — Needleman-Wunsch Alignment**:
- Chỉ áp dụng trên các "gap" (vùng không khớp) giữa các anchor.
- Xây dựng cosine similarity matrix từ embedding vectors (có L2-normalize).
- Thuật toán NW thực hiện global sequence alignment, ghép cặp tối ưu các chunk.
- Mỗi cặp được phân loại:

| Cosine Similarity | Phân loại | Mô tả |
|:------------------|:----------|:------|
| >= 0.95 | UNCHANGED | Không thay đổi (kiểm tra thêm bằng text-ratio) |
| 0.75 — 0.95 | MODIFIED | Nội dung đã bị sửa đổi |
| < 0.75 | DELETED + ADDED | Quá khác biệt, tách thành xóa và thêm mới |

**Bonus — Text-ratio check**: Sau NW, nếu embedding similarity cao nhưng text thực sự thay đổi (character-level ratio < 0.99), hệ thống sẽ downgrade từ UNCHANGED xuống MODIFIED để tránh false negative.

### Bước 4 — LLM Analysis

Module `llm/` thực hiện:

- Chỉ phân tích các cặp có trạng thái `MODIFIED`.
- Gọi Qwen3-4B qua Ollama Chat API với system prompt chuyên biệt cho so sánh pháp lý.
- Xử lý bất đồng bộ (async) với semaphore giới hạn concurrent requests.
- Mỗi điều khoản sửa đổi được phân tích theo format:
  - Nội dung thay đổi
  - Bảng so sánh chi tiết
  - Phân tích tác động (rủi ro/lợi ích, điểm cần lưu ý)

### Bước 5 — Report Generation

Sinh báo cáo Markdown đầy đủ bao gồm:
- Tổng quan thay đổi (thống kê số lượng)
- Phân tích chi tiết từng điều khoản sửa đổi (output của LLM)
- Bảng điều khoản thêm mới
- Bảng điều khoản bị loại bỏ
- Trích dẫn nguồn (tên file, số trang)

---

## API Reference

### `POST /api/compare`

So sánh hai văn bản pháp lý. Response là SSE stream.

**Request**: `multipart/form-data`

| Field | Type | Mô tả |
|:------|:-----|:------|
| `file_a` | File | Văn bản phiên bản gốc (v1). Hỗ trợ `.pdf`, `.docx` |
| `file_b` | File | Văn bản phiên bản mới (v2). Hỗ trợ `.pdf`, `.docx` |

**Giới hạn**: Mỗi file tối đa 50 MB.

**SSE Events**:

| Event | Mô tả |
|:------|:------|
| `progress` | Cập nhật tiến trình từng bước (step, status, title, detail) |
| `stats` | Thống kê số lượng thay đổi (modified, added, deleted, unchanged) |
| `report` | Nội dung báo cáo dạng Markdown |
| `done` | Tín hiệu hoàn thành |
| `error` | Thông báo lỗi (nếu có) |

### `GET /`

Trả về giao diện web (HTML).

### `GET /api/health`

Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

## Cong nghe su dung

### Core

| Thành phần | Công nghệ | Vai trò |
|:-----------|:-----------|:--------|
| Embedding | Qwen3-Embedding 0.6B (qua Ollama) | Chuyển đổi text thành vector ngữ nghĩa |
| LLM | Qwen3-4B-Instruct GGUF Q4_K_M (qua Ollama) | Phân tích và sinh báo cáo pháp lý |
| Vector DB | ChromaDB | Lưu trữ và truy vấn vector embedding |
| Sequence Alignment | Needleman-Wunsch (custom) | Ghép cặp tối ưu các điều khoản |
| Anchor Detection | difflib.SequenceMatcher | Tìm nhanh các chunk không đổi |

### Backend

| Thành phần | Công nghệ |
|:-----------|:-----------|
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Streaming | Server-Sent Events (SSE) |
| PDF Parser | pymupdf4llm |
| DOCX Parser | python-docx |
| Async HTTP | aiohttp |
| Retry Logic | tenacity |

### Frontend

| Thành phần | Công nghệ |
|:-----------|:-----------|
| UI | Single-page HTML (vanilla JS + CSS) |
| Markdown Render | Client-side rendering |

---

## Ghi chu

- Toàn bộ quá trình xử lý chạy **offline** trên máy cục bộ thông qua Ollama. Không có dữ liệu nào được gửi ra bên ngoài hệ thống.
- Chất lượng kết quả phụ thuộc vào model embedding và LLM. Có thể thay đổi model trong `config.py` nếu cần.
- Báo cáo mang tính chất **tham khảo** — cần được kiểm chứng bởi chuyên gia pháp lý trước khi sử dụng chính thức.
