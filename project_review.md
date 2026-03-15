# 🔍 REVIEW TOÀN DIỆN DỰ ÁN TTCS — Legal RAG Comparison System

## Tổng quan dự án

Hệ thống so sánh hai văn bản pháp lý (PDF/DOCX) sử dụng pipeline:
**Ingestion → Embedding → Semantic Matching → LLM Analysis → Report**

| Module | File chính | Chức năng |
|:---|:---|:---|
| `ingestion/` | `loader.py`, `chunker.py`, `normalizer.py`, `models.py` | Đọc tài liệu, chunking theo cấu trúc pháp lý VN |
| `embedding/` | `embedder.py` | Nhúng vector qua Ollama, lưu ChromaDB |
| `retrieval/` | `matcher.py` | Anchor + Needleman-Wunsch semantic alignment |
| `llm/` | `generator.py` | Sinh báo cáo so sánh bằng Qwen3-4B qua Ollama |
| `ui/` | `index.html` | Giao diện web dark-mode với SSE streaming |
| Root | `api.py`, `main.py` | FastAPI backend + CLI entry point |

---

## 🐛 I. LỖI VÀ BUG CẦN SỬA NGAY

### 1. Temp file không bao giờ được cleanup (BUG — `api.py:286-289`)

```python
# Hiện tại - KHÔNG BAO GIỜ XÓA temp file
finally:
    pass  # ← Bug! temp files tồn tại vĩnh viễn
```

Mỗi lần gọi API tạo 2 file tạm trong `%TEMP%` mà không bao giờ xóa. Sau nhiều lần sử dụng → đầy ổ đĩa.

> [!CAUTION]
> `StreamingResponse` trả về response TRƯỚC khi generator hoàn thành, nên `finally` chạy ngay → không thể xóa file trong finally. Cần dùng `background task` hoặc cleanup sau generator.

---

### 2. `embed_chunks()` gọi với sai tham số (`matcher.py:215,218`)

```python
# matcher.py dòng 215
embeds_a = embed_chunks(chunks_a, instruction=INSTRUCTION_DOC)  # ❌ sai
```

Nhưng hàm `embed_chunks()` trong `embedder.py` chỉ nhận `(chunks, batch_size=100)`, **KHÔNG** có tham số `instruction`. Nếu `embeds_a` là `None` khi gọi `build_comparison_pairs()` mà không truyền embedding → **crash lỗi TypeError**.

---

### 3. Hardcoded ngày trong báo cáo (`generator.py:118`)

```python
f"**Ngày lập:** 2026-03-04",  # ← Ngày cứng
```

Phải dùng `datetime.now()` để tự động lấy ngày hiện tại.

---

### 4. Missing dependency `nest_asyncio` và `aiohttp` trong `requirements.txt`

[generator.py](file:///c:/Users/Admin/Nextcloud/Study/TTCS/TTCS_Code/llm/generator.py) import `aiohttp`, `nest_asyncio`, `tqdm` nhưng `requirements.txt` không khai báo:

```diff
 # LLM local
 llama-cpp-python>=0.3.0
 tenacity>=9.0.0
+aiohttp>=3.9.0
+nest-asyncio>=1.6.0
+tqdm>=4.60.0
```

→ Cài đặt từ `requirements.txt` trên máy mới sẽ bị `ModuleNotFoundError`.

---

### 5. Fallback embedding dùng vector zero sai chiều (`embedder.py:73`)

```python
return [[0.0] * 1024] * len(chunks)  # Giả định dim=1024
```

Model `qwen3-embedding:0.6b` có thể trả về chiều **khác** 1024. Nếu sai → crash khi lưu ChromaDB hoặc tính cosine similarity. Nên lấy dim động từ response thực.

---

### 6. `_get_batch_embeddings` retry decorator áp lên toàn hàm (`embedder.py:21`)

```python
@retry(stop=stop_after_attempt(3), ...)
def _get_batch_embeddings(texts, batch_size=50):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        ...  # Nếu batch 3/5 fail → retry LẠI TỪ batch 1!
```

Retry decorator bao toàn hàm → nếu batch cuối fail, sẽ retry lại **TẤT CẢ** từ đầu, lãng phí kết quả các batch trước. Nên retry ở cấp **mỗi batch**.

---

### 7. SSE data parsing thiếu xử lý multi-line data (`ui/index.html:1201`)

```javascript
else if (line.startsWith("data: ")) data = line.slice(6).trim();
```

Chỉ lấy dòng `data:` **cuối cùng**. Chuẩn SSE cho phép multi-line data, nhưng hiện tại nếu có nhiều `data:` lines thì chỉ giữ dòng cuối → mất dữ liệu.

---

## ⚠️ II. VẤN ĐỀ THIẾT KẾ CẦN CẢI THIỆN

### 8. Không validate embedding response length (`embedder.py`)

Sau khi gọi `_get_batch_embeddings()`, không kiểm tra `len(embeddings) == len(texts)`. Nếu Ollama trả thiếu vector (timeout một phần, lỗi mạng) → index lệch → kết quả sai hoàn toàn mà không có lỗi.

---

### 9. ChromaDB bị xóa collection mỗi lần chạy (`embedder.py:86-89`)

```python
try:
    client.delete_collection(collection_name)
except Exception:
    pass
```

Xóa hết dữ liệu cũ trước khi thêm mới → không thể sử dụng lại index cho incremental comparison.

---

### 10. Event loop handling không an toàn (`generator.py:158-168`)

```python
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()

if loop.is_running():
    import nest_asyncio
    nest_asyncio.apply()

loop.run_until_complete(...)
```

- Khi gọi từ CLI (`main.py`): OK.
- Khi gọi từ FastAPI (`api.py`): event loop **đang chạy** → cần `nest_asyncio` → **hacky** và có thể deadlock.
- Nên tách thành 2 path: sync cho CLI, async cho FastAPI.

---

### 11. `_sim_matrix` giả định vectors đã L2-normalized (`matcher.py:82`)

```python
def _sim_matrix(embeds_a, embeds_b):
    """Cosine similarity matrix. Giả định L2-normalized."""
    return (embeds_a @ embeds_b.T)  # Chỉ dot product
```

Nếu Ollama trả về vectors **chưa** normalize → kết quả sai. Cần thêm bước normalize hoặc dùng công thức cosine đầy đủ.

---

### 12. `comparison/` module rỗng, `offload_llm/` rỗng, `test.py` không liên quan

- `comparison/__init__.py`: Chỉ có TODO comment, không dùng ở đâu.
- `offload_llm/`: Thư mục trống.
- `test.py`: Test thử Qwen3 standalone, không phải unit test cho pipeline.

→ Dọn dẹp để tránh nhầm lẫn.

---

### 13. Không có `.env` hoặc config management

Các hằng số quan trọng nằm rải rác:
- `MODEL_NAME` trong `embedder.py`
- `OLLAMA_MODEL`, `OLLAMA_API_URL` trong `generator.py`
- `CHROMA_DIR`, `COLLECTION_NAME` trong `embedder.py`

→ Nên tập trung vào 1 file `config.py` hoặc dùng `.env` + `python-dotenv`.

---

### 14. Không có error handling cho Ollama service down

Nếu Ollama không chạy, cả embedding lẫn LLM đều fail với `ConnectionRefusedError` mà không có thông báo rõ ràng cho user.

---

### 15. `_strip_markdown` regex có thể fail với nested patterns (`loader.py:45-50`)

```python
re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # lazy match
```

Với text `**A** và **B**`, lazy `.*?` sẽ match đúng. Nhưng với `**A **và** B**` (markdown lồng nhau) → kết quả không dự đoán được.

---

## 📈 III. GỢI Ý CẢI THIỆN

### A. Code Quality

| # | Cải thiện | Ưu tiên |
|:--|:---|:---:|
| 1 | Thêm **type hints** đầy đủ cho tất cả function | Trung bình |
| 2 | Thêm **logging** thay vì `print()` xuyên suốt | Cao |
| 3 | Viết **unit tests** cho từng module (`pytest`) | Cao |
| 4 | Thêm **docstring chuẩn** cho public functions | Trung bình |
| 5 | Dùng `pathlib.Path` nhất quán thay vì mix `os.path` | Thấp |

### B. Performance

| # | Cải thiện | Chi tiết |
|:--|:---|:---|
| 1 | **Normalize vectors** trước khi tính cosine | Đảm bảo kết quả chính xác |
| 2 | **Cache embedding model** giữa các lần gọi | Tránh reload model mỗi request |
| 3 | Giảm `num_ctx` nếu prompt ngắn | Tiết kiệm VRAM |

### C. Security & Production

| # | Cải thiện | Chi tiết |
|:--|:---|:---|
| 1 | **Giới hạn file size** upload (e.g., 50MB max) | Tránh DoS |
| 2 | **Rate limiting** cho `/api/compare` | Tránh quá tải server |
| 3 | **CORS** cần thu hẹp từ `"*"` | Chỉ cho phép origin cụ thể |
| 4 | **XSS** — `marked.parse()` render HTML từ LLM | Cần sanitize output |

### D. UX & Features

| # | Cải thiện | Chi tiết |
|:--|:---|:---|
| 1 | Hiển thị **progress %** cho từng bước | Dùng SSE progress chi tiết hơn |
| 2 | Cho phép **so sánh lại** mà không reload trang | Reset state sau mỗi lần so sánh |
| 3 | **Export PDF** ngoài Markdown | Thêm endpoint hoặc client-side PDF gen |
| 4 | **Dark/Light mode toggle** | UI hiện tại chỉ hỗ trợ dark |

---

## 📊 IV. TỔNG KẾT

| Mức độ | Số lượng | Chi tiết |
|:---|:---:|:---|
| 🔴 **Bug cần sửa ngay** | 7 | Temp file leak, sai tham số, missing deps, etc. |
| 🟡 **Vấn đề thiết kế** | 8 | Event loop, vector normalize, config mgmt, etc. |
| 🟢 **Gợi ý cải thiện** | 16+ | Logging, tests, security, UX, etc. |

> [!IMPORTANT]
> **Top 3 ưu tiên cần sửa:**
> 1. Fix temp file cleanup trong `api.py` (memory leak)
> 2. Fix sai tham số `embed_chunks()` trong `matcher.py` (crash potential)
> 3. Thêm missing dependencies vào `requirements.txt` (install broken)
