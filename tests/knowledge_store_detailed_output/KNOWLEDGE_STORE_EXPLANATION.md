# Báo Cáo Phân Tích Chi Tiết `knowledge_store.py` (HybridGraphBuilder)

Tài liệu này giải thích chi tiết vai trò, chức năng của từng class và hàm bên trong file `src/ingestion/knowledge_store.py` dựa trên kết quả chạy test thực tế (xem các file `.json` trong cùng thư mục).

Hybrid Knowledge Store là Module 3 (cuối cùng) của quá trình Ingestion trong dự án L-RAG. Vai trò cốt lõi của nó là tổ chức dữ liệu từ dạng phẳng (chunks) vào một kiến trúc lai (Hybrid) gồm:
1. **Graph DB (Kuzu)**: Lưu trữ cấu trúc phân cấp (CONTAINS) và tham chiếu chéo (REFERENCES) giữa các điều khoản.
2. **Vector DB (ChromaDB)**: Lưu trữ văn bản được mã hoá (embeddings) để tìm kiếm theo ngữ nghĩa.

---

## 1. Các Regex Patterns
Đầu file, hệ thống định nghĩa các mẫu Regex (Biểu thức chính quy) để dò tìm các cụm từ tham chiếu chéo trong văn bản pháp lý.
- `_RE_ARTICLE_REF`: Bắt cụm "theo quy định tại Điều 5", "tại Điều 15 Khoản 2".
- `_RE_CLAUSE_FIRST_REF`: Bắt cụm "khoản 3 Điều 15" (theo thứ tự ngược).
- `_RE_REF_CONTEXT`: Lấy một phần nhỏ văn bản phía trước làm ngữ cảnh (context) cho liên kết.

---

## 2. Placeholder Embedding (`placeholder_embedding_fn`)

**Vai trò:** Là một hàm giả lập (mock) quá trình nhúng vector (embedding). 
- Hiện tại nó nhận một danh sách văn bản và trả về các mảng toàn số `0.0` với độ dài `EMBEDDING_DIM` (mặc định 1024). 
- Khi đưa lên môi trường thật, hàm này sẽ được thay thế bởi các model nhúng thực sự (như OpenAI, BGE-M3, hoặc Sentence-Transformers).

---

## 3. Class `_KuzuSchemaManager`

**Vai trò:** Định nghĩa cấu trúc bảng (Schema/DDL) cho KuzuDB. KuzuDB yêu cầu phải định nghĩa bảng cứng trước khi chèn dữ liệu (khác với Neo4j).
- `Document`: Chứa meta của toàn bộ file văn bản.
- `LegalNode`: Bảng chung chứa tất cả các node (Điều, Khoản).
- `CONTAINS`: Cạnh nối từ Document -> LegalNode, hoặc LegalNode -> LegalNode (Parent-Child).
- `REFERENCES`: Cạnh nối khi có sự tham chiếu chéo (Cross-reference).
- `PRECEDES`: Cạnh nối tuần tự (Khoản 1 -> Khoản 2).

---

## 4. Class `_ArticleIndex`

**Vai trò:** Giống như một danh bạ điện thoại. Trong lúc tạo đồ thị, Kuzu cần biết node_id cụ thể của "Điều 5" để nối cạnh. Hệ thống dùng `_ArticleIndex` để mapping từ `"5"` sang `article_uuid_xxx`. 
- Nó có thể tra cứu theo Điều (`lookup_article`) hoặc tra cứu sâu đến tận Khoản (`lookup_clause`).

---

## 5. Class `HybridGraphBuilder` (Class Chính)

Đây là bộ máy chính thực hiện tất cả công đoạn lưu trữ dữ liệu. Các logic được tổ chức như sau:

### 5.1. Khởi tạo & Build (`build`)
Hàm `build(document, chunks)` là đầu vào chính:
1. Gọi `_init_kuzu` và `_init_chroma` để mở kết nối Database.
2. Gọi `_build_article_index` để tạo danh bạ ID cho việc nối tham chiếu.
3. Chạy `_build_graph` để xây Graph nodes và quan hệ cấu trúc.
4. Chạy `_build_reference_edges` để scan text và nối Graph references.
5. Chạy `_build_vector_store` để lưu vector vào ChromaDB.

### 5.2. Nhóm hàm Upsert Graph (KuzuDB)
- `_upsert_document_node`, `_upsert_article`, `_upsert_clause`: Dùng câu lệnh Cypher `MERGE` để chèn hoặc cập nhật node (idempotent - không sợ lỗi trùng lặp nếu chạy lại 2 lần).
- `_create_contains_edge_...`: Nối các cấp độ lại với nhau (VD: Điều 1 chứa Khoản 1, Khoản 2).
- `_create_precedes_edge`: Nối thứ tự tuần tự.

### 5.3. Nhóm hàm Tạo Tham Chiếu (REFERENCES)
- `_build_reference_edges` và `_scan_and_create_refs`: Quét tất cả `text` của từng Điều, Khoản. Bất cứ khi nào Regex bắt được từ khoá "tại Điều X Khoản Y", nó dùng `_ArticleIndex` để tìm ID của Điều X Khoản Y đó, sau đó tạo một cạnh `REFERENCES` trong đồ thị (lưu kèm đoạn văn bản chứa cụm từ đó vào trường `context`).

### 5.4. Nhóm hàm Vector Store (ChromaDB)
- `_build_vector_store`: Quản lý việc đưa `LsuChunk` vào ChromaDB. Chạy theo batch (mặc định 32) để tránh tràn RAM (OOM) nếu mô hình Embedding quá lớn.
- **CRITICAL LINK**: Thuộc tính `metadata.node_id` của VectorRecord được gắn đúng bằng ID của node trên Kuzu Graph. Đây là "cầu nối" duy nhất! (Khi truy vấn ra vector chunk, L-RAG sẽ dùng `node_id` này chạy sang Kuzu để lấy ra nguyên cái cây tham chiếu liên quan).

### 5.5. Các hàm Utility (Truy vấn thử nghiệm)
- `query_similar_chunks(query_text)`: Thử nghiệm tìm kiếm bằng Vector.
- `get_node_references(node_id)`: Dùng Cypher để lấy tất cả các node đích mà node gốc đang trích dẫn tới.
- `get_graph_stats()`: Gom số liệu thống kê trả về.

---
Bạn có thể xem trực tiếp các file `.json` trong thư mục này để quan sát ví dụ dữ liệu (Node ID, các Cạnh sinh ra, kết quả truy vấn, Vector Metadata) được pipeline sinh ra trong quá trình chạy tự động.
