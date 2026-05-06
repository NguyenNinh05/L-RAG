# Báo Cáo Phân Tích Chi Tiết `lsu_chunker.py` (LSU Chunker)

Tài liệu này giải thích chi tiết vai trò, chức năng của từng class và hàm bên trong file `src/ingestion/lsu_chunker.py` dựa trên kết quả chạy test thực tế (xem các file `.json` trong cùng thư mục).

LSU (Logical Semantic Unit) Chunker là Module 2 của dự án L-RAG. Vai trò cốt lõi của nó là biến cây DOM Pháp lý (`LegalDocument`) thành các khối ngữ nghĩa (`LsuChunk`) có thể nhúng vector (embedding).

---

## 1. Class `_BreadcrumbBuilder`

**Vai trò:** Xây dựng một chuỗi "đường dẫn ngữ cảnh" (breadcrumb) cho mỗi node trong tài liệu. 
**Tại sao cần thiết?** Khi cắt nhỏ một văn bản pháp lý thành các đoạn nhỏ (chunk), đoạn đó sẽ mất đi bối cảnh. Ví dụ một chunk chỉ ghi "2. Quyền yêu cầu bồi thường thiệt hại", model sẽ không biết đây là quyền của ai, ở điều nào. Breadcrumb giúp bổ sung bối cảnh này (VD: `[Chương II > Điều 5 > Khoản 2]`).

- `for_article(article, section)`: Nối tên Section (nếu có) và tên Article.
  - *Ví dụ:* `[Chương II. Quyền và Nghĩa vụ > Điều 5. Quyền của Bên A]`
- `for_clause(clause, article, section)`: Tương tự nhưng thêm cấp độ Khoản.
  - *Ví dụ:* `[Chương II > Điều 5 > Khoản 2]`
- `for_point(point, clause, article, section)`: Thêm cấp độ Điểm (a, b, c...).

> **Lưu ý:** Prefix này sẽ được nối thẳng vào `content` của chunk khi tạo thành `content_with_prefix` mang đi nhúng Vector.

---

## 2. Text Utilities (Các hàm tiện ích)

Các hàm xử lý dữ liệu thô cục bộ, hỗ trợ việc tạo nội dung chunk:

- `_flatten_clause_content(clause)`: Gom toàn bộ text của một Khoản. Vì Khoản có thể chứa nhiều Điểm con (a, b, c...), hàm này gom nội dung của khoản và các điểm con lại thành một chuỗi văn bản hoàn chỉnh.
- `_serialize_tables(tables)`: Convert các đối tượng Pydantic `TableData` thành chuẩn dictionary (JSON) để lưu trữ vào ChromaDB hoặc Kuzu DB mà không bị lỗi kiểu dữ liệu.
- `_detect_content_type(text, tables)`: Gắn nhãn thể loại cho chunk. Nếu chỉ có chữ là `TEXT`, chỉ có bảng là `TABLE`, có cả hai là `MIXED`. Hỗ trợ việc filter/routing lúc retrieval sau này.

---

## 3. Class `_SentenceSplitter`

**Vai trò:** Ngay cả khi đã chia theo cấp độ Khoản, một số Khoản pháp lý (hoặc Điều) quá dài, vượt quá giới hạn mô hình nhúng (VD: > 2000 ký tự). Class này làm nhiệm vụ "chẻ" chuỗi dài thành các phần ngắn hơn, đảm bảo không làm đứt mạch ý tưởng.

- **Khởi tạo:** Nhận vào `max_chars` (kích thước tối đa) và `overlap_chars` (kích thước giao nhau giữa 2 chunk để giữ liên kết ngữ cảnh).
- `split(text)`: Sử dụng Biểu thức chính quy (Regex) `_RE_SENTENCE_BOUNDARY` để nhận diện ranh giới câu (dựa trên dấu chấm, phẩy, hỏi chấm... tiếp nối bởi chữ in hoa). Nó gom các câu lại sao cho tổng chiều dài <= `max_chars`.
- `_hard_split(text)`: *Fallback (Dự phòng).* Nếu một văn bản bị dính liền, hoặc một bảng dài không hề có dấu chấm câu thì regex bị vô hiệu. Hàm này sẽ ép cắt thô (cứ đếm đủ 50 ký tự là cắt cái rụp, lùi lại 10 ký tự overlap rồi cắt tiếp). *(Lưu ý: Quá trình test đã phát hiện và fix một lỗi lặp vô hạn (infinite loop) tiềm ẩn trong hàm này khi chiều dài chuỗi chạm mép độ dài max_chars).*

---

## 4. Class `LsuChunker` (Class Chính)

**Vai trò:** Giám đốc điều phối (Orchestrator). Chạy từ trên xuống dưới cây tài liệu và tạo ra danh sách các `LsuChunk`.

- **`chunk(document)`**: Đầu vào là một `LegalDocument`. Hàm này duyệt qua tất cả các `sections` (Chương/Phần) và cả `orphan_articles` (các Điều đứng tự do không thuộc Chương nào). Gọi `_process_section` hoặc `_process_article`.
- **`_process_section(section)`**: Lặp qua tất cả các `ArticleNode` bên trong section.
- **`_process_article(article, ...)`**: Hàm xử lý cho 1 Điều. Có hai logic phân nhánh:
  1. **Article-Level Chunk (`create_article_level_chunks`):** Gọi `_make_article_chunk()`. Tạo một chunk tổng quan cho nguyên cái Điều đó. Nội dung bao gồm đoạn mở đầu của Điều + Tóm tắt ngắn gọn của từng Khoản bên trong. Phục vụ cho câu hỏi *"Điều X nói về cái gì tổng quát?"*.
  2. **Clause-Level Chunk (`create_clause_level_chunks`):** Duyệt từng `ClauseNode` và gọi `_process_clause()`. Tạo các chunk chi tiết nhất. Phục vụ cho câu hỏi *"Theo điểm a khoản 2 điều 5 thì phạt bao nhiêu?"*.
- **`_process_clause(clause, ...)`**: 
  - Lấy nội dung bằng `_flatten_clause_content`.
  - Nếu nội dung ngắn gọn <= `max_chunk_chars`, gọi `_make_clause_chunk` để gói thành 1 chunk.
  - Nếu nội dung **dài hơn**, nó sẽ gọi `_SentenceSplitter.split()` để chẻ nhỏ. Mỗi mảnh nhỏ sẽ được gói vào một `_make_clause_chunk` với thông báo phần tử: VD `[phần 1/3]`. (Lưu ý: table chỉ đính kèm ở chunk phần đầu tiên để tránh trùng lặp dư thừa).
- **`get_stats(chunks)`**: Hàm tiện ích để thống kê (bao nhiêu chunk Điều, bao nhiêu chunk Khoản, max độ dài, v.v.) để log ra console theo dõi.

### Kết Luận Kiến Trúc
Pipeline của file `lsu_chunker.py` rất rõ ràng và chặt chẽ:
**Cây cấp bậc (Hierarchy) -> Cấp bối cảnh (Breadcrumbs) -> Gom chữ (Flatten) -> Ép khuôn (Split) -> Tạo đối tượng Chunk -> Lưu DB.**

Bạn có thể xem các file `.json` sinh ra từ script `test_lsu_chunker_detailed.py` trong thư mục này để thấy hình thù cụ thể của Data lúc đi qua từng hàm.
