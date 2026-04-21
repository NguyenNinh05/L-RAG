╔══════════════════════════════════════════════════════════════════════╗
║  L-RAG — Chương 4: Demo truy xuất & Đánh giá hiệu năng            ║
║  Cặp tài liệu: docs_test/v1.pdf ↔ docs_test/v2.pdf               ║
║  Thời gian:                                   2026-03-17 21:18:57 ║
╚══════════════════════════════════════════════════════════════════════╝

======================================================================
  4.3 | BENCHMARK: INGESTION & CHUNKING
======================================================================

  📄 File A: docs_test\v1.pdf  (211.4 KB)
     Đoạn thô:  202 paragraphs
     Load time: 2.015s

  📄 File B: docs_test\v2.pdf  (225.1 KB)
     Đoạn thô:  225 paragraphs
     Load time: 1.795s

──────────────────────────────────────────────────────────────────────
  Danh sách chunks
──────────────────────────────────────────────────────────────────────

  Doc A (25 chunks):
    [ 0] Mở đầu                                   |   915 chars | page=None
    [ 1] Phần I > Điều 1                          |   906 chars | page=1
    [ 2] Phần I > Điều 2                          |   586 chars | page=2
    [ 3] Phần I > Điều 3                          |   478 chars | page=2
    [ 4] Phần I > Điều 4                          |   527 chars | page=2
    [ 5] Phần I > Điều 5                          |   512 chars | page=3
    [ 6] Phần I > Điều 6                          |   355 chars | page=3
    [ 7] Phần II > Điều 1                         |   685 chars | page=4
    [ 8] Phần II > Điều 2                         |   644 chars | page=4
    [ 9] Phần II > Điều 3                         |   501 chars | page=4
    [10] Phần II > Điều 4                         |   901 chars | page=4
    [11] Phần II > Điều 5                         |   468 chars | page=5
    [12] Phần II > Điều 6                         |   449 chars | page=5
    [13] Phần II > Điều 7                         |   644 chars | page=5
    [14] Phần II > Điều 8                         |   414 chars | page=6
    [15] Phần III > Điều 1                        |   649 chars | page=7
    [16] Phần III > Điều 2                        |   602 chars | page=7
    [17] Phần III > Điều 3                        |   746 chars | page=7
    [18] Phần III > Điều 4                        |   347 chars | page=8
    [19] Phần III > Điều 5                        |   481 chars | page=8
    [20] Phần III > Điều 6                        |   372 chars | page=8
    [21] Phần III > Điều 7                        |   439 chars | page=8
    [22] Phần III > Điều 8                        |   369 chars | page=8
    [23] Phụ lục A                                |   763 chars | page=9
    [24] Phụ lục B                                |   977 chars | page=10

  Doc B (26 chunks):
    [ 0] Mở đầu                                   |   915 chars | page=None
    [ 1] Phần I > Điều 1                          |   906 chars | page=1
    [ 2] Phần I > Điều 2                          |   586 chars | page=2
    [ 3] Phần I > Điều 3                          |   478 chars | page=2
    [ 4] Phần I > Điều 4                          |   527 chars | page=2
    [ 5] Phần I > Điều 5                          |   512 chars | page=3
    [ 6] Phần I > Điều 6                          |   355 chars | page=3
    [ 7] Phần II > Điều 1                         |   782 chars | page=4
    [ 8] Phần II > Điều 2                         |   644 chars | page=4
    [ 9] Phần II > Điều 3                         |   501 chars | page=4
    [10] Phần II > Điều 4                         |   923 chars | page=4
    [11] Phần II > Điều 5                         |   478 chars | page=5
    [12] Phần II > Điều 6                         |   449 chars | page=5
    [13] Phần II > Điều 7                         |   704 chars | page=5
    [14] Phần II > Điều 8                         |   412 chars | page=6
    [15] Phần III > Điều 1                        |   649 chars | page=7
    [16] Phần III > Điều 2                        |   602 chars | page=7
    [17] Phần III > Điều 3                        |   748 chars | page=7
    [18] Phần III > Điều 4                        |   347 chars | page=8
    [19] Phần III > Điều 5                        |   490 chars | page=8
    [20] Phần III > Điều 6                        |   372 chars | page=8
    [21] Phần III > Điều 7                        |   439 chars | page=8
    [22] Phần III > Điều 8                        |   369 chars | page=8
    [23] Phụ lục A                                |   788 chars | page=9
    [24] Phụ lục B                                |   909 chars | page=10
    [25] Phụ lục C                                |  1402 chars | page=11

  Tổng thời gian ingestion+chunking: 3.390s

======================================================================
  4.3 | BENCHMARK: EMBEDDING & VECTOR DB
======================================================================

  Model:           qwen3-embedding:0.6b
  Tổng chunks:     51
  INSTRUCTION_DOC: "Represent this legal document for retrieval"

──────────────────────────────────────────────────────────────────────
  Cold-start: Nạp model lần đầu
──────────────────────────────────────────────────────────────────────
  Cold-start latency: 3.16s

──────────────────────────────────────────────────────────────────────
  Batch Embedding toàn bộ
──────────────────────────────────────────────────────────────────────
  Thời gian embedding + store: 3.50s
  Throughput:                  14.6 chunks/s
  Vector dimension:            1024
  Avg time per chunk:          68.6 ms
  RAM trước:                   383 MB
  RAM sau:                     405 MB
  RAM tăng thêm:               23 MB
  ChromaDB on disk:            9616.6 KB (9.39 MB)

======================================================================
  4.1 | DEMO TRUY XUẤT THEO ĐIỀU KHOẢN
======================================================================

──────────────────────────────────────────────────────────────────────
  Query 1/10
──────────────────────────────────────────────────────────────────────
  🔍 "Mức phạt khi chậm tiến độ triển khai là bao nhiêu?"
     Mô tả:    Kiểm tra khả năng tìm đúng điều khoản phạt dù dùng từ đồng nghĩa
     Kỳ vọng:  Phần III > Điều 3
     Bộ lọc:   doc_A
     [1] Phần III > Điều 3                        | sim=0.6802 ✅ HIT
         ── FULL TEXT (746 chars) ──
         │ Điều 3: Phạt vi phạm và bồi thường thiệt hại
         │ 3.1. Phạt chậm tiến độ: Bên A chậm bàn giao mỗi ngày chịu phạt 0,1% trên giá trị hạng mục bị chậm.
         │ - 3.2. Phạt vi phạm bảo mật: Nếu Bên A vi phạm nghĩa vụ bảo mật, phạt tối thiểu 500.000.000 VNĐ và bồi thường toàn bộ thiệt hại thực tế.
         │ - 3.3. Bảng phạt vi phạm:
         │ **Loại vi phạm** | **Mức phạt** | **Tối đa**
         │ Chậm tiến độ bàn giao | 0,1%/ngày/hạng mục | 10% giá trị HĐ
         │ Chậm thanh toán (Bên B) | 0,03%/ngày trên số tiền chậm | 5% giá trị HĐ
         │ Vi phạm bảo mật | 500.000.000 VNĐ + bồi<br>thường thực tế | Không giới hạn
         │ Đơn phương chấm dứt HĐ | 15% tổng giá trị HĐ | Không giới hạn
         │ 3.4. Mức phạt tối đa cho tất cả vi phạm (trừ vi phạm bảo mật và đơn phương chấm dứt) không vượt quá 10% tổng giá trị hợp đồng.
         ── END ──
     [2] Phần I > Điều 3                          | sim=0.4950
         ── FULL TEXT (478 chars) ──
         │ Điều 3: Thời hạn hợp đồng
         │ 3.1. Hợp đồng có hiệu lực kể từ ngày ký và có thời hạn 24 tháng.
         │ 3.2. Cơ cấu thời gian thực hiện:
         │ **Giai đoạn** | **Nội dung** | **Thời gian**
         │ Giai đoạn 1 | Phân tích & thiết kế hệ thống | Tháng 1-3
         │ Giai đoạn 2 | Phát triển & kiểm thử | Tháng 4-9
         │ Giai đoạn 3 | Triển khai & đào tạo | Tháng 10-12
         │ Giai đoạn 4 | Vận hành & bảo trì | Tháng 13-24
         │ 3.3. Hợp đồng có thể được gia hạn theo thỏa thuận bằng văn bản của hai bên ít nhất 60 ngày trước khi hết hạn.
         ── END ──
     [3] Phần II > Điều 7                         | sim=0.4940
         ── FULL TEXT (644 chars) ──
         │ Điều 7: Cam kết mức độ dịch vụ (SLA)
         │ 7.1. Bên A cam kết đảm bảo tính sẵn sàng của hệ thống theo các mức độ sau:
         │ **Loại sự cố** | **Mô tả** | **Thời gian phản**<br>**hồi** | **Thời gian khắc**<br>**phục**
         │ Cấp 1 - Nghiêm<br>trọng | Hệ thống ngừng hoạt động<br>hoàn toàn | 30 phút | 4 giờ
         │ Cấp 2 - Cao | Chức năng quan trọng bị lỗi | 2 giờ | 8 giờ
         │ Cấp 3 - Trung bình | Chức năng phụ bị lỗi, có<br>workaround | 4 giờ | 24 giờ
         │ Cấp 4 - Thấp | Lỗi nhỏ, không ảnh hưởng<br>nghiệp vụ | 8 giờ | 5 ngày
         │ - 7.2. Uptime cam kết: 99.5%/tháng (không tính thời gian bảo trì theo lịch).
         │ 7.3. Bên A cung cấp báo cáo SLA hàng tháng trước ngày 5 của tháng tiếp theo.
         ── END ──
     ⏱ Latency tổng: 2157.5 ms  (Embed: 2150.9 ms | ChromaDB: 6.6 ms)

──────────────────────────────────────────────────────────────────────
  Query 2/10
──────────────────────────────────────────────────────────────────────
  🔍 "Nghĩa vụ bảo mật và bảo vệ dữ liệu cá nhân"
     Mô tả:    Tìm điều khoản bảo mật (khác nhau giữa v1 và v2)
     Kỳ vọng:  Phần II > Điều 7
     Bộ lọc:   doc_B
     [1] Phụ lục C                                | sim=0.6940
         ── FULL TEXT (1402 chars) ──
         │ PHỤ LỤC C
         │ CAM KẾT BẢO MẬT VÀ XỬ LÝ DỮ LIỆU CÁ NHÂN
         │ (Bổ sung theo yêu cầu Nghị định 13/2023/NĐ-CP - Chỉ áp dụng cho Phiên bản 2.0)
         │ C.1. Phân loại dữ liệu:
         │ **Loại dữ liệu** | **Mô tả** | **Biện pháp bảo vệ** | **Thời gian lưu trữ**
         │ Dữ liệu nhân sự | CCCD, lương, bảo hiểm NV | Mã hóa AES-256, RBAC | 10 năm sau nghỉ việc
         │ Dữ liệu tài chính | Sổ sách, báo cáo, giao dịch | Mã hóa + audit log | 10 năm theo luật kế<br>toán
         │ Dữ liệu khách hàng | Thông tin mua hàng, hợp<br>đồng | Mã hóa, tách biệt tenant | 5 năm sau hết HĐ
         │ Log hệ thống | Access log, error log | WORM storage | 2 năm
         │ C.2. Quyền của chủ thể dữ liệu:
         │ Bên A hỗ trợ Bên B thực hiện các quyền của chủ thể dữ liệu theo Điều 9 Nghị định 13/2023/NĐ-CP:
         │ - Quyền biết: Cung cấp thông tin trong vòng 5 ngày làm việc.
         │ - Quyền truy cập: Xuất dữ liệu theo định dạng chuẩn trong vòng 10 ngày.
         │ - Quyền xóa: Thực hiện xóa trong vòng 15 ngày (trừ dữ liệu bắt buộc lưu theo luật).
         │ - Quyền hạn chế xử lý: Áp dụng trong vòng 24 giờ nhận yêu cầu.
         │ C.3. Quy trình xử lý sự cố dữ liệu:
         │ **Mức độ sự cố** | **Phạm vi ảnh hưởng** | **Thời gian thông báo nội bộ** | **Thời gian báo cáo cơ**<br>**quan**
         │ Cấp 1 - Nghiêm trọng | > 1000 chủ thể DL | Ngay lập tức | 24 giờ → Bộ TTTT
         │ Cấp 2 - Cao | 100-1000 chủ thể DL | 2 giờ | 48 giờ → Bộ TTTT
         │ Cấp 3 - Trung bình | < 100 chủ thể DL | 4 giờ | 72 giờ (nếu cần)
         │ Hợp đồng này được lập và ký kết tại Hà Nội vào ngày tháng năm 2024.
         ── END ──
     [2] Phần II > Điều 7                         | sim=0.5941 ✅ HIT
         ── FULL TEXT (704 chars) ──
         │ Điều 7: Bảo mật và bảo vệ dữ liệu
         │ 7.1. Bên A cam kết tuân thủ các quy định về bảo vệ dữ liệu cá nhân theo Nghị định 13/2023/NĐ-CP và các văn bản hướng dẫn liên quan.
         │ 7.2. Các biện pháp bảo mật kỹ thuật bắt buộc:
         │ Biện pháp Tiêu chuẩn áp dụng Tần suất kiểm tra
         │ Mã hóa dữ liệu lưu trữ | AES-256 | Hàng quý
         │ Mã hóa kết nối | TLS 1.3+ | Liên tục
         │ Kiểm soát truy cập | RBAC + MFA | Hàng tháng
         │ Sao lưu dữ liệu | 3-2-1 Backup Rule | Hàng ngày
         │ Kiểm thử xâm nhập | OWASP Top 10 | 6 tháng/lần
         │ 7.3. Trong trường hợp xảy ra sự cố rò rỉ dữ liệu, Bên A phải thông báo cho Bên B trong vòng 24 giờ và cơ quan có thẩm quyền trong vòng 72 giờ.
         │ 7.4. Nghĩa vụ bảo mật tiếp tục có hiệu lực trong 5 năm sau khi hợp đồng chấm dứt.
         ── END ──
     [3] Phần III > Điều 1                        | sim=0.5166
         ── FULL TEXT (649 chars) ──
         │ Điều 1: Quyền và nghĩa vụ của Bên A
         │ - 1.1. Nghĩa vụ của Bên A:
         │ - a) Thực hiện dịch vụ đúng phạm vi, tiến độ và chất lượng đã cam kết.
         │ - b) Bố trí đội ngũ nhân sự đủ năng lực, có kinh nghiệm phù hợp.
         │ - c) Cung cấp đầy đủ tài liệu kỹ thuật, hướng dẫn sử dụng và mã nguồn. d) Bảo mật toàn bộ thông tin mật của Bên B theo quy định tại Phần II. e) Báo cáo tiến độ hàng tuần và tham dự họp điều phối dự án định kỳ.
         │ - 1.2. Quyền của Bên A:
         │ - a) Nhận thanh toán đầy đủ và đúng hạn theo lịch đã thỏa thuận.
         │ - b) Từ chối các yêu cầu ngoài phạm vi hợp đồng cho đến khi có thỏa thuận thay đổi. c) Tạm dừng cung cấp dịch vụ nếu Bên B chậm thanh toán quá 30 ngày.
         ── END ──
     ⏱ Latency tổng: 2171.9 ms  (Embed: 2171.9 ms | ChromaDB: 0.0 ms)

──────────────────────────────────────────────────────────────────────
  Query 3/10
──────────────────────────────────────────────────────────────────────
  🔍 "Giá trị hợp đồng và lịch thanh toán theo đợt"
     Mô tả:    Tìm điều khoản tài chính
     Kỳ vọng:  Phần II > Điều 4
     Bộ lọc:   doc_A
     [1] Phần II > Điều 4                         | sim=0.5772 ✅ HIT
         ── FULL TEXT (901 chars) ──
         │ Điều 4: Giá trị hợp đồng và cơ cấu chi phí
         │ 4.1. Cơ cấu chi phí chi tiết:
         │ Hạng mục dịch vụ Mô tả
         │ Giá trị (VNĐ)
         │ Phân tích & Thiết kế | Bao gồm phân tích yêu cầu, thiết kế<br>kiến trúc | 800.000.000
         │ Phát triển phần mềm | Lập trình, tích hợp API, kiểm thử đơn<br>vị | 1.200.000.000
         │ Triển khai hệ thống | Cài đặt, cấu hình, migration dữ liệu | 400.000.000
         │ Đào tạo người dùng | Đào tạo tối đa 50 người, 3 ngày/lớp | 200.000.000
         │ Bảo trì năm 1 | Hỗ trợ 8/5, SLA 4 giờ | 300.000.000
         │ Bảo trì năm 2+ | Hỗ trợ 8/5, SLA 8 giờ | 250.000.000
         │ - 4.2. Tổng giá trị hợp đồng: 3.150.000.000 VNĐ (đã bao gồm VAT 10%).
         │ - 4.3. Phương thức thanh toán:
         │ Đợt 1: 20% khi ký hợp đồng.
         │ Đợt 2: 30% khi hoàn thành Giai đoạn 1 (có biên bản nghiệm thu).
         │ Đợt 3: 30% khi hoàn thành Giai đoạn 2 (có biên bản nghiệm thu). Đợt 4: 20% khi go-live và nghiệm thu tổng thể.
         │ - 4.4. Phí bảo trì hàng năm thanh toán trước vào đầu mỗi năm hợp đồng.
         ── END ──
     [2] Phụ lục B                                | sim=0.5464
         ── FULL TEXT (977 chars) ──
         │ PHỤ LỤC B TIÊU CHÍ NGHIỆM THU CHI TIẾT
         │ (Đính kèm và là bộ phận không tách rời của Hợp đồng số 2024/HĐDV/CNTT/001)
         │ B.1. Tiêu chí nghiệm thu chức năng:
         │ **Module** | **Chức năng** | **Tiêu chí đạt** | **Trọng số**
         │ Kế toán | Ghi nhận chứng từ | 100% ca kiểm thử đạt | 20%
         │ Kế toán | Báo cáo tài chính | Khớp số liệu 100% | 15%
         │ Nhân sự | Quản lý hồ sơ NV | 100% ca kiểm thử đạt | 15%
         │ Nhân sự | Tính lương tự động | Sai lệch < 0.01% | 20%
         │ Kho vận | Nhập/xuất kho | 100% ca kiểm thử đạt | 15%
         │ Kho vận | Báo cáo tồn kho | Thời gian < 30 giây | 15%
         │ B.2. Tiêu chí nghiệm thu phi chức năng:
         │ **Chỉ số** | **Yêu cầu** | **Phương pháp đo**
         │ Thời gian phản hồi | < 3 giây cho 95% request | JMeter load test 500 users
         │ Throughput | > 1000 transaction/phút | JMeter stress test
         │ Uptime | > 99.5%/tháng | Monitoring 24/7
         │ Recovery Time | < 4 giờ (RTO) | DR drill test
         │ Recovery Point | < 1 giờ dữ liệu mất (RPO) | Backup restore test
         │ Hợp đồng này được lập và ký kết tại Hà Nội vào ngày tháng năm 2024.
         ── END ──
     [3] Phần I > Điều 3                          | sim=0.5250
         ── FULL TEXT (478 chars) ──
         │ Điều 3: Thời hạn hợp đồng
         │ 3.1. Hợp đồng có hiệu lực kể từ ngày ký và có thời hạn 24 tháng.
         │ 3.2. Cơ cấu thời gian thực hiện:
         │ **Giai đoạn** | **Nội dung** | **Thời gian**
         │ Giai đoạn 1 | Phân tích & thiết kế hệ thống | Tháng 1-3
         │ Giai đoạn 2 | Phát triển & kiểm thử | Tháng 4-9
         │ Giai đoạn 3 | Triển khai & đào tạo | Tháng 10-12
         │ Giai đoạn 4 | Vận hành & bảo trì | Tháng 13-24
         │ 3.3. Hợp đồng có thể được gia hạn theo thỏa thuận bằng văn bản của hai bên ít nhất 60 ngày trước khi hết hạn.
         ── END ──
     ⏱ Latency tổng: 2195.6 ms  (Embed: 2195.0 ms | ChromaDB: 0.7 ms)

──────────────────────────────────────────────────────────────────────
  Query 4/10
──────────────────────────────────────────────────────────────────────
  🔍 "Thời hạn hiệu lực và các giai đoạn triển khai"
     Mô tả:    Tìm điều khoản thời hạn
     Kỳ vọng:  Phần I > Điều 3
     Bộ lọc:   doc_A
     [1] Phần II > Điều 5                         | sim=0.5694
         ── FULL TEXT (468 chars) ──
         │ Điều 5: Nghiệm thu và bàn giao
         │ 5.1. Quy trình nghiệm thu từng giai đoạn:
         │ a) Bên A gửi thông báo hoàn thành và tài liệu nghiệm thu.
         │ b) Bên B có 10 ngày làm việc để kiểm tra và phản hồi.
         │ c) Nếu không có phản hồi sau 10 ngày, coi như nghiệm thu đạt.
         │ d) Nếu có vấn đề, Bên A có 7 ngày làm việc để khắc phục.
         │ 5.2. Tiêu chí nghiệm thu tổng thể bao gồm: đáp ứng 100% yêu cầu chức năng bắt buộc, hiệu năng đạt theo benchmark đã thống nhất, và đào tạo hoàn thành theo kế hoạch.
         ── END ──
     [2] Phần I > Điều 3                          | sim=0.5679 ✅ HIT
         ── FULL TEXT (478 chars) ──
         │ Điều 3: Thời hạn hợp đồng
         │ 3.1. Hợp đồng có hiệu lực kể từ ngày ký và có thời hạn 24 tháng.
         │ 3.2. Cơ cấu thời gian thực hiện:
         │ **Giai đoạn** | **Nội dung** | **Thời gian**
         │ Giai đoạn 1 | Phân tích & thiết kế hệ thống | Tháng 1-3
         │ Giai đoạn 2 | Phát triển & kiểm thử | Tháng 4-9
         │ Giai đoạn 3 | Triển khai & đào tạo | Tháng 10-12
         │ Giai đoạn 4 | Vận hành & bảo trì | Tháng 13-24
         │ 3.3. Hợp đồng có thể được gia hạn theo thỏa thuận bằng văn bản của hai bên ít nhất 60 ngày trước khi hết hạn.
         ── END ──
     [3] Phần II > Điều 7                         | sim=0.5503
         ── FULL TEXT (644 chars) ──
         │ Điều 7: Cam kết mức độ dịch vụ (SLA)
         │ 7.1. Bên A cam kết đảm bảo tính sẵn sàng của hệ thống theo các mức độ sau:
         │ **Loại sự cố** | **Mô tả** | **Thời gian phản**<br>**hồi** | **Thời gian khắc**<br>**phục**
         │ Cấp 1 - Nghiêm<br>trọng | Hệ thống ngừng hoạt động<br>hoàn toàn | 30 phút | 4 giờ
         │ Cấp 2 - Cao | Chức năng quan trọng bị lỗi | 2 giờ | 8 giờ
         │ Cấp 3 - Trung bình | Chức năng phụ bị lỗi, có<br>workaround | 4 giờ | 24 giờ
         │ Cấp 4 - Thấp | Lỗi nhỏ, không ảnh hưởng<br>nghiệp vụ | 8 giờ | 5 ngày
         │ - 7.2. Uptime cam kết: 99.5%/tháng (không tính thời gian bảo trì theo lịch).
         │ 7.3. Bên A cung cấp báo cáo SLA hàng tháng trước ngày 5 của tháng tiếp theo.
         ── END ──
     ⏱ Latency tổng: 2256.8 ms  (Embed: 2255.6 ms | ChromaDB: 1.2 ms)

──────────────────────────────────────────────────────────────────────
  Query 5/10
──────────────────────────────────────────────────────────────────────
  🔍 "Tiêu chí nghiệm thu phần mềm ERP"
     Mô tả:    Tìm điều khoản nghiệm thu
     Kỳ vọng:  Phần II > Điều 5
     Bộ lọc:   doc_A
     [1] Phụ lục B                                | sim=0.6127
         ── FULL TEXT (977 chars) ──
         │ PHỤ LỤC B TIÊU CHÍ NGHIỆM THU CHI TIẾT
         │ (Đính kèm và là bộ phận không tách rời của Hợp đồng số 2024/HĐDV/CNTT/001)
         │ B.1. Tiêu chí nghiệm thu chức năng:
         │ **Module** | **Chức năng** | **Tiêu chí đạt** | **Trọng số**
         │ Kế toán | Ghi nhận chứng từ | 100% ca kiểm thử đạt | 20%
         │ Kế toán | Báo cáo tài chính | Khớp số liệu 100% | 15%
         │ Nhân sự | Quản lý hồ sơ NV | 100% ca kiểm thử đạt | 15%
         │ Nhân sự | Tính lương tự động | Sai lệch < 0.01% | 20%
         │ Kho vận | Nhập/xuất kho | 100% ca kiểm thử đạt | 15%
         │ Kho vận | Báo cáo tồn kho | Thời gian < 30 giây | 15%
         │ B.2. Tiêu chí nghiệm thu phi chức năng:
         │ **Chỉ số** | **Yêu cầu** | **Phương pháp đo**
         │ Thời gian phản hồi | < 3 giây cho 95% request | JMeter load test 500 users
         │ Throughput | > 1000 transaction/phút | JMeter stress test
         │ Uptime | > 99.5%/tháng | Monitoring 24/7
         │ Recovery Time | < 4 giờ (RTO) | DR drill test
         │ Recovery Point | < 1 giờ dữ liệu mất (RPO) | Backup restore test
         │ Hợp đồng này được lập và ký kết tại Hà Nội vào ngày tháng năm 2024.
         ── END ──
     [2] Phần II > Điều 1                         | sim=0.6035
         ── FULL TEXT (685 chars) ──
         │ Điều 1: Phạm vi và nội dung dịch vụ
         │ Bên A cung cấp cho Bên B các dịch vụ sau:
         │ - a) Phân tích yêu cầu nghiệp vụ và thiết kế kiến trúc hệ thống ERP.
         │ - b) Phát triển và tùy chỉnh phần mềm ERP theo đặc tả yêu cầu được hai bên phê duyệt.
         │ - c) Tích hợp hệ thống ERP với các hệ thống hiện có của Bên B (kế toán, nhân sự, kho).
         │ - d) Triển khai, kiểm thử chấp nhận (UAT) và go-live hệ thống.
         │ - f) Bảo trì và hỗ trợ kỹ thuật theo cam kết SLA trong suốt thời hạn hợp đồng.
         │ - 1.2. Các dịch vụ sau nằm ngoài phạm vi hợp đồng (Out of Scope):
         │ - Phần cứng, hạ tầng server và thiết bị mạng.
         │ - Bản quyền phần mềm của bên thứ ba (Oracle, SAP, Microsoft).
         │ - Dịch vụ tại các chi nhánh nước ngoài của Bên B.
         ── END ──
     [3] Phần II > Điều 5                         | sim=0.5377 ✅ HIT
         ── FULL TEXT (468 chars) ──
         │ Điều 5: Nghiệm thu và bàn giao
         │ 5.1. Quy trình nghiệm thu từng giai đoạn:
         │ a) Bên A gửi thông báo hoàn thành và tài liệu nghiệm thu.
         │ b) Bên B có 10 ngày làm việc để kiểm tra và phản hồi.
         │ c) Nếu không có phản hồi sau 10 ngày, coi như nghiệm thu đạt.
         │ d) Nếu có vấn đề, Bên A có 7 ngày làm việc để khắc phục.
         │ 5.2. Tiêu chí nghiệm thu tổng thể bao gồm: đáp ứng 100% yêu cầu chức năng bắt buộc, hiệu năng đạt theo benchmark đã thống nhất, và đào tạo hoàn thành theo kế hoạch.
         ── END ──
     ⏱ Latency tổng: 2237.8 ms  (Embed: 2237.8 ms | ChromaDB: 0.0 ms)

──────────────────────────────────────────────────────────────────────
  Query 6/10
──────────────────────────────────────────────────────────────────────
  🔍 "Trường hợp nào được miễn trách nhiệm hợp đồng?"
     Mô tả:    Hỏi gián tiếp về bất khả kháng — không dùng từ khóa chính xác
     Kỳ vọng:  Phần II > Điều 8
     Bộ lọc:   doc_A
     [1] Phần II > Điều 8                         | sim=0.6353 ✅ HIT
         ── FULL TEXT (414 chars) ──
         │ Điều 8: Bất khả kháng
         │ 8.1. Các bên được miễn trách nhiệm thực hiện nghĩa vụ hợp đồng trong thời gian xảy ra Sự kiện bất khả kháng theo định nghĩa tại Phần I, Điều 1.
         │ - 8.2. Bên bị ảnh hưởng phải thông báo cho bên kia trong vòng 48 giờ kể từ khi sự kiện xảy ra, kèm theo bằng chứng xác nhận.
         │ 8.3. Nếu sự kiện bất khả kháng kéo dài quá 60 ngày, mỗi bên có quyền đơn phương chấm dứt hợp đồng mà không phải bồi thường.
         ── END ──
     [2] Phần I > Điều 6                          | sim=0.5568
         ── FULL TEXT (355 chars) ──
         │ Điều 6: Tính độc lập của các điều khoản
         │ 6.1. Nếu bất kỳ điều khoản nào của hợp đồng bị tuyên bố vô hiệu hoặc không thể thực thi theo pháp luật hiện hành, các điều khoản còn lại vẫn tiếp tục có hiệu lực đầy đủ.
         │ 6.2. Các bên sẽ thương lượng thiện chí để thay thế điều khoản vô hiệu bằng điều khoản hợp lệ có nội dung gần nhất với ý định ban đầu của các bên.
         ── END ──
     [3] Phần III > Điều 7                        | sim=0.5317
         ── FULL TEXT (439 chars) ──
         │ Điều 7: Chấm dứt hợp đồng
         │ 7.1. Hợp đồng chấm dứt trong các trường hợp:
         │ a) Hết thời hạn theo Phần I, Điều 3 và không có thỏa thuận gia hạn.
         │ b) Hai bên thỏa thuận bằng văn bản.
         │ c) Một bên vi phạm nghiêm trọng và không khắc phục trong vòng 20 ngày sau khi nhận thông báo.
         │ d) Sự kiện bất khả kháng kéo dài quá 60 ngày.
         │ 7.2. Khi chấm dứt hợp đồng, Bên A phải bàn giao toàn bộ tài liệu, mã nguồn và dữ liệu của Bên B trong vòng 10 ngày làm việc.
         ── END ──
     ⏱ Latency tổng: 2262.2 ms  (Embed: 2261.2 ms | ChromaDB: 1.0 ms)

──────────────────────────────────────────────────────────────────────
  Query 7/10
──────────────────────────────────────────────────────────────────────
  🔍 "Danh sách nhân sự chủ chốt thực hiện dự án"
     Mô tả:    Tìm phụ lục nhân sự
     Kỳ vọng:  Phụ lục > Phụ lục A
     Bộ lọc:   doc_A
     [1] Phụ lục A                                | sim=0.5755 ✅ HIT
         ── FULL TEXT (763 chars) ──
         │ PHỤ LỤC A DANH SÁCH NHÂN SỰ CHỦ CHỐT THỰC HIỆN DỰA ÁN
         │ (Đính kèm và là bộ phận không tách rời của Hợp đồng số 2024/HĐDV/CNTT/001)
         │ Danh sách nhân sự chủ chốt được Bên A cam kết bố trí để thực hiện dự án:
         │ **Họ và tên** | **Vị trí trong dự án** | **Kinh nghiệm** | **Chứng chỉ**
         │ Nguyễn Minh Tuấn | Project Manager | 15 năm | PMP, PRINCE2
         │ Trần Văn Dũng | Solution Architect | 12 năm | AWS SA, TOGAF
         │ Lê Thị Hương | Lead Developer | 8 năm | Oracle Certified
         │ Phạm Quốc Huy | Business Analyst | 7 năm | CBAP
         │ Đỗ Thị Mai | QA Lead | 6 năm | ISTQB Advanced
         │ Vũ Hoàng Nam | DevOps Engineer | 5 năm | CKA, AWS DevOps
         │ Lưu ý: Bên A không được thay thế nhân sự chủ chốt mà không có sự đồng ý trước bằng văn bản của Bên B. Nhân sự thay thế phải có năng lực tương đương hoặc cao hơn.
         ── END ──
     [2] Phần I > Điều 4                          | sim=0.5417
         ── FULL TEXT (527 chars) ──
         │ Điều 4: Đại diện liên lạc và quản lý dự án
         │ 4.1. Mỗi bên chỉ định một Quản lý dự án (Project Manager) làm đầu mối liên lạc chính:
         │ **Bên** | **Họ tên** | **Chức vụ** | **Email** | **Điện thoại**
         │ Bên A | Trần Văn Dũng | PM Cấp cao | dungTV@benA.vn | 0912 345 678
         │ Bên B | Phạm Thị Lan | IT Director | lanPT@phuhung.vn | 0987 654 321
         │ 4.2. Thay đổi đại diện liên lạc phải được thông báo bằng văn bản trước ít nhất 5 ngày làm việc.
         │ 4.3. Các quyết định kỹ thuật trong phạm vi ngân sách được duyệt không cần phê duyệt thêm của cấp trên.
         ── END ──
     [3] Phần III > Điều 1                        | sim=0.5268
         ── FULL TEXT (649 chars) ──
         │ Điều 1: Quyền và nghĩa vụ của Bên A
         │ - 1.1. Nghĩa vụ của Bên A:
         │ - a) Thực hiện dịch vụ đúng phạm vi, tiến độ và chất lượng đã cam kết.
         │ - b) Bố trí đội ngũ nhân sự đủ năng lực, có kinh nghiệm phù hợp.
         │ - c) Cung cấp đầy đủ tài liệu kỹ thuật, hướng dẫn sử dụng và mã nguồn. d) Bảo mật toàn bộ thông tin mật của Bên B theo quy định tại Phần II. e) Báo cáo tiến độ hàng tuần và tham dự họp điều phối dự án định kỳ.
         │ - 1.2. Quyền của Bên A:
         │ - a) Nhận thanh toán đầy đủ và đúng hạn theo lịch đã thỏa thuận.
         │ - b) Từ chối các yêu cầu ngoài phạm vi hợp đồng cho đến khi có thỏa thuận thay đổi. c) Tạm dừng cung cấp dịch vụ nếu Bên B chậm thanh toán quá 30 ngày.
         ── END ──
     ⏱ Latency tổng: 2263.3 ms  (Embed: 2262.5 ms | ChromaDB: 0.7 ms)

──────────────────────────────────────────────────────────────────────
  Query 8/10
──────────────────────────────────────────────────────────────────────
  🔍 "Nơi giải quyết tranh chấp hợp đồng"
     Mô tả:    Tìm điều khoản tranh chấp (Hà Nội vs TP.HCM)
     Kỳ vọng:  Phần III > Điều 5
     Bộ lọc:   doc_A
     [1] Phần III > Điều 5                        | sim=0.5917 ✅ HIT
         ── FULL TEXT (481 chars) ──
         │ Điều 5: Giải quyết tranh chấp
         │ 5.1. Các bên cam kết giải quyết tranh chấp theo trình tự sau:
         │ Bước 1: Thương lượng trực tiếp giữa đại diện có thẩm quyền trong vòng 30 ngày.
         │ Bước 2: Hòa giải thông qua Trung tâm Hòa giải Thương mại Việt Nam trong vòng 30 ngày tiếp theo.
         │ Bước 3: Nếu hòa giải không thành, tranh chấp được đưa ra Tòa án nhân dân có thẩm quyền tại Hà Nội.
         │ 5.2. Trong thời gian giải quyết tranh chấp, các bên tiếp tục thực hiện các nghĩa vụ không liên quan đến tranh chấp.
         ── END ──
     [2] Phần I > Điều 5                          | sim=0.5339
         ── FULL TEXT (512 chars) ──
         │ Điều 5: Ngôn ngữ và tài liệu hợp đồng
         │ 5.1. Hợp đồng được lập bằng tiếng Việt. Trong trường hợp có mâu thuẫn giữa bản tiếng Việt và bản dịch sang ngôn ngữ khác, bản tiếng Việt có giá trị pháp lý cao hơn.
         │ - 5.2. Hồ sơ hợp đồng bao gồm các tài liệu sau đây theo thứ tự ưu tiên pháp lý:
         │ - a) Hợp đồng này và các phụ lục đính kèm
         │ - b) Biên bản thỏa thuận bổ sung (nếu có)
         │ - c) Tài liệu kỹ thuật và đặc tả yêu cầu đã được hai bên phê duyệt
         │ - 5.3. Hợp đồng được lập thành 04 bản gốc, Bên A giữ 02 bản, Bên B giữ 02 bản.
         ── END ──
     [3] Phần I > Điều 2                          | sim=0.5068
         ── FULL TEXT (586 chars) ──
         │ Điều 2: Mục đích và phạm vi hợp đồng
         │ 2.1. Hợp đồng này điều chỉnh quan hệ hợp tác giữa Bên A và Bên B trong việc triển khai hệ thống ERP tích hợp cho toàn bộ hoạt động của Bên B tại Việt Nam.
         │ 2.2. Phạm vi địa lý: Hợp đồng áp dụng cho tất cả các chi nhánh, văn phòng và cơ sở sản xuất của Bên B tại Việt Nam, bao gồm:
         │ a) Trụ sở chính tại TP. Hồ Chí Minh
         │ - b) Chi nhánh miền Bắc tại Hà Nội
         │ c) Chi nhánh miền Trung tại Đà Nẵng
         │ d) Nhà máy sản xuất tại Bình Dương và Đồng Nai
         │ 2.3. Hợp đồng không áp dụng cho các hoạt động của Bên B tại nước ngoài, trừ khi có thỏa thuận bổ sung bằng văn bản.
         ── END ──
     ⏱ Latency tổng: 2238.9 ms  (Embed: 2238.9 ms | ChromaDB: 0.0 ms)

──────────────────────────────────────────────────────────────────────
  Query 9/10
──────────────────────────────────────────────────────────────────────
  🔍 "Đền bù thiệt hại khi vi phạm thông tin mật"
     Mô tả:    Dùng 'đền bù' thay vì 'phạt', 'thông tin mật' thay vì 'bảo mật'
     Kỳ vọng:  Phần III > Điều 3
     Bộ lọc:   doc_A
     [1] Phần III > Điều 3                        | sim=0.5585 ✅ HIT
         ── FULL TEXT (746 chars) ──
         │ Điều 3: Phạt vi phạm và bồi thường thiệt hại
         │ 3.1. Phạt chậm tiến độ: Bên A chậm bàn giao mỗi ngày chịu phạt 0,1% trên giá trị hạng mục bị chậm.
         │ - 3.2. Phạt vi phạm bảo mật: Nếu Bên A vi phạm nghĩa vụ bảo mật, phạt tối thiểu 500.000.000 VNĐ và bồi thường toàn bộ thiệt hại thực tế.
         │ - 3.3. Bảng phạt vi phạm:
         │ **Loại vi phạm** | **Mức phạt** | **Tối đa**
         │ Chậm tiến độ bàn giao | 0,1%/ngày/hạng mục | 10% giá trị HĐ
         │ Chậm thanh toán (Bên B) | 0,03%/ngày trên số tiền chậm | 5% giá trị HĐ
         │ Vi phạm bảo mật | 500.000.000 VNĐ + bồi<br>thường thực tế | Không giới hạn
         │ Đơn phương chấm dứt HĐ | 15% tổng giá trị HĐ | Không giới hạn
         │ 3.4. Mức phạt tối đa cho tất cả vi phạm (trừ vi phạm bảo mật và đơn phương chấm dứt) không vượt quá 10% tổng giá trị hợp đồng.
         ── END ──
     [2] Phần II > Điều 8                         | sim=0.5363
         ── FULL TEXT (414 chars) ──
         │ Điều 8: Bất khả kháng
         │ 8.1. Các bên được miễn trách nhiệm thực hiện nghĩa vụ hợp đồng trong thời gian xảy ra Sự kiện bất khả kháng theo định nghĩa tại Phần I, Điều 1.
         │ - 8.2. Bên bị ảnh hưởng phải thông báo cho bên kia trong vòng 48 giờ kể từ khi sự kiện xảy ra, kèm theo bằng chứng xác nhận.
         │ 8.3. Nếu sự kiện bất khả kháng kéo dài quá 60 ngày, mỗi bên có quyền đơn phương chấm dứt hợp đồng mà không phải bồi thường.
         ── END ──
     [3] Phần I > Điều 1                          | sim=0.5352
         ── FULL TEXT (906 chars) ──
         │ Điều 1: Định nghĩa và giải thích thuật ngữ
         │ Trong hợp đồng này, các thuật ngữ dưới đây được hiểu như sau:
         │ "Bên A" có nghĩa là Công ty TNHH Giải pháp Công nghệ ABC, pháp nhân được thành lập và hoạt động hợp pháp theo pháp luật Việt Nam.
         │ "Bên B" có nghĩa là Tập đoàn Sản xuất và Thương mại Phú Hưng.
         │ "Dịch vụ" là toàn bộ các hoạt động triển khai, vận hành và hỗ trợ hệ thống CNTT được quy định tại Phần II.
         │ "Hệ thống" là phần mềm ERP tích hợp quản lý toàn bộ hoạt động doanh nghiệp Bên B. "Ngày làm việc" là các ngày từ thứ Hai đến thứ Sáu, không bao gồm ngày lễ theo quy định pháp luật Việt Nam.
         │ "Sự kiện bất khả kháng" là sự kiện nằm ngoài tầm kiểm soát của các bên, bao gồm thiên tai, chiến tranh, dịch bệnh, hoặc các quyết định hành chính của cơ quan nhà nước có thẩm quyền.
         │ "Thông tin mật" là toàn bộ dữ liệu kinh doanh, kỹ thuật, tài chính của Bên B mà Bên A tiếp cận trong quá trình thực hiện hợp đồng.
         ── END ──
     ⏱ Latency tổng: 2230.0 ms  (Embed: 2229.5 ms | ChromaDB: 0.5 ms)

──────────────────────────────────────────────────────────────────────
  Query 10/10
──────────────────────────────────────────────────────────────────────
  🔍 "Cam kết mức độ dịch vụ SLA và uptime"
     Mô tả:    Tìm SLA — chỉ có trong v1, bị xóa ở v2
     Kỳ vọng:  Phần II > Điều 7
     Bộ lọc:   doc_A
     [1] Phần II > Điều 7                         | sim=0.7704 ✅ HIT
         ── FULL TEXT (644 chars) ──
         │ Điều 7: Cam kết mức độ dịch vụ (SLA)
         │ 7.1. Bên A cam kết đảm bảo tính sẵn sàng của hệ thống theo các mức độ sau:
         │ **Loại sự cố** | **Mô tả** | **Thời gian phản**<br>**hồi** | **Thời gian khắc**<br>**phục**
         │ Cấp 1 - Nghiêm<br>trọng | Hệ thống ngừng hoạt động<br>hoàn toàn | 30 phút | 4 giờ
         │ Cấp 2 - Cao | Chức năng quan trọng bị lỗi | 2 giờ | 8 giờ
         │ Cấp 3 - Trung bình | Chức năng phụ bị lỗi, có<br>workaround | 4 giờ | 24 giờ
         │ Cấp 4 - Thấp | Lỗi nhỏ, không ảnh hưởng<br>nghiệp vụ | 8 giờ | 5 ngày
         │ - 7.2. Uptime cam kết: 99.5%/tháng (không tính thời gian bảo trì theo lịch).
         │ 7.3. Bên A cung cấp báo cáo SLA hàng tháng trước ngày 5 của tháng tiếp theo.
         ── END ──
     [2] Phụ lục B                                | sim=0.5572
         ── FULL TEXT (977 chars) ──
         │ PHỤ LỤC B TIÊU CHÍ NGHIỆM THU CHI TIẾT
         │ (Đính kèm và là bộ phận không tách rời của Hợp đồng số 2024/HĐDV/CNTT/001)
         │ B.1. Tiêu chí nghiệm thu chức năng:
         │ **Module** | **Chức năng** | **Tiêu chí đạt** | **Trọng số**
         │ Kế toán | Ghi nhận chứng từ | 100% ca kiểm thử đạt | 20%
         │ Kế toán | Báo cáo tài chính | Khớp số liệu 100% | 15%
         │ Nhân sự | Quản lý hồ sơ NV | 100% ca kiểm thử đạt | 15%
         │ Nhân sự | Tính lương tự động | Sai lệch < 0.01% | 20%
         │ Kho vận | Nhập/xuất kho | 100% ca kiểm thử đạt | 15%
         │ Kho vận | Báo cáo tồn kho | Thời gian < 30 giây | 15%
         │ B.2. Tiêu chí nghiệm thu phi chức năng:
         │ **Chỉ số** | **Yêu cầu** | **Phương pháp đo**
         │ Thời gian phản hồi | < 3 giây cho 95% request | JMeter load test 500 users
         │ Throughput | > 1000 transaction/phút | JMeter stress test
         │ Uptime | > 99.5%/tháng | Monitoring 24/7
         │ Recovery Time | < 4 giờ (RTO) | DR drill test
         │ Recovery Point | < 1 giờ dữ liệu mất (RPO) | Backup restore test
         │ Hợp đồng này được lập và ký kết tại Hà Nội vào ngày tháng năm 2024.
         ── END ──
     [3] Phần II > Điều 1                         | sim=0.5454
         ── FULL TEXT (685 chars) ──
         │ Điều 1: Phạm vi và nội dung dịch vụ
         │ Bên A cung cấp cho Bên B các dịch vụ sau:
         │ - a) Phân tích yêu cầu nghiệp vụ và thiết kế kiến trúc hệ thống ERP.
         │ - b) Phát triển và tùy chỉnh phần mềm ERP theo đặc tả yêu cầu được hai bên phê duyệt.
         │ - c) Tích hợp hệ thống ERP với các hệ thống hiện có của Bên B (kế toán, nhân sự, kho).
         │ - d) Triển khai, kiểm thử chấp nhận (UAT) và go-live hệ thống.
         │ - f) Bảo trì và hỗ trợ kỹ thuật theo cam kết SLA trong suốt thời hạn hợp đồng.
         │ - 1.2. Các dịch vụ sau nằm ngoài phạm vi hợp đồng (Out of Scope):
         │ - Phần cứng, hạ tầng server và thiết bị mạng.
         │ - Bản quyền phần mềm của bên thứ ba (Oracle, SAP, Microsoft).
         │ - Dịch vụ tại các chi nhánh nước ngoài của Bên B.
         ── END ──
     ⏱ Latency tổng: 2255.8 ms  (Embed: 2254.2 ms | ChromaDB: 1.6 ms)

======================================================================
  4.1.2 | TRUY XUẤT CHÉO: Chunk Doc_A → Tìm trên Doc_B
======================================================================
  Mục tiêu: Lấy nội dung 1 điều khoản từ Bản A, tìm đoạn tương đồng ở Bản B.
  Mô phỏng bước đệm cho thuật toán căn chỉnh (Chương 5).


──────────────────────────────────────────────────────────────────────
  Cross-query 1: Phần I > Điều 3
──────────────────────────────────────────────────────────────────────
  Nguồn: Doc_A | Phần I > Điều 3
  Nội dung truy vấn (100 chars): Điều 3: Thời hạn hợp đồng 3.1. Hợp đồng có hiệu lực kể từ ngày ký và có thời hạn 24 tháng. 3.2. Cơ c...
  → [1] Doc_B | Phần I > Điều 3                          | sim=0.9519
  → [2] Doc_B | Phần II > Điều 4                         | sim=0.6302
  → [3] Doc_B | Phần III > Điều 7                        | sim=0.6162
  ⏱ Latency: 2421.3 ms

──────────────────────────────────────────────────────────────────────
  Cross-query 2: Phần I > Điều 4
──────────────────────────────────────────────────────────────────────
  Nguồn: Doc_A | Phần I > Điều 4
  Nội dung truy vấn (100 chars): Điều 4: Đại diện liên lạc và quản lý dự án 4.1. Mỗi bên chỉ định một Quản lý dự án (Project Manager)...
  → [1] Doc_B | Phần I > Điều 4                          | sim=0.9820
  → [2] Doc_B | Phụ lục A                                | sim=0.6044
  → [3] Doc_B | Phần II > Điều 3                         | sim=0.5540
  ⏱ Latency: 2204.8 ms

──────────────────────────────────────────────────────────────────────
  Cross-query 3: Phần I > Điều 5
──────────────────────────────────────────────────────────────────────
  Nguồn: Doc_A | Phần I > Điều 5
  Nội dung truy vấn (100 chars): Điều 5: Ngôn ngữ và tài liệu hợp đồng 5.1. Hợp đồng được lập bằng tiếng Việt. Trong trường hợp có mâ...
  → [1] Doc_B | Phần I > Điều 5                          | sim=0.9802
  → [2] Doc_B | Phần III > Điều 8                        | sim=0.6207
  → [3] Doc_B | Phần III > Điều 6                        | sim=0.5328
  ⏱ Latency: 2274.9 ms

──────────────────────────────────────────────────────────────────────
  Cross-query 4: Phần II > Điều 3
──────────────────────────────────────────────────────────────────────
  Nguồn: Doc_A | Phần II > Điều 3
  Nội dung truy vấn (100 chars): Điều 3: Quy trình quản lý thay đổi - 3.1. Mọi yêu cầu thay đổi phạm vi dịch vụ (Change Request - CR)...
  → [1] Doc_B | Phần II > Điều 3                         | sim=0.9687
  → [2] Doc_B | Phần I > Điều 4                          | sim=0.5674
  → [3] Doc_B | Phần III > Điều 2                        | sim=0.5502
  ⏱ Latency: 2316.0 ms

──────────────────────────────────────────────────────────────────────
  Cross-query 5: Phần II > Điều 4
──────────────────────────────────────────────────────────────────────
  Nguồn: Doc_A | Phần II > Điều 4
  Nội dung truy vấn (100 chars): Điều 4: Giá trị hợp đồng và cơ cấu chi phí 4.1. Cơ cấu chi phí chi tiết: Hạng mục dịch vụ Mô tả Giá ...
  → [1] Doc_B | Phần II > Điều 4                         | sim=0.9390
  → [2] Doc_B | Phần II > Điều 1                         | sim=0.5967
  → [3] Doc_B | Phần I > Điều 3                          | sim=0.5803
  ⏱ Latency: 2304.5 ms

======================================================================
  4.2 | ĐÁNH GIÁ CHẤT LƯỢNG TRUY XUẤT
======================================================================

──────────────────────────────────────────────────────────────────────
  4.2.1 | Bảng tổng hợp kết quả truy xuất
──────────────────────────────────────────────────────────────────────
  ───────────────────────────────────────────────────────────────────────────────────────────────
   STT | Truy vấn                                           | Kỳ vọng                | Hit@K  | Kết quả 
  ───────────────────────────────────────────────────────────────────────────────────────────────
     1 | Mức phạt khi chậm tiến độ triển khai là bao nhiêu? | Phần III > Điều 3      | Top-1  |    ✅    
     2 | Nghĩa vụ bảo mật và bảo vệ dữ liệu cá nhân         | Phần II > Điều 7       | Top-2  |    ✅    
     3 | Giá trị hợp đồng và lịch thanh toán theo đợt       | Phần II > Điều 4       | Top-1  |    ✅    
     4 | Thời hạn hiệu lực và các giai đoạn triển khai      | Phần I > Điều 3        | Top-2  |    ✅    
     5 | Tiêu chí nghiệm thu phần mềm ERP                   | Phần II > Điều 5       | Top-3  |    ✅    
     6 | Trường hợp nào được miễn trách nhiệm hợp đồng?     | Phần II > Điều 8       | Top-1  |    ✅    
     7 | Danh sách nhân sự chủ chốt thực hiện dự án         | Phụ lục > Phụ lục A    | Top-1  |    ✅    
     8 | Nơi giải quyết tranh chấp hợp đồng                 | Phần III > Điều 5      | Top-1  |    ✅    
     9 | Đền bù thiệt hại khi vi phạm thông tin mật         | Phần III > Điều 3      | Top-1  |    ✅    
    10 | Cam kết mức độ dịch vụ SLA và uptime               | Phần II > Điều 7       | Top-1  |    ✅    
  ───────────────────────────────────────────────────────────────────────────────────────────────

  📊 Metrics tổng hợp:
     Hit Rate (Top-3):  10/10 = 100.0%
     Hit Rate (Top-1):  7/10 = 70.0%
     MRR (Mean Reciprocal Rank): 0.8333

──────────────────────────────────────────────────────────────────────
  4.2.2 | Phân tích rủi ro
──────────────────────────────────────────────────────────────────────

  ✅ Tất cả 10 truy vấn đều hit trong Top-3!

──────────────────────────────────────────────────────────────────────
  4.3.2 | Thống kê Query Latency (phân tách)
──────────────────────────────────────────────────────────────────────
  Số truy vấn:           10

  ┌─ TỔNG CỘNG ─────────────────────────────┐
  │  Min:         2157.5 ms               │
  │  Max:         2263.3 ms               │
  │  Trung bình:  2227.0 ms               │
  ├─ OLLAMA EMBEDDING ──────────────────────┤
  │  Min:         2150.9 ms               │
  │  Max:         2262.5 ms               │
  │  Trung bình:  2225.7 ms               │
  ├─ CHROMADB HNSW SEARCH ───────────────────┤
  │  Min:            0.0 ms               │
  │  Max:            6.6 ms               │
  │  Trung bình:     1.2 ms               │
  └──────────────────────────────────────────┘

  📊 Phân bổ: Ollama chiếm 99.9% | ChromaDB chiếm 0.1%

======================================================================
  TỔNG HỢP BENCHMARK — CHƯƠNG 4
======================================================================

  ┌───────────────────────────────────────────────────────────────┐
  │              BẢNG TỔNG HỢP HIỆU NĂNG HỆ THỐNG              │
  ├─────────────────────────────────────┬─────────────────────────┤
  │  CẤU HÌNH                          │                         │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  File test                          │  v1.pdf ↔ v2.pdf        │
  │  Embedding model                    │  qwen3-embedding:0.6b    │
  │  Chunks (A + B)                     │  25 + 26 = 51               │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  THỜI GIAN                          │                         │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  Ingestion + Chunking               │       3.390s             │
  │  Cold-start (nạp model)             │        3.16s             │
  │  Embedding + ChromaDB store         │        3.50s             │
  │  Avg query latency                  │      2227.0 ms           │
  │    ├ Ollama embedding                │      2225.7 ms           │
  │    └ ChromaDB HNSW search            │         1.2 ms           │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  CHẤT LƯỢNG TRUY XUẤT              │                         │
  ├─────────────────────────────────────┼─────────────────────────┤
  │  Hit Rate (Top-1)                   │  7/10 =  70.0%           │
  │  Hit Rate (Top-3)                   │  10/10 = 100.0%           │
  │  MRR                                │      0.8333              │
  └─────────────────────────────────────┴─────────────────────────┘

  Cấu hình ngưỡng (tham khảo):
    UNCHANGED_THRESHOLD  = 0.95
    MODIFIED_THRESHOLD   = 0.75
    TEXT_UNCHANGED_RATIO = 0.998

======================================================================
  HOÀN TẤT
======================================================================

  ⏱  Tổng thời gian test: 47.68s
  📄 Kết quả lưu tại:     test_results.md