### 1. Cải thiện Thẩm mỹ (Visual Aesthetics)

- **Hiệu ứng Glassmorphism** : Sử dụng `backdrop-filter: blur()` cho thanh Sidebar và Topbar để tạo chiều sâu. Thay vì các khối màu đặc, chúng ta có thể dùng các dải màu Gradient chuyển tiếp nhẹ nhàng giữa các sắc độ xám-xanh (`Slate` & `Zinc`).
- **Card Design** : Các thẻ "Feature Card" ở màn hình chờ nên có hiệu ứng `hover` (nổi lên hoặc phát sáng nhẹ ở viền) để tăng tính tương tác.
- **Cải thiện Bảng** : Bảng so sánh hiện tại trong Markdown có thể trông hơi "thô". Chúng ta có thể dùng CSS để custom lại bảng Markdown: bo góc, kẻ dòng mảnh, và bôi màu nền khác nhau cho cột V1 và V2 để dễ phân biệt.

### 2. Cải thiện Trải nghiệm (UX) & Chức năng

- **Chế độ Xem Song Song (Split-View)** : Thay vì một "Source Panel" nhỏ ở bên phải, hãy cho phép người dùng chuyển đổi sang chế độ **Side-by-side** . Khi click vào một trích dẫn, màn hình sẽ chia đôi: Bên trái là văn bản gốc, bên phải là văn bản mới, cả hai đều cuộn đến đúng vị trí thay đổi (Scroll Sync).
- **Skeleton Loading** : Thay vì chỉ hiện một chấm "Processing", hãy sử dụng hiệu ứng Skeleton (các khối xám nhấp nháy theo hình dáng báo cáo) để người dùng có cảm giác hệ thống đang "viết" báo cáo trong thời gian thực.
- **Interactive Stats** : Các thẻ thông số (Sửa đổi, Thêm mới, Xóa) ở phần tổng quan nên đóng vai trò là bộ lọc nhanh. Ví dụ: Click vào số "5" ở ô "Sửa đổi" sẽ tự động ẩn các phần khác và chỉ hiện 5 điều khoản bị sửa.

### 3. Micro-interactions (Hiệu ứng nhỏ nhưng tinh tế)

- **Highlight chuyên sâu** : Khi di chuột qua một dòng trong bảng so sánh của AI, đoạn văn bản tương ứng ở panel nguồn bên phải sẽ tự động "nháy" nhẹ hoặc đổi màu viền để người dùng biết chính xác dữ liệu lấy từ đâu.
- **Trạng thái Upload** : Thêm hiệu ứng sóng (wave animation) hoặc progress bar chạy xung quanh slot upload khi file đang được xử lý.

### 4. Tùy chọn Giao diện (Theming)

- **Light Mode** : Thêm một nút chuyển đổi sang giao diện sáng (Light Mode) với tone màu trắng-xám sạch sẽ, vì nhiều luật sư/nhân viên văn phòng vẫn ưu tiên làm việc với nền sáng trong thời gian dài.
