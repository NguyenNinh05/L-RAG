# Hướng dẫn chạy Backend LegalDiff

Tài liệu này hướng dẫn cách chạy backend tại `/home/haipd/L-RAG/backend` bằng Conda env `l_rag`. Mục tiêu là chạy được API để test các chức năng cơ bản như health check, auth, upload document, tạo job; chưa cần bật LLM server.

## 1. Backend này gồm những thành phần nào?

Backend LegalDiff không chỉ có một process FastAPI. Khi chạy đầy đủ, nó gồm nhiều thành phần phối hợp với nhau:

| Thành phần | Vai trò | Có cần khi test cơ bản không? |
|---|---|---|
| FastAPI | API server chính, nhận request từ frontend hoặc curl | Có |
| PostgreSQL | Lưu user, document metadata, comparison job, report | Có |
| Redis | Message broker/result backend cho Celery, đồng thời dùng cho progress/WebSocket | Có |
| Alembic | Công cụ tạo/cập nhật schema database | Có, chạy trước khi start API |
| Celery worker | Process xử lý pipeline so sánh tài liệu ở background | Chỉ cần khi muốn job chạy thật |
| LLM server | Local model server cho Phase 3/generative comparison | Chưa cần nếu chỉ test API/auth/upload |

Vì vậy khi test backend cơ bản, tối thiểu cần:

1. PostgreSQL đang chạy.
2. Redis đang chạy.
3. Database đã migration.
4. FastAPI backend đang chạy.

## 2. Kích hoạt môi trường `l_rag`

```bash
cd /home/haipd/L-RAG
conda activate l_rag
```

### Vai trò của lệnh

- `cd /home/haipd/L-RAG`: đưa terminal về thư mục gốc của project.
- `conda activate l_rag`: bật đúng môi trường Python đã cài dependency cho dự án.

### Tại sao phải dùng `l_rag`?

Backend cần các package như `fastapi`, `uvicorn`, `sqlalchemy`, `alembic`, `redis`, `celery`, `python-jose`, `bcrypt`, v.v. Nếu không activate đúng env, terminal có thể dùng Python/package của môi trường khác và sinh lỗi kiểu `ModuleNotFoundError`.

Kiểm tra Python đang dùng:

```bash
which python
python --version
```

Kỳ vọng đường dẫn Python nằm trong env `l_rag`.

## 3. Kiểm tra PostgreSQL

```bash
pg_isready -h localhost -p 5432
```

Nếu PostgreSQL đang chạy, kết quả sẽ giống:

```text
localhost:5432 - accepting connections
```

### Vai trò của PostgreSQL

PostgreSQL là database chính của backend. Backend dùng nó để lưu:

- tài khoản user,
- thông tin file upload,
- comparison job,
- trạng thái job,
- catalog/report sau khi pipeline chạy.

Nếu PostgreSQL chưa chạy hoặc sai cấu hình, các API như register/login/upload/job sẽ lỗi.

### Giải thích lệnh

- `pg_isready`: công cụ kiểm tra PostgreSQL có đang nhận kết nối không.
- `-h localhost`: kiểm tra PostgreSQL trên máy local.
- `-p 5432`: kiểm tra port mặc định của PostgreSQL.

### Lỗi gõ nhầm thường gặp

Sai:

```bash
pg_isready -h -localhost -p 5432
```

Lỗi vì host bị gõ thành `-localhost`.

Sai:

```bash
pg_isready -h localhose -p 5432
```

Lỗi vì gõ nhầm `localhost` thành `localhose`.

Đúng:

```bash
pg_isready -h localhost -p 5432
```

## 4. Chạy Redis

Mở một terminal riêng và chạy:

```bash
conda activate l_rag
redis-server --port 6379 --save "" --appendonly no
```

Giữ terminal này mở trong lúc test backend.

Kiểm tra Redis ở một terminal khác:

```bash
redis-cli -h localhost -p 6379 ping
```

Kết quả đúng:

```text
PONG
```

### Vai trò của Redis

Redis trong backend này có 2 vai trò chính:

1. Là broker/backend cho Celery: FastAPI gửi job vào Redis, Celery worker lấy job ra xử lý.
2. Là kênh pub/sub để backend gửi progress của job qua WebSocket.

Ngay cả khi chưa chạy Celery worker, readiness endpoint vẫn kiểm tra Redis để biết hạ tầng backend đã sẵn sàng chưa.

### Giải thích lệnh Redis

```bash
redis-server --port 6379 --save "" --appendonly no
```

- `redis-server`: start Redis server.
- `--port 6379`: chạy Redis trên port mặc định `6379`.
- `--save ""`: tắt snapshot persistence để test nhẹ hơn, không ghi RDB snapshot ra disk.
- `--appendonly no`: tắt AOF persistence để Redis chạy đơn giản cho môi trường dev/test.

Cấu hình này phù hợp để test local. Nếu chạy production thì cần cấu hình persistence/security đầy đủ hơn.

## 5. Load biến môi trường backend

Mở terminal dùng để chạy backend:

```bash
cd /home/haipd/L-RAG
conda activate l_rag

set -a
source backend/.env
set +a
```

### Vai trò của `.env`

File `backend/.env` chứa cấu hình runtime cho backend, ví dụ:

- `DATABASE_URL`: database URL async cho FastAPI.
- `DATABASE_URL_SYNC`: database URL sync cho Alembic/Celery.
- `REDIS_URL`: địa chỉ Redis.
- `JWT_SECRET_KEY`: key để ký JWT token.
- `LLM_BASE_URL`: địa chỉ LLM server local.
- `STORAGE_ROOT`: thư mục lưu file upload.

### Giải thích lệnh

- `set -a`: tự động export các biến shell được khai báo sau đó.
- `source backend/.env`: đọc biến môi trường từ file `.env` vào shell hiện tại.
- `set +a`: tắt chế độ auto-export.

Nếu không source `.env`, backend có thể dùng default config không đúng với môi trường local hiện tại.

## 6. Chạy database migration

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini upgrade head
```

Nếu không có traceback/error là migration đã chạy xong.

### Vai trò của Alembic migration

Alembic tạo/cập nhật các bảng database cần cho backend. Ví dụ:

- `users`,
- `documents`,
- `comparison_jobs`,
- `comparison_reports`.

Nếu chưa chạy migration, API có thể start được nhưng khi gọi register/login/upload sẽ lỗi vì database chưa có bảng.

### Giải thích lệnh

- `PYTHONPATH=/home/haipd/L-RAG`: thêm project root vào Python import path để Python import được package `backend`.
- `alembic`: CLI migration của SQLAlchemy.
- `-c backend/alembic.ini`: chỉ định file cấu hình Alembic của backend.
- `upgrade head`: nâng database lên migration mới nhất.

### Khi nào cần chạy lại migration?

Chạy lại khi:

- mới clone project,
- database mới tạo,
- có migration mới trong `backend/alembic/versions`,
- lỗi API liên quan thiếu bảng/cột database.

Nếu database đã ở version mới nhất, lệnh này chạy lại vẫn an toàn và thường không làm gì thêm.

## 7. Chạy FastAPI backend

```bash
PYTHONPATH=/home/haipd/L-RAG uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

Sau khi chạy, mở Swagger UI:

```text
http://localhost:8001/docs
```

### Vai trò của FastAPI

FastAPI là API server chính. Nó expose các endpoint như:

- `/api/health`,
- `/api/auth/register`,
- `/api/auth/login`,
- `/api/documents/upload`,
- `/api/jobs`,
- `/api/reports`,
- `/ws/jobs/{job_id}`.

Frontend hoặc công cụ test như `curl`/Swagger sẽ gọi vào FastAPI.

### Giải thích lệnh

- `PYTHONPATH=/home/haipd/L-RAG`: giúp import `backend.main` đúng package.
- `uvicorn`: ASGI server dùng để chạy FastAPI.
- `backend.main:app`: trỏ tới biến `app` trong file `backend/main.py`.
- `--host 0.0.0.0`: cho phép server listen trên tất cả network interface, không chỉ `127.0.0.1`.
- `--port 8001`: chạy backend ở port `8001`.
- `--reload`: tự reload server khi code thay đổi, tiện cho dev.

### Khi nào không dùng `--reload`?

Khi chạy production hoặc benchmark performance, bỏ `--reload` để process ổn định hơn:

```bash
PYTHONPATH=/home/haipd/L-RAG uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

## 8. Kiểm tra backend

Health check:

```bash
curl http://localhost:8001/api/health
```

Readiness check:

```bash
curl http://localhost:8001/api/health/ready
```

Nếu chưa bật LLM, kết quả readiness có thể giống:

```json
{
  "status": "ok",
  "database": "connected",
  "redis": "connected",
  "worker": "available",
  "llm_server": "disconnected"
}
```

### Vai trò của health check

`/api/health` kiểm tra API server có sống không. Endpoint này đơn giản, thường dùng để biết FastAPI đã start thành công.

### Vai trò của readiness check

`/api/health/ready` kiểm tra các dependency quan trọng hơn:

- database có kết nối được không,
- Redis có kết nối được không,
- GPU lock/worker state có đọc được không,
- LLM server có trả lời không.

### Vì sao `llm_server: disconnected` vẫn chấp nhận được?

Trong giai đoạn test API cơ bản, chưa cần LLM. Vì vậy `llm_server: disconnected` là bình thường nếu bạn chưa start local model server. Khi test Phase 3/generative comparison mới cần bật LLM.

## 9. Test auth cơ bản

Đăng ký user:

```bash
curl -X POST http://localhost:8001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "password123",
    "full_name": "Test User"
  }'
```

Login lấy access token:

```bash
TOKEN=$(curl -s -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123"
  }' | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
```

Kiểm tra user hiện tại:

```bash
curl http://localhost:8001/api/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### Vai trò của auth

Các API upload document và tạo job yêu cầu user đã login. Backend dùng JWT access token để xác định request thuộc user nào.

### Giải thích các lệnh

#### Register

- `curl -X POST`: gửi HTTP POST request.
- `/api/auth/register`: endpoint tạo user mới.
- `-H "Content-Type: application/json"`: báo cho backend biết body là JSON.
- `-d '{...}'`: payload đăng ký user.

#### Login

- `/api/auth/login`: endpoint xác thực username/password.
- Backend trả về `access_token` và `refresh_token`.
- Lệnh `python3 -c ...` lấy riêng field `access_token` từ JSON response rồi lưu vào biến shell `TOKEN`.

#### Me

- `/api/auth/me`: endpoint kiểm tra token hiện tại thuộc user nào.
- `Authorization: Bearer $TOKEN`: gửi JWT token cho backend.

Nếu `/api/auth/me` trả thông tin user thì auth flow đã chạy đúng.

## 10. Upload tài liệu

Upload file V1:

```bash
curl -X POST http://localhost:8001/api/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/duong/dan/toi/file_v1.pdf"
```

Upload file V2:

```bash
curl -X POST http://localhost:8001/api/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/duong/dan/toi/file_v2.pdf"
```

List documents:

```bash
curl http://localhost:8001/api/documents \
  -H "Authorization: Bearer $TOKEN"
```

### Vai trò của upload document

Backend cần lưu 2 phiên bản văn bản pháp lý trước khi tạo comparison job:

- V1: bản cũ/bản gốc,
- V2: bản mới/bản sửa đổi.

Khi upload, backend lưu file vào storage local và lưu metadata vào PostgreSQL.

### Giải thích lệnh

- `-X POST`: gửi request upload.
- `/api/documents/upload`: endpoint upload tài liệu.
- `Authorization: Bearer $TOKEN`: bắt buộc vì document thuộc về một user.
- `-F "file=@..."`: gửi file dạng multipart/form-data.

Backend hiện cho phép file `.pdf` và `.docx`.

Sau khi upload, response sẽ có `id`. Cần giữ `id` của V1 và V2 để tạo comparison job.

## 11. Tạo comparison job

```bash
curl -X POST http://localhost:8001/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "document_v1_id": "UUID_FILE_V1",
    "document_v2_id": "UUID_FILE_V2",
    "skip_phase3": true
  }'
```

### Vai trò của comparison job

Comparison job là bản ghi yêu cầu backend so sánh 2 tài liệu. FastAPI tạo job trong PostgreSQL và gửi task sang Celery để xử lý background.

### Giải thích payload

- `document_v1_id`: ID file V1 đã upload.
- `document_v2_id`: ID file V2 đã upload.
- `skip_phase3`: nếu `true`, có thể dùng để bỏ qua phase generative comparison trong một số flow test.

### Lưu ý quan trọng

Nếu chưa chạy Celery worker, job có thể được tạo nhưng pipeline sẽ chưa được xử lý thật. Muốn job chạy thật thì cần chạy thêm worker ở bước tiếp theo.

## 12. Chạy Celery worker khi cần xử lý pipeline

Mở terminal riêng:

```bash
cd /home/haipd/L-RAG
conda activate l_rag

set -a
source backend/.env
set +a

PYTHONPATH=/home/haipd/L-RAG celery -A backend.celery_app worker --concurrency=1 --loglevel=info
```

### Vai trò của Celery worker

FastAPI chỉ nhận request và tạo job. Công việc nặng như parse tài liệu, alignment, comparison sẽ do Celery worker xử lý ở background.

Thiết kế này giúp API không bị treo khi pipeline chạy lâu.

### Giải thích lệnh

- `celery`: CLI chạy worker.
- `-A backend.celery_app`: chỉ định Celery app trong file `backend/celery_app.py`.
- `worker`: chạy process worker để nhận task.
- `--concurrency=1`: chỉ chạy 1 job một lúc, phù hợp vì pipeline dùng GPU/tài nguyên nặng.
- `--loglevel=info`: in log ở mức vừa đủ để debug.

### Khi nào cần LLM server?

LLM server chỉ cần khi chạy Phase 3/generative comparison. Nếu chỉ test health, auth, upload, hoặc tạo job đơn giản thì chưa cần bật LLM.

## 13. Tắt backend

Trong terminal đang chạy FastAPI:

```text
Ctrl+C
```

Trong terminal đang chạy Redis:

```text
Ctrl+C
```

Nếu đang chạy Celery worker:

```text
Ctrl+C
```

Kiểm tra backend đã tắt:

```bash
curl http://localhost:8001/api/health
```

Nếu thấy lỗi kiểu này là backend đã tắt:

```text
Failed to connect to localhost port 8001
Connection refused
```

### Vai trò của bước tắt

Khi test xong nên tắt các process dev để tránh chiếm port `8001`, port `6379`, hoặc giữ tài nguyên GPU/CPU không cần thiết.

## 14. Redis end-to-end

### Redis là gì trong backend này?

Redis là một in-memory data store. Trong dự án này Redis không phải database chính để lưu dữ liệu nghiệp vụ lâu dài. Dữ liệu lâu dài nằm ở PostgreSQL. Redis được dùng cho các phần cần nhanh, tạm thời, hoặc cần truyền tín hiệu giữa các process.

Trong backend LegalDiff, Redis có các vai trò sau:

1. **Celery broker**: FastAPI gửi task vào Redis, Celery worker lấy task từ Redis ra xử lý.
2. **Celery result backend**: Celery có thể ghi trạng thái/kết quả task tạm thời vào Redis.
3. **Pub/Sub progress channel**: worker publish tiến độ job vào Redis, WebSocket layer subscribe để đẩy progress ra frontend.
4. **GPU lock / trạng thái tạm thời**: một số logic có thể dùng Redis để biết worker/GPU đang bận hay rảnh.

### Luồng Redis end-to-end khi tạo comparison job

Khi user gọi API tạo job:

```text
User/Frontend
  -> FastAPI POST /api/jobs
  -> PostgreSQL: tạo row comparison_jobs
  -> Redis: push Celery task vào queue
  -> Celery worker: lấy task từ Redis
  -> Worker chạy pipeline
  -> Worker publish progress vào Redis pub/sub
  -> WebSocket nhận progress và gửi về frontend
  -> PostgreSQL: cập nhật trạng thái job/report cuối cùng
```

Điểm quan trọng: Redis chỉ đóng vai trò trung gian điều phối và truyền tín hiệu. Kết quả chính thống cuối cùng vẫn phải nằm trong PostgreSQL.

### Vì sao backend vẫn cần Redis dù chưa chạy LLM?

Ngay cả khi chưa bật LLM, backend vẫn cần Redis để:

- kiểm tra readiness,
- chuẩn bị queue cho job,
- hỗ trợ Celery worker nếu chạy Phase 1/Phase 2,
- hỗ trợ progress/WebSocket.

Nếu chỉ test `/api/health` đơn giản thì có thể không cần Redis. Nhưng nếu test `/api/health/ready`, job, Celery, WebSocket thì cần Redis.

### Các lệnh Redis hay dùng

#### Start Redis local cho dev

```bash
redis-server --port 6379 --save "" --appendonly no
```

- `--port 6379`: dùng port mặc định.
- `--save ""`: tắt snapshot RDB.
- `--appendonly no`: tắt append-only file.

Cách này phù hợp để test local vì không cần lưu dữ liệu Redis sau khi tắt máy.

#### Kiểm tra Redis còn sống không

```bash
redis-cli -h localhost -p 6379 ping
```

Kết quả đúng:

```text
PONG
```

#### Xem thông tin Redis server

```bash
redis-cli -h localhost -p 6379 info server
redis-cli -h localhost -p 6379 info memory
redis-cli -h localhost -p 6379 info clients
```

Dùng khi muốn biết version Redis, RAM đang dùng, số client đang kết nối.

#### Xem số key trong Redis

```bash
redis-cli -h localhost -p 6379 dbsize
```

Dùng để biết Redis hiện có bao nhiêu key trong database mặc định.

#### Liệt kê key để debug

```bash
redis-cli -h localhost -p 6379 scan 0
```

Không nên dùng `KEYS *` trên Redis production vì có thể block server nếu có quá nhiều key. Với dev local ít key thì có thể dùng:

```bash
redis-cli -h localhost -p 6379 keys '*'
```

#### Đọc một key cụ thể

```bash
redis-cli -h localhost -p 6379 get TEN_KEY
redis-cli -h localhost -p 6379 ttl TEN_KEY
redis-cli -h localhost -p 6379 type TEN_KEY
```

- `get`: đọc value nếu key là string.
- `ttl`: xem key còn sống bao lâu.
- `type`: xem kiểu dữ liệu của key.

#### Xóa một key cụ thể

```bash
redis-cli -h localhost -p 6379 del TEN_KEY
```

Dùng khi cần xóa key debug cụ thể.

#### Xóa toàn bộ Redis database dev

```bash
redis-cli -h localhost -p 6379 flushdb
```

Chỉ dùng ở local/dev. Không dùng tùy tiện ở môi trường có dữ liệu thật.

#### Theo dõi lệnh Redis realtime

```bash
redis-cli -h localhost -p 6379 monitor
```

Lệnh này in mọi command Redis nhận được. Hữu ích để debug Celery/FastAPI có đang đẩy gì vào Redis không. Không nên bật lâu vì rất nhiều output.

#### Kiểm tra pub/sub channel

```bash
redis-cli -h localhost -p 6379 pubsub channels
```

Dùng để xem hiện có channel pub/sub nào. Với job progress, channel thường có dạng liên quan đến job id, ví dụ `job:{job_id}:progress`.

#### Subscribe thử một channel

```bash
redis-cli -h localhost -p 6379 subscribe 'job:JOB_ID:progress'
```

Dùng để kiểm tra worker có publish progress không.

### Khi Redis lỗi thì backend thường biểu hiện thế nào?

- `/api/health/ready` báo `redis: disconnected`.
- Tạo job có thể lỗi hoặc job không được worker nhận.
- WebSocket không nhận progress.
- Celery worker log báo không connect được broker.

### Checklist debug Redis

Chạy lần lượt:

```bash
redis-cli -h localhost -p 6379 ping
redis-cli -h localhost -p 6379 dbsize
redis-cli -h localhost -p 6379 info clients
```

Nếu `ping` không trả `PONG`, cần start Redis trước.

## 15. PostgreSQL end-to-end

### PostgreSQL là gì trong backend này?

PostgreSQL là database chính thống của backend. Nếu Redis là nơi truyền tín hiệu tạm thời, PostgreSQL là nơi lưu trạng thái thật và dữ liệu nghiệp vụ lâu dài.

Backend dùng PostgreSQL để lưu:

- user account,
- password hash,
- document metadata,
- đường dẫn file upload,
- comparison job,
- trạng thái job,
- catalog kết quả alignment,
- report kết quả comparison.

### Luồng PostgreSQL end-to-end

Ví dụ flow `register -> upload -> create job`:

```text
POST /api/auth/register
  -> INSERT INTO users

POST /api/documents/upload
  -> lưu file vào storage local
  -> INSERT INTO documents

POST /api/jobs
  -> kiểm tra document_v1/document_v2 thuộc user
  -> INSERT INTO comparison_jobs
  -> gửi task sang Redis/Celery

Celery worker chạy xong
  -> UPDATE comparison_jobs
  -> INSERT INTO comparison_reports
```

Như vậy, PostgreSQL là nguồn sự thật để biết backend hiện có user nào, file nào, job nào, job đang ở trạng thái nào.

### Async URL và sync URL

Trong `.env` có hai URL database:

```env
DATABASE_URL=postgresql+asyncpg://...
DATABASE_URL_SYNC=postgresql://...
```

Vai trò:

- `DATABASE_URL`: dùng cho FastAPI async code thông qua SQLAlchemy async engine.
- `DATABASE_URL_SYNC`: dùng cho Alembic migration và Celery worker sync code.

Cả hai cùng trỏ tới một database, nhưng dùng driver khác nhau.

### Các bảng chính

| Bảng | Vai trò |
|---|---|
| `users` | Lưu thông tin user và password hash |
| `documents` | Lưu metadata file upload, owner, storage path |
| `comparison_jobs` | Lưu yêu cầu so sánh, trạng thái, progress, catalog |
| `comparison_reports` | Lưu report chi tiết sau khi pipeline chạy |
| `alembic_version` | Lưu database hiện đang ở migration version nào |

### Các lệnh PostgreSQL hay dùng

#### Kiểm tra PostgreSQL sẵn sàng

```bash
pg_isready -h localhost -p 5432
```

Kết quả đúng:

```text
localhost:5432 - accepting connections
```

#### Kết nối vào database bằng psql

Tùy user/password trong `.env`, có thể dùng URL:

```bash
psql "$DATABASE_URL_SYNC"
```

Hoặc truyền từng phần:

```bash
psql -h localhost -p 5432 -U legaldiff -d legaldiff
```

Nếu PostgreSQL yêu cầu password, nhập password tương ứng trong `.env`.

#### Liệt kê database

Trong `psql`:

```sql
\l
```

#### Chuyển database

```sql
\c legaldiff
```

#### Liệt kê bảng

```sql
\dt
```

#### Xem schema của một bảng

```sql
\d users
\d documents
\d comparison_jobs
\d comparison_reports
```

#### Xem migration version hiện tại

```sql
SELECT * FROM alembic_version;
```

Nếu bảng này có version mới nhất thì migration đã được apply.

#### Đếm số row trong bảng

```sql
SELECT count(*) FROM users;
SELECT count(*) FROM documents;
SELECT count(*) FROM comparison_jobs;
SELECT count(*) FROM comparison_reports;
```

#### Xem vài row mới nhất

```sql
SELECT id, username, email, created_at
FROM users
ORDER BY created_at DESC
LIMIT 5;
```

```sql
SELECT id, original_filename, user_id, created_at
FROM documents
ORDER BY created_at DESC
LIMIT 5;
```

```sql
SELECT id, status, current_phase, progress_pct, created_at
FROM comparison_jobs
ORDER BY created_at DESC
LIMIT 5;
```

#### Bật expanded display trong psql

```sql
\x on
```

Dùng khi row có nhiều cột, output sẽ dễ đọc hơn.

#### Thoát psql

```sql
\q
```

### Lệnh kiểm tra kết nối đúng database

Sau khi `source backend/.env`, chạy:

```bash
psql "$DATABASE_URL_SYNC" -c 'SELECT current_database(), current_user;'
```

Dùng để xác nhận shell đang dùng đúng database URL.

### Khi PostgreSQL lỗi thì backend thường biểu hiện thế nào?

- `/api/health/ready` báo `database: disconnected`.
- Register/login trả lỗi 500.
- Upload document lỗi khi insert metadata.
- Alembic migration lỗi connection refused/auth failed.

### Checklist debug PostgreSQL

```bash
pg_isready -h localhost -p 5432
psql "$DATABASE_URL_SYNC" -c 'SELECT 1;'
psql "$DATABASE_URL_SYNC" -c 'SELECT * FROM alembic_version;'
psql "$DATABASE_URL_SYNC" -c '\dt'
```

Nếu `pg_isready` OK nhưng `psql "$DATABASE_URL_SYNC"` fail, thường là sai database name/user/password trong `.env`.

## 16. Database migration và Alembic end-to-end

### Database migration là gì?

Database migration là cách quản lý thay đổi schema database theo phiên bản. Schema là cấu trúc database: bảng, cột, index, foreign key, constraint.

Ví dụ khi code cần thêm cột mới vào `documents`, ta không sửa database thủ công trên từng máy. Thay vào đó tạo một migration file. Khi chạy Alembic, migration đó được apply vào database.

### Vì sao cần migration?

Không có migration, mỗi máy có thể có schema khác nhau:

- máy A có bảng `users`, máy B chưa có,
- máy A có cột `progress_pct`, máy B chưa có,
- code mới chạy trên database cũ sẽ lỗi.

Migration giúp đảm bảo:

- schema database khớp với code,
- thay đổi database có lịch sử rõ ràng,
- có thể reproduce database từ đầu,
- nhiều người cùng team chạy cùng một cấu trúc database.

### Alembic là gì?

Alembic là migration tool phổ biến cho SQLAlchemy. Trong backend này:

- SQLAlchemy model định nghĩa bảng bằng Python class.
- Alembic đọc metadata từ model.
- Alembic tạo hoặc apply migration vào PostgreSQL.
- PostgreSQL lưu version hiện tại trong bảng `alembic_version`.

### Các file Alembic trong project

| File/thư mục | Vai trò |
|---|---|
| `backend/alembic.ini` | Config chính của Alembic |
| `backend/alembic/env.py` | Script khởi tạo môi trường migration, load model metadata và `.env` |
| `backend/alembic/versions/` | Chứa các migration script |
| `backend/alembic/versions/001_initial_migration.py` | Migration đầu tiên tạo bảng chính |

### Luồng migration end-to-end

```text
SQLAlchemy models
  -> Alembic đọc Base.metadata
  -> Migration file trong alembic/versions
  -> alembic upgrade head
  -> PostgreSQL schema được tạo/cập nhật
  -> alembic_version lưu revision đã apply
```

### Lệnh migration quan trọng nhất

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini upgrade head
```

Lệnh này nghĩa là: dùng config `backend/alembic.ini`, apply toàn bộ migration còn thiếu cho tới revision mới nhất.

### Giải thích chi tiết lệnh `upgrade head`

- `PYTHONPATH=/home/haipd/L-RAG`: để `env.py` import được package `backend`.
- `alembic`: chạy Alembic CLI.
- `-c backend/alembic.ini`: dùng đúng config Alembic của backend.
- `upgrade`: apply migration lên database.
- `head`: revision mới nhất trong migration history.

### Các lệnh Alembic thường dùng

#### Xem database đang ở revision nào

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini current
```

Dùng để biết database đã apply migration nào.

#### Xem revision mới nhất trong code

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini heads
```

Nếu `current` khác `heads`, database chưa được upgrade tới bản mới nhất.

#### Xem lịch sử migration

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini history
```

Dùng để xem thứ tự migration từ cũ tới mới.

#### Upgrade lên bản mới nhất

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini upgrade head
```

Đây là lệnh thường dùng nhất khi chạy backend.

#### Downgrade một revision

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini downgrade -1
```

Chỉ dùng ở local/dev khi hiểu rõ migration sẽ rollback gì. Không dùng tùy tiện trên database có dữ liệu quan trọng.

#### Tạo migration mới bằng autogenerate

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini revision --autogenerate -m "describe schema change"
```

Dùng khi đã sửa SQLAlchemy model và muốn Alembic sinh migration tương ứng.

Sau khi autogenerate, phải mở file mới trong `backend/alembic/versions/` để review. Không nên tin autogenerate 100% vì có thể thiếu hoặc hiểu sai một số thay đổi phức tạp.

#### Tạo migration rỗng

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini revision -m "manual migration"
```

Dùng khi cần tự viết SQL hoặc migration logic thủ công.

#### Stamp database là đã ở head

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini stamp head
```

Lệnh này chỉ cập nhật bảng `alembic_version`, không thật sự chạy migration. Chỉ dùng khi chắc chắn schema database đã đúng nhưng thiếu version marker. Không dùng để né lỗi migration.

### Quy trình đúng khi thay đổi schema

1. Sửa SQLAlchemy model trong `backend/models/`.
2. Tạo migration:

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini revision --autogenerate -m "short description"
```

3. Review file migration mới trong `backend/alembic/versions/`.
4. Chạy migration:

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini upgrade head
```

5. Kiểm tra schema bằng `psql`:

```bash
psql "$DATABASE_URL_SYNC" -c '\dt'
psql "$DATABASE_URL_SYNC" -c 'SELECT * FROM alembic_version;'
```

6. Chạy backend và test endpoint liên quan.

### Vì sao không nên sửa database thủ công thay cho migration?

Ví dụ nếu tự chạy:

```sql
ALTER TABLE users ADD COLUMN phone text;
```

trên máy local nhưng không tạo migration, người khác pull code sẽ không có cột đó. Production cũng không có cột đó. Code mới có thể chạy trên máy bạn nhưng lỗi ở nơi khác.

Migration giúp thay đổi schema trở thành một phần của source code.

### Khi migration lỗi thì debug như thế nào?

#### Lỗi connection refused

PostgreSQL chưa chạy hoặc sai host/port.

Kiểm tra:

```bash
pg_isready -h localhost -p 5432
```

#### Lỗi authentication failed

Sai user/password trong `DATABASE_URL_SYNC`.

Kiểm tra:

```bash
psql "$DATABASE_URL_SYNC" -c 'SELECT 1;'
```

#### Lỗi import `backend`

Thiếu `PYTHONPATH` hoặc đang chạy sai thư mục.

Dùng:

```bash
cd /home/haipd/L-RAG
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini current
```

#### Lỗi thiếu bảng/cột khi gọi API

Database chưa upgrade hoặc migration chưa tạo đúng.

Kiểm tra:

```bash
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini current
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini heads
psql "$DATABASE_URL_SYNC" -c '\d comparison_jobs'
```

### Quy tắc an toàn với migration

- Luôn review migration file trước khi chạy.
- Cẩn thận với `drop_table`, `drop_column`, `alter_column(nullable=False)` trên bảng đã có dữ liệu.
- Không dùng `stamp head` để bỏ qua lỗi thật.
- Không downgrade database có dữ liệu quan trọng nếu chưa backup.
- Với local/dev có thể reset dễ hơn, nhưng production cần quy trình backup/review riêng.

## 17. Lệnh chạy nhanh sau khi PostgreSQL đã sẵn sàng

Phần này dùng khi bạn đã hiểu các bước ở trên và chỉ muốn chạy nhanh.

### Terminal 1: Redis

```bash
conda activate l_rag
redis-server --port 6379 --save "" --appendonly no
```

### Terminal 2: FastAPI

```bash
cd /home/haipd/L-RAG
conda activate l_rag
set -a
source backend/.env
set +a
PYTHONPATH=/home/haipd/L-RAG alembic -c backend/alembic.ini upgrade head
PYTHONPATH=/home/haipd/L-RAG uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

### Terminal 3: kiểm tra

```bash
curl http://localhost:8001/api/health
curl http://localhost:8001/api/health/ready
```

## 18. Thứ tự chạy khuyến nghị

Chạy theo đúng thứ tự này để dễ debug:

1. `conda activate l_rag`
2. `pg_isready -h localhost -p 5432`
3. Start Redis.
4. `redis-cli -h localhost -p 6379 ping`
5. `source backend/.env`
6. `alembic upgrade head`
7. Start `uvicorn`.
8. Mở `http://localhost:8001/docs`.
9. Test `/api/health` và `/api/health/ready`.
10. Test auth/register/login.

Nếu lỗi xảy ra, kiểm tra theo thứ tự dependency trước: PostgreSQL → Redis → migration → FastAPI → auth/upload/job.
