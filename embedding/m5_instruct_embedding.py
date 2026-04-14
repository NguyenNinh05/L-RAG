import time
import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'

def evaluate_speed_and_memory(model):
    print("\n" + "="*60)
    print(f"1. ĐÁNH GIÁ TỐC ĐỘ VÀ TÀI NGUYÊN")
    print("="*60)
    
    device = model.device
    print(f"Device đang chạy: {device}")
    
    # Tạo mố dữ liệu giả lập là 100 chunks dài tương đương 1 đoạn văn (khoảng vài chục tokens)
    dummy_text = "Đây là một đoạn văn bản mẫu quy định về quyền bảo mật dữ liệu khách hàng trong hợp đồng bảo mật thông tin (NDA) của dự án phần mềm. " * 5
    texts = [dummy_text] * 100 
    
    # Warm-up (Khởi động GPU để đo chính xác hơn)
    _ = model.encode(texts[:2])
    
    start_time = time.time()
    # Batch size 16 là phổ biến, bạn có thể chỉnh để xem max throughput
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = len(texts) / total_time
    
    print(f"\n- Tốc độ (Throughput): {throughput:.2f} đoạn (chunks) / giây (Batch size = 16)")
    print(f"- Thời gian trung bình 1 đoạn latency: {(total_time/len(texts))*1000:.2f} ms")
    
    if device.type == "cuda":
        max_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"- VRAM tiêu thụ tối đa: {max_vram:.2f} MB")


def evaluate_retrieval_quality(model):
    print("\n" + "="*60)
    print("2. ĐÁNH GIÁ CHẤT LƯỢNG TIẾNG VIỆT (RETRIEVAL METRICS)")
    print("="*60)
    
    # Tập tài liệu (Corpus) giả lập
    corpus = [
        "Cơ sở dữ liệu người dùng phải được mã hóa trước khi lưu vào server.",                                       # 0
        "Trong trường hợp vi phạm, bên vi phạm phải chịu phạt 8% giá trị lỗi.",                                       # 1
        "Hợp đồng có hiệu lực kể từ ngày ký và kết thúc khi nghiệm thu mốc số 3.",                                    # 2
        "Bên A có quyền đơn phương chấm dứt hợp đồng nếu bên B trễ hạn bàn giao 30 ngày.",                           # 3
        "Mã nguồn phần mềm và quyền sở hữu trí tuệ hoàn toàn thuộc về Bên A sau khi thanh toán đủ."                   # 4
    ]
    
    # Các câu hỏi và ID tài liệu tương ứng đúng (Ground Truth)
    queries = {
        "Khi nào thì hợp đồng bắt đầu chạy?": 2,
        "Nếu đối tác gởi sản phẩm muộn 1 tháng thì tôi có được hủy hợp đồng không?": 3,
        "Vấn đề bản quyền sau khi hoàn thành dự án được giải quyết như thế nào?": 4,
        "Chỉ số phần trăm bồi thường là bao nhiêu khi làm sai nguyên tắc?": 1,
        "Thông tin khách hàng có phải mã hóa không?": 0
    }
    
    # LƯU Ý QUAN TRỌNG: Multilingual-E5-Instruct yêu cầu Query phải có Prefix instruction!
    instruction = "Instruct: Given a legal query in Vietnamese, retrieve the relevant contract provision.\nQuery: "
    
    print("Tiến hành Embed Corpus...")
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    correct_at_1 = 0
    mrr_score = 0
    
    for q, expected_id in queries.items():
        # Thêm tiền tố vào query
        full_query = instruction + q
        q_emb = model.encode(full_query, convert_to_tensor=True)
        
        # Tìm top 3
        hits = util.semantic_search(q_emb, corpus_embeddings, top_k=3)[0]
        
        rank = 0
        for i, hit in enumerate(hits):
            if hit['corpus_id'] == expected_id:
                rank = i + 1
                break
                
        if rank == 1:
            correct_at_1 += 1
        if rank > 0:
            mrr_score += 1.0 / rank # Tính MRR
            
        print(f"\nCâu hỏi: '{q}'")
        print(f" -> Văn bản đáp án lọt vào Top: {rank if rank > 0 else 'Ngoài Top 3'}")
        print(f" -> Tương đồng với đáp án đúng: {hits[rank-1]['score']:.4f}" if rank > 0 else "")

    print("\n--- KẾT QUẢ TỔNG QUAN ---")
    print(f"1. Độ chính xác tuyệt đối (Recall@1): {(correct_at_1 / len(queries)) * 100:.2f}%")
    print(f"2. Điểm MRR (Mean Reciprocal Rank) : {mrr_score / len(queries):.4f} (Max là 1.0)")
    print("-------------------------")


if __name__ == "__main__":
    print(f"Đang tải model: {MODEL_NAME}...\n")
    model = SentenceTransformer(MODEL_NAME)
    
    evaluate_speed_and_memory(model)
    evaluate_retrieval_quality(model)
