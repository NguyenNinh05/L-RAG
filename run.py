import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Chạy hệ thống L-RAG (LegalDiff Pipeline)")
    parser.add_argument("--v1", type=str, required=True, help="Đường dẫn đến file văn bản Pháp lý V1 (gốc)")
    parser.add_argument("--v2", type=str, required=True, help="Đường dẫn đến file văn bản Pháp lý V2 (sửa đổi)")
    parser.add_argument("--skip-phase3", action="store_true", help="Bỏ qua Phase 3 (LLM) để chỉ chạy Phase 1 (Ingestion) và Phase 2 (Alignment)")
    args = parser.parse_args()

    # Thêm src vào PYTHONPATH
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from src.pipeline import LegalDiffPipeline

    try:
        print("Khởi tạo LegalDiff Pipeline...")
        pipeline = LegalDiffPipeline.from_config()

        print(f"\n[BẮT ĐẦU] Xử lý với file:\nV1: {args.v1}\nV2: {args.v2}")
        result = pipeline.run(
            file_v1=args.v1,
            file_v2=args.v2,
            skip_phase3=args.skip_phase3
        )
        
        print("\n[HOÀN THÀNH] Xử lý thành công!")
        
        if not args.skip_phase3 and "reports" in result:
            print(f"Tổng số Report đã được tạo: {len(result['reports'])}")
            
    except Exception as e:
        print(f"\n[LỖI] Đã xảy ra lỗi trong quá trình thực thi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
