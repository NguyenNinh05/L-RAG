"""
evaluation/ragas_eval.py
=========================
RAGAS-based faithfulness scoring cho Phase 3 output.

Đánh giá mức độ faithful của các ACU được tạo ra bởi LLM
so với nội dung gốc (raw_text_v1, raw_text_v2).

Metrics:
    - Faithfulness:  ACU có bám sát văn bản gốc không?
    - Precision:     Tỷ lệ ACU hợp lệ / tổng ACU được tạo
    - Recall:        Tỷ lệ thay đổi thực sự được phát hiện

Usage:
    python evaluation/ragas_eval.py --reports data/reports/latest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_faithfulness(reports_path: str) -> dict:
    """
    Stub: Tính RAGAS faithfulness score từ report JSON.

    TODO: Integrate với RAGAS library.
    """
    logger.info("[RAGAS] TODO: Chưa implement. Reports: %s", reports_path)
    return {"status": "stub", "faithfulness": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for LegalDiff reports")
    parser.add_argument("--reports", required=True, help="Path đến report JSON")
    args = parser.parse_args()

    result = evaluate_faithfulness(args.reports)
    print(json.dumps(result, indent=2, ensure_ascii=False))
