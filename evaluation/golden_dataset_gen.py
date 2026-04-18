"""
evaluation/golden_dataset_gen.py
==================================
Generate golden dataset cho evaluation bằng cách tạo synthetic V2 từ V1.

Workflow:
    1. Load một tài liệu gốc (V1)
    2. Tự động tạo V2 bằng các thay đổi có kiểm soát:
       - Thay đổi số liệu (numerical changes)
       - Thêm/xoá khoản (addition/deletion)
       - Đổi thuật ngữ (terminology changes)
    3. Lưu cặp (V1, V2) kèm ground-truth ACU list vào data/golden/

Usage:
    python evaluation/golden_dataset_gen.py --input data/raw/v1.docx --n-changes 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_golden_pair(
    input_file: str,
    n_numerical: int = 3,
    n_additions: int = 1,
    n_deletions: int = 1,
    output_dir: str = "./data/golden",
) -> dict:
    """
    Stub: Tạo golden pair từ file đầu vào.

    TODO: Implement full synthetic change generation.
    """
    logger.info("[GoldenGen] TODO: Chưa implement. Input: %s", input_file)
    return {"status": "stub", "input": input_file}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden dataset pairs")
    parser.add_argument("--input", required=True, help="File V1 đầu vào (PDF/DOCX)")
    parser.add_argument("--n-numerical", type=int, default=3)
    parser.add_argument("--n-additions", type=int, default=1)
    parser.add_argument("--n-deletions", type=int, default=1)
    parser.add_argument("--output-dir", default="./data/golden")
    args = parser.parse_args()

    result = generate_golden_pair(
        input_file=args.input,
        n_numerical=args.n_numerical,
        n_additions=args.n_additions,
        n_deletions=args.n_deletions,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
