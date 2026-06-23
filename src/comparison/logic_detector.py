"""
src/comparison/logic_detector.py
================================
Deterministic detector cho dấu hiệu "thay đổi logic / nghĩa pháp lý"
(Category D) giữa hai đoạn văn bản V1 ↔ V2.

Mục đích (Strategy S4):
    LLM thường BỎ SOT các thay đổi tinh tế đảo nghĩa pháp lý
    (đổi "không" ↔ "có thể", thêm/bớt "nếu…", "trừ khi…", đổi phạm vi…).
    Module này quét token-level để phát hiện các "logic token" thay đổi giữa
    V1 và V2, rồi trả về danh sách tín hiệu để INJECT vào Tier-1 prompt
    như một HINT — buộc LLM kiểm tra kỹ các thay đổi nghĩa.

Đây là tín hiệu deterministic, rẻ, KHÔNG dùng để gate/reject — chỉ gợi ý
cho LLM. Vẫn tuân thủ Zero-Hallucination (ACU phải có verbatim evidence +
qua VerificationEngine như thường).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Logic lexicon — các từ "đòn bẩy nghĩa" trong văn bản pháp lý VN
# ---------------------------------------------------------------------------
# Phân nhóm để dễ guard, nhưng detect dùng chung.
LOGIC_LEXICON: list[str] = [
    # Phủ định / cho phép / bắt buộc
    "không được phép", "không được", "không thể", "không",
    "được phép", "được", "phải", "có thể", "có quyền", "được quyền",
    "bắt buộc", "cấm", "cho phép", "không cấm",
    # Điều kiện / ngoại lệ
    "trừ khi", "chỉ khi", "nếu không", "nếu", "ngoại trừ",
    "trong trường hợp", "khi", "điều kiện là",
    # Phạm vi / số lượng
    "chỉ", "duy nhất", "tất cả", "mọi", "toàn bộ", "một phần", "không quá",
]


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace để đếm token ổn định."""
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_logic_signals(v1: str, v2: str, max_signals: int = 6) -> list[str]:
    """
    Phát hiện "logic token" có sự thay đổi số lượng giữa V1 và V2.

    Args:
        v1, v2: raw text của hai phiên bản (thường là 1 điều/khoản).
        max_signals: giới hạn số tín hiệu trả về.

    Returns:
        Danh sách mô tả dạng "thêm 'có thể'", "bỏ 'không'", ...
        Rỗng nếu không phát hiện (hoặc một trong hai text trống).
    """
    if not v1 or not v2:
        return []

    t1 = _normalize(v1)
    t2 = _normalize(v2)
    if not t1 or not t2:
        return []

    signals: list[str] = []
    for tok in LOGIC_LEXICON:
        # token có khoảng trắng (vd "không được") đếm bằng substring;
        # từ đơn cũng đếm bằng substring trên text đã normalize (đã lowercase,
        # collapse ws) — chấp nhận matches trong từ ghép (an toàn cho hint).
        c1 = t1.count(tok)
        c2 = t2.count(tok)
        if c1 == c2:
            continue
        if c2 > c1:
            signals.append(f"thêm '{tok}'")
        else:
            signals.append(f"bỏ '{tok}'")

    # Ưu tiên token dài hơn (specific hơn) lên đầu
    signals.sort(key=lambda s: -len(s))
    return signals[:max_signals]


def build_logic_hint(v1: str, v2: str) -> str:
    """
    Trả về chuỗi hint để nối vào Tier-1 user prompt, hoặc "" nếu không có tín hiệu.
    """
    signals = detect_logic_signals(v1, v2)
    if not signals:
        return ""
    return (
        "\n\n⚠️ Dấu hiệu thay đổi LOGIC/NGHĨA phát hiện (V1→V2): "
        + ", ".join(signals)
        + ". Hãy kiểm tra kỹ các thay đổi nghĩa pháp lý này và tạo ACU riêng "
        "(change_type 'terminology' hoặc 'structural', confidence cao) khi thấy "
        "nghĩa thực sự thay đổi — kèm verbatim evidence."
    )
