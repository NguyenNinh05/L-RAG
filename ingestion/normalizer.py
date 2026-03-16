import re
import unicodedata

_NOISE_PATTERNS = [
    # Số trang
    r"^trang\s+\d+\s*(\/\s*\d+)?$",
    r"^\d+$",
    # Đường kẻ
    r"^-{3,}$",
    r"^_{3,}$",
    r"^\.{3,}$",
    # Nhãn bảo mật
    r"^bảo\s*mật$",
    r"^confidential$",
    # Dòng ngày ký
    r"^(tp\.|thành\s*phố|hà\s*nội|hồ\s*chí\s*minh).{0,30}ngày\s+\d+",
    r"^ngày\s+\d+\s+tháng\s+\d+\s+năm\s+\d{4}$",
    # Ký hiệu hợp đồng đứng một mình
    r"^số:\s*[\w\/\-]+$",
    r"^số\s+hợp\s*đồng",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)


def normalize_table_row(row_text: str) -> str:
    """
    Chuẩn hóa một dòng bảng dạng 'col1 | col2 | col3'.
    - Strip từng cell
    - Loại cell rỗng
    - Join lại bằng ' | ' chuẩn
    Giúp tránh False Positive khi cùng bảng nhưng PDF của 2 file
    cho ra whitespace/format khác nhau.
    """
    cells = [c.strip() for c in row_text.split("|")]
    cells = [c for c in cells if c]
    return " | ".join(cells)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r'[\u201c\u201d\u201e]', '"', text)
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u2013\u2014]', '-', text)
    text = re.sub(r'\u2026', '...', text)
    return text


def is_noise(text: str) -> bool:
    return bool(_NOISE_RE.match(text.strip()))
