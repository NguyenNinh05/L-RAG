import re
from dataclasses import dataclass
from typing import Final
from typing import Optional
from ingestion.models import ArticleChunk
from ingestion.normalizer import normalize_text, is_noise

@dataclass(frozen=True)
class ChunkingConfig:
    MAX_CHARS: Final[int] = 1500
    MIN_CHARS: Final[int] = 100
    TABLE_DETECTION_THRESHOLD: Final[float] = 0.5

CONFIG = ChunkingConfig()
MAX_CHUNK_CHARS = CONFIG.MAX_CHARS
MIN_CHUNK_CHARS = CONFIG.MIN_CHARS


# Phân cấp cao nhất: Phần, Phụ lục
_TOP_LEVEL_RE = re.compile(
    r"^(ph[aầ]n|PH[AẦ]N|phụ\s+lục|PHỤ\s+LỤC)\s*([\dIVXivx\w]*)\s*[:\-–\.]?\s*(.*)$",
    re.IGNORECASE
)

# Phân cấp trung bình: Chương, Mục, Tiểu mục, Căn cứ
_MID_LEVEL_PATTERNS = [
    r"^(ch[ươ]ng|CH[ƯƠ]NG|m[uụ]c|M[UỤ]C|ti[ể]u\s+m[uụ]c|TI[EỂ]U\s+M[UỤ]C)\s+([IVXLC]+|\d+[\w]*)\s*[:\-–]?\s*(.*)$",
    r"^(c[ă]n\s+c[ứ]|CĂN\s+CỨ)\s+.*$",
]
_MID_LEVEL_RE = [re.compile(p, re.IGNORECASE) for p in _MID_LEVEL_PATTERNS]

# Phân cấp chi tiết: Điều, Mẫu số
_ARTICLE_PATTERNS = [
    r"^(đi[eề]u\s+\d+[\w]*)\s*[:\-–\.]\s*(.*)$",
    r"^(ĐI[EỀ]U\s+\d+[\w]*)\s+(.+)$",
    r"^(ĐI[EỀ]U\s+\d+[\w]*)$",
    r"^(m[ẫ]u\s+s[ố]\s+[\d\w\/\-]+)\s*[:\-–]?\s*(.*)$",
]
_ARTICLE_RE = [re.compile(p, re.IGNORECASE) for p in _ARTICLE_PATTERNS]

# Tách nhỏ Khoản, Điểm (dùng khi Điều quá dài)
_SUB_CLAUSE_RE = re.compile(
    r"^\s*(?:[a-zđ]\d+[\).]|[a-zđ][\).]|\([ivxIVX]+\)|[ivxIVX][\).]|\d+(?:\.\d+)+|\d+[\).]|[•●\-—])\s+.{5,}",
    re.IGNORECASE
)

# Nhận diện phần ký tên / xác nhận (thường ở cuối văn bản)
_SIGNATURE_RE = re.compile(
    r"^(đ[ạa]i\s+di[ệê]n\s+bên\s+[ab]|đ[ạa]i\s+di[ệê]n\s+các\s+bên|người\s+đại\s+diện|phần\s+ký\s+tên|ký\s+tên|đóng\s+dấu|xác\s+nhận\s+của|thay\s+mặt\s+công\s+ty|giám\s+đốc|tổng\s+giám\s+đốc|người\s+lập\s+biểu)",
    re.IGNORECASE
)

def detect_top_level(text: str) -> Optional[str]:
    """Nhận diện Phần hoặc Phụ lục."""
    m = _TOP_LEVEL_RE.match(text.strip())
    if m:
        label = m.group(1).strip().capitalize()
        num = m.group(2).strip().upper()
        return f"{label} {num}"
    return None


def detect_mid_level(text: str) -> Optional[str]:
    """Nhận diện Chương, Mục, Tiểu mục, Căn cứ."""
    for pattern in _MID_LEVEL_RE:
        m = pattern.match(text.strip())
        if m:
            return text.strip()
    return None


def detect_signature(text: str) -> Optional[str]:
    """Nhận diện phần ký tên, xác nhận."""
    if _SIGNATURE_RE.match(text.strip()):
        return "Phần ký tên & Xác nhận"
    return None


def detect_article_header(text: str) -> tuple[Optional[str], Optional[str]]:
    """Nhận diện Điều hoặc Mẫu số."""
    for pattern in _ARTICLE_RE:
        m = pattern.match(text.strip())
        if m:
            article_number = m.group(1).strip()
            try:
                title = m.group(2).strip() or None
            except IndexError:
                title = None
            return article_number, title
    return None, None


def _is_table_chunk(lines: list[str]) -> bool:
    """Kiểm tra xem chunk có phải bảng Markdown không (đa số dòng bắt đầu bằng '|')."""
    if not lines:
        return False
    table_lines = sum(1 for l in lines if l.strip().startswith("|"))
    return table_lines / len(lines) >= 0.5


def _resolve_line_pages(chunk: ArticleChunk, line_count: int) -> list[int | None]:
    """Map page theo từng dòng; fallback về chunk.page nếu không có metadata chi tiết."""
    line_pages = chunk.metadata.get("line_pages") if isinstance(chunk.metadata, dict) else None
    if isinstance(line_pages, list) and len(line_pages) == line_count:
        return [p if isinstance(p, int) else None for p in line_pages]
    return [chunk.page for _ in range(line_count)]


def _line_offsets(lines: list[str]) -> list[int]:
    """Vị trí ký tự bắt đầu của từng dòng trong chunk gốc."""
    offsets: list[int] = []
    cursor = 0
    for idx, line in enumerate(lines):
        offsets.append(cursor)
        cursor += len(line)
        if idx < len(lines) - 1:
            cursor += 1  # newline
    return offsets


def _is_strong_noise_chunk(content: str) -> bool:
    """Nhận diện chunk cực ngắn có xác suất cao là noise để cho phép drop an toàn."""
    normalized = " ".join(content.split()).strip().lower()
    if not normalized:
        return True

    if re.fullmatch(r"trang\s+\d+\s*(/\s*\d+)?", normalized):
        return True
    if re.fullmatch(r"\d+", normalized):
        return True
    if re.fullmatch(r"[-_.]{3,}", normalized):
        return True
    if re.fullmatch(r"số:\s*[\w/\-]+", normalized):
        return True

    alpha_count = sum(1 for ch in normalized if ch.isalpha())
    if len(normalized) < 30 and alpha_count < 5:
        return True

    return False


def _should_keep_chunk(content: str, article_number: str | None) -> bool:
    """Giữ chunk theo nguyên tắc keep-by-default, chỉ drop khi có bằng chứng noise mạnh."""
    if article_number:
        return True
    if len(content) >= MIN_CHUNK_CHARS:
        return True
    if len(content) >= 30:
        return True
    return not _is_strong_noise_chunk(content)


def split_into_subchunks(
    chunk: ArticleChunk,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[ArticleChunk]:
    if len(chunk.content) <= max_chars:
        return [chunk]

    lines = chunk.content.split("\n")
    line_pages = _resolve_line_pages(chunk, len(lines))
    offsets = _line_offsets(lines)

    groups: list[list[str]] = []
    group_pages: list[list[int | None]] = []
    group_line_indices: list[list[int]] = []
    current_group: list[str] = []
    current_group_pages: list[int | None] = []
    current_group_line_indices: list[int] = []

    if _is_table_chunk(lines):
        # Bảng Markdown: cắt theo từng row (dòng | ... |)
        # Giữ header row đầu tiên ở mọi sub-chunk
        header = lines[0] if lines and lines[0].strip().startswith("|") else ""
        current_chars = 0
        for line_idx, line in enumerate(lines):
            line_page = line_pages[line_idx] if line_idx < len(line_pages) else chunk.page
            if (current_chars + len(line) > max_chars
                    and current_group
                    and line.strip().startswith("|")):
                groups.append(current_group)
                group_pages.append(current_group_pages)
                group_line_indices.append(current_group_line_indices)
                current_group = [header, line] if header else [line]
                # Header lặp lại chỉ để tăng ngữ cảnh, không đại diện vị trí gốc.
                current_group_pages = [line_page]
                current_group_line_indices = [line_idx]
                current_chars = len(header) + len(line)
            else:
                current_group.append(line)
                current_group_pages.append(line_page)
                current_group_line_indices.append(line_idx)
                current_chars += len(line)
        if current_group:
            groups.append(current_group)
            group_pages.append(current_group_pages)
            group_line_indices.append(current_group_line_indices)
    else:
        # Text thường: cắt theo khoản a), b), c)
        for line_idx, line in enumerate(lines):
            line_page = line_pages[line_idx] if line_idx < len(line_pages) else chunk.page
            if _SUB_CLAUSE_RE.match(line) and current_group:
                groups.append(current_group)
                group_pages.append(current_group_pages)
                group_line_indices.append(current_group_line_indices)
                current_group = [line]
                current_group_pages = [line_page]
                current_group_line_indices = [line_idx]
            else:
                current_group.append(line)
                current_group_pages.append(line_page)
                current_group_line_indices.append(line_idx)
        if current_group:
            groups.append(current_group)
            group_pages.append(current_group_pages)
            group_line_indices.append(current_group_line_indices)

    if len(groups) <= 1:
        return [chunk]

    result = []
    prefix = f"[{chunk.article_number} – {chunk.title}]\n" if chunk.title else f"[{chunk.article_number}]\n"
    for i, group in enumerate(groups):
        content = "\n".join(group).strip()
        if not content:
            continue
        if i > 0:
            content = prefix + content

        page_candidates = [p for p in group_pages[i] if isinstance(p, int)]
        sub_page = min(page_candidates) if page_candidates else chunk.page
        sub_page_end = max(page_candidates) if page_candidates else (chunk.page_end or chunk.page)

        line_indices = group_line_indices[i]
        line_start = line_indices[0] + 1 if line_indices else chunk.line_start
        line_end = line_indices[-1] + 1 if line_indices else chunk.line_end
        char_start = offsets[line_indices[0]] if line_indices else (chunk.char_start or 0)
        if line_indices:
            last_idx = line_indices[-1]
            char_end = offsets[last_idx] + len(lines[last_idx])
        else:
            char_end = chunk.char_end if chunk.char_end is not None else len(chunk.content)

        meta = dict(chunk.metadata)
        meta["page"] = sub_page
        meta["page_end"] = sub_page_end
        meta["line_start"] = line_start
        meta["line_end"] = line_end
        meta["char_start"] = char_start
        meta["char_end"] = char_end
        meta["line_pages"] = list(group_pages[i])
        meta["line_indices"] = list(line_indices)
        meta["sub_index"] = i + 1
        result.append(ArticleChunk(
            doc_label=chunk.doc_label,
            doc_id=chunk.doc_id,
            article_number=chunk.article_number,
            title=chunk.title,
            content=content,
            page=sub_page,
            page_end=sub_page_end,
            line_start=line_start,
            line_end=line_end,
            char_start=char_start,
            char_end=char_end,
            raw_index=chunk.raw_index,
            sub_index=i + 1,
            metadata=meta,
        ))
    return result



def structure_document(
    paragraphs: list[dict],
    doc_id: str,
    doc_label: str,
) -> list[ArticleChunk]:
    raw_chunks: list[ArticleChunk] = []
    
    current_top: Optional[str] = None
    current_mid: Optional[str] = None
    current_art_num: Optional[str] = None
    current_title: Optional[str] = None

    current_lines: list[str] = []
    current_pages: list[int | None] = []

    def make_breadcrumb() -> str:
        parts = []
        if current_top: parts.append(current_top)
        if current_mid: parts.append(current_mid)
        if current_art_num: parts.append(current_art_num)
        return " > ".join(parts) if parts else "Thông tin chung"

    def flush():
        nonlocal current_lines, current_pages
        if not current_lines:
            return

        line_count = len(current_lines)
        line_pages = list(current_pages)
        content = "\n".join(current_lines).strip()
        page_candidates = [p for p in current_pages if isinstance(p, int)]
        page_first = page_candidates[0] if page_candidates else None
        page_last = page_candidates[-1] if page_candidates else None
        current_lines = []
        current_pages = []

        if not _should_keep_chunk(content, current_art_num):
            return

        breadcrumb = make_breadcrumb()

        raw_chunks.append(ArticleChunk(
            doc_label=doc_label,
            doc_id=doc_id,
            article_number=breadcrumb,
            title=current_title,
            content=content,
            page=page_first,
            page_end=page_last,
            line_start=1 if line_count else None,
            line_end=line_count if line_count else None,
            char_start=0,
            char_end=len(content),
            raw_index=len(raw_chunks),
            sub_index=0,
            metadata={
                "doc_label": doc_label,
                "doc_id": doc_id,
                "top_level": current_top or "",
                "mid_level": current_mid or "",
                "article_number": current_art_num or "",
                "breadcrumb": breadcrumb,
                "title": current_title or "",
                "page": page_first,
                "page_last": page_last,
                "page_end": page_last,
                "line_start": 1 if line_count else None,
                "line_end": line_count if line_count else None,
                "char_start": 0,
                "char_end": len(content),
                "line_pages": line_pages,
            }
        ))

    for para in paragraphs:
        text = normalize_text(para.get("text", ""))
        page = para.get("page")

        if not text or is_noise(text):
            continue

        # 1. Check Phần / Phụ lục (Cấp 1)
        top = detect_top_level(text)
        if top:
            flush()
            current_top = top
            current_mid = None
            current_art_num = None
            current_title = None
            current_lines.append(text)
            current_pages.append(page)
            continue

        # 2. Check Chương / Mục / Tiểu mục / Căn cứ (Cấp 2)
        mid = detect_mid_level(text)
        if mid:
            flush()
            current_mid = mid
            current_art_num = None
            current_title = None
            current_lines.append(text)
            current_pages.append(page)
            continue

        # 2.5 Check Phần ký tên
        sig = detect_signature(text)
        if sig:
            flush()
            current_top = sig
            current_mid = None
            current_art_num = None
            current_title = None
            current_lines.append(text)
            current_pages.append(page)
            continue

        # 3. Check Điều / Mẫu số (Cấp 3)
        art_num, art_title = detect_article_header(text)
        if art_num:
            flush()
            current_art_num = art_num
            current_title = art_title
            current_lines.append(text)
            current_pages.append(page)
        else:
            if not any([current_top, current_mid, current_art_num]):
                current_top = "Mở đầu"
            current_lines.append(text)
            current_pages.append(page)

    flush()

    final_chunks: list[ArticleChunk] = []
    for chunk in raw_chunks:
        final_chunks.extend(split_into_subchunks(chunk))
    return final_chunks
