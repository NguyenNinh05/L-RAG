from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArticleChunk:
    doc_label: str                  # "doc_A" hoặc "doc_B"
    doc_id: str                     # tên file nguồn
    article_number: Optional[str]   # "Điều 1", "Khoản 2", ...
    title: Optional[str]            # tiêu đề điều khoản
    content: str                    # nội dung
    page: Optional[int] = None      # trang (PDF)
    raw_index: int = 0              # thứ tự trong tài liệu
    sub_index: int = 0              # thứ tự sub-chunk (0 = không tách)
    metadata: dict = field(default_factory=dict)

    def chunk_id(self) -> str:
        raw = (self.article_number or "mo_dau")
        safe = raw.replace(" > ", "_").replace(" ", "_").replace(">", "_")
        base = f"{self.doc_label}__{self.doc_id}__{safe}__idx{self.raw_index}"
        return f"{base}__sub{self.sub_index}"

    def __repr__(self):
        sub = f" [sub{self.sub_index}]" if self.sub_index > 0 else ""
        return (
            f"[{self.doc_label}] {self.article_number or 'N/A'}{sub} – {self.title or ''}\n"
            f"  {self.content[:120].replace(chr(10), ' ')}..."
        )
