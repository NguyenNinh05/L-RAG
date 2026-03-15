from pathlib import Path
import re

from ingestion.normalizer import normalize_table_row

try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
except ImportError:
    DocxDocument = None
    qn = None


def _check_file(file_path: str) -> Path:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    return p


def _table_to_rows(table) -> list[str]:
    """Chuyển bảng DOCX thành list các dòng text 'cột1 | cột2 | ...'.
    Dedup theo từng cell liền kề để xử lý merged cells trong Word.
    """
    rows = []
    for row in table.rows:
        cells = []
        prev = None
        for cell in row.cells:
            val = cell.text.strip()
            if val != prev:
                cells.append(val)
                prev = val
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _strip_markdown(text: str) -> str:
    """Xóa ký tự Markdown khỏi text để chunker regex không bị trượt.
    Xử lý theo thứ tự dài → ngắn để tránh lazy match sai.
    Dùng re.DOTALL để xử lý an toàn trường hợp markdown lồng nhau multiline.
    VD: **A** và **B** → A và B  (không bị: A** và **B)
    """
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text, flags=re.DOTALL)  # ***bold italic*** trước
    text = re.sub(r'\*\*(.*?)\*\*',     r'\1', text, flags=re.DOTALL)  # **bold**
    text = re.sub(r'\*(.*?)\*',         r'\1', text, flags=re.DOTALL)  # *italic*
    text = re.sub(r'__(.*?)__',         r'\1', text, flags=re.DOTALL)  # __bold__
    text = re.sub(r'_(.*?)_',           r'\1', text, flags=re.DOTALL)  # _italic_
    text = re.sub(r'`([^`]+)`',         r'\1', text)                   # `code` (không cần DOTALL)
    return text.strip()


def _detect_style_pdf(line: str) -> str:
    """Nhận diện style đơn giản cho từng dòng PDF dựa trên pattern pháp lý VN."""
    clean_line = re.sub(r'^[\*\_\#\s]+', '', line)

    if re.match(r'^(CHƯƠNG|Chương|PH[AẦ]N|Phần)\s+[IVXLC\d]+', clean_line):
        return "Heading 1"
    if re.match(r'^(Đi[eề]u|ĐI[EỀ]U)\s+\d+', clean_line):
        return "Heading 2"
    if re.match(r'^\d+\.\d+[\. ]', clean_line):
        return "Heading 3"

    return "Normal"


def load_docx(file_path: str) -> list[dict]:
    """
    Đọc DOCX và trả về list dict, giữ đúng thứ tự xuất hiện trong tài liệu.
    Cả đoạn văn (paragraphs) lẫn bảng (tables) đều được đọc theo thứ tự XML.
    """
    if DocxDocument is None:
        raise ImportError("pip install python-docx")
    _check_file(file_path)
    doc = DocxDocument(file_path)
    result = []

    for child in doc.element.body:
        tag = child.tag
        if tag == qn("w:p"):
            from docx.text.paragraph import Paragraph
            para = Paragraph(child, doc)
            text = para.text.strip()
            if text:
                style = para.style.name if para.style else "Normal"
                result.append({"text": text, "style": style, "page": None})

        elif tag == qn("w:tbl"):
            from docx.table import Table
            table = Table(child, doc)
            rows_text = _table_to_rows(table)
            for row_text in rows_text:
                result.append({"text": row_text, "style": "Table", "page": None})

    return result


def load_pdf(file_path: str) -> list[dict]:
    """
    Đọc PDF bằng pymupdf4llm -> Markdown, sau đó parse thành list[dict].

    pymupdf4llm giải quyết 3 lỗi của pdfplumber:
      1. Giữ đúng thứ tự đọc: Text và Table xen kẽ đúng vị trí trên trang
      2. Tự động nối dòng bị gãy vật lý (un-break lines)
      3. Không dedup sai - giữ nguyên tất cả dòng của bảng
    Output Markdown: bảng dạng | col1 | col2 | LLM/embedding đọc tốt hơn.
    """
    try:
        import pymupdf4llm
    except ImportError:
        raise ImportError("pip install pymupdf4llm")

    _check_file(file_path)

    # page_chunks=True  → list[dict] mỗi trang, lấy được page number
    # ignore_images=True → tăng tốc, không cần ảnh cho RAG text
    # hdr_info mặc định  → pymupdf4llm tự detect heading → sinh ra # ## ###
    try:
        pages_data: list[dict] = pymupdf4llm.to_markdown(
            str(file_path),
            page_chunks=True,
            ignore_images=True,
        )
        if not pages_data or not isinstance(pages_data, list):
            raise ValueError(f"Invalid PDF structure or empty data returning from {file_path}")
    except Exception as e:
        print(f"[Loader Error] PDF parsing failed for {file_path}: {e}")
        raise RuntimeError(f"Cannot process PDF: {e}") from e
    # print(pages_data)
    paragraphs: list[dict] = []

    for page_idx, page_data in enumerate(pages_data):
        meta = page_data.get("metadata", {})
        page_num = meta.get("page") or (page_idx + 1)
        md_text: str = page_data["text"]
        blocks = md_text.split("\n\n")

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")

            # ── Bảng Markdown: mỗi dòng bắt đầu bằng "|" ──
            if lines[0].startswith("|"):
                for line in lines:
                    line = line.strip()
                    # Bỏ dòng phân cách |---|---|
                    if re.match(r'^\|[\s\-:|]+\|$', line):
                        continue
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    cells = [c for c in cells if c]
                    if not cells:
                        continue
                    row_text = normalize_table_row(" | ".join(cells))
                    if row_text:
                        paragraphs.append({
                            "text": row_text,
                            "style": "Table",
                            "page": page_num,
                        })

            # ── Heading Markdown: bắt đầu bằng # ──
            elif lines[0].startswith("#"):
                text = _strip_markdown(re.sub(r'^#+\s*', '', lines[0]))
                if text:
                    style = _detect_style_pdf(text)
                    paragraphs.append({"text": text, "style": style, "page": page_num})
                for line in lines[1:]:
                    line = _strip_markdown(line)
                    if line:
                        paragraphs.append({"text": line, "style": _detect_style_pdf(line), "page": page_num})

            # ── Block text thường: xử lý từng line và strip markdown ──
            else:
                for line in lines:
                    line = _strip_markdown(line)  # <-- strip ** __ * trước khi detect
                    if not line:
                        continue
                    style = _detect_style_pdf(line)
                    paragraphs.append({"text": line, "style": style, "page": page_num})

    return paragraphs



def load_document(file_path: str) -> list[dict]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".docx":
        return load_docx(file_path)
    elif suffix == ".pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {suffix}. Dùng .docx hoặc .pdf")
