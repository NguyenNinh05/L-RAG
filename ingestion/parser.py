"""
ingestion/parser.py
===================
Module 1: LegalDocumentParser

Chịu trách nhiệm chuyển đổi file PDF/DOCX thành Legal DOM tree (LegalDocument).

Luồng xử lý:
┌─────────────────────────────────────────────────────┐
│  Input: PDF hoặc DOCX file path                     │
│                                                     │
│  1. Gọi docling để parse → DoclingDocument          │
│  2. Tính avg confidence của text                    │
│     ├─ Đủ cao → _build_dom_from_docling()           │
│     └─ Quá thấp → Fallback: marker-pdf OCR          │
│                   → _build_dom_from_markdown()      │
│  3. Trả về LegalDocument (Pydantic)                 │
└─────────────────────────────────────────────────────┘

Quy tắc bất biến:
  - Bảng LUÔN được trích xuất thành TableData (JSON), không bao giờ flatten.
  - Confidence check được thực hiện trên từng trang.
  - Module này KHÔNG ghi ra disk, chỉ parse và trả về model.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from ingestion.models import (
    ArticleNode,
    ClauseNode,
    DocumentSection,
    LegalDocument,
    ParseEngine,
    ParseQualityMetrics,
    PointNode,
    TableCell,
    TableData,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns — nhận diện cấu trúc văn bản pháp lý VN
# ---------------------------------------------------------------------------

# Điều: "Điều 1.", "Điều 15.", "Điều 1:" — có hoặc không có tiêu đề theo sau
_RE_ARTICLE = re.compile(
    r"^(?P<prefix>Điều)\s+(?P<number>\d+|[IVXLCDM]+)\s*[.:]?\s*(?P<title>.*)$",
    re.IGNORECASE | re.UNICODE,
)

# Khoản: "1.", "2.", "3." — số đứng đầu dòng, SAU đó là text
# Phân biệt với số thứ tự trong bảng bằng context
_RE_CLAUSE = re.compile(
    r"^(?P<number>\d{1,2})\.\s+(?P<content>.+)$",
    re.UNICODE,
)

# Điểm: "a)", "b)", "c)" — có thể có khoảng trắng đầu dòng
_RE_POINT = re.compile(
    r"^\s*(?P<label>[a-zđ])\)\s+(?P<content>.+)$",
    re.UNICODE,
)

# Chương / Phần / Mục:
# "CHƯƠNG I", "Chương II.", "PHẦN 1", "Mục 1"
_RE_SECTION = re.compile(
    r"^(?P<section_type>Chương|CHƯƠNG|Phần|PHẦN|Mục|MỤC|Tiểu mục)\s+"
    r"(?P<number>[IVXLCDM\d]+)\.?\s*(?P<title>.*)$",
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Confidence Checker
# ---------------------------------------------------------------------------

class _ConfidenceResult:
    """Kết quả đánh giá chất lượng text từ docling."""

    def __init__(
        self,
        avg_confidence: float,
        low_conf_pages: list[int],
        total_pages: int,
    ) -> None:
        self.avg_confidence = avg_confidence
        self.low_conf_pages = low_conf_pages
        self.total_pages = total_pages


# ---------------------------------------------------------------------------
# Parser chính
# ---------------------------------------------------------------------------


class LegalDocumentParser:
    """
    Parser văn bản pháp lý với docling làm engine chính
    và marker-pdf làm OCR fallback.

    Usage:
        parser = LegalDocumentParser(confidence_threshold=0.75)
        doc: LegalDocument = parser.parse("/path/to/contract.pdf")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        low_conf_page_ratio: float = 0.30,
        marker_timeout: int = 300,
    ) -> None:
        """
        Args:
            confidence_threshold: Nếu avg confidence < threshold này, trigger OCR.
            low_conf_page_ratio:  Nếu tỷ lệ trang low-conf > ratio này, trigger OCR.
            marker_timeout:       Timeout (giây) khi gọi marker-pdf.
        """
        self.confidence_threshold = confidence_threshold
        self.low_conf_page_ratio = low_conf_page_ratio
        self.marker_timeout = marker_timeout

        # Lazy import để không bắt buộc cài đặt nếu chỉ dùng một engine
        self._docling_converter: Any = None
        self._marker_available: bool | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str | Path) -> LegalDocument:
        """
        Parse một file PDF/DOCX thành LegalDocument.

        Args:
            file_path: Đường dẫn tới file cần parse.

        Returns:
            LegalDocument — cấu trúc Legal DOM đầy đủ.

        Raises:
            FileNotFoundError: Nếu file không tồn tại.
            ValueError:        Nếu định dạng file không được hỗ trợ.
        """
        path = Path(file_path).resolve()
        self._validate_file(path)

        start_ts = time.perf_counter()
        logger.info(f"[Parser] Bắt đầu parse: {path.name}")

        # --- Thử docling trước ---
        try:
            doc, metrics = self._parse_with_docling(path)
        except Exception as exc:
            logger.warning(
                f"[Parser] docling thất bại ({exc}), thử marker-pdf fallback..."
            )
            doc, metrics = self._parse_with_marker(path, ocr_triggered=True)
            metrics.warnings.append(f"docling failed: {exc!s}")
        else:
            # Kiểm tra confidence — có cần OCR không?
            if self._should_trigger_ocr(metrics):
                logger.warning(
                    f"[Parser] Confidence thấp ({metrics.avg_confidence:.2f}), "
                    f"chuyển sang OCR (marker-pdf)..."
                )
                doc, metrics = self._parse_with_marker(path, ocr_triggered=True)

        metrics.parse_duration_seconds = time.perf_counter() - start_ts
        doc.quality_metrics = metrics

        logger.info(
            f"[Parser] Hoàn thành '{path.name}' | engine={metrics.engine_used.value} "
            f"| {len(doc.iter_all_articles())} điều | {metrics.parse_duration_seconds:.1f}s"
        )
        return doc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File không tồn tại: {path}")
        if path.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            raise ValueError(
                f"Định dạng không được hỗ trợ: '{path.suffix}'. "
                f"Chỉ hỗ trợ: .pdf, .docx, .doc"
            )

    # ------------------------------------------------------------------
    # OCR trigger logic
    # ------------------------------------------------------------------

    def _should_trigger_ocr(self, metrics: ParseQualityMetrics) -> bool:
        """Quyết định có cần fallback OCR không dựa trên metrics."""
        if metrics.avg_confidence is None:
            return False

        low_conf_ratio = (
            len(metrics.low_conf_pages) / metrics.total_pages
            if metrics.total_pages > 0
            else 0.0
        )

        return (
            metrics.avg_confidence < self.confidence_threshold
            or low_conf_ratio > self.low_conf_page_ratio
        )

    # ------------------------------------------------------------------
    # Engine 1: docling
    # ------------------------------------------------------------------

    def _get_docling_converter(self) -> Any:
        """Lazy-load DocumentConverter từ docling."""
        if self._docling_converter is None:
            try:
                from docling.document_converter import DocumentConverter  # type: ignore[import]
                from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore[import]
                from docling.datamodel.base_models import InputFormat  # type: ignore[import]
                from docling.document_converter import PdfFormatOption  # type: ignore[import]

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False          # docling xử lý text layer trước
                pipeline_options.do_table_structure = True  # BẮT BUỘC để trích xuất bảng

                self._docling_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )
                logger.debug("[Parser] docling DocumentConverter đã khởi tạo.")
            except ImportError as e:
                raise RuntimeError(
                    "docling chưa được cài đặt. Chạy: pip install docling"
                ) from e

        return self._docling_converter

    def _parse_with_docling(
        self, path: Path
    ) -> tuple[LegalDocument, ParseQualityMetrics]:
        """
        Parse file bằng docling và build Legal DOM.

        Returns:
            (LegalDocument, ParseQualityMetrics)
        """
        converter = self._get_docling_converter()

        logger.debug(f"[docling] Đang convert: {path}")
        result = converter.convert(str(path))
        docling_doc = result.document

        # --- Tính confidence ---
        metrics = self._compute_docling_confidence(docling_doc)
        metrics.engine_used = ParseEngine.DOCLING

        # --- Build Legal DOM ---
        doc = self._build_dom_from_docling(docling_doc, path)
        return doc, metrics

    def _compute_docling_confidence(
        self, docling_doc: Any
    ) -> ParseQualityMetrics:
        """
        Tính toán confidence score từ docling document.

        docling cung cấp confidence qua các text element provenance.
        Nếu confidence không có sẵn, trả về None (không trigger OCR).
        """
        confidences: list[float] = []
        low_conf_pages: set[int] = set()
        total_pages: int = 0

        try:
            # docling >= 2.x: texts có .prov với .confidence
            for text_item in docling_doc.texts:
                for prov in getattr(text_item, "prov", []):
                    page_no = getattr(prov, "page_no", None)
                    conf = getattr(prov, "confidence", None)

                    if page_no is not None:
                        total_pages = max(total_pages, page_no)

                    if conf is not None:
                        confidences.append(float(conf))
                        if float(conf) < 0.5 and page_no is not None:
                            low_conf_pages.add(int(page_no))
        except Exception as exc:
            logger.debug(f"[docling] Không lấy được confidence: {exc}")
            confidences = []

        avg_conf = sum(confidences) / len(confidences) if confidences else None

        return ParseQualityMetrics(
            engine_used=ParseEngine.DOCLING,
            avg_confidence=avg_conf,
            low_conf_pages=sorted(low_conf_pages),
            total_pages=total_pages if total_pages > 0 else self._count_pages(None),
            ocr_triggered=False,
        )

    def _build_dom_from_docling(
        self, docling_doc: Any, path: Path
    ) -> LegalDocument:
        """
        Duyệt DoclingDocument và xây dựng Legal DOM tree.

        Docling trả về document với các phần tử:
          - texts (SectionHeaderItem, TextItem, ListItem...)
          - tables (TableItem với GridCell)

        Chiến lược:
          1. Export toàn bộ document thành Markdown (để parse cấu trúc pháp lý).
          2. Trích xuất tables RIÊNG vào TableData (không bị flatten trong export).
          3. Xây dựng Legal DOM từ Markdown lines.
          4. Map tables vào đúng node.
        """
        logger.debug("[docling] Đang build Legal DOM...")

        # Bước 1: Export markdown (giữ hierarchy headers)
        try:
            md_content: str = docling_doc.export_to_markdown()
        except Exception:
            # Fallback: ghép text từ các element
            md_content = "\n".join(
                getattr(item, "text", "") for item in getattr(docling_doc, "texts", [])
            )

        # Bước 2: Trích xuất tables từ docling (KHÔNG flatten)
        tables_by_page: dict[int, list[TableData]] = {}
        try:
            tables_by_page = self._extract_tables_from_docling(docling_doc)
        except Exception as exc:
            logger.warning(f"[docling] Lỗi trích xuất bảng: {exc}")

        # Bước 3: Build DOM từ markdown text
        doc = self._build_dom_from_lines(
            lines=md_content.splitlines(),
            path=path,
            tables_by_page=tables_by_page,
        )

        # Bước 4: Điền metadata tài liệu từ docling nếu có
        try:
            meta = docling_doc.metadata  # type: ignore[attr-defined]
            if hasattr(meta, "title") and meta.title:
                doc.doc_title = doc.doc_title or meta.title
        except Exception:
            pass

        return doc

    def _extract_tables_from_docling(
        self, docling_doc: Any
    ) -> dict[int, list[TableData]]:
        """
        Trích xuất tất cả bảng từ DoclingDocument thành TableData (JSON).

        Returns:
            Dict mapping: page_number → list[TableData]
        """
        result: dict[int, list[TableData]] = {}

        for table_item in getattr(docling_doc, "tables", []):
            # Lấy trang của bảng
            page_no: int = 0
            for prov in getattr(table_item, "prov", []):
                if hasattr(prov, "page_no"):
                    page_no = int(prov.page_no)
                    break

            table_data = self._docling_table_to_model(table_item, page_no)
            result.setdefault(page_no, []).append(table_data)

        return result

    def _docling_table_to_model(
        self, table_item: Any, page_no: int
    ) -> TableData:
        """
        Convert một docling TableItem → TableData.

        docling TableItem có:
          - table_item.data: TableData với grid_cells list
          - grid_cells[i].text, .start_row_offset_idx, .start_col_offset_idx,
            .row_span, .col_span, .column_header, .row_header
        """
        headers: list[str] = []
        rows_dict: dict[int, dict[int, str]] = {}
        cells: list[TableCell] = []
        num_rows = 0
        num_cols = 0

        try:
            grid = getattr(table_item.data, "grid", None) or []

            # grid là list[list[GridCell]]
            for row_idx, row_cells in enumerate(grid):
                for col_idx, cell in enumerate(row_cells):
                    text = str(getattr(cell, "text", "") or "").strip()
                    is_header = bool(
                        getattr(cell, "column_header", False)
                        or getattr(cell, "row_header", False)
                    )
                    row_span = int(getattr(cell, "row_span", 1) or 1)
                    col_span = int(getattr(cell, "col_span", 1) or 1)

                    cells.append(
                        TableCell(
                            row=row_idx,
                            col=col_idx,
                            row_span=row_span,
                            col_span=col_span,
                            content=text,
                            is_header=is_header,
                        )
                    )

                    if is_header and row_idx == 0:
                        headers.append(text)

                    rows_dict.setdefault(row_idx, {})[col_idx] = text
                    num_rows = max(num_rows, row_idx + 1)
                    num_cols = max(num_cols, col_idx + 1)

        except Exception as exc:
            logger.debug(f"[docling] Lỗi đọc grid cell: {exc}")

        # Chuyển rows_dict thành list[list[str]]
        rows: list[list[Any]] = []
        for r in range(num_rows):
            row_data = [rows_dict.get(r, {}).get(c, "") for c in range(num_cols)]
            rows.append(row_data)

        # Bỏ header row ra khỏi rows data nếu headers đã được tách
        data_rows = rows[1:] if headers and rows else rows

        return TableData(
            headers=headers,
            rows=data_rows,
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
            source_page=page_no if page_no > 0 else None,
        )

    # ------------------------------------------------------------------
    # Engine 2: marker-pdf (OCR fallback)
    # ------------------------------------------------------------------

    def _parse_with_marker(
        self, path: Path, ocr_triggered: bool = False
    ) -> tuple[LegalDocument, ParseQualityMetrics]:
        """
        Parse file bằng marker-pdf (OCR cho scan PDF).

        marker-pdf trả về Markdown → parse tương tự _build_dom_from_lines().
        """
        marker_md = self._run_marker(path)

        metrics = ParseQualityMetrics(
            engine_used=ParseEngine.MARKER_PDF,
            avg_confidence=None,            # marker không cung cấp confidence
            low_conf_pages=[],
            total_pages=self._count_pages_from_md(marker_md),
            ocr_triggered=ocr_triggered,
        )

        doc = self._build_dom_from_lines(
            lines=marker_md.splitlines(),
            path=path,
            tables_by_page={},              # marker parse bảng thành MD, sẽ handle riêng
        )

        # Parse markdown tables từ marker output
        doc = self._inject_markdown_tables(doc, marker_md)

        return doc, metrics

    def _run_marker(self, path: Path) -> str:
        """
        Gọi marker-pdf để OCR và trả về Markdown string.

        Ưu tiên dùng Python API, fallback sang subprocess CLI.
        """
        # Thử Python API của marker
        try:
            return self._run_marker_python_api(path)
        except ImportError:
            logger.debug("[marker] Python API không có, thử CLI subprocess...")

        # Fallback: subprocess CLI
        return self._run_marker_cli(path)

    def _run_marker_python_api(self, path: Path) -> str:
        """Gọi marker qua Python API (marker-pdf >= 0.2.x)."""
        try:
            from marker.convert import convert_single_pdf  # type: ignore[import]
            from marker.models import load_all_models        # type: ignore[import]
        except ImportError:
            raise ImportError("marker-pdf Python API không khả dụng")

        logger.debug(f"[marker] Đang OCR (Python API): {path}")
        models = load_all_models()
        full_text, _images, _metadata = convert_single_pdf(
            fname=str(path),
            model_lst=models,
            max_pages=None,
            langs=["vi", "en"],
        )
        return full_text

    def _run_marker_cli(self, path: Path) -> str:
        """
        Gọi marker qua CLI subprocess.
        Lưu kết quả vào thư mục temp và đọc lại.
        """
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                "marker_single",
                str(path),
                "--output_dir", tmp_dir,
                "--langs", "vi,en",
            ]
            logger.debug(f"[marker] CLI: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.marker_timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"marker CLI thất bại (code={result.returncode}): {result.stderr[:500]}"
                )

            # Tìm file .md trong output dir
            md_files = list(Path(tmp_dir).glob("**/*.md"))
            if not md_files:
                raise RuntimeError("marker CLI không tạo ra file .md nào.")

            return md_files[0].read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # DOM Builder (dùng chung cho cả 2 engine)
    # ------------------------------------------------------------------

    def _build_dom_from_lines(
        self,
        lines: list[str],
        path: Path,
        tables_by_page: dict[int, list[TableData]],
    ) -> LegalDocument:
        """
        Core parser: Duyệt từng line của Markdown/text và build Legal DOM.

        State machine:
            current_section → current_article → current_clause → current_point

        Mỗi khi gặp pattern mới ở cấp cao hơn, flush buffer của cấp thấp hơn.
        """
        doc = LegalDocument(
            doc_id=f"doc_{path.stem}_{int(time.time())}",
            source_path=str(path),
            file_name=path.name,
        )

        current_section: DocumentSection | None = None
        current_article: ArticleNode | None = None
        current_clause: ClauseNode | None = None
        current_point: PointNode | None = None

        preamble_lines: list[str] = []
        preamble_done: bool = False

        def flush_point() -> None:
            nonlocal current_point
            if current_point and current_clause:
                current_clause.points.append(current_point)
                current_point = None

        def flush_clause() -> None:
            nonlocal current_clause
            flush_point()
            if current_clause and current_article:
                current_article.clauses.append(current_clause)
                current_clause = None

        def flush_article() -> None:
            nonlocal current_article
            flush_clause()
            if current_article:
                if current_section:
                    current_section.articles.append(current_article)
                else:
                    doc.orphan_articles.append(current_article)
                current_article = None

        def flush_section() -> None:
            nonlocal current_section
            flush_article()
            if current_section:
                doc.sections.append(current_section)
                current_section = None

        for raw_line in lines:
            line = raw_line.strip()

            # Bỏ qua markdown horizontal rules, empty formatting
            if line in {"---", "***", "___", ""}:
                continue

            # Strip markdown heading markers (#, ##, ###)
            line = re.sub(r"^#{1,6}\s+", "", line)

            # Strip markdown bold/italic markers mà docling thêm vào
            # Ví dụ: "**Điều 5: Tên điều**" → "Điều 5: Tên điều"
            #        "*Số: 001*"            → "Số: 001"
            line = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", line).strip()

            # Strip dấu gạch ngang đặc biệt (separator unicode) mà docling dùng
            if re.match(r"^[─━─\-]{3,}$", line):
                continue

            # ── 1. Chương / Phần / Mục ──────────────────────────
            m = _RE_SECTION.match(line)
            if m:
                flush_section()
                preamble_done = True
                current_section = DocumentSection(
                    section_type=m.group("section_type").capitalize(),
                    number=m.group("number").strip(),
                    title=m.group("title").strip(),
                )
                logger.debug(
                    f"  [DOM] Section: {current_section.section_type} {current_section.number}"
                )
                continue

            # ── 2. Điều ────────────────────────────────────────────
            m = _RE_ARTICLE.match(line)
            if m:
                flush_article()
                preamble_done = True
                current_article = ArticleNode(
                    number=m.group("number").strip(),
                    title=m.group("title").strip(),
                )
                logger.debug(f"  [DOM] Article: Điều {current_article.number}")
                continue

            # ── 3. Khoản ───────────────────────────────────────────
            m = _RE_CLAUSE.match(line)
            if m and current_article:
                flush_clause()
                current_clause = ClauseNode(
                    number=m.group("number"),
                    content=m.group("content").strip(),
                )
                continue

            # ── 4. Điểm ────────────────────────────────────────────
            m = _RE_POINT.match(line)
            if m and current_clause:
                flush_point()
                current_point = PointNode(
                    label=m.group("label"),
                    number=f"{m.group('label')})",
                    content=m.group("content").strip(),
                )
                continue

            # ── 5. Continuation text (nối vào node hiện tại) ───────
            if line:
                if current_point:
                    current_point.content += " " + line
                elif current_clause:
                    current_clause.content += " " + line
                elif current_article:
                    current_article.intro += " " + line
                elif not preamble_done:
                    preamble_lines.append(line)

        # Flush tất cả state còn lại
        flush_section()
        if current_article:
            flush_article()

        doc.preamble = "\n".join(preamble_lines).strip()

        # Gắn tables vào article theo page number (best-effort mapping)
        self._attach_tables_to_nodes(doc, tables_by_page)

        # Trích xuất metadata tài liệu từ preamble
        self._extract_doc_metadata(doc)

        return doc

    def _inject_markdown_tables(
        self, doc: LegalDocument, markdown: str
    ) -> LegalDocument:
        """
        Parse markdown tables (từ marker output) và gán vào orphan nếu chưa được gán.
        Markdown table format:
            | col1 | col2 |
            |------|------|
            | val1 | val2 |
        """
        md_tables = self._parse_markdown_tables(markdown)
        if not md_tables:
            return doc

        # Gán vào article cuối cùng chưa có table (heuristic đơn giản)
        articles = doc.iter_all_articles()
        if not articles:
            return doc

        # Phân phối tables theo thứ tự xuất hiện
        for i, tbl in enumerate(md_tables):
            target_article = articles[min(i, len(articles) - 1)]
            target_article.tables.append(tbl)

        return doc

    def _parse_markdown_tables(self, markdown: str) -> list[TableData]:
        """Parse markdown pipe tables thành list[TableData]."""
        tables: list[TableData] = []
        lines = markdown.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Phát hiện header row: | col1 | col2 |
            if line.startswith("|") and line.endswith("|"):
                # Kiểm tra dòng tiếp theo là separator: |---|---|
                if i + 1 < len(lines) and re.match(
                    r"^\|[-:| ]+\|$", lines[i + 1].strip()
                ):
                    header_row = [
                        c.strip() for c in line.strip("|").split("|")
                    ]
                    rows: list[list[str]] = []
                    j = i + 2
                    while j < len(lines):
                        row_line = lines[j].strip()
                        if row_line.startswith("|") and row_line.endswith("|"):
                            rows.append(
                                [
                                    c.strip()
                                    for c in row_line.strip("|").split("|")
                                ]
                            )
                            j += 1
                        else:
                            break

                    tables.append(
                        TableData(
                            headers=header_row,
                            rows=rows,
                            num_rows=len(rows) + 1,
                            num_cols=len(header_row),
                        )
                    )
                    i = j
                    continue
            i += 1

        return tables

    # ------------------------------------------------------------------
    # Metadata extraction helpers
    # ------------------------------------------------------------------

    def _extract_doc_metadata(self, doc: LegalDocument) -> None:
        """
        Trích xuất metadata (tiêu đề, số hiệu, ngày ký) từ preamble
        bằng các regex pattern phổ biến trong văn bản pháp lý VN.
        """
        text = doc.preamble + "\n" + (doc.doc_title or "")

        # Số hợp đồng: "Số 2024/HĐ-ABC", "Hợp đồng số: 01/2024/HĐMB"
        m_num = re.search(
            r"(?:Số|số)\s*[:\s]+([A-Z0-9/\-\.]+)",
            text, re.IGNORECASE
        )
        if m_num and not doc.doc_number:
            doc.doc_number = m_num.group(1).strip()

        # Ngày ký: "ngày 01 tháng 01 năm 2024"
        m_date = re.search(
            r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
            text, re.IGNORECASE | re.UNICODE,
        )
        if m_date and not doc.signing_date:
            d, mo, y = m_date.groups()
            doc.signing_date = f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    def _attach_tables_to_nodes(
        self,
        doc: LegalDocument,
        tables_by_page: dict[int, list[TableData]],
    ) -> None:
        """
        Gắn TableData vào ArticleNode dựa trên page number.
        Heuristic: Table ở trang X được gắn vào Article gần nhất trước trang X.
        """
        if not tables_by_page:
            return

        # Build danh sách (page_number, article) để mapping
        articles = doc.iter_all_articles()
        article_pages: list[tuple[int, ArticleNode]] = []
        for art in articles:
            if art.page_number is not None:
                article_pages.append((art.page_number, art))

        article_pages.sort(key=lambda x: x[0])

        for page_no, tables in sorted(tables_by_page.items()):
            # Tìm article có page_number <= page_no, gần nhất
            target: ArticleNode | None = None
            for art_page, art in article_pages:
                if art_page <= page_no:
                    target = art
                else:
                    break

            if target is None and articles:
                target = articles[0]

            if target:
                target.tables.extend(tables)
                logger.debug(
                    f"[DOM] Gắn {len(tables)} bảng (trang {page_no}) → "
                    f"Điều {target.number}"
                )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _count_pages(docling_doc: Any) -> int:
        """Đếm số trang từ docling document."""
        try:
            return len(docling_doc.pages) if docling_doc else 0
        except Exception:
            return 0

    @staticmethod
    def _count_pages_from_md(markdown: str) -> int:
        """Ước lượng số trang từ marker markdown (mỗi ~3000 chars ≈ 1 trang)."""
        return max(1, len(markdown) // 3000)
