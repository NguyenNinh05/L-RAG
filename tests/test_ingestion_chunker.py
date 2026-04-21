from __future__ import annotations

import unittest

from ingestion.chunker import split_into_subchunks, structure_document
from ingestion.models import ArticleChunk


class IngestionChunkerProvenanceTests(unittest.TestCase):
    def test_structure_document_keeps_line_pages_metadata(self) -> None:
        paragraphs = [
            {"text": "Điều 1: Nghĩa vụ bảo mật", "page": 1},
            {
                "text": "a) Bên A phải bảo mật toàn bộ dữ liệu khách hàng và thông tin giao dịch trong toàn bộ vòng đời hợp đồng.",
                "page": 1,
            },
            {
                "text": "b) Bên A phải thông báo sự cố rò rỉ dữ liệu trong vòng 24 giờ kể từ khi phát hiện sự cố.",
                "page": 2,
            },
        ]

        chunks = structure_document(paragraphs, doc_id="v1.pdf", doc_label="doc_A")

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.page, 1)
        self.assertEqual(chunk.page_end, 2)
        self.assertEqual(chunk.metadata.get("line_pages"), [1, 1, 2])
        self.assertEqual(chunk.line_start, 1)
        self.assertEqual(chunk.line_end, 3)

    def test_split_into_subchunks_preserves_page_and_char_spans(self) -> None:
        lines = [
            "Điều 1: Nghĩa vụ bảo mật",
            "a) Dòng số một ở trang một.",
            "b) Dòng số hai ở trang hai.",
            "c) Dòng số ba ở trang hai.",
        ]
        content = "\n".join(lines)

        chunk = ArticleChunk(
            doc_label="doc_A",
            doc_id="v1.pdf",
            article_number="Phần II > Điều 1",
            title="Nghĩa vụ bảo mật",
            content=content,
            page=1,
            page_end=2,
            line_start=1,
            line_end=4,
            char_start=0,
            char_end=len(content),
            raw_index=3,
            metadata={"line_pages": [1, 1, 2, 2]},
        )

        subchunks = split_into_subchunks(chunk, max_chars=80)

        self.assertGreaterEqual(len(subchunks), 2)
        self.assertEqual(subchunks[0].page, 1)
        self.assertEqual(subchunks[0].page_end, 1)

        spans_second_page = [sub for sub in subchunks if sub.page == 2]
        self.assertTrue(spans_second_page)
        for sub in spans_second_page:
            self.assertEqual(sub.page_end, 2)
            self.assertIsNotNone(sub.char_start)
            self.assertIsNotNone(sub.char_end)
            excerpt = content[sub.char_start:sub.char_end]
            self.assertTrue(excerpt.strip())
            self.assertIn(lines[sub.line_start - 1], excerpt)

    def test_structure_document_keeps_short_legal_clause(self) -> None:
        paragraphs = [
            {"text": "Điều 12: Xử lý vi phạm", "page": 1},
            {"text": "Vi phạm bị xử lý theo quy định.", "page": 1},
        ]

        chunks = structure_document(paragraphs, doc_id="v1.pdf", doc_label="doc_A")

        self.assertEqual(len(chunks), 1)
        self.assertIn("Điều 12", chunks[0].article_number)
        self.assertIn("xử lý", chunks[0].content.lower())

    def test_split_into_subchunks_supports_mixed_numbering(self) -> None:
        lines = [
            "Điều 9: Phạm vi áp dụng",
            "b1) Ảnh phông nền trắng theo quy chuẩn.",
            "(i) Hồ sơ kỹ thuật phải được lưu trữ tối thiểu 24 tháng.",
            "1.2.3 Trường hợp đặc biệt do cơ quan có thẩm quyền quyết định.",
        ]
        content = "\n".join(lines)
        chunk = ArticleChunk(
            doc_label="doc_A",
            doc_id="v1.pdf",
            article_number="Phần II > Điều 9",
            title="Phạm vi áp dụng",
            content=content,
            page=1,
            page_end=1,
            line_start=1,
            line_end=4,
            char_start=0,
            char_end=len(content),
            raw_index=5,
            metadata={"line_pages": [1, 1, 1, 1]},
        )

        subchunks = split_into_subchunks(chunk, max_chars=60)

        self.assertGreaterEqual(len(subchunks), 3)
        joined = "\n".join(sub.content for sub in subchunks)
        self.assertIn("b1)", joined)
        self.assertIn("(i)", joined)
        self.assertIn("1.2.3", joined)

    def test_chunk_id_stable_when_raw_index_changes(self) -> None:
        payload = {
            "doc_label": "doc_A",
            "doc_id": "v1.pdf",
            "article_number": "Phần II > Điều 3",
            "title": "Hiệu lực",
            "content": "Điều khoản này có hiệu lực kể từ ngày ký.",
            "page": 1,
            "page_end": 1,
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 44,
        }
        chunk_first = ArticleChunk(raw_index=1, sub_index=0, **payload)
        chunk_second = ArticleChunk(raw_index=99, sub_index=0, **payload)

        self.assertEqual(chunk_first.chunk_id(), chunk_second.chunk_id())
        self.assertNotEqual(
            chunk_first.metadata.get("legacy_chunk_id"),
            chunk_second.metadata.get("legacy_chunk_id"),
        )


if __name__ == "__main__":
    unittest.main()
