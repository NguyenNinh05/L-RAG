from __future__ import annotations

import unittest

from comparison.analyzer import _confidence_score, build_comparison_result
from ingestion.models import ArticleChunk
from retrieval.matcher import ComparedPair


def make_chunk(
    doc_label: str,
    article_number: str,
    content: str,
    page: int = 1,
    raw_index: int = 0,
) -> ArticleChunk:
    return ArticleChunk(
        doc_label=doc_label,
        doc_id=f"{doc_label}.pdf",
        article_number=article_number,
        title=article_number,
        content=content,
        page=page,
        raw_index=raw_index,
    )


class ComparisonAnalyzerUnitTests(unittest.TestCase):
    def test_personnel_clause_detects_replacements_not_added_duplicates(self) -> None:
        text_a = "\n".join(
            [
                "PHỤ LỤC A DANH SÁCH NHÂN SỰ CHỦ CHỐT",
                "| Họ tên | Vai trò | Kinh nghiệm | Chứng chỉ |",
                "| Lê Thị Hương | Lead Developer | 8 năm | Oracle Certified |",
                "| Đỗ Thị Mai | QA Lead | 6 năm | ISTQB Advanced |",
            ]
        )
        text_b = "\n".join(
            [
                "PHỤ LỤC A",
                "DANH SÁCH NHÂN SỰ CHỦ CHỐT",
                "| Họ tên | Vai trò | Kinh nghiệm | Chứng chỉ |",
                "| Hoàng Đức Anh | Lead Developer | 10 năm | Oracle Certified, Java EE |",
                "| Nguyễn Thị Thu | QA Lead | 8 năm | ISTQB Advanced, Selenium |",
            ]
        )
        pair = ComparedPair(
            chunk_a=make_chunk("doc_A", "Phụ lục A", text_a, raw_index=10),
            chunk_b=make_chunk("doc_B", "Phụ lục A", text_b, raw_index=10),
            match_type="MODIFIED",
            similarity=0.88,
        )

        result = build_comparison_result([pair], file_a="v1.pdf", file_b="v2.pdf")
        clause = result.clauses[0]
        replaced_records = [record for record in clause.records if record.change_kind == "REPLACED"]
        formal_records = [record for record in clause.records if record.impact_level == "formal"]

        self.assertEqual(clause.clause_change_kind, "REPLACED")
        self.assertEqual(len(replaced_records), 2)
        self.assertEqual(len(formal_records), 1)
        self.assertTrue(all("Nhân sự" in record.tags for record in replaced_records))
        self.assertTrue(all(record.change_kind != "ADDED" for record in clause.records))

    def test_location_change_is_tagged_as_location_not_time(self) -> None:
        old_text = "Mọi tranh chấp phải được giải quyết trong vòng 30 ngày tại Hà Nội."
        new_text = "Mọi tranh chấp phải được giải quyết trong vòng 30 ngày tại TP. Hồ Chí Minh."
        pair = ComparedPair(
            chunk_a=make_chunk("doc_A", "Phần III > Điều 5", old_text, raw_index=5),
            chunk_b=make_chunk("doc_B", "Phần III > Điều 5", new_text, raw_index=5),
            match_type="MODIFIED",
            similarity=0.91,
        )

        result = build_comparison_result([pair], file_a="v1.pdf", file_b="v2.pdf")
        tags = {tag for change in result.changes for tag in change.tags}

        self.assertIn("Địa điểm", tags)
        self.assertNotIn("Thời hạn", tags)
        self.assertTrue(all(change.semantic_effect == "SUBSTANTIVE" for change in result.changes))
        self.assertTrue(all(change.review_status == "AUTO" for change in result.changes))

    def test_deleted_added_same_clause_collapses_to_replaced(self) -> None:
        deleted_pair = ComparedPair(
            chunk_a=make_chunk("doc_A", "Phần II > Điều 7", "Nội dung cũ", raw_index=7),
            chunk_b=None,
            match_type="DELETED",
            similarity=0.22,
        )
        added_pair = ComparedPair(
            chunk_a=None,
            chunk_b=make_chunk("doc_B", "Phần II > Điều 7", "Nội dung mới", raw_index=7),
            match_type="ADDED",
            similarity=0.0,
        )

        result = build_comparison_result([deleted_pair, added_pair], file_a="v1.pdf", file_b="v2.pdf")

        self.assertEqual(result.stats.replaced, 1)
        self.assertEqual(result.stats.added, 0)
        self.assertEqual(result.stats.deleted, 0)
        self.assertEqual(result.clauses[0].clause_change_kind, "REPLACED")

    def test_boilerplate_only_difference_is_formal(self) -> None:
        old_text = "\n".join(
            [
                "Phụ lục B: Quy trình nghiệm thu",
                "Bước 1: Kiểm tra nội bộ",
                "Hợp đồng này được lập và ký kết tại Hà Nội vào ngày tháng năm 2024.",
            ]
        )
        new_text = "\n".join(
            [
                "Phụ lục B: Quy trình nghiệm thu",
                "Bước 1: Kiểm tra nội bộ",
            ]
        )
        pair = ComparedPair(
            chunk_a=make_chunk("doc_A", "Phụ lục B", old_text, raw_index=11),
            chunk_b=make_chunk("doc_B", "Phụ lục B", new_text, raw_index=11),
            match_type="MODIFIED",
            similarity=0.97,
        )

        result = build_comparison_result([pair], file_a="v1.pdf", file_b="v2.pdf")
        clause = result.clauses[0]

        self.assertTrue(all(record.impact_level == "formal" for record in clause.records))
        self.assertTrue(all(record.semantic_effect == "FORMAL" for record in clause.records))
        self.assertGreaterEqual(result.stats.formal, 1)

    def test_ambiguous_wording_change_marks_review_needed(self) -> None:
        old_text = "Bên A có quyền chấm dứt hợp đồng nếu bên kia vi phạm nghiêm trọng."
        new_text = "Bên A có quyền kết thúc hợp đồng nếu bên kia vi phạm nghiêm trọng."
        pair = ComparedPair(
            chunk_a=make_chunk("doc_A", "Phần II > Điều 9", old_text, raw_index=9),
            chunk_b=make_chunk("doc_B", "Phần II > Điều 9", new_text, raw_index=9),
            match_type="MODIFIED",
            similarity=0.93,
        )

        result = build_comparison_result([pair], file_a="v1.pdf", file_b="v2.pdf")
        change = result.changes[0]

        self.assertEqual(change.impact_level, "substantive")
        self.assertEqual(change.semantic_effect, "SUBSTANTIVE")
        self.assertEqual(change.review_status, "REVIEW_NEEDED")

    def test_confidence_bands_are_stable(self) -> None:
        _, high_band = _confidence_score(1.0, 0.95, 0.9, 0.9)
        _, low_band = _confidence_score(0.4, 0.3, 0.3, 0.3)
        _, suspect_band = _confidence_score(0.1, 0.1, 0.1, 0.1)

        self.assertEqual(high_band, "HIGH")
        self.assertEqual(low_band, "LOW")
        self.assertEqual(suspect_band, "SUSPECT")

    def test_sla_support_window_changes_are_detected_as_critical(self) -> None:
        old_text = "Bảo trì năm 1 | Hỗ trợ 8/5, SLA 4 giờ | 300.000.000"
        new_text = "Bảo trì năm 1 | Hỗ trợ 24/7, SLA 2 giờ | 500.000.000"
        pair = ComparedPair(
            chunk_a=make_chunk("doc_A", "Phần II > Điều 4", old_text, raw_index=10),
            chunk_b=make_chunk("doc_B", "Phần II > Điều 4", new_text, raw_index=10),
            match_type="MODIFIED",
            similarity=0.90,
        )

        result = build_comparison_result([pair], file_a="v1.pdf", file_b="v2.pdf")
        clause = result.clauses[0]

        critical_records = [
            record for record in clause.records if "CRITICAL_VALUE_DELTA" in record.meta_tags
        ]
        self.assertTrue(critical_records)
        self.assertTrue(any("Số liệu" in record.tags for record in critical_records))
        self.assertTrue(any("Thời hạn" in record.tags for record in critical_records))
        self.assertTrue(all(record.review_status == "REVIEW_NEEDED" for record in critical_records))

    def test_uptime_delta_generates_substantive_record(self) -> None:
        old_text = "Uptime cam kết: 99.5%/tháng"
        new_text = "Uptime cam kết: 99.9%/tháng"
        pair = ComparedPair(
            chunk_a=make_chunk("doc_A", "Phần II > Điều 7", old_text, raw_index=13),
            chunk_b=make_chunk("doc_B", "Phần II > Điều 7", new_text, raw_index=13),
            match_type="MODIFIED",
            similarity=0.92,
        )

        result = build_comparison_result([pair], file_a="v1.pdf", file_b="v2.pdf")
        tags = {tag for change in result.changes for tag in change.tags}

        self.assertIn("Số liệu", tags)
        self.assertTrue(any("CRITICAL_VALUE_DELTA" in change.meta_tags for change in result.changes))
