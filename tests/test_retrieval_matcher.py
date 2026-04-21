from __future__ import annotations

import unittest

import numpy as np

from ingestion.models import ArticleChunk
from retrieval.matcher import _apply_clause_hint_bonus, _classify_pair, _normalize_for_anchor


def make_chunk(content: str, article_number: str = "Phần I > Điều 1") -> ArticleChunk:
    return ArticleChunk(
        doc_label="doc_A",
        doc_id="doc_A.pdf",
        article_number=article_number,
        title=article_number,
        content=content,
        page=1,
        raw_index=0,
    )


class MatcherNearThresholdBandTests(unittest.TestCase):
    def test_near_above_unchanged_threshold_is_marked_modified(self) -> None:
        chunk_a = make_chunk("Điều khoản giữ nguyên nội dung.")
        chunk_b = make_chunk("Điều khoản giữ nguyên nội dung.")

        pairs = _classify_pair(
            chunk_a,
            chunk_b,
            sim_score=0.955,
            unchanged_threshold=0.95,
            modified_threshold=0.75,
            near_unchanged_band=0.02,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].match_type, "MODIFIED")
        self.assertTrue(pairs[0].near_threshold)

    def test_high_confident_match_above_band_stays_unchanged(self) -> None:
        chunk_a = make_chunk("Điều khoản giữ nguyên nội dung.")
        chunk_b = make_chunk("Điều khoản giữ nguyên nội dung.")

        pairs = _classify_pair(
            chunk_a,
            chunk_b,
            sim_score=0.98,
            unchanged_threshold=0.95,
            modified_threshold=0.75,
            near_unchanged_band=0.02,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].match_type, "UNCHANGED")
        self.assertFalse(pairs[0].near_threshold)

    def test_near_below_unchanged_threshold_is_marked_modified(self) -> None:
        chunk_a = make_chunk("Nội dung điều khoản a.")
        chunk_b = make_chunk("Nội dung điều khoản a.")

        pairs = _classify_pair(
            chunk_a,
            chunk_b,
            sim_score=0.94,
            unchanged_threshold=0.95,
            modified_threshold=0.75,
            near_unchanged_band=0.02,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].match_type, "MODIFIED")
        self.assertTrue(pairs[0].near_threshold)

    def test_far_below_threshold_splits_into_deleted_added(self) -> None:
        chunk_a = make_chunk("Điều khoản cũ hoàn toàn.")
        chunk_b = make_chunk("Điều khoản mới hoàn toàn.")

        pairs = _classify_pair(
            chunk_a,
            chunk_b,
            sim_score=0.5,
            unchanged_threshold=0.95,
            modified_threshold=0.75,
            near_unchanged_band=0.02,
        )

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].match_type, "DELETED")
        self.assertEqual(pairs[1].match_type, "ADDED")


class MatcherClauseHintTests(unittest.TestCase):
    def test_clause_hint_bonus_increases_matching_cells(self) -> None:
        sim = np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float64)
        gap_a = [
            make_chunk("A", article_number="Phần I > Điều 1"),
            make_chunk("B", article_number="Phần I > Điều 2"),
        ]
        gap_b = [
            make_chunk("X", article_number="Phần I > Điều 1"),
            make_chunk("Y", article_number="Phần I > Điều 2"),
        ]

        adjusted = _apply_clause_hint_bonus(sim, gap_a, gap_b, alpha=0.1, min_gap=2)

        self.assertGreater(adjusted[0, 0], sim[0, 0])
        self.assertGreater(adjusted[1, 1], sim[1, 1])

    def test_clause_hint_bonus_not_applied_when_gap_too_small(self) -> None:
        sim = np.array([[0.9]], dtype=np.float64)
        gap_a = [make_chunk("A", article_number="Phần I > Điều 1")]
        gap_b = [make_chunk("B", article_number="Phần I > Điều 1")]

        adjusted = _apply_clause_hint_bonus(sim, gap_a, gap_b, alpha=0.1, min_gap=2)

        self.assertAlmostEqual(float(adjusted[0, 0]), 0.9)


class MatcherAnchorNormalizationTests(unittest.TestCase):
    def test_anchor_normalization_handles_diacritics_and_punctuation(self) -> None:
        left = "Điều 7: Bảo mật và bảo vệ dữ liệu — bắt buộc."
        right = "Dieu 7 Bao mat va bao ve du lieu bat buoc"

        self.assertEqual(_normalize_for_anchor(left), _normalize_for_anchor(right))


if __name__ == "__main__":
    unittest.main()
