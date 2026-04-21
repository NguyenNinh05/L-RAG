from __future__ import annotations

import unittest

from embedding.embedder import _sanitize_metadata_for_chroma


class EmbeddingMetadataSanitizerTests(unittest.TestCase):
    def test_drops_line_pages_when_all_values_are_none(self) -> None:
        metadata = {
            "line_pages": [None, None, None, None, None],
            "line_indices": [0, 1, 2],
            "doc_label": "doc_A",
        }

        sanitized = _sanitize_metadata_for_chroma(metadata)

        self.assertNotIn("line_pages", sanitized)
        self.assertEqual(sanitized["line_indices"], [0, 1, 2])
        self.assertEqual(sanitized["doc_label"], "doc_A")

    def test_normalizes_mixed_primitive_lists_to_string_list(self) -> None:
        metadata = {
            "mixed": [1, "2", True],
        }

        sanitized = _sanitize_metadata_for_chroma(metadata)

        self.assertEqual(sanitized["mixed"], ["1", "2", "True"])


if __name__ == "__main__":
    unittest.main()
