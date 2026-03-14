from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from src.preprocess import common
from src.preprocess.dataset_specs import (
    assign_self_split_records,
    clean_annotations_text_for_021,
    chunk_novel24_text,
    split_novel24_files,
)


class PreprocessCoreTests(unittest.TestCase):
    def test_serialize_messages(self) -> None:
        serialized = common.serialize_messages(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]
        )
        self.assertEqual(serialized, "<|system|>s<|user|>u<|assistant|>a")

    def test_progress_bar_uses_stdout(self) -> None:
        bar = common.progress_bar(total=1, desc="test")
        try:
            wrapped = getattr(bar.fp, "_wrapped", None)
            self.assertIs(wrapped, sys.stdout)
        finally:
            bar.close()

    def test_clean_annotations_text_for_021(self) -> None:
        raw = "A. 안녕하세요  \n\n B : 문의 드립니다   \nA. 네 도와드리겠습니다"
        cleaned = clean_annotations_text_for_021(raw)
        self.assertEqual(cleaned, "안녕하세요\n문의 드립니다\n네 도와드리겠습니다")

    def test_split_novel24_files(self) -> None:
        from pathlib import Path

        files = [Path(f"f{i}.txt") for i in range(10)]
        train, val = split_novel24_files(files)
        self.assertEqual(len(train), 9)
        self.assertEqual(len(val), 1)
        self.assertTrue(files[-1] in val)

    def test_chunk_novel24_text_preserves_lines(self) -> None:
        text = "첫 줄\n둘째 줄\n셋째 줄"
        with patch("src.preprocess.common.compute_token_count", return_value=3):
            chunks = chunk_novel24_text(text)
        self.assertEqual(chunks, ["첫 줄\n둘째 줄\n셋째 줄"])

    def test_assign_self_split_records_is_stable_and_exact(self) -> None:
        from pathlib import Path

        candidates = [(f"doc-{index}", Path(f"f{index}.json")) for index in range(11)]
        first = assign_self_split_records(source="국립국어원 구어 말뭉치", candidates=candidates)
        second = assign_self_split_records(source="국립국어원 구어 말뭉치", candidates=list(reversed(candidates)))
        first_train = {
            record_id
            for record_ids in first["train"].values()
            for record_id in record_ids
        }
        first_val = {
            record_id
            for record_ids in first["val"].values()
            for record_id in record_ids
        }
        second_train = {
            record_id
            for record_ids in second["train"].values()
            for record_id in record_ids
        }
        second_val = {
            record_id
            for record_ids in second["val"].values()
            for record_id in record_ids
        }
        self.assertEqual(len(first_train), 9)
        self.assertEqual(len(first_val), 2)
        self.assertSetEqual(first_train, second_train)
        self.assertSetEqual(first_val, second_val)
        self.assertTrue(first_train.isdisjoint(first_val))


if __name__ == "__main__":
    unittest.main()
