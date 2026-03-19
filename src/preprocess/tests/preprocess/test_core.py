from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from src.preprocess import common
from src.preprocess.dataset_specs import assign_split_records


class PreprocessCoreTests(unittest.TestCase):
    def tearDown(self) -> None:
        common._TOKENIZER_BY_PATH.clear()
        common.TOKENIZER_JSON_PATH = Path("data/tokenizers/korean_bbpe_v1/tokenizer.json")

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

    def test_rows_to_table_uses_staging_schema_when_token_count_is_null(self) -> None:
        table = common.rows_to_table(
            [
                {
                    "source": "src",
                    "data_usage": "PT",
                    "split": "train",
                    "content": "본문",
                    "messages": None,
                    "token_count": None,
                }
            ]
        )
        self.assertEqual(table.schema, common.STAGING_SCHEMA)
        self.assertIsNone(table.to_pylist()[0]["token_count"])

    def test_assign_split_records_is_stable_and_uses_one_percent_val(self) -> None:
        candidates = [
            (f"doc-{index}", f"split-{index}", Path(f"f{index}.json"))
            for index in range(101)
        ]
        first = assign_split_records(source="novel24", candidates=candidates)
        second = assign_split_records(source="novel24", candidates=list(reversed(candidates)))
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
        self.assertEqual(len(first_train), 100)
        self.assertEqual(len(first_val), 1)
        self.assertSetEqual(first_train, second_train)
        self.assertSetEqual(first_val, second_val)
        self.assertTrue(first_train.isdisjoint(first_val))

    def test_compute_token_count_uses_content_only_even_when_messages_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer_path = Path(tmp_dir) / "tokenizer.json"
            self._write_test_tokenizer(tokenizer_path)

            token_count = common.compute_token_count(
                content="안녕 세상",
                messages=[{"role": "assistant", "content": "이 값은 무시"}],
                tokenizer_json_path=tokenizer_path,
            )

        self.assertEqual(token_count, 2)

    def test_validate_row_uses_content_based_token_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer_path = Path(tmp_dir) / "tokenizer.json"
            self._write_test_tokenizer(tokenizer_path)
            common.TOKENIZER_JSON_PATH = tokenizer_path

            row = {
                "source": "src",
                "data_usage": "SFT",
                "split": "train",
                "content": "안녕 세상",
                "messages": [{"role": "assistant", "content": "다른 문자열"}],
                "token_count": 2,
            }
            common.validate_row(row)

    def _write_test_tokenizer(self, path: Path) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer = Tokenizer(
            WordLevel(
                vocab={
                    "[UNK]": 0,
                    "안녕": 1,
                    "세상": 2,
                    "질문": 3,
                    "답변": 4,
                },
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(path))


if __name__ == "__main__":
    unittest.main()
