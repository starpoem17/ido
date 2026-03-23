from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow.parquet as pq

from src.preprocess import add_token_count, common


class _FakeTokenizer:
    def encode(self, text: str):  # type: ignore[no-untyped-def]
        return SimpleNamespace(ids=text.split())


def _write_input_shard(root: Path, name: str, rows: list[dict[str, object]]) -> Path:
    shard_path = root / name
    common.ensure_parent_dir(shard_path)
    pq.write_table(common.rows_to_table(rows), shard_path)
    return shard_path


def _read_rows(shard_path: Path) -> list[dict[str, object]]:
    return pq.read_table(shard_path).to_pylist()


class AddTokenCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_common_tqdm = common.ENABLE_TQDM
        self._old_common_debug = common.ENABLE_DEBUG_LOG
        common.ENABLE_TQDM = False
        common.ENABLE_DEBUG_LOG = False
        common._TOKENIZER_BY_PATH.clear()

    def tearDown(self) -> None:
        common.ENABLE_TQDM = self._old_common_tqdm
        common.ENABLE_DEBUG_LOG = self._old_common_debug
        common.TOKENIZER_JSON_PATH = Path("data/tokenizers/korean_bbpe_v1/tokenizer.json")
        common._TOKENIZER_BY_PATH.clear()

    def test_run_add_token_count_preserves_shards_and_overwrites_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "near_dedup"
            output_root = tmp_root / "token_count_added"
            tokenizer_path = tmp_root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")

            _write_input_shard(
                input_root,
                "part-000001.parquet",
                [
                    {
                        "source": "alpha",
                        "data_usage": "SFT",
                        "split": "train",
                        "content": "안녕 세상",
                        "messages": [{"role": "assistant", "content": "답변"}],
                        "token_count": None,
                    },
                    {
                        "source": "beta",
                        "data_usage": "PT",
                        "split": "val",
                        "content": "질문 답변 추가",
                        "messages": None,
                        "token_count": 999,
                    },
                ],
            )
            _write_input_shard(
                input_root,
                "part-000002.parquet",
                [
                    {
                        "source": "gamma",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "문장",
                        "messages": None,
                        "token_count": 111,
                    }
                ],
            )

            with patch("src.preprocess.add_token_count.common.get_tokenizer", return_value=_FakeTokenizer()):
                summary = add_token_count.run_add_token_count(
                    add_token_count.AddTokenCountConfig(
                        input_root=input_root,
                        output_root=output_root,
                        tokenizer_json_path=tokenizer_path,
                        num_workers=1,
                        row_batch_rows=2,
                        overwrite_output=False,
                    )
                )

            self.assertEqual(summary.input_shard_count, 2)
            self.assertEqual(summary.output_shard_count, 2)
            self.assertEqual(summary.input_row_count, 3)
            self.assertEqual(summary.output_row_count, 3)
            self.assertEqual(summary.input_null_token_count_row_count, 1)
            self.assertEqual(summary.input_non_null_token_count_row_count, 2)
            self.assertEqual(summary.overwritten_token_count_row_count, 3)
            self.assertTrue((output_root / "part-000001.parquet").exists())
            self.assertTrue((output_root / "part-000002.parquet").exists())
            self.assertTrue((output_root / "_meta" / "add_token_count_manifest.json").exists())
            self.assertTrue((output_root / "_meta" / "add_token_count_summary.json").exists())

            shard1_rows = _read_rows(output_root / "part-000001.parquet")
            shard2_rows = _read_rows(output_root / "part-000002.parquet")
            self.assertEqual([row["source"] for row in shard1_rows], ["alpha", "beta"])
            self.assertEqual([row["source"] for row in shard2_rows], ["gamma"])
            self.assertEqual(shard1_rows[0]["token_count"], 2)
            self.assertEqual(shard1_rows[1]["token_count"], 3)
            self.assertEqual(shard2_rows[0]["token_count"], 1)

            manifest = json.loads(
                (output_root / "_meta" / "add_token_count_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["input_row_count"], 3)
            self.assertEqual(manifest["output_row_count"], 3)
            self.assertEqual(len(manifest["output_shards"]), 2)

    def test_run_add_token_count_fails_when_content_is_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "near_dedup"
            output_root = tmp_root / "token_count_added"
            tokenizer_path = tmp_root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")

            _write_input_shard(
                input_root,
                "part-000001.parquet",
                [
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": None,
                        "messages": [{"role": "assistant", "content": "답변"}],
                        "token_count": None,
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "content must be a non-null string"):
                with patch(
                    "src.preprocess.add_token_count.common.get_tokenizer",
                    return_value=_FakeTokenizer(),
                ):
                    add_token_count.run_add_token_count(
                        add_token_count.AddTokenCountConfig(
                            input_root=input_root,
                            output_root=output_root,
                            tokenizer_json_path=tokenizer_path,
                            num_workers=1,
                            row_batch_rows=2,
                            overwrite_output=False,
                        )
                    )

    def test_run_add_token_count_fails_when_output_root_exists_and_overwrite_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "near_dedup"
            output_root = tmp_root / "token_count_added"
            tokenizer_path = tmp_root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")
            _write_input_shard(
                input_root,
                "part-000001.parquet",
                [
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "안녕",
                        "messages": None,
                        "token_count": None,
                    }
                ],
            )
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "existing.txt").write_text("already here", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "OVERWRITE_OUTPUT=True"):
                with patch(
                    "src.preprocess.add_token_count.common.get_tokenizer",
                    return_value=_FakeTokenizer(),
                ):
                    add_token_count.run_add_token_count(
                        add_token_count.AddTokenCountConfig(
                            input_root=input_root,
                            output_root=output_root,
                            tokenizer_json_path=tokenizer_path,
                            num_workers=1,
                            row_batch_rows=2,
                            overwrite_output=False,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
