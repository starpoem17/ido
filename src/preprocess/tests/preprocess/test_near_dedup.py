from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from src.preprocess import common
from src.preprocess import near_dedup


def _write_input_shard(root: Path, rows: list[dict[str, object]]) -> Path:
    shard_path = root / "part-000001.parquet"
    common.ensure_parent_dir(shard_path)
    pq.write_table(common.rows_to_table(rows), shard_path)
    return shard_path


def _read_output_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shard_path in sorted(output_root.glob("part-*.parquet"), key=lambda path: str(path)):
        rows.extend(pq.read_table(shard_path).to_pylist())
    return rows


class NearDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_common_tqdm = common.ENABLE_TQDM
        self._old_common_debug = common.ENABLE_DEBUG_LOG
        common.ENABLE_TQDM = False
        common.ENABLE_DEBUG_LOG = False

    def tearDown(self) -> None:
        common.ENABLE_TQDM = self._old_common_tqdm
        common.ENABLE_DEBUG_LOG = self._old_common_debug

    def test_build_shingles_uses_ngram_join_and_short_text_fallback(self) -> None:
        with patch("src.preprocess.near_dedup._tokenize_korean_text", return_value=["가", "나", "다"]):
            shingles = near_dedup._build_shingles(text="가 나 다", n=2)
            short_shingles = near_dedup._build_shingles(text="가 나 다", n=5)

        self.assertEqual(shingles, {"가 나", "나 다"})
        self.assertEqual(short_shingles, {"가 나 다"})

    def test_select_representative_prefers_usage_then_length_then_locator(self) -> None:
        members = [
            near_dedup.RowMetadata(
                record_locator="b#2",
                source="src",
                data_usage="PT",
                split="train",
                content="짧다",
                content_length=2,
            ),
            near_dedup.RowMetadata(
                record_locator="c#1",
                source="src",
                data_usage="SFT",
                split="train",
                content="보통 길이",
                content_length=5,
            ),
            near_dedup.RowMetadata(
                record_locator="a#9",
                source="src",
                data_usage="SFT",
                split="train",
                content="더 긴 텍스트",
                content_length=7,
            ),
        ]

        representative = near_dedup._select_representative(members)
        self.assertEqual(representative.record_locator, "a#9")

    def test_run_near_dedup_removes_verified_duplicates_and_keeps_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "exact_dedup"
            output_root = tmp_root / "near_dedup"
            intermediate_root = output_root / "_intermediate"

            _write_input_shard(
                input_root,
                [
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "가 나 다 라 마",
                        "messages": None,
                        "token_count": None,
                    },
                    {
                        "source": "beta",
                        "data_usage": "SFT",
                        "split": "train",
                        "content": "가 나 다 라 마",
                        "messages": [
                            {"role": "user", "content": "질문"},
                            {"role": "assistant", "content": "답변"},
                        ],
                        "token_count": None,
                    },
                    {
                        "source": "gamma",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "서로 다른 문장",
                        "messages": None,
                        "token_count": None,
                    },
                    {
                        "source": "delta",
                        "data_usage": "PT",
                        "split": "train",
                        "content": None,
                        "messages": [{"role": "user", "content": "null content"}],
                        "token_count": None,
                    },
                ],
            )

            with patch(
                "src.preprocess.near_dedup._tokenize_korean_text",
                side_effect=lambda text: text.split(),
            ):
                summary = near_dedup.run_near_dedup(
                    input_root=input_root,
                    output_root=output_root,
                    intermediate_root=intermediate_root,
                    target_shard_bytes=1024 * 1024,
                    minhash_ngrams=2,
                    minhash_num_buckets=2,
                    minhash_hashes_per_bucket=2,
                    minhash_hash_precision=64,
                    jaccard_threshold=1.0,
                    fetch_record_batch_rows=2,
                    duckdb_threads=1,
                    overwrite_output=False,
                )

            retained_rows = _read_output_rows(output_root)
            retained_pairs = {
                (row["source"], row["data_usage"], row["content"]) for row in retained_rows
            }
            self.assertEqual(summary.input_row_count, 4)
            self.assertEqual(summary.deleted_row_count, 1)
            self.assertEqual(summary.near_cluster_count, 1)
            self.assertEqual(summary.retained_row_count, 3)
            self.assertIn(("beta", "SFT", "가 나 다 라 마"), retained_pairs)
            self.assertIn(("gamma", "PT", "서로 다른 문장"), retained_pairs)
            self.assertIn(("delta", "PT", None), retained_pairs)

            deleted_rows = [
                json.loads(line)
                for line in (output_root / "_logs" / "deleted_rows_minhash.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(deleted_rows), 1)
            self.assertEqual(deleted_rows[0]["reason_code"], "near_duplicate")


if __name__ == "__main__":
    unittest.main()
