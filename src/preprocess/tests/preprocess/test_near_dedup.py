from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


def _build_config(input_root: Path, output_root: Path, **overrides: object) -> near_dedup.NearDedupConfig:
    payload = {
        "input_root": input_root,
        "output_root": output_root,
        "minhash_lsh_root": output_root / "_minhash_lsh",
        "target_shard_bytes": 1024 * 1024,
        "shingle_ngram_size": 2,
        "minhash_num_permutations": 4,
        "lsh_num_bands": 2,
        "lsh_rows_per_band": 2,
        "minhash_hash_bits": 64,
        "minhash_random_seed": 17,
        "jaccard_threshold": 1.0,
        "num_workers": 1,
        "row_batch_rows": 2,
        "signature_shingle_chunk_size": 64,
        "giant_bucket_max_rows": 8,
        "giant_bucket_rebucket_fanout": 4,
        "giant_bucket_max_passes": 1,
        "verify_pair_batch_rows": 32,
        "verify_cache_max_rows": 128,
        "duckdb_threads": 1,
        "duckdb_memory_limit": "1GB",
        "duckdb_arrow_large_buffer_size": True,
        "overwrite_output": False,
    }
    payload.update(overrides)
    return near_dedup.NearDedupConfig(**payload)


class NearDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_common_tqdm = common.ENABLE_TQDM
        self._old_common_debug = common.ENABLE_DEBUG_LOG
        common.ENABLE_TQDM = False
        common.ENABLE_DEBUG_LOG = False

    def tearDown(self) -> None:
        common.ENABLE_TQDM = self._old_common_tqdm
        common.ENABLE_DEBUG_LOG = self._old_common_debug

    def test_build_char_ngrams_removes_whitespace_and_uses_short_text_fallback(self) -> None:
        shingles = near_dedup._build_char_ngrams("가 나 다", 2)
        short_shingles = near_dedup._build_char_ngrams("가 나", 3)

        self.assertEqual(shingles, {"가나", "나다"})
        self.assertEqual(short_shingles, {"가나"})

    def test_validate_config_rejects_mismatched_permutation_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config = _build_config(
                tmp_root / "exact_dedup",
                tmp_root / "near_dedup",
                minhash_num_permutations=5,
                lsh_num_bands=2,
                lsh_rows_per_band=2,
            )
            with self.assertRaisesRegex(ValueError, "minhash_num_permutations"):
                near_dedup.run_near_dedup(config)

    def test_run_near_dedup_prefers_higher_usage_and_keeps_null_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "exact_dedup"
            output_root = tmp_root / "near_dedup"
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
                        "content": "가나다라마",
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

            summary = near_dedup.run_near_dedup(_build_config(input_root, output_root))
            retained_rows = _read_output_rows(output_root)
            retained_keys = {
                (row["source"], row["data_usage"], row["content"])
                for row in retained_rows
            }

            self.assertEqual(summary.input_row_count, 4)
            self.assertEqual(summary.candidate_pair_count, 1)
            self.assertEqual(summary.verified_pair_count, 1)
            self.assertEqual(summary.near_cluster_count, 1)
            self.assertEqual(summary.deleted_row_count, 1)
            self.assertEqual(summary.retained_row_count, 3)
            self.assertIn(("beta", "SFT", "가나다라마"), retained_keys)
            self.assertIn(("gamma", "PT", "서로 다른 문장"), retained_keys)
            self.assertIn(("delta", "PT", None), retained_keys)

            deleted_rows = [
                json.loads(line)
                for line in (output_root / "_logs" / "deleted_rows_minhash.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(deleted_rows), 1)
            self.assertEqual(deleted_rows[0]["source"], "alpha")
            self.assertEqual(deleted_rows[0]["reason_code"], "near_duplicate")

    def test_run_near_dedup_uses_longer_content_tiebreak_with_same_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "exact_dedup"
            output_root = tmp_root / "near_dedup"
            _write_input_shard(
                input_root,
                [
                    {
                        "source": "shorter",
                        "data_usage": "SFT",
                        "split": "train",
                        "content": "가나다라마",
                        "messages": [{"role": "assistant", "content": "a"}],
                        "token_count": None,
                    },
                    {
                        "source": "longer",
                        "data_usage": "SFT",
                        "split": "train",
                        "content": "가나 다라마",
                        "messages": [{"role": "assistant", "content": "b"}],
                        "token_count": None,
                    },
                ],
            )

            summary = near_dedup.run_near_dedup(_build_config(input_root, output_root))
            retained_rows = _read_output_rows(output_root)

            self.assertEqual(summary.deleted_row_count, 1)
            self.assertEqual(len(retained_rows), 1)
            self.assertEqual(retained_rows[0]["source"], "longer")

    def test_run_near_dedup_records_giant_bucket_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "exact_dedup"
            output_root = tmp_root / "near_dedup"
            _write_input_shard(
                input_root,
                [
                    {
                        "source": "a",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "가나다라마바사",
                        "messages": None,
                        "token_count": None,
                    },
                    {
                        "source": "b",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "가나다라마바사",
                        "messages": None,
                        "token_count": None,
                    },
                    {
                        "source": "c",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "가나다라마바사",
                        "messages": None,
                        "token_count": None,
                    },
                ],
            )

            config = _build_config(
                input_root,
                output_root,
                minhash_num_permutations=2,
                lsh_num_bands=1,
                lsh_rows_per_band=2,
                giant_bucket_max_rows=2,
            )
            summary = near_dedup.run_near_dedup(config)

            self.assertGreaterEqual(summary.giant_bucket_count, 1)
            self.assertEqual(summary.deleted_row_count, 2)
            self.assertEqual(summary.retained_row_count, 1)


if __name__ == "__main__":
    unittest.main()
