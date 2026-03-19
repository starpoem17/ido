from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.tokenizer import bench_tokenizer
from src.tokenizer import build_tokenizer
from src.tokenizer import common


def _write_near_dedup_shard(root: Path, texts: list[str | None]) -> Path:
    path = root / "part-000001.parquet"
    common.ensure_parent_dir(path)
    pq.write_table(pa.table({"content": texts}), path)
    return path


def _write_benchmark_parquet(path: Path, texts: list[str], token_counts: list[int]) -> None:
    common.ensure_parent_dir(path)
    pq.write_table(pa.table({"text": texts, "token_count": token_counts}), path)


class BenchTokenizerTests(unittest.TestCase):
    def test_run_bench_tokenizer_writes_compression_oriented_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            near_dedup_root = tmp_root / "near_dedup"
            tokenizer_root = tmp_root / "tokenizers"
            _write_near_dedup_shard(near_dedup_root, ["가나다라마바사", "가나다라마바사", "라마바"])
            build_summary = build_tokenizer.run_build_tokenizer(
                build_tokenizer.BuildTokenizerConfig(
                    near_dedup_root=near_dedup_root,
                    tokenizer_root=tokenizer_root,
                    tokenizer_name_prefix="korean_bbpe_v",
                    debug_run_root=tmp_root / "_tmp",
                    vocab_size=64,
                    top_token_count=5,
                    num_workers=1,
                    train_chunk_enable=True,
                    train_chunk_max_chars=8,
                    train_chunk_max_utf8_bytes=24,
                    train_chunk_break_priority=("\n\n", "\n", " "),
                    trainer_min_frequency=1,
                    trainer_max_token_length=16,
                    trainer_limit_alphabet=256,
                    top_long_rows=10,
                    memory_warning_available_mb=256.0,
                    memory_warning_swap_free_mb=512.0,
                    debug_heartbeat_interval_sec=0.05,
                    train_progress_log_every_chunks=2,
                    special_tokens=tuple(build_tokenizer.SPECIAL_TOKENS),
                    parquet_batch_rows=2,
                    enable_tqdm=False,
                    enable_debug_log=False,
                    tqdm_mininterval_sec=0.0,
                )
            )

            benchmark_parquet = tmp_root / "benchmark.parquet"
            tokenizer = common.load_tokenizer(build_summary.output_dir / "tokenizer.json")
            exact = len(tokenizer.encode("가나다라마바사").ids)
            short_exact = len(tokenizer.encode("라마바").ids)
            _write_benchmark_parquet(
                benchmark_parquet,
                ["가나다라마바사", "라마바", "라마바"],
                [exact, short_exact + 1, short_exact - 2],
            )

            summary = bench_tokenizer.run_bench_tokenizer(
                bench_tokenizer.BenchTokenizerConfig(
                    tokenizer_json_path=build_summary.output_dir / "tokenizer.json",
                    benchmark_parquet_path=benchmark_parquet,
                    delta_sample_limit=10,
                    num_workers=1,
                    parquet_batch_rows=2,
                    enable_tqdm=False,
                    enable_debug_log=False,
                    tqdm_mininterval_sec=0.0,
                )
            )

            self.assertEqual(summary.compared_row_count, 3)
            self.assertEqual(summary.more_compressive_row_count, 1)
            self.assertEqual(summary.same_token_count_row_count, 1)
            self.assertEqual(summary.less_compressive_row_count, 1)
            self.assertAlmostEqual(summary.mean_delta, 1 / 3)
            self.assertAlmostEqual(summary.mean_abs_delta, 1.0)
            summary_json = json.loads(
                (summary.benchmark_out_dir / "benchmark_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary_json["more_compressive_row_count"], 1)
            self.assertEqual(summary_json["same_token_count_row_count"], 1)
            self.assertEqual(summary_json["less_compressive_row_count"], 1)
            self.assertIn("mean_delta_ratio", summary_json)
            self.assertTrue(summary_json["build_meta_path"].endswith("meta.json"))
            delta_rows = [
                json.loads(line)
                for line in (
                    summary.benchmark_out_dir / "benchmark_delta_samples.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(delta_rows), 2)
            self.assertEqual(delta_rows[0]["delta"], 2)
            self.assertEqual(delta_rows[0]["compression_label"], "less_compressive")
            self.assertEqual(delta_rows[1]["delta"], -1)
            self.assertEqual(delta_rows[1]["compression_label"], "more_compressive")
            self.assertIn("text_preview", delta_rows[0])
            self.assertNotIn("content_preview", delta_rows[0])
