from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.tokenizer import build_tokenizer
from src.tokenizer import common
from src.tokenizer import simrun


def _write_near_dedup_shard(root: Path, texts: list[str | None]) -> Path:
    path = root / "part-000001.parquet"
    common.ensure_parent_dir(path)
    pq.write_table(pa.table({"content": texts}), path)
    return path


class SimrunTests(unittest.TestCase):
    def test_compute_compression_stats(self) -> None:
        stats = simrun.compute_compression_stats("가나다", 2)
        self.assertEqual(stats["char_count"], 3)
        self.assertEqual(stats["token_count"], 2)
        self.assertGreater(stats["utf8_byte_count"], 0)

    def test_run_simrun_handles_empty_input_and_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            near_dedup_root = tmp_root / "near_dedup"
            tokenizer_root = tmp_root / "tokenizers"
            _write_near_dedup_shard(near_dedup_root, ["가나다라마바사", "가나다라마바사"])
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

            answers = iter(["", "가나다", "quit"])
            output = io.StringIO()

            simrun.run_simrun(
                simrun.SimrunConfig(
                    tokenizer_json_path=build_summary.output_dir / "tokenizer.json",
                    max_display_tokens=4,
                    show_token_breakdown=True,
                    exit_commands=("quit",),
                ),
                input_fn=lambda _prompt: next(answers),
                output_stream=output,
            )

            rendered = output.getvalue()
            self.assertIn("empty input", rendered)
            self.assertIn("token_count", rendered)
            self.assertIn("token_breakdown", rendered)
            self.assertIn("bye", rendered)
