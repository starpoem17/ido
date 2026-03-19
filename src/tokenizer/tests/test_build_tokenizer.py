from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from src.tokenizer import build_tokenizer
from src.tokenizer import common


def _write_near_dedup_shard(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "part-000001.parquet"
    common.ensure_parent_dir(path)
    table = pa.table({"content": [row["content"] for row in rows]})
    pq.write_table(table, path)
    return path


class BuildTokenizerTests(unittest.TestCase):
    def test_chunk_training_text_prefers_soft_breaks_then_hard_cuts(self) -> None:
        self.assertEqual(
            build_tokenizer._chunk_training_text(
                "aaaa\n\nbbbb",
                max_chars=6,
                max_utf8_bytes=24,
                break_priority=("\n\n", "\n", " "),
            ),
            ["aaaa\n\n", "bbbb"],
        )
        self.assertEqual(
            build_tokenizer._chunk_training_text(
                "aaa bbb ccc",
                max_chars=7,
                max_utf8_bytes=24,
                break_priority=("\n\n", "\n", " "),
            ),
            ["aaa ", "bbb ccc"],
        )
        self.assertEqual(
            build_tokenizer._chunk_training_text(
                "abcdefghij",
                max_chars=4,
                max_utf8_bytes=4,
                break_priority=("\n\n", "\n", " "),
            ),
            ["abcd", "efgh", "ij"],
        )
        self.assertEqual(
            build_tokenizer._chunk_training_text(
                "가나다라마바사",
                max_chars=7,
                max_utf8_bytes=9,
                break_priority=("\n\n", "\n", " "),
            ),
            ["가나다", "라마바", "사"],
        )

    def test_summarize_child_exit_marks_sigkill_as_oom(self) -> None:
        self.assertEqual(
            build_tokenizer._summarize_child_exit(-9),
            {
                "child_exit_code": -9,
                "child_signal": 9,
                "oom_suspected": True,
            },
        )
        self.assertEqual(
            build_tokenizer._classify_oom_risk(
                mem_available_mb=128.0,
                swap_free_mb=512.0,
                warning_available_mb=1024.0,
                warning_swap_free_mb=4096.0,
            ),
            "critical",
        )

    def test_find_next_version_dir_skips_existing_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "korean_bbpe_v1").mkdir()
            (root / "korean_bbpe_v3").mkdir()

            target = common.find_next_version_dir(root, "korean_bbpe_v")

            self.assertEqual(target.name, "korean_bbpe_v2")

    def test_run_build_tokenizer_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            near_dedup_root = tmp_root / "near_dedup"
            tokenizer_root = tmp_root / "tokenizers"
            _write_near_dedup_shard(
                near_dedup_root,
                [
                    {"content": "가나다라마바사"},
                    {"content": "가나다라마바사"},
                    {"content": "라마바"},
                    {"content": None},
                    {"content": "   "},
                ],
            )

            config = build_tokenizer.BuildTokenizerConfig(
                near_dedup_root=near_dedup_root,
                tokenizer_root=tokenizer_root,
                tokenizer_name_prefix="korean_bbpe_v",
                debug_run_root=tmp_root / "_tmp",
                vocab_size=64,
                top_token_count=5,
                num_workers=1,
                train_chunk_enable=True,
                train_chunk_max_chars=4,
                train_chunk_max_utf8_bytes=16,
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
            summary = build_tokenizer.run_build_tokenizer(config)

            output_dir = summary.output_dir
            self.assertTrue((output_dir / "tokenizer.json").exists())
            self.assertTrue((output_dir / "vocab.json").exists())
            self.assertTrue((output_dir / "merges.txt").exists())
            self.assertTrue((output_dir / "meta.json").exists())
            self.assertTrue((output_dir / "top_tokens.jsonl").exists())

            meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["input_row_count"], 5)
            self.assertEqual(meta["valid_row_count"], 3)
            self.assertEqual(meta["filtered_row_count"], 2)
            self.assertEqual(meta["top_token_count"], 5)
            self.assertTrue(meta["train_chunk_enable"])
            self.assertEqual(meta["train_chunk_max_chars"], 4)
            self.assertEqual(meta["train_chunk_max_utf8_bytes"], 16)
            self.assertGreaterEqual(meta["train_chunk_count"], meta["valid_row_count"])
            self.assertEqual(meta["trainer_min_frequency"], 1)
            self.assertEqual(meta["trainer_max_token_length"], 16)
            self.assertEqual(meta["trainer_limit_alphabet"], 256)
            self.assertIn("build_tokenizer_run_", meta["debug_run_dir"])
            self.assertIn("finished_at", meta)
            self.assertIn("duration_sec", meta)

            top_rows = [
                json.loads(line)
                for line in (output_dir / "top_tokens.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreaterEqual(len(top_rows), 1)
            self.assertIn("raw_token", top_rows[0])
            self.assertIn("escaped_token", top_rows[0])
            self.assertIn("decoded_token", top_rows[0])

            debug_run_dir = Path(meta["debug_run_dir"])
            self.assertTrue((debug_run_dir / "run_config.json").exists())
            self.assertTrue((debug_run_dir / "events.jsonl").exists())
            self.assertTrue((debug_run_dir / "memory_heartbeat.jsonl").exists())
            self.assertTrue((debug_run_dir / "long_rows.jsonl").exists())
            self.assertTrue(meta["top_long_rows_path"].endswith("long_rows.jsonl"))
            self.assertEqual(meta["child_exit_code"], 0)
            self.assertFalse(meta["oom_suspected"])

    def test_run_build_tokenizer_rejects_missing_content_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            near_dedup_root = tmp_root / "near_dedup"
            path = near_dedup_root / "part-000001.parquet"
            common.ensure_parent_dir(path)
            pq.write_table(pa.table({"text": ["abc"]}), path)

            config = build_tokenizer.BuildTokenizerConfig(
                near_dedup_root=near_dedup_root,
                tokenizer_root=tmp_root / "tokenizers",
                tokenizer_name_prefix="korean_bbpe_v",
                debug_run_root=tmp_root / "_tmp",
                vocab_size=64,
                top_token_count=5,
                num_workers=1,
                train_chunk_enable=True,
                train_chunk_max_chars=4,
                train_chunk_max_utf8_bytes=16,
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
            with self.assertRaisesRegex(ValueError, "missing content column"):
                build_tokenizer.run_build_tokenizer(config)

    def test_run_build_tokenizer_records_child_failure_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            near_dedup_root = tmp_root / "near_dedup"
            tokenizer_root = tmp_root / "tokenizers"
            _write_near_dedup_shard(near_dedup_root, [{"content": "가나다라마바사"}])

            config = build_tokenizer.BuildTokenizerConfig(
                near_dedup_root=near_dedup_root,
                tokenizer_root=tokenizer_root,
                tokenizer_name_prefix="korean_bbpe_v",
                debug_run_root=tmp_root / "_tmp",
                vocab_size=64,
                top_token_count=5,
                num_workers=1,
                train_chunk_enable=True,
                train_chunk_max_chars=4,
                train_chunk_max_utf8_bytes=16,
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

            class FakeProcess:
                pid = 4321
                exitcode = -9

                def join(self) -> None:
                    return None

            with mock.patch.object(
                build_tokenizer,
                "_spawn_training_child",
                return_value=FakeProcess(),
            ):
                with self.assertRaisesRegex(RuntimeError, "oom_suspected=True"):
                    build_tokenizer.run_build_tokenizer(config)

            debug_run_dirs = sorted((tmp_root / "_tmp").glob("build_tokenizer_run_*"))
            self.assertEqual(len(debug_run_dirs), 1)
            failure_context = json.loads(
                (debug_run_dirs[0] / "failure_context.json").read_text(encoding="utf-8")
            )
            self.assertTrue(failure_context["oom_suspected"])
            self.assertEqual(failure_context["child_signal"], 9)
