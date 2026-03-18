from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow.parquet as pq

from src.preprocess import common
from src.preprocess import exact_dedup


def _write_input_shard(root: Path, dataset_id: str, split: str, rows: list[dict[str, object]]) -> Path:
    shard_path = root / dataset_id / split / "part-000001.parquet"
    common.ensure_parent_dir(shard_path)
    pq.write_table(common.rows_to_table(rows), shard_path)
    return shard_path


def _read_output_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shard_path in sorted(output_root.glob("part-*.parquet"), key=lambda path: str(path)):
        rows.extend(pq.read_table(shard_path).to_pylist())
    return rows


class ExactDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_common_tqdm = common.ENABLE_TQDM
        self._old_common_debug = common.ENABLE_DEBUG_LOG
        common.ENABLE_TQDM = False
        common.ENABLE_DEBUG_LOG = False

    def tearDown(self) -> None:
        common.ENABLE_TQDM = self._old_common_tqdm
        common.ENABLE_DEBUG_LOG = self._old_common_debug

    def test_run_exact_dedup_retains_best_row_and_null_content_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "input"
            output_root = tmp_root / "output"

            _write_input_shard(
                input_root,
                "alpha",
                "train",
                [
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "dup-usage",
                        "messages": None,
                        "token_count": None,
                    },
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": None,
                        "messages": [{"role": "user", "content": "질문"}],
                        "token_count": None,
                    },
                ],
            )
            _write_input_shard(
                input_root,
                "beta",
                "val",
                [
                    {
                        "source": "beta",
                        "data_usage": "SFT",
                        "split": "val",
                        "content": "dup-usage",
                        "messages": [
                            {"role": "user", "content": "질문"},
                            {"role": "assistant", "content": "답변"},
                        ],
                        "token_count": None,
                    },
                    {
                        "source": "beta",
                        "data_usage": "REASONING",
                        "split": "val",
                        "content": "unique-reasoning",
                        "messages": None,
                        "token_count": None,
                    },
                ],
            )

            summary = exact_dedup.run_exact_dedup(
                input_root=input_root,
                output_root=output_root,
                target_shard_bytes=1024 * 1024,
                overwrite_output=False,
                duckdb_threads=1,
                duckdb_preserve_insertion_order=False,
                duckdb_memory_limit="26GB",
                duckdb_arrow_large_buffer_size=True,
                fetch_record_batch_rows=2,
            )

            retained_rows = _read_output_rows(output_root)
            retained_pairs = {(row["content"], row["data_usage"]) for row in retained_rows}
            self.assertEqual(summary.input_row_count, 4)
            self.assertEqual(summary.deleted_row_count, 1)
            self.assertEqual(summary.retained_row_count, 3)
            self.assertEqual(summary.null_content_passthrough_count, 1)
            self.assertEqual(summary.exact_cluster_count, 1)
            self.assertIn(("dup-usage", "SFT"), retained_pairs)
            self.assertIn(("unique-reasoning", "REASONING"), retained_pairs)
            self.assertIn((None, "PT"), retained_pairs)

            deleted_rows = [
                json.loads(line)
                for line in (output_root / "_logs" / "deleted_rows_exact.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(deleted_rows), 1)
            self.assertEqual(deleted_rows[0]["reason_code"], "exact_duplicate")
            self.assertEqual(deleted_rows[0]["reason_detail"], "duplicate removed because representative row had higher data_usage priority")
            self.assertEqual(summary.representative_reason_counts, {"higher_data_usage": 1})

    def test_run_exact_dedup_uses_numeric_file_row_tiebreak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "input"
            output_root = tmp_root / "output"

            rows: list[dict[str, object]] = []
            for index in range(11):
                content = f"unique-{index}"
                if index in {2, 10}:
                    content = "dup-locator"
                rows.append(
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": content,
                        "messages": None,
                        "token_count": None,
                    }
                )
            _write_input_shard(input_root, "alpha", "train", rows)

            summary = exact_dedup.run_exact_dedup(
                input_root=input_root,
                output_root=output_root,
                target_shard_bytes=1024 * 1024,
                overwrite_output=False,
                duckdb_threads=1,
                duckdb_preserve_insertion_order=False,
                duckdb_memory_limit="26GB",
                duckdb_arrow_large_buffer_size=True,
                fetch_record_batch_rows=4,
            )

            retained_rows = _read_output_rows(output_root)
            dup_rows = [row for row in retained_rows if row["content"] == "dup-locator"]
            self.assertEqual(len(dup_rows), 1)
            self.assertEqual(summary.deleted_row_count, 1)
            self.assertEqual(summary.representative_reason_counts, {"stable_locator_tiebreak": 1})

            deleted_row = json.loads(
                (output_root / "_logs" / "deleted_rows_exact.jsonl").read_text(
                    encoding="utf-8"
                ).strip()
            )
            self.assertTrue(str(deleted_row["record_locator"]).endswith("#10"))
            self.assertTrue(str(deleted_row["representative_locator"]).endswith("#2"))

    def test_run_exact_dedup_rolls_shard_after_crossing_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            rows = [
                {
                    "source": "alpha",
                    "data_usage": "PT",
                    "split": "train",
                    "content": "a" * 64,
                    "messages": None,
                    "token_count": None,
                },
                {
                    "source": "alpha",
                    "data_usage": "PT",
                    "split": "train",
                    "content": "b" * 64,
                    "messages": None,
                    "token_count": None,
                },
                {
                    "source": "alpha",
                    "data_usage": "PT",
                    "split": "train",
                    "content": "c" * 64,
                    "messages": None,
                    "token_count": None,
                },
            ]
            measurement_input_root = tmp_root / "measurement_input"
            measurement_output_root = tmp_root / "measurement_output"
            _write_input_shard(measurement_input_root, "alpha", "train", rows[:2])

            measurement_summary = exact_dedup.run_exact_dedup(
                input_root=measurement_input_root,
                output_root=measurement_output_root,
                target_shard_bytes=1024 * 1024,
                overwrite_output=False,
                duckdb_threads=1,
                duckdb_preserve_insertion_order=False,
                duckdb_memory_limit="26GB",
                duckdb_arrow_large_buffer_size=True,
                fetch_record_batch_rows=8,
            )
            two_row_nbytes = measurement_summary.output_shards[0]["approx_nbytes"]

            input_root = tmp_root / "input"
            output_root = tmp_root / "output"
            _write_input_shard(input_root, "alpha", "train", rows)

            summary = exact_dedup.run_exact_dedup(
                input_root=input_root,
                output_root=output_root,
                target_shard_bytes=two_row_nbytes - 1,
                overwrite_output=False,
                duckdb_threads=1,
                duckdb_preserve_insertion_order=False,
                duckdb_memory_limit="26GB",
                duckdb_arrow_large_buffer_size=True,
                fetch_record_batch_rows=8,
            )

            self.assertEqual(summary.output_shard_count, 2)
            self.assertEqual([shard["row_count"] for shard in summary.output_shards], [2, 1])

    def test_run_exact_dedup_emits_stage_logs_and_progress_bars(self) -> None:
        class FakeProgressBar:
            def __init__(self, *, total: int | None, desc: str) -> None:
                self.total = total
                self.desc = desc
                self.updated = 0
                self.closed = False

            def update(self, value: int) -> None:
                self.updated += value

            def close(self) -> None:
                self.closed = True

        created_bars: list[FakeProgressBar] = []

        def fake_progress_bar(*, total: int | None, desc: str) -> FakeProgressBar:
            bar = FakeProgressBar(total=total, desc=desc)
            created_bars.append(bar)
            return bar

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "input"
            output_root = tmp_root / "output"

            _write_input_shard(
                input_root,
                "alpha",
                "train",
                [
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "dup-text",
                        "messages": None,
                        "token_count": None,
                    },
                    {
                        "source": "alpha",
                        "data_usage": "SFT",
                        "split": "train",
                        "content": "dup-text",
                        "messages": [
                            {"role": "user", "content": "질문"},
                            {"role": "assistant", "content": "답변"},
                        ],
                        "token_count": None,
                    },
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "unique-text",
                        "messages": None,
                        "token_count": None,
                    },
                ],
            )

            with (
                patch("src.preprocess.exact_dedup.progress_bar", side_effect=fake_progress_bar),
                patch("src.preprocess.exact_dedup.log") as mock_log,
            ):
                summary = exact_dedup.run_exact_dedup(
                    input_root=input_root,
                    output_root=output_root,
                    target_shard_bytes=1024 * 1024,
                    overwrite_output=False,
                    duckdb_threads=1,
                    duckdb_preserve_insertion_order=False,
                    duckdb_memory_limit="26GB",
                    duckdb_arrow_large_buffer_size=True,
                    fetch_record_batch_rows=2,
                )

        self.assertEqual(summary.deleted_row_count, 1)
        self.assertEqual([bar.desc for bar in created_bars], ["exact_dedup/deleted_log", "exact_dedup/retained"])
        self.assertEqual(created_bars[0].total, 1)
        self.assertEqual(created_bars[0].updated, 1)
        self.assertTrue(created_bars[0].closed)
        self.assertEqual(created_bars[1].total, 2)
        self.assertEqual(created_bars[1].updated, 2)
        self.assertTrue(created_bars[1].closed)

        logged_messages = [call.args[0] for call in mock_log.call_args_list]
        self.assertTrue(any("[exact_dedup] start " in message for message in logged_messages))
        self.assertTrue(any("[exact_dedup/deleted_log] start " in message for message in logged_messages))
        self.assertTrue(any("[exact_dedup/deleted_log] completed " in message for message in logged_messages))
        self.assertTrue(any("[exact_dedup/retained] completed " in message for message in logged_messages))
        self.assertTrue(any("[exact_dedup] completed " in message for message in logged_messages))
        self.assertTrue(any("duckdb_preserve_insertion_order=False" in message for message in logged_messages))
        self.assertTrue(any("duckdb_memory_limit=26GB" in message for message in logged_messages))
        self.assertTrue(any("duckdb_arrow_large_buffer_size=True" in message for message in logged_messages))

    def test_configure_duckdb_applies_memory_insertion_and_arrow_pragmas(self) -> None:
        con = MagicMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir) / "tmp"
            exact_dedup._configure_duckdb(
                con=con,
                temp_root=temp_root,
                duckdb_threads=2,
                duckdb_preserve_insertion_order=False,
                duckdb_memory_limit="26GB",
                duckdb_arrow_large_buffer_size=True,
            )

        executed = [call.args[0] for call in con.execute.call_args_list]
        self.assertEqual(executed[0], "PRAGMA threads=2")
        self.assertTrue(executed[1].startswith("PRAGMA temp_directory='"))
        self.assertEqual(executed[2], "PRAGMA preserve_insertion_order=false")
        self.assertEqual(executed[3], "PRAGMA memory_limit='26GB'")
        self.assertEqual(executed[4], "SET arrow_large_buffer_size=true")


if __name__ == "__main__":
    unittest.main()
