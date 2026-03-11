from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from preprocess.add_data_usage import build_data_usage_column, rewrite_parquet_file, transform_table


MESSAGE_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("content", pa.string(), nullable=False),
            pa.field("role", pa.string(), nullable=False),
        ]
    )
)


class AddDataSplitTests(unittest.TestCase):
    def test_build_data_usage_column_assigns_pt_ft_and_empty_list(self) -> None:
        column, stats = build_data_usage_column(
            [
                None,
                [],
                [{"content": "u", "role": "user"}],
            ]
        )
        self.assertEqual(column.to_pylist(), ["PT", "PT", "FT"])
        self.assertEqual(stats["total_rows"], 3)
        self.assertEqual(stats["pt_rows"], 2)
        self.assertEqual(stats["ft_rows"], 1)
        self.assertEqual(stats["empty_messages_rows"], 1)

    def test_transform_table_places_data_usage_first(self) -> None:
        table = pa.Table.from_arrays(
            [
                pa.array(["src-a", "src-b"], type=pa.large_string()),
                pa.array(["train", "train"], type=pa.large_string()),
                pa.array(["c1", "c2"], type=pa.large_string()),
                pa.array(
                    [
                        None,
                        [{"content": "u", "role": "user"}],
                    ],
                    type=MESSAGE_TYPE,
                ),
                pa.array([1, 2], type=pa.int32()),
            ],
            names=["source", "split", "content", "messages", "token_count"],
        )
        transformed, stats = transform_table(table)
        self.assertEqual(transformed.column_names[0], "data_usage")
        self.assertEqual(transformed.column("data_usage").to_pylist(), ["PT", "FT"])
        self.assertEqual(stats["had_existing_data_usage"], 0)

    def test_transform_table_overwrites_existing_data_usage(self) -> None:
        table = pa.Table.from_arrays(
            [
                pa.array(["FT"], type=pa.large_string()),
                pa.array(["src-a"], type=pa.large_string()),
                pa.array(["train"], type=pa.large_string()),
                pa.array(["c1"], type=pa.large_string()),
                pa.array([None], type=MESSAGE_TYPE),
                pa.array([1], type=pa.int32()),
            ],
            names=["data_usage", "source", "split", "content", "messages", "token_count"],
        )
        transformed, stats = transform_table(table)
        self.assertEqual(transformed.column("data_usage").to_pylist(), ["PT"])
        self.assertEqual(stats["had_existing_data_usage"], 1)

    def test_rewrite_parquet_file_replaces_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "part-000001.parquet"
            table = pa.Table.from_arrays(
                [
                    pa.array(["src-a", "src-b"], type=pa.large_string()),
                    pa.array(["train", "val"], type=pa.large_string()),
                    pa.array(["c1", "c2"], type=pa.large_string()),
                    pa.array(
                        [
                            None,
                            [{"content": "a", "role": "assistant"}],
                        ],
                        type=MESSAGE_TYPE,
                    ),
                    pa.array([1, 2], type=pa.int32()),
                ],
                names=["source", "split", "content", "messages", "token_count"],
            )
            pq.write_table(table, path)

            result = rewrite_parquet_file(path)

            rewritten = pq.read_table(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(rewritten.column_names[0], "data_usage")
            self.assertEqual(rewritten.column("data_usage").to_pylist(), ["PT", "FT"])
            self.assertFalse(path.with_suffix(".parquet.tmp").exists())


if __name__ == "__main__":
    unittest.main()
