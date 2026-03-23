from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow.parquet as pq

from src.preprocess import build_lance, common


class _FakeTokenizer:
    def encode(self, text: str):  # type: ignore[no-untyped-def]
        return SimpleNamespace(ids=text.split())


def _write_input_shard(root: Path, name: str, rows: list[dict[str, object]]) -> Path:
    shard_path = root / name
    common.ensure_parent_dir(shard_path)
    pq.write_table(common.rows_to_table(rows), shard_path)
    return shard_path


class _FakeLanceModule:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def write_dataset(self, table, uri: str, *, mode: str) -> None:  # type: ignore[no-untyped-def]
        target = Path(uri)
        target.mkdir(parents=True, exist_ok=True)
        self.calls.append(
            {
                "uri": uri,
                "mode": mode,
                "row_count": table.num_rows,
                "sources": table.column("source").to_pylist(),
            }
        )
        (target / f"{len(self.calls):06d}-{mode}.json").write_text(
            json.dumps(self.calls[-1], ensure_ascii=False),
            encoding="utf-8",
        )


class BuildLanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_common_tqdm = common.ENABLE_TQDM
        self._old_common_debug = common.ENABLE_DEBUG_LOG
        common.ENABLE_TQDM = False
        common.ENABLE_DEBUG_LOG = False
        common._TOKENIZER_BY_PATH.clear()

    def tearDown(self) -> None:
        common.ENABLE_TQDM = self._old_common_tqdm
        common.ENABLE_DEBUG_LOG = self._old_common_debug
        common._TOKENIZER_BY_PATH.clear()

    def test_expand_sft_row_splits_on_pair_boundaries_and_rebuilds_content(self) -> None:
        system = {"role": "system", "content": "s" * 40}
        pair1 = [
            {"role": "user", "content": "u" * 150},
            {"role": "assistant", "content": "a" * 150},
        ]
        pair2 = [
            {"role": "user", "content": "v" * 150},
            {"role": "assistant", "content": "b" * 150},
        ]
        pair3 = [
            {"role": "user", "content": "w" * 150},
            {"role": "assistant", "content": "c" * 150},
        ]
        pair4 = [
            {"role": "user", "content": "x" * 150},
            {"role": "assistant", "content": "d" * 150},
        ]
        row = {
            "source": "alpha",
            "data_usage": "SFT",
            "split": "train",
            "content": "ignored",
            "messages": [system, *pair1, *pair2, *pair3, *pair4],
            "token_count": None,
        }

        rows = build_lance._expand_input_row(row, count_tokens=len)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(output_row["data_usage"] == "SFT" for output_row in rows))
        self.assertTrue(all(output_row["token_count"] <= build_lance.SFT_MAX_TOKENS for output_row in rows))
        self.assertTrue(all(str(output_row["messages"]).startswith("<|bos|><|system|>") for output_row in rows))
        self.assertTrue(all(str(output_row["messages"]).endswith("<|eos|>") for output_row in rows))
        self.assertEqual(
            rows[0]["content"],
            " ".join([pair1[0]["content"], pair1[1]["content"], pair2[0]["content"], pair2[1]["content"]]),
        )
        self.assertEqual(
            rows[1]["content"],
            " ".join([pair3[0]["content"], pair3[1]["content"], pair4[0]["content"], pair4[1]["content"]]),
        )

    def test_split_pt_content_prefers_period_boundaries(self) -> None:
        content = ("a" * 500) + ". " + ("b" * 500) + ". " + ("c" * 500) + "."

        parts = build_lance._split_pt_content(content, count_tokens=len)

        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(all(part.strip() for part in parts))
        self.assertTrue(all(len(part) <= build_lance.PT_MAX_TOKENS for part in parts))
        self.assertIn("a" * 500, parts[0])

    def test_expand_reasoning_row_produces_pt_and_reasoning_rows(self) -> None:
        row = {
            "source": "reasoning-src",
            "data_usage": "REASONING",
            "split": "val",
            "content": "질문 답변 추가",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "문제"},
                {"role": "assistant", "content": "풀이"},
            ],
            "token_count": None,
        }

        rows = build_lance._expand_input_row(row, count_tokens=len)

        self.assertEqual([output_row["data_usage"] for output_row in rows], ["PT", "REASONING"])
        self.assertEqual(rows[0]["messages"], None)
        self.assertEqual(rows[0]["content"], "질문 답변 추가")
        self.assertIsNone(rows[1]["content"])
        self.assertIsInstance(rows[1]["messages"], str)
        self.assertEqual(rows[1]["token_count"], len(rows[1]["messages"]))

    def test_run_build_lance_writes_source_shards_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_root = tmp_root / "near_dedup"
            output_root = tmp_root / "final_lancedb"
            temp_source_root = tmp_root / "lance_by_source"
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
                        "content": "안녕 세상 추가 문장",
                        "messages": None,
                        "token_count": None,
                    },
                    {
                        "source": "beta",
                        "data_usage": "PT",
                        "split": "val",
                        "content": "질문 답변",
                        "messages": None,
                        "token_count": None,
                    },
                ],
            )
            _write_input_shard(
                input_root,
                "part-000002.parquet",
                [
                    {
                        "source": "alpha",
                        "data_usage": "PT",
                        "split": "train",
                        "content": "첫째 둘째 셋째 추가 문장",
                        "messages": None,
                        "token_count": None,
                    }
                ],
            )

            fake_lance = _FakeLanceModule()
            with (
                patch("src.preprocess.build_lance._load_lance_module", return_value=fake_lance),
                patch("src.preprocess.build_lance.common.get_tokenizer", return_value=_FakeTokenizer()),
            ):
                summary = build_lance.run_build_lance(
                    build_lance.BuildLanceConfig(
                        input_root=input_root,
                        output_root=output_root,
                        temp_source_root=temp_source_root,
                        tokenizer_json_path=tokenizer_path,
                        source_shard_target_bytes=80,
                        read_batch_rows=16,
                        overwrite_output=False,
                    )
                )

            self.assertEqual(summary.input_shard_count, 2)
            self.assertEqual(summary.input_row_count, 3)
            self.assertEqual(summary.output_row_count, 3)
            self.assertEqual(summary.source_count, 2)
            self.assertEqual(summary.data_usage_counts, {"PT": 3})
            self.assertEqual(summary.split_counts, {"train": 2, "val": 1})
            self.assertTrue((output_root / "_meta" / "build_lance_manifest.json").exists())
            self.assertTrue((output_root / "_meta" / "build_lance_progress.json").exists())
            self.assertTrue((output_root / "dataset.lance").exists())
            self.assertFalse(temp_source_root.exists())
            self.assertEqual(fake_lance.calls[0]["mode"], "create")
            self.assertTrue(all(call["uri"] == str(output_root / "dataset.lance") for call in fake_lance.calls))
            self.assertGreaterEqual(len(fake_lance.calls), 2)

            progress = json.loads(
                (output_root / "_meta" / "build_lance_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["phase"], "completed")
            self.assertEqual(progress["last_completed_source"], "beta")


if __name__ == "__main__":
    unittest.main()
