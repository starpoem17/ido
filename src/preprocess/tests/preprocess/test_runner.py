from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.preprocess import runner
from src.preprocess.staging import StageResult


class RunnerTests(unittest.TestCase):
    def test_run_stage_entrypoint_collects_split_results(self) -> None:
        def fake_stage_dataset(config):  # type: ignore[no-untyped-def]
            return StageResult(
                dataset_id=config.dataset_id,
                split=config.target_split,
                source="fake-source",
                input_files=1,
                raw_candidate_count=10,
                valid_candidate_count=9,
                output_rows=8 if config.target_split == "train" else 1,
                shard_files=1,
                quality_event_count=0,
                manifest_path=f"/tmp/{config.dataset_id}-{config.target_split}.json",
                quality_summary_path=f"/tmp/{config.dataset_id}-{config.target_split}-summary.json",
                quality_events_path=f"/tmp/{config.dataset_id}-{config.target_split}-events.jsonl",
            )

        with patch("src.preprocess.runner.stage_dataset", side_effect=fake_stage_dataset):
            result = runner.run_stage_entrypoint(
                dataset_id="009",
                output_root=Path("data/korean_processed/_staging"),
                target_splits=("train", "val"),
                workers=7,
                shard_row_limit=5000,
                overwrite_output=False,
                limit_files=None,
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(sorted(result.split_results), ["train", "val"])
        self.assertEqual(result.split_results["train"]["output_rows"], 8)
        self.assertEqual(result.split_results["val"]["output_rows"], 1)

    def test_run_all_stage_entrypoint_continues_after_failure(self) -> None:
        def fake_stage_dataset(config):  # type: ignore[no-untyped-def]
            if config.dataset_id == "broken":
                raise RuntimeError("boom")
            return StageResult(
                dataset_id=config.dataset_id,
                split=config.target_split,
                source="fake-source",
                input_files=1,
                raw_candidate_count=10,
                valid_candidate_count=9,
                output_rows=5,
                shard_files=1,
                quality_event_count=0,
                manifest_path=f"/tmp/{config.dataset_id}-{config.target_split}.json",
                quality_summary_path=f"/tmp/{config.dataset_id}-{config.target_split}-summary.json",
                quality_events_path=f"/tmp/{config.dataset_id}-{config.target_split}-events.jsonl",
            )

        with patch("src.preprocess.runner.stage_dataset", side_effect=fake_stage_dataset):
            summary = runner.run_all_stage_entrypoint(
                output_root=Path("data/korean_processed/_staging"),
                target_splits=("train", "val"),
                workers=7,
                shard_row_limit=5000,
                overwrite_output=False,
                limit_files=None,
                dataset_ids=("ok-a", "broken", "ok-b"),
            )

        self.assertEqual(summary.total_datasets, 3)
        self.assertEqual(summary.succeeded_datasets, 2)
        self.assertEqual(summary.failed_datasets, 1)
        self.assertEqual(summary.results[1]["dataset_id"], "broken")
        self.assertEqual(summary.results[1]["status"], "failed")
        self.assertIn("RuntimeError: boom", summary.results[1]["error"])
        self.assertEqual(summary.results[2]["dataset_id"], "ok-b")
        self.assertEqual(summary.results[2]["status"], "success")

    def test_dataset_wrappers_use_project_root_three_levels_up(self) -> None:
        wrapper_path = Path("src/preprocess/datasets/dataset_009.py")
        content = wrapper_path.read_text(encoding="utf-8")
        self.assertIn("PROJECT_ROOT = Path(__file__).resolve().parents[3]", content)


if __name__ == "__main__":
    unittest.main()
