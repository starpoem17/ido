from __future__ import annotations

import unittest
from pathlib import Path

from src.preprocess.dataset_specs import (
    DATASET_SPECS,
    _process_019,
    clean_annotations_text_for_021,
    finalize_dialog_turns,
    make_row,
    merge_consecutive_turns,
)
from src.preprocess.quality import QualityRecorder


class DatasetRuleTests(unittest.TestCase):
    def test_finalize_dialog_turns_trims_trailing_user(self) -> None:
        recorder = QualityRecorder()
        turns = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        final = finalize_dialog_turns(
            turns=turns,
            dataset="x",
            split="train",
            file_path=Path("dummy.json"),
            record_id="1",
            recorder=recorder,
        )
        self.assertEqual(final, [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}])
        self.assertEqual(recorder.events[0].reason_code, "trim_trailing_user")

    def test_merge_consecutive_turns(self) -> None:
        merged = merge_consecutive_turns(
            [
                {"role": "user", "content": "u1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a1"},
            ]
        )
        self.assertEqual(
            merged,
            [
                {"role": "user", "content": "u1 u2"},
                {"role": "assistant", "content": "a1"},
            ],
        )

    def test_make_row_for_pt_only_dataset(self) -> None:
        row = make_row(
            source=DATASET_SPECS["019"].source,
            split="train",
            content="판결문 본문",
            messages=None,
        )
        self.assertEqual(row["messages"], None)
        self.assertGreater(row["token_count"], 0)

    def test_019_allows_partial_fields(self) -> None:
        recorder = QualityRecorder()
        rows = _process_019(
            split="train",
            file_path=Path("019.json"),
            obj={
                "info": {"caseNo": "2026가단1"},
                "mentionedItems": {"rqestObjet": ["  청구 취지  "]},
                "disposal": {"disposalcontent": []},
                "assrs": {"dedatAssrs": None},
                "facts": {"bsisFacts": ["", "사실관계"]},
                "dcss": {"courtDcss": ["판단"]},
                "close": {"cnclsns": [123, "결론"]},
            },
            recorder=recorder,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "청구 취지\n사실관계\n판단\n결론")
        self.assertIsNone(rows[0]["messages"])
        self.assertCountEqual(
            [event.reason_code for event in recorder.events],
            ["skip_field", "skip_item", "skip_item"],
        )

    def test_019_skips_only_when_all_fields_empty(self) -> None:
        recorder = QualityRecorder()
        rows = _process_019(
            split="train",
            file_path=Path("019.json"),
            obj={
                "info": {"caseNo": "2026가단2"},
                "mentionedItems": {"rqestObjet": []},
                "disposal": {"disposalcontent": ["   "]},
                "assrs": {"dedatAssrs": None},
                "facts": {"bsisFacts": []},
                "dcss": {"courtDcss": [None]},
                "close": {"cnclsns": []},
            },
            recorder=recorder,
        )
        self.assertEqual(rows, [])
        self.assertEqual(recorder.events[-1].reason_code, "empty_content")

    def test_021_content_cleaning_keeps_newlines(self) -> None:
        content = clean_annotations_text_for_021("A. 첫줄\nB. 둘째 줄\n\nA : 셋째줄")
        self.assertEqual(content.splitlines(), ["첫줄", "둘째 줄", "셋째줄"])


if __name__ == "__main__":
    unittest.main()
