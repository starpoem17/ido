from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.preprocess.dataset_specs import (
    DATASET_SPECS,
    _process_019,
    _process_020,
    _process_021,
    clean_annotations_text_for_021,
    finalize_dialog_turns,
    make_row,
    merge_consecutive_turns,
)
from src.preprocess.quality import QualityRecorder


class DatasetRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compute_token_count_patcher = patch("src.preprocess.dataset_specs.compute_token_count", return_value=1)
        self.compute_token_count_patcher.start()

    def tearDown(self) -> None:
        self.compute_token_count_patcher.stop()

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
                "clauseArticle": ["조항 본문"],
                "comProvision": ["공통 규정"],
            },
            recorder=recorder,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "청구 취지\n사실관계\n판단\n결론\n조항 본문\n공통 규정")
        self.assertIsNone(rows[0]["messages"])
        self.assertCountEqual(
            [event.reason_code for event in recorder.events],
            ["skip_field", "skip_item", "skip_item"],
        )
        self.assertTrue(all(event.severity == "fixup" for event in recorder.events))
        self.assertIn(
            "assrs.dedatAssrs ignored because value is missing or not a list",
            [event.reason_detail for event in recorder.events],
        )

    def test_019_keeps_rows_with_clause_article_and_com_provision_only(self) -> None:
        recorder = QualityRecorder()
        rows = _process_019(
            split="train",
            file_path=Path("019.json"),
            obj={
                "info": {"caseNo": "2026가단3"},
                "clauseArticle": ["제1조"],
                "comProvision": ["공통규정"],
            },
            recorder=recorder,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "제1조\n공통규정")
        self.assertEqual(len(recorder.events), 6)
        self.assertTrue(all(event.reason_code == "skip_field" for event in recorder.events))
        self.assertTrue(all(event.severity == "fixup" for event in recorder.events))

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
                "clauseArticle": [],
                "comProvision": ["   "],
            },
            recorder=recorder,
        )
        self.assertEqual(rows, [])
        self.assertEqual(recorder.events[-1].reason_code, "empty_content")
        self.assertEqual(
            recorder.events[-1].reason_detail,
            (
                "row skipped because no usable text remained across 019 candidate fields "
                "(mentionedItems.rqestObjet, disposal.disposalcontent, assrs.dedatAssrs, "
                "facts.bsisFacts, dcss.courtDcss, close.cnclsns, clauseArticle, comProvision) "
                "after normalization"
            ),
        )
        self.assertTrue(all(event.severity == "skip" for event in recorder.events))

    def test_021_content_cleaning_keeps_newlines(self) -> None:
        content = clean_annotations_text_for_021("A. 첫줄\nB. 둘째 줄\n\nA : 셋째줄")
        self.assertEqual(content.splitlines(), ["첫줄", "둘째 줄", "셋째줄"])

    def test_021_salvages_pt_content_when_bad_start_role_occurs(self) -> None:
        recorder = QualityRecorder()
        rows = _process_021(
            split="train",
            file_path=Path("021.json"),
            obj={
                "info": [
                    {
                        "id": "2026-1",
                        "annotations": {
                            "text": "B. 문의 내용\nA. 상담 안내",
                            "lines": [
                                {"speaker": {"id": "A"}, "norm_text": "A. 첫 상담 멘트"},
                                {"speaker": {"id": "A"}, "norm_text": "A. 추가 안내"},
                                {"speaker": {"id": "B"}, "norm_text": "B. 문의 내용"},
                            ],
                        },
                    }
                ]
            },
            recorder=recorder,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "문의 내용\n상담 안내")
        self.assertIsNone(rows[0]["messages"])
        self.assertEqual(recorder.events[-1].reason_code, "bad_start_role")

    def test_021_keeps_messages_for_valid_dialog(self) -> None:
        recorder = QualityRecorder()
        rows = _process_021(
            split="train",
            file_path=Path("021.json"),
            obj={
                "info": [
                    {
                        "id": "2026-2",
                        "annotations": {
                            "text": "B. 배송이 늦어요\nA. 확인해드릴게요",
                            "lines": [
                                {"speaker": {"id": "A"}, "text": "A. 상담 시작"},
                                {"speaker": {"id": "B"}, "text": "B. 배송이 늦어요"},
                                {"speaker": {"id": "A"}, "text": "A. 확인해드릴게요"},
                            ],
                        },
                    }
                ]
            },
            recorder=recorder,
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["messages"])
        self.assertEqual(
            rows[0]["messages"],
            [
                {"role": "system", "content": "당신은 콜센터 상담원입니다. 사용자의 문의에 정확하고 친절하게 답변하세요."},
                {"role": "user", "content": "배송이 늦어요"},
                {"role": "assistant", "content": "확인해드릴게요"},
            ],
        )

    def test_021_still_drops_when_assistant_missing(self) -> None:
        recorder = QualityRecorder()
        rows = _process_021(
            split="train",
            file_path=Path("021.json"),
            obj={
                "info": [
                    {
                        "id": "2026-3",
                        "annotations": {
                            "text": "B. 문의만 남음",
                            "lines": [
                                {"speaker": {"id": "B"}, "norm_text": "B. 문의만 남음"},
                            ],
                        },
                    }
                ]
            },
            recorder=recorder,
        )
        self.assertEqual(rows, [])
        self.assertEqual(recorder.events[-1].reason_code, "missing_assistant")

    def test_020_salvages_multispeaker_content_as_pt_only(self) -> None:
        recorder = QualityRecorder()
        rows = _process_020(
            split="train",
            file_path=Path("020.json"),
            obj={
                "info": [
                    {
                        "id": "2026-4",
                        "annotations": {
                            "speaker_type": "다자간 대화",
                            "lines": [
                                {"speaker": {"id": "1"}, "text": "1 : 첫 번째 발화"},
                                {"speaker": {"id": "2"}, "text": "2 : 두 번째 발화"},
                                {"speaker": {"id": "3"}, "text": "3 : 세 번째 발화"},
                            ],
                        },
                    }
                ]
            },
            recorder=recorder,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "첫 번째 발화 두 번째 발화 세 번째 발화")
        self.assertIsNone(rows[0]["messages"])
        skip_speaker_type = next(event for event in recorder.events if event.reason_code == "skip_speaker_type")
        self.assertEqual(skip_speaker_type.severity, "fixup")

    def test_020_salvages_third_speaker_content_as_pt_only(self) -> None:
        recorder = QualityRecorder()
        rows = _process_020(
            split="train",
            file_path=Path("020.json"),
            obj={
                "info": [
                    {
                        "id": "2026-5",
                        "annotations": {
                            "speaker_type": "1:1",
                            "lines": [
                                {"speaker": {"id": "1"}, "text": "1 : 안녕하세요"},
                                {"speaker": {"id": "2"}, "text": "2 : 반갑습니다"},
                                {"speaker": {"id": "3"}, "text": "3 : 저도 왔어요"},
                            ],
                        },
                    }
                ]
            },
            recorder=recorder,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "안녕하세요 반갑습니다 저도 왔어요")
        self.assertIsNone(rows[0]["messages"])
        third_speaker = next(event for event in recorder.events if event.reason_code == "third_speaker")
        self.assertEqual(third_speaker.severity, "fixup")

    def test_020_keeps_messages_for_valid_one_to_one_dialog(self) -> None:
        recorder = QualityRecorder()
        rows = _process_020(
            split="train",
            file_path=Path("020.json"),
            obj={
                "info": [
                    {
                        "id": "2026-6",
                        "annotations": {
                            "speaker_type": "1:1",
                            "lines": [
                                {"speaker": {"id": "1"}, "text": "1 : 안녕"},
                                {"speaker": {"id": "2"}, "text": "2 : 반가워"},
                            ],
                        },
                    }
                ]
            },
            recorder=recorder,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["messages"],
            [
                {"role": "system", "content": "당신은 일상 대화 상대입니다. 사용자의 말에 자연스럽고 친근하게 반응하세요."},
                {"role": "user", "content": "안녕"},
                {"role": "assistant", "content": "반가워"},
            ],
        )

    def test_020_keeps_skip_when_multispeaker_has_no_usable_turns(self) -> None:
        recorder = QualityRecorder()
        rows = _process_020(
            split="train",
            file_path=Path("020.json"),
            obj={
                "info": [
                    {
                        "id": "2026-7",
                        "annotations": {
                            "speaker_type": "다자간 대화",
                            "lines": [
                                {"speaker": {"id": None}, "text": None},
                            ],
                        },
                    }
                ]
            },
            recorder=recorder,
        )
        self.assertEqual(rows, [])
        skip_speaker_type = next(event for event in recorder.events if event.reason_code == "skip_speaker_type")
        self.assertEqual(skip_speaker_type.severity, "skip")


if __name__ == "__main__":
    unittest.main()
