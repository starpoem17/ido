from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocess.dataset_specs import (
    DATASET_SPECS,
    _build_019_records,
    _build_020_records,
    _build_021_records,
    _build_030_records,
    _build_045_records,
    _build_hr_math_records,
    _build_namu_records,
    _build_nikl_newspaper_records,
    _build_nikl_spoken_records,
    _build_nikl_written_records,
    _build_nohurry_records,
    _build_novel24_records,
    _build_webtext_records,
    finalize_dialog_turns,
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

    def test_019_allows_partial_fields(self) -> None:
        recorder = QualityRecorder()
        records, raw_count = _build_019_records(
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
            split="train",
            recorder=recorder,
        )
        self.assertEqual(raw_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "청구 취지\n사실관계\n판단\n결론\n조항 본문\n공통 규정")
        self.assertIsNone(records[0].messages)
        self.assertCountEqual(
            [event.reason_code for event in recorder.events],
            ["skip_field", "skip_item", "skip_item"],
        )
        self.assertTrue(all(event.severity == "fixup" for event in recorder.events))

    def test_020_skips_record_when_third_speaker_appears(self) -> None:
        recorder = QualityRecorder()
        records, _ = _build_020_records(
            file_path=Path("020.json"),
            obj={
                "info": [
                    {
                        "id": "rec-1",
                        "annotations": {
                            "speaker_type": "1:1",
                            "lines": [
                                {"speaker": {"id": "A"}, "norm_text": "안녕"},
                                {"speaker": {"id": "B"}, "norm_text": "반가워"},
                                {"speaker": {"id": "C"}, "norm_text": "끼어들기"},
                            ],
                        },
                    }
                ]
            },
            split="train",
            recorder=recorder,
        )
        self.assertEqual(records, [])
        self.assertEqual(recorder.events[-1].reason_code, "third_speaker")

    def test_021_keeps_annotations_text_as_content_and_requires_ft_dialog(self) -> None:
        recorder = QualityRecorder()
        records, _ = _build_021_records(
            file_path=Path("021.json"),
            obj={
                "info": [
                    {
                        "id": "call-1",
                        "annotations": {
                            "text": "A. 인사\nB. 문의\nA. 답변",
                            "lines": [
                                {"speaker": {"id": "A"}, "norm_text": "인사"},
                                {"speaker": {"id": "B"}, "norm_text": "문의"},
                                {"speaker": {"id": "A"}, "norm_text": "답변"},
                            ],
                        },
                    }
                ]
            },
            split="train",
            recorder=recorder,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "A. 인사\nB. 문의\nA. 답변")
        self.assertEqual(records[0].data_usage, "SFT")
        self.assertEqual(records[0].messages[1:], [{"role": "user", "content": "문의"}, {"role": "assistant", "content": "답변"}])

    def test_030_builds_pt_rows_without_messages(self) -> None:
        recorder = QualityRecorder()
        records, _ = _build_030_records(
            file_path=Path("030.json"),
            obj={
                "named_entity": [
                    {
                        "title": [{"sentence": "기사 제목"}],
                        "content": [{"sentence": "첫 문장"}, {"sentence": "둘째 문장"}],
                    }
                ]
            },
            split="train",
            recorder=recorder,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "기사 제목\n첫 문장 둘째 문장")
        self.assertIsNone(records[0].messages)
        self.assertEqual(records[0].data_usage, "PT")

    def test_045_uses_only_question_reference_text(self) -> None:
        recorder = QualityRecorder()
        records, _ = _build_045_records(
            file_path=Path("045.json"),
            obj={
                "info": {"id": "row-1"},
                "utterances": [
                    {"role": "질문자", "text": "질문", "reference_text": [{"value": "근거1"}, {"value": "근거2"}]},
                    {"role": "전문가", "text": "답변", "reference_text": [{"value": "무시"}]},
                ],
            },
            split="train",
            recorder=recorder,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].messages[1]["content"], "질문\n근거: 근거1 근거2")
        self.assertEqual(records[0].messages[2]["content"], "답변")

    def test_nikl_newspaper_uses_first_valid_paragraph_as_title(self) -> None:
        recorder = QualityRecorder()
        records, _ = _build_nikl_newspaper_records(
            file_path=Path("news.json"),
            obj={
                "document": [
                    {
                        "id": "doc-1",
                        "paragraph": [
                            {"form": "  기사 제목  "},
                            {"form": "  첫 번째 본문  "},
                            {"form": ""},
                            {"form": "두 번째 본문"},
                        ],
                    }
                ]
            },
            split="train",
            recorder=recorder,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "제목: 기사 제목\n내용: 첫 번째 본문\n두 번째 본문")
        skip_event = next(event for event in recorder.events if event.reason_code == "skip_paragraph")
        self.assertEqual(skip_event.severity, "fixup")

    def test_nikl_spoken_builds_pt_content(self) -> None:
        recorder = QualityRecorder()
        records, _ = _build_nikl_spoken_records(
            file_path=Path("spoken.json"),
            obj={
                "document": [
                    {
                        "id": "doc-1",
                        "utterance": [
                            {"form": " 첫 발화 "},
                            {"form": ""},
                            {"form": "둘째 발화"},
                        ],
                    }
                ]
            },
            split="train",
            recorder=recorder,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "첫 발화\n둘째 발화")

    def test_nikl_written_builds_content_from_title_and_paragraphs(self) -> None:
        recorder = QualityRecorder()
        records, _ = _build_nikl_written_records(
            file_path=Path("written.json"),
            obj={
                "document": [
                    {
                        "id": "doc-1",
                        "metadata": {"title": "  폭력과 존엄 사이  "},
                        "paragraph": [
                            {"form": " 들어가는 말 "},
                            {"form": ""},
                            {"form": "잠깐 내린 눈"},
                        ],
                    }
                ]
            },
            split="train",
            recorder=recorder,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "제목: 폭력과 존엄 사이\n내용: 들어가는 말\n잠깐 내린 눈")

    def test_novel24_keeps_whole_file_as_single_pt_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "novel.txt"
            path.write_text("제목\n\n본문 1\n본문 2\n", encoding="utf-8")
            recorder = QualityRecorder()
            records, raw_count = _build_novel24_records(
                file_path=path,
                split="train",
                recorder=recorder,
            )
        self.assertEqual(raw_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "제목\n\n본문 1\n본문 2")
        self.assertEqual(records[0].source, DATASET_SPECS["novel24"].source)

    def test_hr_math_builds_reasoning_rows_from_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "hr.parquet"
            table = pa.Table.from_pylist(
                [{"instruction": "문제", "response": "풀이"}],
                schema=pa.schema(
                    [
                        pa.field("instruction", pa.string(), nullable=False),
                        pa.field("response", pa.string(), nullable=False),
                    ]
                ),
            )
            pq.write_table(table, path)
            recorder = QualityRecorder()
            records, raw_count = _build_hr_math_records(file_path=path, split="train", recorder=recorder)
        self.assertEqual(raw_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].data_usage, "REASONING")
        self.assertEqual(records[0].content, "문제\n풀이")

    def test_webtext_ignores_original_token_count_and_keeps_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "web.parquet"
            table = pa.Table.from_pylist(
                [{"text": "웹 텍스트", "source": "row-1"}],
                schema=pa.schema(
                    [
                        pa.field("text", pa.string(), nullable=False),
                        pa.field("source", pa.string(), nullable=False),
                    ]
                ),
            )
            pq.write_table(table, path)
            recorder = QualityRecorder()
            records, raw_count = _build_webtext_records(file_path=path, split="train", recorder=recorder)
        self.assertEqual(raw_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, DATASET_SPECS["HAERAE-HUB-KOREAN-WEBTEXT"].source)
        self.assertEqual(records[0].content, "웹 텍스트")

    def test_namu_rebuilds_canonical_like_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "namu.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "source": "namu.문서.섹션",
                            "data_usage": "PT",
                            "split": "train",
                            "content": "상식 본문",
                            "messages": None,
                            "token_count": None,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            recorder = QualityRecorder()
            records, raw_count = _build_namu_records(file_path=path, split="train", recorder=recorder)
        self.assertEqual(raw_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "namu.문서.섹션")
        self.assertEqual(records[0].content, "상식 본문")

    def test_nohurry_builds_reasoning_row_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "reasoning.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "row-1",
                        "problem": "문제",
                        "thinking": "생각",
                        "solution": "정답",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            recorder = QualityRecorder()
            records, raw_count = _build_nohurry_records(file_path=path, split="train", recorder=recorder)
        self.assertEqual(raw_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "문제\n생각\n정답")
        self.assertEqual(records[0].data_usage, "REASONING")


if __name__ == "__main__":
    unittest.main()
