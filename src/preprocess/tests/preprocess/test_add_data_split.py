from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.preprocess.dataset_specs import process_file


class ProcessFileSelectionTests(unittest.TestCase):
    def test_process_file_filters_selected_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "009.json"
            path.write_text(
                json.dumps(
                    {
                        "sessionInfo": [
                            {
                                "sessionID": "session-a",
                                "dialog": [
                                    {"speaker": "speaker1", "utterance": "질문 A"},
                                    {"speaker": "speaker2", "utterance": "답변 A"},
                                ],
                            },
                            {
                                "sessionID": "session-b",
                                "dialog": [
                                    {"speaker": "speaker1", "utterance": "질문 B"},
                                    {"speaker": "speaker2", "utterance": "답변 B"},
                                ],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rows, events, stats = process_file(
                "009",
                "val",
                path,
                selected_record_ids=("session-b",),
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "val")
        self.assertEqual(rows[0]["content"], "질문 B 답변 B")
        self.assertEqual(stats["raw_candidate_count"], 2)
        self.assertEqual(stats["valid_candidate_count"], 2)
        self.assertEqual(stats["output_rows"], 1)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
