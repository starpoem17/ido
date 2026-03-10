from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import ensure_parent_dir


# =========================
# User configuration block
# =========================
MAX_EVENT_SAMPLES_PER_REASON = 20


@dataclass
class QualityEvent:
    dataset: str
    split: str | None
    file_path: str
    record_id: str | None
    reason_code: str
    reason_detail: str
    sample_text: str | None
    severity: str


class QualityRecorder:
    def __init__(self) -> None:
        self.events: list[QualityEvent] = []

    def add(
        self,
        *,
        dataset: str,
        split: str | None,
        file_path: str,
        record_id: str | None,
        reason_code: str,
        reason_detail: str,
        sample_text: str | None,
        severity: str,
    ) -> None:
        self.events.append(
            QualityEvent(
                dataset=dataset,
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code=reason_code,
                reason_detail=reason_detail,
                sample_text=sample_text,
                severity=severity,
            )
        )

    def extend(self, events: list[QualityEvent]) -> None:
        self.events.extend(events)

    def to_summary(self) -> dict[str, Any]:
        counts = Counter(event.reason_code for event in self.events)
        severity_counts = Counter(event.severity for event in self.events)
        samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            bucket = samples[event.reason_code]
            if len(bucket) >= MAX_EVENT_SAMPLES_PER_REASON:
                continue
            bucket.append(asdict(event))
        return {
            "total_events": len(self.events),
            "counts_by_reason": dict(sorted(counts.items())),
            "counts_by_severity": dict(sorted(severity_counts.items())),
            "samples_by_reason": dict(sorted(samples.items())),
        }

    def write_jsonl(self, path: Path) -> None:
        ensure_parent_dir(path)
        with path.open("w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
