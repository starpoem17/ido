from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .common import (
    build_content_from_messages,
    compact_spaces,
    stable_sorted_paths,
    strip_text,
    truncate_sample,
)
from .quality import QualityEvent, QualityRecorder


MULTI_SESSION_BASE_SYSTEM = "당신은 사용자의 대화 상대로서 친절하고 긍정적으로 반응합니다."
MATH_REASONING_SYSTEM = "사용자의 질문을 읽고 단계 별로 사고하여 논리적인 답변을 제시합니다."
TEXT_PREFIX_RE = re.compile(r"^\s*(?:[AB]|\d+)\s*[.:：]\s*")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    source: str
    input_patterns: tuple[str, ...]


@dataclass(frozen=True)
class InputFileItem:
    split: str
    path: Path
    selected_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplitPlan:
    assignments: dict[str, dict[Path, tuple[str, ...]]]
    stats: dict[str, int]


@dataclass(frozen=True)
class BuiltRecord:
    record_id: str
    split_key: str
    source: str
    data_usage: str
    content: str | None
    messages: list[dict[str, str]] | None

    def to_row(self, split: str) -> dict[str, Any]:
        return {
            "source": self.source,
            "data_usage": self.data_usage,
            "split": split,
            "content": self.content,
            "messages": self.messages,
            "token_count": None,
        }


DATASET_SPECS: dict[str, DatasetSpec] = {
    "009": DatasetSpec(
        dataset_id="009",
        source="009.전문분야_기술과학_한국어 멀티세션 데이터",
        input_patterns=(
            "data/korean_raw/009.전문분야_기술과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json",
            "data/korean_raw/009.전문분야_기술과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json",
        ),
    ),
    "010": DatasetSpec(
        dataset_id="010",
        source="010.전문분야_사회과학_한국어 멀티세션 데이터",
        input_patterns=(
            "data/korean_raw/010.전문분야_사회과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json",
            "data/korean_raw/010.전문분야_사회과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json",
        ),
    ),
    "011": DatasetSpec(
        dataset_id="011",
        source="011.일상대화 한국어 멀티세션 데이터",
        input_patterns=(
            "data/korean_raw/011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json",
            "data/korean_raw/011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json",
        ),
    ),
    "019": DatasetSpec(
        dataset_id="019",
        source="019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터",
        input_patterns=(
            "data/korean_raw/019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터/01.데이터/1.Training/라벨링데이터_230510_add/**/*.json",
            "data/korean_raw/019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터/01.데이터/2.Validation/라벨링데이터_230510_add/**/*.json",
        ),
    ),
    "020": DatasetSpec(
        dataset_id="020",
        source="020.주제별 텍스트 일상 대화 데이터",
        input_patterns=(
            "data/korean_raw/020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/라벨링데이터/**/*.json",
            "data/korean_raw/020.주제별 텍스트 일상 대화 데이터/01.데이터/2.Validation/라벨링데이터/**/*.json",
        ),
    ),
    "021": DatasetSpec(
        dataset_id="021",
        source="021.용도별 목적대화 데이터",
        input_patterns=(
            "data/korean_raw/021.용도별 목적대화 데이터/01.데이터/1.Training/라벨링데이터/**/*.json",
            "data/korean_raw/021.용도별 목적대화 데이터/01.데이터/2.Validation/라벨링데이터/**/*.json",
        ),
    ),
    "023": DatasetSpec(
        dataset_id="023",
        source="023.국회 회의록 기반 지식검색 데이터",
        input_patterns=(
            "data/korean_raw/023.국회 회의록 기반 지식검색 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/**/*.json",
            "data/korean_raw/023.국회 회의록 기반 지식검색 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/**/*.json",
        ),
    ),
    "030": DatasetSpec(
        dataset_id="030",
        source="030.웹데이터 기반 한국어 말뭉치 데이터",
        input_patterns=(
            "data/korean_raw/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/**/*.json",
            "data/korean_raw/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/2.Validation/라벨링데이터/**/*.json",
        ),
    ),
    "045": DatasetSpec(
        dataset_id="045",
        source="045.지식검색 대화",
        input_patterns=(
            "data/korean_raw/045.지식검색 대화/01-1.정식개방데이터/Training/02.라벨링데이터/**/*.json",
            "data/korean_raw/045.지식검색 대화/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*.json",
        ),
    ),
    "046": DatasetSpec(
        dataset_id="046",
        source="046.공감형 대화",
        input_patterns=(
            "data/korean_raw/046.공감형 대화/01-1.정식개방데이터/Training/02.라벨링데이터/**/*.json",
            "data/korean_raw/046.공감형 대화/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*.json",
        ),
    ),
    "141": DatasetSpec(
        dataset_id="141",
        source="141.한국어 멀티세션 대화",
        input_patterns=(
            "data/korean_raw/141.한국어 멀티세션 대화/01-1.정식개방데이터/Training/02.라벨링데이터/**/*.json",
            "data/korean_raw/141.한국어 멀티세션 대화/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*.json",
        ),
    ),
    "gsm8k": DatasetSpec(
        dataset_id="gsm8k",
        source="gsm8k",
        input_patterns=(
            "data/korean_raw/gsm8k/train-00000-of-00001.parquet",
            "data/korean_raw/gsm8k/test-00000-of-00001.parquet",
        ),
    ),
    "novel24": DatasetSpec(
        dataset_id="novel24",
        source="novel24",
        input_patterns=("data/korean_raw/novel24/*.txt",),
    ),
    "nikl_newspaper_2020": DatasetSpec(
        dataset_id="nikl_newspaper_2020",
        source="국립국어원 신문 말뭉치 2020",
        input_patterns=(
            "data/korean_raw/국립국어원 신문 말뭉치 2020/*.json",
            "data/korean_raw/국립국어원 신문 말뭉치 2020(버전 1.1)/*.json",
        ),
    ),
    "nikl_spoken": DatasetSpec(
        dataset_id="nikl_spoken",
        source="국립국어원 구어 말뭉치",
        input_patterns=(
            "data/korean_raw/국립국어원 구어 말뭉치/*.json",
            "data/korean_raw/국립국어원 구어 말뭉치(버전 1.2)/*.json",
        ),
    ),
    "nikl_written": DatasetSpec(
        dataset_id="nikl_written",
        source="국립국어원 문어 말뭉치",
        input_patterns=("data/korean_raw/국립국어원 문어 말뭉치/*.json",),
    ),
    "HAERAE-HUB-KOREAN-WEBTEXT": DatasetSpec(
        dataset_id="HAERAE-HUB-KOREAN-WEBTEXT",
        source="HAERAE-HUB-KOREAN-WEBTEXT",
        input_patterns=("data/korean_raw/HAERAE-HUB-KOREAN-WEBTEXT/*.parquet",),
    ),
    "HAERAE-HUB-HR-Instruct-Math-v0.1": DatasetSpec(
        dataset_id="HAERAE-HUB-HR-Instruct-Math-v0.1",
        source="HAERAE-HUB-HR-Instruct-Math-v0.1",
        input_patterns=("data/korean_raw/HAERAE-HUB-HR-Instruct-Math-v0.1/*.parquet",),
    ),
    "namu": DatasetSpec(
        dataset_id="namu",
        source="namu",
        input_patterns=("data/korean_raw/namu/*.json",),
    ),
    "nohurry-Opus-4.6-Reasoning-3000x-filtered": DatasetSpec(
        dataset_id="nohurry-Opus-4.6-Reasoning-3000x-filtered",
        source="nohurry-Opus-4.6-Reasoning-3000x-filtered",
        input_patterns=("data/korean_raw/nohurry-Opus-4.6-Reasoning-3000x-filtered/*.jsonl",),
    ),
}

MULTI_SESSION_SYSTEM_BY_DATASET = {
    "009": "당신은 기술과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다.",
    "010": "당신은 사회과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다.",
    "011": MULTI_SESSION_BASE_SYSTEM,
}

_SPLIT_PLAN_CACHE: dict[str, DatasetSplitPlan] = {}


def _add_event(
    recorder: QualityRecorder | None,
    *,
    dataset: str,
    split: str | None,
    file_path: Path,
    record_id: str | None,
    reason_code: str,
    reason_detail: str,
    sample_text: str | None,
    severity: str,
) -> None:
    if recorder is None:
        return
    recorder.add(
        dataset=dataset,
        split=split,
        file_path=str(file_path),
        record_id=record_id,
        reason_code=reason_code,
        reason_detail=reason_detail,
        sample_text=sample_text,
        severity=severity,
    )


def list_input_files(spec: DatasetSpec) -> list[Path]:
    files: list[Path] = []
    for pattern in spec.input_patterns:
        files.extend(Path().glob(pattern))
    unique_files = stable_sorted_paths(dict.fromkeys(files))
    if spec.dataset_id == "023":
        unique_files = [path for path in unique_files if path.name.startswith("LAB_")]
    if spec.dataset_id == "novel24":
        unique_files = [
            path
            for path in unique_files
            if path.suffix == ".txt" and not path.name.startswith(".") and path.name != ".DS_Store"
        ]
    return unique_files


def stable_sorted_assignment_items(
    assignments: dict[Path, tuple[str, ...]],
) -> list[tuple[Path, tuple[str, ...]]]:
    return sorted(assignments.items(), key=lambda item: str(item[0]))


def resolve_input_files(spec: DatasetSpec, target_split: str) -> list[InputFileItem]:
    plan = get_dataset_split_plan(spec.dataset_id)
    return [
        InputFileItem(split=target_split, path=path, selected_record_ids=record_ids)
        for path, record_ids in stable_sorted_assignment_items(plan.assignments[target_split])
        if record_ids
    ]


def assign_split_records(
    *,
    source: str,
    candidates: Sequence[tuple[str, str, Path]],
) -> dict[str, dict[Path, tuple[str, ...]]]:
    ranked: list[tuple[str, str, Path]] = []
    for record_id, split_key, file_path in candidates:
        digest = hashlib.sha256(f"{source}\t{split_key}".encode("utf-8")).hexdigest()
        ranked.append((digest, record_id, file_path))
    ranked.sort(key=lambda item: (item[0], item[1], str(item[2])))
    total = len(ranked)
    train_assignments: dict[Path, list[str]] = defaultdict(list)
    val_assignments: dict[Path, list[str]] = defaultdict(list)
    if total == 0:
        return {"train": {}, "val": {}}
    train_cut = total if total == 1 else ((total * 99) + 99) // 100
    if total >= 2 and train_cut >= total:
        train_cut = total - 1
    for index, (_, record_id, file_path) in enumerate(ranked):
        target = train_assignments if index < train_cut else val_assignments
        target[file_path].append(record_id)
    return {
        "train": {path: tuple(record_ids) for path, record_ids in train_assignments.items()},
        "val": {path: tuple(record_ids) for path, record_ids in val_assignments.items()},
    }


def get_dataset_split_plan(dataset_id: str) -> DatasetSplitPlan:
    cached = _SPLIT_PLAN_CACHE.get(dataset_id)
    if cached is not None:
        return cached
    spec = DATASET_SPECS[dataset_id]
    files = list_input_files(spec)
    candidates: list[tuple[str, str, Path]] = []
    raw_candidate_count = 0
    for file_path in files:
        raw_count, file_candidates = collect_split_candidates(dataset_id, file_path)
        raw_candidate_count += raw_count
        for record_id, split_key in file_candidates:
            candidates.append((record_id, split_key, file_path))
    assignments = assign_split_records(source=spec.source, candidates=candidates)
    stats = {
        "input_files": len(files),
        "raw_candidate_count": raw_candidate_count,
        "valid_candidate_count": len(candidates),
        "pre_split_excluded_count": raw_candidate_count - len(candidates),
        "train_row_count": sum(len(record_ids) for record_ids in assignments["train"].values()),
        "val_row_count": sum(len(record_ids) for record_ids in assignments["val"].values()),
    }
    plan = DatasetSplitPlan(assignments=assignments, stats=stats)
    _SPLIT_PLAN_CACHE[dataset_id] = plan
    return plan


def get_split_plan_stats(dataset_id: str) -> dict[str, int]:
    return dict(get_dataset_split_plan(dataset_id).stats)


def clean_text_prefix(text: str) -> str:
    return TEXT_PREFIX_RE.sub("", text).strip()


def clean_annotations_text_for_021(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = strip_text(line)
        if stripped is None:
            continue
        stripped = clean_text_prefix(stripped)
        stripped = compact_spaces(stripped)
        if stripped:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def merge_consecutive_turns(turns: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"] = merged[-1]["content"] + " " + turn["content"]
        else:
            merged.append({"role": turn["role"], "content": turn["content"]})
    return merged


def finalize_dialog_turns(
    *,
    turns: Sequence[dict[str, str]],
    dataset: str,
    split: str | None,
    file_path: Path,
    record_id: str | None,
    recorder: QualityRecorder | None,
) -> list[dict[str, str]] | None:
    if not turns:
        _add_event(
            recorder,
            dataset=dataset,
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="empty_turns",
            reason_detail="no valid turns after normalization",
            sample_text=None,
            severity="skip",
        )
        return None
    if turns[0]["role"] != "user":
        _add_event(
            recorder,
            dataset=dataset,
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="bad_start_role",
            reason_detail="dialog must start with user",
            sample_text=truncate_sample(turns[0]["content"]),
            severity="skip",
        )
        return None
    for left, right in zip(turns, turns[1:]):
        if left["role"] == right["role"]:
            _add_event(
                recorder,
                dataset=dataset,
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="role_alternation_broken",
                reason_detail="adjacent turns share the same role",
                sample_text=truncate_sample(left["content"]),
                severity="skip",
            )
            return None
    output = [{"role": turn["role"], "content": turn["content"]} for turn in turns]
    if output[-1]["role"] == "user":
        removed = output.pop()
        _add_event(
            recorder,
            dataset=dataset,
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="trim_trailing_user",
            reason_detail="removed trailing user turn",
            sample_text=truncate_sample(removed["content"]),
            severity="fixup",
        )
    if not output or output[-1]["role"] != "assistant":
        _add_event(
            recorder,
            dataset=dataset,
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="no_final_assistant",
            reason_detail="dialog does not end in assistant after fixup",
            sample_text=None,
            severity="skip",
        )
        return None
    if not any(turn["role"] == "assistant" for turn in output):
        _add_event(
            recorder,
            dataset=dataset,
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="missing_assistant",
            reason_detail="assistant turn not found",
            sample_text=None,
            severity="skip",
        )
        return None
    return output


def _make_record(
    *,
    record_id: str,
    split_key: str,
    source: str,
    data_usage: str,
    content: str | None,
    messages: list[dict[str, str]] | None,
) -> BuiltRecord:
    if content is None and messages is None:
        raise ValueError("content/messages cannot both be null")
    return BuiltRecord(
        record_id=record_id,
        split_key=split_key,
        source=source,
        data_usage=data_usage,
        content=content,
        messages=messages,
    )


def _read_json(
    file_path: Path,
    *,
    dataset_id: str,
    split: str | None,
    recorder: QualityRecorder | None,
) -> Any | None:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        _add_event(
            recorder,
            dataset=dataset_id,
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="json_parse_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return None


def _extract_sentence_values(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    sentences: list[str] = []
    for item in items:
        sentence = strip_text(item.get("sentence") if isinstance(item, dict) else None)
        if sentence is not None:
            sentences.append(sentence)
    return sentences


def _collect_turn_text(
    line: dict[str, Any],
    *,
    strip_prefix: bool,
) -> str | None:
    text = strip_text(line.get("norm_text"))
    if text is not None:
        return text
    raw_text = strip_text(line.get("text"))
    if raw_text is None:
        return None
    if not strip_prefix:
        return raw_text
    cleaned = clean_text_prefix(raw_text)
    return cleaned or None


def collect_split_candidates(dataset_id: str, file_path: Path) -> tuple[int, list[tuple[str, str]]]:
    records, raw_count = _build_records_for_file(
        dataset_id=dataset_id,
        file_path=file_path,
        split=None,
        recorder=None,
    )
    return raw_count, [(record.record_id, record.split_key) for record in records]


def process_file(
    dataset_id: str,
    split: str,
    file_path: Path,
    selected_record_ids: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[QualityEvent], dict[str, int]]:
    recorder = QualityRecorder()
    stats = Counter(input_files=1, output_rows=0)
    try:
        records, raw_count = _build_records_for_file(
            dataset_id=dataset_id,
            file_path=file_path,
            split=split,
            recorder=recorder,
        )
    except Exception as exc:  # noqa: BLE001
        _add_event(
            recorder,
            dataset=dataset_id,
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="file_processing_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return [], recorder.events, dict(stats)
    allowed_record_ids = set(selected_record_ids) if selected_record_ids is not None else None
    selected_records = [
        record for record in records if allowed_record_ids is None or record.record_id in allowed_record_ids
    ]
    rows = [record.to_row(split) for record in selected_records]
    stats["raw_candidate_count"] = raw_count
    stats["valid_candidate_count"] = len(records)
    stats["output_rows"] = len(rows)
    return rows, recorder.events, dict(stats)


def _build_records_for_file(
    *,
    dataset_id: str,
    file_path: Path,
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    if dataset_id == "gsm8k":
        return _build_gsm8k_records(file_path=file_path, split=split, recorder=recorder)
    if dataset_id == "novel24":
        return _build_novel24_records(file_path=file_path, split=split, recorder=recorder)
    if dataset_id == "HAERAE-HUB-HR-Instruct-Math-v0.1":
        return _build_hr_math_records(file_path=file_path, split=split, recorder=recorder)
    if dataset_id == "HAERAE-HUB-KOREAN-WEBTEXT":
        return _build_webtext_records(file_path=file_path, split=split, recorder=recorder)
    if dataset_id == "namu":
        return _build_namu_records(file_path=file_path, split=split, recorder=recorder)
    if dataset_id == "nohurry-Opus-4.6-Reasoning-3000x-filtered":
        return _build_nohurry_records(file_path=file_path, split=split, recorder=recorder)
    obj = _read_json(file_path, dataset_id=dataset_id, split=split, recorder=recorder)
    if obj is None:
        return [], 0
    if dataset_id in {"009", "010", "011"}:
        return _build_multi_session_records(
            dataset_id=dataset_id,
            file_path=file_path,
            obj=obj,
            split=split,
            recorder=recorder,
        )
    if dataset_id == "141":
        return _build_141_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "019":
        return _build_019_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "020":
        return _build_020_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "021":
        return _build_021_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "023":
        return _build_023_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "030":
        return _build_030_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "045":
        return _build_045_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "046":
        return _build_046_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "nikl_newspaper_2020":
        return _build_nikl_newspaper_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "nikl_spoken":
        return _build_nikl_spoken_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    if dataset_id == "nikl_written":
        return _build_nikl_written_records(file_path=file_path, obj=obj, split=split, recorder=recorder)
    raise ValueError(f"unsupported dataset_id: {dataset_id}")


def _build_multi_session_records(
    *,
    dataset_id: str,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    sessions = obj.get("sessionInfo")
    if not isinstance(sessions, list):
        _add_event(
            recorder,
            dataset=dataset_id,
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_session_info",
            reason_detail="sessionInfo must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 0
    records: list[BuiltRecord] = []
    system = MULTI_SESSION_SYSTEM_BY_DATASET[dataset_id]
    for session_index, session in enumerate(sessions):
        session_id = str(session.get("sessionID") or f"{file_path.stem}:{session_index}")
        dialog = session.get("dialog")
        if not isinstance(dialog, list):
            _add_event(
                recorder,
                dataset=dataset_id,
                split=split,
                file_path=file_path,
                record_id=session_id,
                reason_code="missing_dialog",
                reason_detail="dialog must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        turns: list[dict[str, str]] = []
        bad_session = False
        for item in dialog:
            utterance = strip_text(item.get("utterance") if isinstance(item, dict) else None)
            speaker = item.get("speaker") if isinstance(item, dict) else None
            if utterance is None:
                _add_event(
                    recorder,
                    dataset=dataset_id,
                    split=split,
                    file_path=file_path,
                    record_id=session_id,
                    reason_code="invalid_utterance",
                    reason_detail="utterance must be non-empty string",
                    sample_text=None,
                    severity="skip",
                )
                bad_session = True
                break
            if speaker == "speaker1":
                role = "user"
            elif speaker == "speaker2":
                role = "assistant"
            else:
                _add_event(
                    recorder,
                    dataset=dataset_id,
                    split=split,
                    file_path=file_path,
                    record_id=session_id,
                    reason_code="invalid_speaker",
                    reason_detail=f"unsupported speaker: {speaker}",
                    sample_text=truncate_sample(utterance),
                    severity="skip",
                )
                bad_session = True
                break
            turns.append({"role": role, "content": utterance})
        if bad_session:
            continue
        final_turns = finalize_dialog_turns(
            turns=turns,
            dataset=dataset_id,
            split=split,
            file_path=file_path,
            record_id=session_id,
            recorder=recorder,
        )
        if final_turns is None:
            continue
        messages = [{"role": "system", "content": system}, *final_turns]
        records.append(
            _make_record(
                record_id=session_id,
                split_key=session_id,
                source=DATASET_SPECS[dataset_id].source,
                data_usage="SFT",
                content=build_content_from_messages(messages),
                messages=messages,
            )
        )
    return records, len(sessions)


def _build_141_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    sessions = obj.get("sessionInfo")
    if not isinstance(sessions, list):
        _add_event(
            recorder,
            dataset="141",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_session_info",
            reason_detail="sessionInfo must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 0
    feature_lines: list[str] = []
    features = obj.get("personaInfo", {}).get("clInfo", {}).get("personaFeatures")
    if isinstance(features, list):
        for feature in features:
            text = strip_text(feature)
            if text is not None:
                feature_lines.append(text)
    system = MULTI_SESSION_BASE_SYSTEM
    if feature_lines:
        system = system + "\n[clInfo persona]\n" + "\n".join(feature_lines)
    else:
        _add_event(
            recorder,
            dataset="141",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_persona_features",
            reason_detail="personaInfo.clInfo.personaFeatures missing or empty",
            sample_text=None,
            severity="fixup",
        )
    records: list[BuiltRecord] = []
    for session_index, session in enumerate(sessions):
        session_id = str(session.get("sessionID") or f"{file_path.stem}:{session_index}")
        dialog = session.get("dialog")
        if not isinstance(dialog, list):
            _add_event(
                recorder,
                dataset="141",
                split=split,
                file_path=file_path,
                record_id=session_id,
                reason_code="missing_dialog",
                reason_detail="dialog must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        turns: list[dict[str, str]] = []
        bad_session = False
        for item in dialog:
            utterance = strip_text(item.get("utterance") if isinstance(item, dict) else None)
            speaker = item.get("speaker") if isinstance(item, dict) else None
            if utterance is None:
                _add_event(
                    recorder,
                    dataset="141",
                    split=split,
                    file_path=file_path,
                    record_id=session_id,
                    reason_code="invalid_utterance",
                    reason_detail="utterance must be non-empty string",
                    sample_text=None,
                    severity="skip",
                )
                bad_session = True
                break
            if speaker == "speaker1":
                role = "user"
            elif speaker == "speaker2":
                role = "assistant"
            else:
                _add_event(
                    recorder,
                    dataset="141",
                    split=split,
                    file_path=file_path,
                    record_id=session_id,
                    reason_code="invalid_speaker",
                    reason_detail=f"unsupported speaker: {speaker}",
                    sample_text=truncate_sample(utterance),
                    severity="skip",
                )
                bad_session = True
                break
            turns.append({"role": role, "content": utterance})
        if bad_session:
            continue
        final_turns = finalize_dialog_turns(
            turns=turns,
            dataset="141",
            split=split,
            file_path=file_path,
            record_id=session_id,
            recorder=recorder,
        )
        if final_turns is None:
            continue
        messages = [{"role": "system", "content": system}, *final_turns]
        records.append(
            _make_record(
                record_id=session_id,
                split_key=session_id,
                source=DATASET_SPECS["141"].source,
                data_usage="SFT",
                content=build_content_from_messages(messages),
                messages=messages,
            )
        )
    return records, len(sessions)


def _build_019_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    case_id = str(obj.get("info", {}).get("caseNo") or file_path.stem)
    field_specs = [
        ("mentionedItems.rqestObjet", "mentionedItems", "rqestObjet"),
        ("disposal.disposalcontent", "disposal", "disposalcontent"),
        ("assrs.dedatAssrs", "assrs", "dedatAssrs"),
        ("facts.bsisFacts", "facts", "bsisFacts"),
        ("dcss.courtDcss", "dcss", "courtDcss"),
        ("close.cnclsns", "close", "cnclsns"),
        ("clauseArticle", "clauseArticle", None),
        ("comProvision", "comProvision", None),
    ]
    parts: list[str] = []
    pending_events: list[QualityEvent] = []
    for field_label, parent_key, child_key in field_specs:
        if child_key is None:
            values = obj.get(parent_key)
        else:
            parent = obj.get(parent_key)
            values = parent.get(child_key) if isinstance(parent, dict) else None
        if not isinstance(values, list):
            pending_events.append(
                QualityEvent(
                    dataset="019",
                    split=split,
                    file_path=str(file_path),
                    record_id=case_id,
                    reason_code="skip_field",
                    reason_detail=f"{field_label} ignored because value is missing or not a list",
                    sample_text=None,
                    severity="skip",
                )
            )
            continue
        for value in values:
            text = strip_text(value)
            if text is not None:
                parts.append(text)
            else:
                pending_events.append(
                    QualityEvent(
                        dataset="019",
                        split=split,
                        file_path=str(file_path),
                        record_id=case_id,
                        reason_code="skip_item",
                        reason_detail=f"{field_label} excluded a non-string or empty item",
                        sample_text=None,
                        severity="skip",
                    )
                )
    event_severity = "fixup" if parts else "skip"
    if recorder is not None:
        recorder.extend(
            [
                QualityEvent(
                    dataset=event.dataset,
                    split=event.split,
                    file_path=event.file_path,
                    record_id=event.record_id,
                    reason_code=event.reason_code,
                    reason_detail=event.reason_detail,
                    sample_text=event.sample_text,
                    severity=event_severity,
                )
                for event in pending_events
            ]
        )
    if not parts:
        _add_event(
            recorder,
            dataset="019",
            split=split,
            file_path=file_path,
            record_id=case_id,
            reason_code="empty_content",
            reason_detail=(
                "row skipped because no usable text remained across 019 candidate fields "
                "(mentionedItems.rqestObjet, disposal.disposalcontent, assrs.dedatAssrs, "
                "facts.bsisFacts, dcss.courtDcss, close.cnclsns, clauseArticle, comProvision) "
                "after normalization"
            ),
            sample_text=None,
            severity="skip",
        )
        return [], 1
    return (
        [
            _make_record(
                record_id=case_id,
                split_key=case_id,
                source=DATASET_SPECS["019"].source,
                data_usage="PT",
                content="\n".join(parts),
                messages=None,
            )
        ],
        1,
    )


def _build_020_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    info = obj.get("info")
    if not isinstance(info, list):
        _add_event(
            recorder,
            dataset="020",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_info",
            reason_detail="info must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 0
    records: list[BuiltRecord] = []
    system = "당신은 일상 대화 상대입니다. 사용자의 말에 자연스럽고 친근하게 반응하세요."
    for record_index, record in enumerate(info):
        record_id = str(record.get("id") or record.get("filename") or f"{file_path.stem}:{record_index}")
        annotations = record.get("annotations")
        if not isinstance(annotations, dict):
            _add_event(
                recorder,
                dataset="020",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_annotations",
                reason_detail="annotations must be an object",
                sample_text=None,
                severity="skip",
            )
            continue
        if annotations.get("speaker_type") != "1:1":
            _add_event(
                recorder,
                dataset="020",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="skip_speaker_type",
                reason_detail=f"speaker_type={annotations.get('speaker_type')}",
                sample_text=None,
                severity="skip",
            )
            continue
        lines = annotations.get("lines")
        if not isinstance(lines, list):
            _add_event(
                recorder,
                dataset="020",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_lines",
                reason_detail="annotations.lines must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        speaker_map: dict[str, str] = {}
        turns: list[dict[str, str]] = []
        third_speaker = False
        for line in lines:
            if not isinstance(line, dict):
                _add_event(
                    recorder,
                    dataset="020",
                    split=split,
                    file_path=file_path,
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail="line item must be an object",
                    sample_text=None,
                    severity="fixup",
                )
                continue
            speaker_id = strip_text(line.get("speaker", {}).get("id"))
            if speaker_id is None:
                _add_event(
                    recorder,
                    dataset="020",
                    split=split,
                    file_path=file_path,
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail="speaker.id missing",
                    sample_text=None,
                    severity="fixup",
                )
                continue
            text = _collect_turn_text(line, strip_prefix=True)
            if text is None:
                _add_event(
                    recorder,
                    dataset="020",
                    split=split,
                    file_path=file_path,
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail="turn text missing",
                    sample_text=None,
                    severity="fixup",
                )
                continue
            if speaker_id not in speaker_map:
                if not speaker_map:
                    speaker_map[speaker_id] = "user"
                elif len(speaker_map) == 1:
                    speaker_map[speaker_id] = "assistant"
                else:
                    _add_event(
                        recorder,
                        dataset="020",
                        split=split,
                        file_path=file_path,
                        record_id=record_id,
                        reason_code="third_speaker",
                        reason_detail=f"third speaker encountered: {speaker_id}",
                        sample_text=truncate_sample(text),
                        severity="skip",
                    )
                    third_speaker = True
                    break
            turns.append({"role": speaker_map[speaker_id], "content": text})
        if third_speaker:
            continue
        merged = merge_consecutive_turns(turns)
        final_turns = finalize_dialog_turns(
            turns=merged,
            dataset="020",
            split=split,
            file_path=file_path,
            record_id=record_id,
            recorder=recorder,
        )
        if final_turns is None:
            continue
        messages = [{"role": "system", "content": system}, *final_turns]
        records.append(
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["020"].source,
                data_usage="SFT",
                content=build_content_from_messages(messages),
                messages=messages,
            )
        )
    return records, len(info)


def _build_021_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    info = obj.get("info")
    if not isinstance(info, list):
        _add_event(
            recorder,
            dataset="021",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_info",
            reason_detail="info must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 0
    system = "당신은 콜센터 상담원입니다. 사용자의 문의에 정확하고 친절하게 답변하세요."
    records: list[BuiltRecord] = []
    for record_index, record in enumerate(info):
        record_id = str(record.get("id") or record.get("filename") or f"{file_path.stem}:{record_index}")
        annotations = record.get("annotations")
        if not isinstance(annotations, dict):
            _add_event(
                recorder,
                dataset="021",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_annotations",
                reason_detail="annotations must be an object",
                sample_text=None,
                severity="skip",
            )
            continue
        content = strip_text(annotations.get("text"))
        if content is None:
            _add_event(
                recorder,
                dataset="021",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_annotations_text",
                reason_detail="annotations.text must be non-empty string",
                sample_text=None,
                severity="skip",
            )
            continue
        lines = annotations.get("lines")
        if not isinstance(lines, list):
            _add_event(
                recorder,
                dataset="021",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_lines",
                reason_detail="annotations.lines must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        turns: list[dict[str, str]] = []
        for line in lines:
            if not isinstance(line, dict):
                _add_event(
                    recorder,
                    dataset="021",
                    split=split,
                    file_path=file_path,
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail="line item must be an object",
                    sample_text=None,
                    severity="fixup",
                )
                continue
            speaker_id = line.get("speaker", {}).get("id")
            text = _collect_turn_text(line, strip_prefix=True)
            if text is None:
                _add_event(
                    recorder,
                    dataset="021",
                    split=split,
                    file_path=file_path,
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail="turn text missing",
                    sample_text=None,
                    severity="fixup",
                )
                continue
            if speaker_id == "B":
                role = "user"
            elif speaker_id == "A":
                role = "assistant"
            else:
                _add_event(
                    recorder,
                    dataset="021",
                    split=split,
                    file_path=file_path,
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail=f"unsupported speaker.id={speaker_id}",
                    sample_text=truncate_sample(text),
                    severity="fixup",
                )
                continue
            turns.append({"role": role, "content": text})
        first_assistant_index = next(
            (index for index, turn in enumerate(turns) if turn["role"] == "assistant"),
            None,
        )
        if first_assistant_index is None:
            _add_event(
                recorder,
                dataset="021",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_assistant",
                reason_detail="assistant turn not found before FT cleanup",
                sample_text=None,
                severity="skip",
            )
            continue
        _add_event(
            recorder,
            dataset="021",
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="drop_initial_assistant",
            reason_detail="removed the first assistant greeting turn",
            sample_text=truncate_sample(turns[first_assistant_index]["content"]),
            severity="fixup",
        )
        ft_turns = turns[:first_assistant_index] + turns[first_assistant_index + 1 :]
        merged = merge_consecutive_turns(ft_turns)
        final_turns = finalize_dialog_turns(
            turns=merged,
            dataset="021",
            split=split,
            file_path=file_path,
            record_id=record_id,
            recorder=recorder,
        )
        if final_turns is None:
            continue
        messages = [{"role": "system", "content": system}, *final_turns]
        records.append(
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["021"].source,
                data_usage="SFT",
                content=content,
                messages=messages,
            )
        )
    return records, len(info)


def _build_023_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    record_id = str(obj.get("id") or obj.get("question_number") or file_path.stem)
    context = strip_text(obj.get("context"))
    question = strip_text(obj.get("question", {}).get("comment"))
    answer = strip_text(obj.get("answer", {}).get("comment"))
    if context is None or question is None or answer is None:
        _add_event(
            recorder,
            dataset="023",
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="missing_fields",
            reason_detail="context/question.comment/answer.comment must be non-empty",
            sample_text=None,
            severity="skip",
        )
        return [], 1
    messages = [
        {
            "role": "system",
            "content": "당신은 국회회의록 기반으로 대답을 하는 위원이야. 질문에 대해서 사실 근거에 기반해서 대답해줘.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return (
        [
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["023"].source,
                data_usage="SFT",
                content=context,
                messages=messages,
            )
        ],
        1,
    )


def _build_030_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    named_entity = obj.get("named_entity")
    if not isinstance(named_entity, list):
        _add_event(
            recorder,
            dataset="030",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_named_entity",
            reason_detail="named_entity must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 0
    records: list[BuiltRecord] = []
    for article_index, article in enumerate(named_entity):
        record_id = str(article.get("id") or f"{file_path.stem}:{article_index}")
        title_sentences = _extract_sentence_values(article.get("title") if isinstance(article, dict) else None)
        body_sentences = _extract_sentence_values(article.get("content") if isinstance(article, dict) else None)
        if not title_sentences:
            _add_event(
                recorder,
                dataset="030",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_title",
                reason_detail="named_entity.title.sentence must contain at least one value",
                sample_text=None,
                severity="skip",
            )
            continue
        if not body_sentences:
            _add_event(
                recorder,
                dataset="030",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="empty_body",
                reason_detail="named_entity.content.sentence all empty",
                sample_text=None,
                severity="skip",
            )
            continue
        title = title_sentences[0]
        body = " ".join(body_sentences)
        records.append(
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["030"].source,
                data_usage="PT",
                content=title + "\n" + body,
                messages=None,
            )
        )
    return records, len(named_entity)


def _collect_reference_text(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    parts: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        text = strip_text(item.get("value"))
        if text is not None:
            parts.append(text)
    if not parts:
        return None
    return " ".join(parts)


def _build_045_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    record_id = str(obj.get("info", {}).get("id") or file_path.stem)
    utterances = obj.get("utterances")
    if not isinstance(utterances, list):
        _add_event(
            recorder,
            dataset="045",
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="missing_utterances",
            reason_detail="utterances must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 1
    normalized_turns: list[dict[str, str]] = []
    user_evidences: list[str] = []
    for item in utterances:
        if not isinstance(item, dict):
            _add_event(
                recorder,
                dataset="045",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="skip_turn",
                reason_detail="utterance item must be an object",
                sample_text=None,
                severity="fixup",
            )
            continue
        role = item.get("role")
        text = strip_text(item.get("text"))
        if text is None:
            _add_event(
                recorder,
                dataset="045",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="skip_turn",
                reason_detail="text missing",
                sample_text=None,
                severity="fixup",
            )
            continue
        if role == "질문자":
            normalized_turns.append({"role": "user", "content": text})
            user_evidences.append(_collect_reference_text(item.get("reference_text")) or "")
        elif role == "전문가":
            normalized_turns.append({"role": "assistant", "content": text})
        else:
            _add_event(
                recorder,
                dataset="045",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="skip_turn",
                reason_detail=f"unsupported role={role}",
                sample_text=truncate_sample(text),
                severity="fixup",
            )
    final_turns = finalize_dialog_turns(
        turns=normalized_turns,
        dataset="045",
        split=split,
        file_path=file_path,
        record_id=record_id,
        recorder=recorder,
    )
    if final_turns is None:
        return [], 1
    evidence_index = 0
    for turn in final_turns:
        if turn["role"] != "user":
            continue
        evidence = user_evidences[evidence_index] if evidence_index < len(user_evidences) else ""
        evidence_index += 1
        if evidence:
            turn["content"] = turn["content"] + "\n근거: " + evidence
    messages = [
        {"role": "system", "content": "user의 질문에 대해서 텍스트 근거 기반으로 대답을 하는 전문가야"},
        *final_turns,
    ]
    return (
        [
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["045"].source,
                data_usage="SFT",
                content=build_content_from_messages(messages),
                messages=messages,
            )
        ],
        1,
    )


def _build_046_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    info = obj.get("info")
    utterances = obj.get("utterances")
    record_id = str(obj.get("info", {}).get("id") or file_path.stem)
    if not isinstance(info, dict) or not isinstance(utterances, list):
        _add_event(
            recorder,
            dataset="046",
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="invalid_root",
            reason_detail="info must be object and utterances must be list",
            sample_text=None,
            severity="skip",
        )
        return [], 1
    situation = strip_text(info.get("situation"))
    behaviors_raw = info.get("listener_behavior")
    behaviors = [text for behavior in behaviors_raw or [] for text in [strip_text(behavior)] if text is not None]
    if situation is None or not behaviors:
        _add_event(
            recorder,
            dataset="046",
            split=split,
            file_path=file_path,
            record_id=record_id,
            reason_code="invalid_system_fields",
            reason_detail="situation/listener_behavior missing",
            sample_text=None,
            severity="skip",
        )
        return [], 1
    turns: list[dict[str, str]] = []
    for item in utterances:
        if not isinstance(item, dict):
            _add_event(
                recorder,
                dataset="046",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="skip_turn",
                reason_detail="utterance item must be an object",
                sample_text=None,
                severity="fixup",
            )
            continue
        text = strip_text(item.get("text"))
        role = item.get("role")
        if text is None:
            _add_event(
                recorder,
                dataset="046",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="skip_turn",
                reason_detail="text missing",
                sample_text=None,
                severity="fixup",
            )
            continue
        if role == "speaker":
            mapped_role = "user"
        elif role == "listener":
            mapped_role = "assistant"
        else:
            _add_event(
                recorder,
                dataset="046",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="skip_turn",
                reason_detail=f"unsupported role={role}",
                sample_text=truncate_sample(text),
                severity="fixup",
            )
            continue
        turns.append({"role": mapped_role, "content": text})
    final_turns = finalize_dialog_turns(
        turns=turns,
        dataset="046",
        split=split,
        file_path=file_path,
        record_id=record_id,
        recorder=recorder,
    )
    if final_turns is None:
        return [], 1
    system = situation + "에 대해서 " + ", ".join(behaviors) + "에 맞추어서 대답을 해주는 친구가 되어줘"
    messages = [{"role": "system", "content": system}, *final_turns]
    return (
        [
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["046"].source,
                data_usage="SFT",
                content=build_content_from_messages(messages),
                messages=messages,
            )
        ],
        1,
    )


def _build_nikl_newspaper_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    documents = obj.get("document")
    if not isinstance(documents, list):
        _add_event(
            recorder,
            dataset="nikl_newspaper_2020",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_document",
            reason_detail="document must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 0
    records: list[BuiltRecord] = []
    for index, document in enumerate(documents):
        record_id = str(document.get("id") or f"{file_path.stem}:{index}") if isinstance(document, dict) else f"{file_path.stem}:{index}"
        if not isinstance(document, dict):
            _add_event(
                recorder,
                dataset="nikl_newspaper_2020",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="invalid_document",
                reason_detail="document item must be an object",
                sample_text=None,
                severity="skip",
            )
            continue
        paragraphs = document.get("paragraph")
        if not isinstance(paragraphs, list):
            _add_event(
                recorder,
                dataset="nikl_newspaper_2020",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_paragraph",
                reason_detail="document.paragraph must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        valid_forms: list[str] = []
        skipped_events: list[QualityEvent] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            form_value = paragraph.get("form") if isinstance(paragraph, dict) else None
            form_text = strip_text(form_value)
            if form_text is not None:
                valid_forms.append(form_text)
                continue
            skipped_events.append(
                QualityEvent(
                    dataset="nikl_newspaper_2020",
                    split=split,
                    file_path=str(file_path),
                    record_id=record_id,
                    reason_code="skip_paragraph",
                    reason_detail=f"paragraph[{paragraph_index}].form missing or empty",
                    sample_text=truncate_sample(form_value) if isinstance(form_value, str) else None,
                    severity="skip",
                )
            )
        if not valid_forms:
            if recorder is not None:
                recorder.extend(skipped_events)
            _add_event(
                recorder,
                dataset="nikl_newspaper_2020",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="empty_paragraphs",
                reason_detail="no usable paragraph.form remained after normalization",
                sample_text=None,
                severity="skip",
            )
            continue
        title = valid_forms[0]
        body_paragraphs = valid_forms[1:]
        if not body_paragraphs:
            if recorder is not None:
                recorder.extend(skipped_events)
            _add_event(
                recorder,
                dataset="nikl_newspaper_2020",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="title_only_document",
                reason_detail="document has a title paragraph but no body paragraphs",
                sample_text=truncate_sample(title),
                severity="skip",
            )
            continue
        if recorder is not None and skipped_events:
            recorder.extend(
                [
                    QualityEvent(
                        dataset=event.dataset,
                        split=event.split,
                        file_path=event.file_path,
                        record_id=event.record_id,
                        reason_code=event.reason_code,
                        reason_detail=event.reason_detail,
                        sample_text=event.sample_text,
                        severity="fixup",
                    )
                    for event in skipped_events
                ]
            )
        records.append(
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["nikl_newspaper_2020"].source,
                data_usage="PT",
                content=f"제목: {title}\n내용: " + "\n".join(body_paragraphs),
                messages=None,
            )
        )
    return records, len(documents)


def _build_nikl_spoken_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    documents = obj.get("document")
    if not isinstance(documents, list):
        _add_event(
            recorder,
            dataset="nikl_spoken",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_document",
            reason_detail="document must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 0
    records: list[BuiltRecord] = []
    for index, document in enumerate(documents):
        record_id = str(document.get("id") or f"{file_path.stem}:{index}") if isinstance(document, dict) else f"{file_path.stem}:{index}"
        if not isinstance(document, dict):
            _add_event(
                recorder,
                dataset="nikl_spoken",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="invalid_document",
                reason_detail="document item must be an object",
                sample_text=None,
                severity="skip",
            )
            continue
        utterances = document.get("utterance")
        if not isinstance(utterances, list):
            _add_event(
                recorder,
                dataset="nikl_spoken",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_utterance",
                reason_detail="document.utterance must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        valid_forms: list[str] = []
        skipped_events: list[QualityEvent] = []
        for utterance_index, utterance in enumerate(utterances):
            form_value = utterance.get("form") if isinstance(utterance, dict) else None
            form_text = strip_text(form_value)
            if form_text is not None:
                valid_forms.append(form_text)
                continue
            skipped_events.append(
                QualityEvent(
                    dataset="nikl_spoken",
                    split=split,
                    file_path=str(file_path),
                    record_id=record_id,
                    reason_code="skip_utterance",
                    reason_detail=f"utterance[{utterance_index}].form missing or empty",
                    sample_text=truncate_sample(form_value) if isinstance(form_value, str) else None,
                    severity="skip",
                )
            )
        if not valid_forms:
            if recorder is not None:
                recorder.extend(skipped_events)
            _add_event(
                recorder,
                dataset="nikl_spoken",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="empty_utterances",
                reason_detail="no usable utterance.form remained after normalization",
                sample_text=None,
                severity="skip",
            )
            continue
        if recorder is not None and skipped_events:
            recorder.extend(
                [
                    QualityEvent(
                        dataset=event.dataset,
                        split=event.split,
                        file_path=event.file_path,
                        record_id=event.record_id,
                        reason_code=event.reason_code,
                        reason_detail=event.reason_detail,
                        sample_text=event.sample_text,
                        severity="fixup",
                    )
                    for event in skipped_events
                ]
            )
        records.append(
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["nikl_spoken"].source,
                data_usage="PT",
                content="\n".join(valid_forms),
                messages=None,
            )
        )
    return records, len(documents)


def _build_nikl_written_records(
    *,
    file_path: Path,
    obj: dict[str, Any],
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    documents = obj.get("document")
    if not isinstance(documents, list):
        _add_event(
            recorder,
            dataset="nikl_written",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="missing_document",
            reason_detail="document must be a list",
            sample_text=None,
            severity="skip",
        )
        return [], 0
    records: list[BuiltRecord] = []
    for index, document in enumerate(documents):
        record_id = str(document.get("id") or f"{file_path.stem}:{index}") if isinstance(document, dict) else f"{file_path.stem}:{index}"
        if not isinstance(document, dict):
            _add_event(
                recorder,
                dataset="nikl_written",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="invalid_document",
                reason_detail="document item must be an object",
                sample_text=None,
                severity="skip",
            )
            continue
        metadata = document.get("metadata")
        title = strip_text(metadata.get("title") if isinstance(metadata, dict) else None)
        if title is None:
            _add_event(
                recorder,
                dataset="nikl_written",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_title",
                reason_detail="document.metadata.title must be a non-empty string",
                sample_text=None,
                severity="skip",
            )
            continue
        paragraphs = document.get("paragraph")
        if not isinstance(paragraphs, list):
            _add_event(
                recorder,
                dataset="nikl_written",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_paragraph",
                reason_detail="document.paragraph must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        valid_forms: list[str] = []
        skipped_events: list[QualityEvent] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            form_value = paragraph.get("form") if isinstance(paragraph, dict) else None
            form_text = strip_text(form_value)
            if form_text is not None:
                valid_forms.append(form_text)
                continue
            skipped_events.append(
                QualityEvent(
                    dataset="nikl_written",
                    split=split,
                    file_path=str(file_path),
                    record_id=record_id,
                    reason_code="skip_paragraph",
                    reason_detail=f"paragraph[{paragraph_index}].form missing or empty",
                    sample_text=truncate_sample(form_value) if isinstance(form_value, str) else None,
                    severity="skip",
                )
            )
        if not valid_forms:
            if recorder is not None:
                recorder.extend(skipped_events)
            _add_event(
                recorder,
                dataset="nikl_written",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="empty_paragraphs",
                reason_detail="no usable paragraph.form remained after normalization",
                sample_text=None,
                severity="skip",
            )
            continue
        if recorder is not None and skipped_events:
            recorder.extend(
                [
                    QualityEvent(
                        dataset=event.dataset,
                        split=event.split,
                        file_path=event.file_path,
                        record_id=event.record_id,
                        reason_code=event.reason_code,
                        reason_detail=event.reason_detail,
                        sample_text=event.sample_text,
                        severity="fixup",
                    )
                    for event in skipped_events
                ]
            )
        records.append(
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["nikl_written"].source,
                data_usage="PT",
                content=f"제목: {title}\n내용: " + "\n".join(valid_forms),
                messages=None,
            )
        )
    return records, len(documents)


def _build_gsm8k_records(
    *,
    file_path: Path,
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    try:
        table = pq.read_table(file_path, columns=["question", "answer"])
    except Exception as exc:  # noqa: BLE001
        _add_event(
            recorder,
            dataset="gsm8k",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="parquet_read_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return [], 0
    items = table.to_pylist()
    records: list[BuiltRecord] = []
    system = "당신은 수학 문제를 입력받으면 문제 풀이 과정을 포함하여 정답을 출력합니다."
    for index, item in enumerate(items):
        record_id = f"{file_path.name}:{index}"
        question = strip_text(item.get("question"))
        answer = strip_text(item.get("answer"))
        if question is None or answer is None:
            _add_event(
                recorder,
                dataset="gsm8k",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="invalid_row",
                reason_detail="question/answer must be non-empty strings",
                sample_text=None,
                severity="skip",
            )
            continue
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        records.append(
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["gsm8k"].source,
                data_usage="SFT",
                content=question + "\n" + answer,
                messages=messages,
            )
        )
    return records, len(items)


def _build_novel24_records(
    *,
    file_path: Path,
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _add_event(
            recorder,
            dataset="novel24",
            split=split,
            file_path=file_path,
            record_id=file_path.name,
            reason_code="file_read_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return [], 1
    content = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        _add_event(
            recorder,
            dataset="novel24",
            split=split,
            file_path=file_path,
            record_id=file_path.name,
            reason_code="empty_content",
            reason_detail="txt content became empty after normalization",
            sample_text=None,
            severity="skip",
        )
        return [], 1
    return (
        [
            _make_record(
                record_id=file_path.name,
                split_key=file_path.name,
                source=DATASET_SPECS["novel24"].source,
                data_usage="PT",
                content=content,
                messages=None,
            )
        ],
        1,
    )


def _build_hr_math_records(
    *,
    file_path: Path,
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    try:
        table = pq.read_table(file_path, columns=["instruction", "response"])
    except Exception as exc:  # noqa: BLE001
        _add_event(
            recorder,
            dataset="HAERAE-HUB-HR-Instruct-Math-v0.1",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="parquet_read_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return [], 0
    items = table.to_pylist()
    records: list[BuiltRecord] = []
    for index, item in enumerate(items):
        record_id = str(item.get("id") or f"{file_path.name}:{index}")
        instruction = strip_text(item.get("instruction"))
        response = strip_text(item.get("response"))
        if instruction is None:
            _add_event(
                recorder,
                dataset="HAERAE-HUB-HR-Instruct-Math-v0.1",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_instruction",
                reason_detail="instruction must be non-empty string",
                sample_text=None,
                severity="skip",
            )
            continue
        if response is None:
            _add_event(
                recorder,
                dataset="HAERAE-HUB-HR-Instruct-Math-v0.1",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="missing_response",
                reason_detail="response must be non-empty string",
                sample_text=None,
                severity="skip",
            )
            continue
        messages = [
            {"role": "system", "content": MATH_REASONING_SYSTEM},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        records.append(
            _make_record(
                record_id=record_id,
                split_key=record_id,
                source=DATASET_SPECS["HAERAE-HUB-HR-Instruct-Math-v0.1"].source,
                data_usage="REASONING",
                content=instruction + "\n" + response,
                messages=messages,
            )
        )
    return records, len(items)


def _build_webtext_records(
    *,
    file_path: Path,
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    try:
        table = pq.read_table(file_path, columns=["text", "source"])
    except Exception as exc:  # noqa: BLE001
        _add_event(
            recorder,
            dataset="HAERAE-HUB-KOREAN-WEBTEXT",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="parquet_read_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return [], 0
    items = table.to_pylist()
    records: list[BuiltRecord] = []
    for index, item in enumerate(items):
        record_id = f"{file_path.name}:{index}"
        split_key = str(item.get("source") or record_id)
        content = strip_text(item.get("text"))
        if content is None:
            _add_event(
                recorder,
                dataset="HAERAE-HUB-KOREAN-WEBTEXT",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="empty_text",
                reason_detail="text must be non-empty string",
                sample_text=None,
                severity="skip",
            )
            continue
        records.append(
            _make_record(
                record_id=record_id,
                split_key=split_key,
                source=DATASET_SPECS["HAERAE-HUB-KOREAN-WEBTEXT"].source,
                data_usage="PT",
                content=content,
                messages=None,
            )
        )
    return records, len(items)


def _build_namu_records(
    *,
    file_path: Path,
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    obj = _read_json(file_path, dataset_id="namu", split=split, recorder=recorder)
    if obj is None:
        return [], 0
    raw_items = obj if isinstance(obj, list) else [obj]
    records: list[BuiltRecord] = []
    for index, item in enumerate(raw_items):
        record_id = f"{file_path.stem}:{index}"
        if not isinstance(item, dict):
            _add_event(
                recorder,
                dataset="namu",
                split=split,
                file_path=file_path,
                record_id=record_id,
                reason_code="malformed_canonical_like_row",
                reason_detail="row must be an object",
                sample_text=None,
                severity="skip",
            )
            continue
        source = strip_text(item.get("source")) or f"namu.{file_path.stem}"
        content = strip_text(item.get("content"))
        if content is None:
            _add_event(
                recorder,
                dataset="namu",
                split=split,
                file_path=file_path,
                record_id=source,
                reason_code="empty_content",
                reason_detail="content must be non-empty string",
                sample_text=None,
                severity="skip",
            )
            continue
        records.append(
            _make_record(
                record_id=source,
                split_key=source,
                source=source,
                data_usage="PT",
                content=content,
                messages=None,
            )
        )
    return records, len(raw_items)


def _build_nohurry_records(
    *,
    file_path: Path,
    split: str | None,
    recorder: QualityRecorder | None,
) -> tuple[list[BuiltRecord], int]:
    records: list[BuiltRecord] = []
    raw_count = 0
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                raw_count += 1
                try:
                    item = json.loads(stripped_line)
                except Exception as exc:  # noqa: BLE001
                    _add_event(
                        recorder,
                        dataset="nohurry-Opus-4.6-Reasoning-3000x-filtered",
                        split=split,
                        file_path=file_path,
                        record_id=f"{file_path.name}:{line_index}",
                        reason_code="jsonl_parse_error",
                        reason_detail=f"{type(exc).__name__}: {exc}",
                        sample_text=truncate_sample(stripped_line),
                        severity="skip",
                    )
                    continue
                record_id = str(item.get("id") or f"{file_path.name}:{line_index}")
                problem = strip_text(item.get("problem"))
                thinking = strip_text(item.get("thinking"))
                solution = strip_text(item.get("solution"))
                if problem is None:
                    _add_event(
                        recorder,
                        dataset="nohurry-Opus-4.6-Reasoning-3000x-filtered",
                        split=split,
                        file_path=file_path,
                        record_id=record_id,
                        reason_code="missing_problem",
                        reason_detail="problem must be non-empty string",
                        sample_text=None,
                        severity="skip",
                    )
                    continue
                if thinking is None or solution is None:
                    _add_event(
                        recorder,
                        dataset="nohurry-Opus-4.6-Reasoning-3000x-filtered",
                        split=split,
                        file_path=file_path,
                        record_id=record_id,
                        reason_code="missing_reasoning_text",
                        reason_detail="thinking and solution must be non-empty strings",
                        sample_text=None,
                        severity="skip",
                    )
                    continue
                assistant = thinking + "\n" + solution
                messages = [
                    {"role": "system", "content": MATH_REASONING_SYSTEM},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": assistant},
                ]
                records.append(
                    _make_record(
                        record_id=record_id,
                        split_key=record_id,
                        source=DATASET_SPECS["nohurry-Opus-4.6-Reasoning-3000x-filtered"].source,
                        data_usage="REASONING",
                        content=problem + "\n" + assistant,
                        messages=messages,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        _add_event(
            recorder,
            dataset="nohurry-Opus-4.6-Reasoning-3000x-filtered",
            split=split,
            file_path=file_path,
            record_id=None,
            reason_code="jsonl_read_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return [], raw_count
    return records, raw_count
