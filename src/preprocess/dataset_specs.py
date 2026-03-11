from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .common import (
    build_content_from_messages,
    compact_spaces,
    compute_token_count,
    stable_sorted_paths,
    strip_text,
    truncate_sample,
)
from .quality import QualityEvent, QualityRecorder


MULTI_SESSION_BASE_SYSTEM = "당신은 사용자의 대화 상대로서 친절하고 긍정적으로 반응합니다."
ARTICLE_SYSTEM = "당신은 기사를 읽고 제목을 짓습니다. 내용을 요약하고, 사람들의 눈길을 끄는 제목을 작성합니다."


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    source: str
    train_patterns: tuple[str, ...]
    val_patterns: tuple[str, ...]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "009": DatasetSpec(
        dataset_id="009",
        source="009.전문분야_기술과학_한국어 멀티세션 데이터",
        train_patterns=(
            "data/korean_raw/009.전문분야_기술과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json",
        ),
        val_patterns=(
            "data/korean_raw/009.전문분야_기술과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json",
        ),
    ),
    "010": DatasetSpec(
        dataset_id="010",
        source="010.전문분야_사회과학_한국어 멀티세션 데이터",
        train_patterns=(
            "data/korean_raw/010.전문분야_사회과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json",
        ),
        val_patterns=(
            "data/korean_raw/010.전문분야_사회과학_한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json",
        ),
    ),
    "011": DatasetSpec(
        dataset_id="011",
        source="011.일상대화 한국어 멀티세션 데이터",
        train_patterns=(
            "data/korean_raw/011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/*.json",
        ),
        val_patterns=(
            "data/korean_raw/011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/*.json",
        ),
    ),
    "019": DatasetSpec(
        dataset_id="019",
        source="019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터",
        train_patterns=(
            "data/korean_raw/019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터/01.데이터/1.Training/라벨링데이터_230510_add/**/*.json",
        ),
        val_patterns=(
            "data/korean_raw/019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터/01.데이터/2.Validation/라벨링데이터_230510_add/**/*.json",
        ),
    ),
    "020": DatasetSpec(
        dataset_id="020",
        source="020.주제별 텍스트 일상 대화 데이터",
        train_patterns=(
            "data/korean_raw/020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/라벨링데이터/**/*.json",
        ),
        val_patterns=(
            "data/korean_raw/020.주제별 텍스트 일상 대화 데이터/01.데이터/2.Validation/라벨링데이터/**/*.json",
        ),
    ),
    "021": DatasetSpec(
        dataset_id="021",
        source="021.용도별 목적대화 데이터",
        train_patterns=(
            "data/korean_raw/021.용도별 목적대화 데이터/01.데이터/1.Training/라벨링데이터/**/*.json",
        ),
        val_patterns=(
            "data/korean_raw/021.용도별 목적대화 데이터/01.데이터/2.Validation/라벨링데이터/**/*.json",
        ),
    ),
    "023": DatasetSpec(
        dataset_id="023",
        source="023.국회 회의록 기반 지식검색 데이터",
        train_patterns=(
            "data/korean_raw/023.국회 회의록 기반 지식검색 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/**/*.json",
        ),
        val_patterns=(
            "data/korean_raw/023.국회 회의록 기반 지식검색 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/**/*.json",
        ),
    ),
    "030": DatasetSpec(
        dataset_id="030",
        source="030.웹데이터 기반 한국어 말뭉치 데이터",
        train_patterns=(
            "data/korean_raw/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/**/*.json",
        ),
        val_patterns=(
            "data/korean_raw/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/2.Validation/라벨링데이터/**/*.json",
        ),
    ),
    "045": DatasetSpec(
        dataset_id="045",
        source="045.지식검색 대화",
        train_patterns=(
            "data/korean_raw/045.지식검색 대화/01-1.정식개방데이터/Training/02.라벨링데이터/**/*.json",
        ),
        val_patterns=(
            "data/korean_raw/045.지식검색 대화/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*.json",
        ),
    ),
    "046": DatasetSpec(
        dataset_id="046",
        source="046.공감형 대화",
        train_patterns=(
            "data/korean_raw/046.공감형 대화/01-1.정식개방데이터/Training/02.라벨링데이터/**/*.json",
        ),
        val_patterns=(
            "data/korean_raw/046.공감형 대화/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*.json",
        ),
    ),
    "141": DatasetSpec(
        dataset_id="141",
        source="141.한국어 멀티세션 대화",
        train_patterns=(
            "data/korean_raw/141.한국어 멀티세션 대화/01-1.정식개방데이터/Training/02.라벨링데이터/**/*.json",
        ),
        val_patterns=(
            "data/korean_raw/141.한국어 멀티세션 대화/01-1.정식개방데이터/Validation/02.라벨링데이터/**/*.json",
        ),
    ),
    "gsm8k": DatasetSpec(
        dataset_id="gsm8k",
        source="gsm8k",
        train_patterns=("data/korean_raw/gsm8k/train-00000-of-00001.parquet",),
        val_patterns=("data/korean_raw/gsm8k/test-00000-of-00001.parquet",),
    ),
    "novel24": DatasetSpec(
        dataset_id="novel24",
        source="novel24",
        train_patterns=("data/korean_raw/novel24/*.txt",),
        val_patterns=(),
    ),
}

MULTI_SESSION_CONFIG = {
    "009": {
        "system": "당신은 기술과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다.",
    },
    "010": {
        "system": "당신은 사회과학 전문 비서입니다. 사용자의 질문에 친절하고 과학적으로 대답합니다.",
    },
    "011": {
        "system": MULTI_SESSION_BASE_SYSTEM,
    },
}

TEXT_PREFIX_RE = re.compile(r"^\s*(?:[AB]|\d+)\s*[.:：]\s*")


def resolve_input_files(spec: DatasetSpec, target_split: str) -> list[tuple[str, Path]]:
    if spec.dataset_id == "novel24":
        files = [
            path
            for path in stable_sorted_paths(Path().glob("data/korean_raw/novel24/*.txt"))
            if path.name != ".DS_Store" and not path.name.startswith(".")
        ]
        train_files, val_files = split_novel24_files(files)
        if target_split == "train":
            return [("train", path) for path in files if path in train_files]
        if target_split == "val":
            return [("val", path) for path in files if path in val_files]
        raise ValueError(f"unsupported split for novel24: {target_split}")

    patterns = spec.train_patterns if target_split == "train" else spec.val_patterns
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path().glob(pattern))
    sorted_files = stable_sorted_paths(files)
    if spec.dataset_id == "023":
        sorted_files = [path for path in sorted_files if path.name.startswith("LAB_")]
    return [(target_split, path) for path in sorted_files]


def clean_text_prefix(text: str) -> str:
    return TEXT_PREFIX_RE.sub("", text).strip()


def merge_consecutive_turns(turns: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"] = merged[-1]["content"] + " " + turn["content"]
        else:
            merged.append(dict(turn))
    return merged


def finalize_dialog_turns(
    *,
    turns: Sequence[dict[str, str]],
    dataset: str,
    split: str,
    file_path: Path,
    record_id: str | None,
    recorder: QualityRecorder,
) -> list[dict[str, str]] | None:
    if not turns:
        recorder.add(
            dataset=dataset,
            split=split,
            file_path=str(file_path),
            record_id=record_id,
            reason_code="empty_turns",
            reason_detail="no valid turns after normalization",
            sample_text=None,
            severity="skip",
        )
        return None
    if turns[0]["role"] != "user":
        recorder.add(
            dataset=dataset,
            split=split,
            file_path=str(file_path),
            record_id=record_id,
            reason_code="bad_start_role",
            reason_detail="dialog must start with user",
            sample_text=truncate_sample(turns[0]["content"]),
            severity="skip",
        )
        return None
    for left, right in zip(turns, turns[1:]):
        if left["role"] == right["role"]:
            recorder.add(
                dataset=dataset,
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="role_alternation_broken",
                reason_detail="adjacent turns share the same role",
                sample_text=truncate_sample(left["content"]),
                severity="skip",
            )
            return None
    output = [dict(turn) for turn in turns]
    if output[-1]["role"] == "user":
        removed = output.pop()
        recorder.add(
            dataset=dataset,
            split=split,
            file_path=str(file_path),
            record_id=record_id,
            reason_code="trim_trailing_user",
            reason_detail="removed trailing user turn",
            sample_text=truncate_sample(removed["content"]),
            severity="fixup",
        )
    if not output or output[-1]["role"] != "assistant":
        recorder.add(
            dataset=dataset,
            split=split,
            file_path=str(file_path),
            record_id=record_id,
            reason_code="no_final_assistant",
            reason_detail="dialog does not end in assistant after fixup",
            sample_text=None,
            severity="skip",
        )
        return None
    if not any(turn["role"] == "assistant" for turn in output):
        recorder.add(
            dataset=dataset,
            split=split,
            file_path=str(file_path),
            record_id=record_id,
            reason_code="missing_assistant",
            reason_detail="assistant turn not found",
            sample_text=None,
            severity="skip",
        )
        return None
    return output


def make_row(
    *,
    source: str,
    split: str,
    content: str | None,
    messages: list[dict[str, str]] | None,
) -> dict[str, Any]:
    token_count = compute_token_count(content=content, messages=messages)
    return {
        "source": source,
        "split": split,
        "content": content,
        "messages": messages,
        "token_count": token_count,
    }


def process_file(
    dataset_id: str,
    split: str,
    file_path: Path,
) -> tuple[list[dict[str, Any]], list[QualityEvent], dict[str, int]]:
    recorder = QualityRecorder()
    stats = Counter(input_files=1, output_rows=0)
    try:
        if dataset_id == "gsm8k":
            rows = _process_gsm8k(file_path=file_path, split=split, recorder=recorder)
        elif dataset_id == "novel24":
            rows = _process_novel24(file_path=file_path, recorder=recorder)
        else:
            rows = _process_json_dataset(
                dataset_id=dataset_id,
                split=split,
                file_path=file_path,
                recorder=recorder,
            )
    except Exception as exc:  # noqa: BLE001
        recorder.add(
            dataset=dataset_id,
            split=split,
            file_path=str(file_path),
            record_id=None,
            reason_code="file_processing_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        rows = []
    stats["output_rows"] = len(rows)
    return rows, recorder.events, dict(stats)


def _read_json(file_path: Path, dataset_id: str, split: str, recorder: QualityRecorder) -> Any | None:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        recorder.add(
            dataset=dataset_id,
            split=split,
            file_path=str(file_path),
            record_id=None,
            reason_code="json_parse_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return None


def _process_json_dataset(
    *,
    dataset_id: str,
    split: str,
    file_path: Path,
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    obj = _read_json(file_path, dataset_id, split, recorder)
    if obj is None:
        return []
    if dataset_id in {"009", "010", "011"}:
        return _process_multi_session(dataset_id, split, file_path, obj, recorder)
    if dataset_id == "141":
        return _process_141(split, file_path, obj, recorder)
    if dataset_id == "019":
        return _process_019(split, file_path, obj, recorder)
    if dataset_id == "020":
        return _process_020(split, file_path, obj, recorder)
    if dataset_id == "021":
        return _process_021(split, file_path, obj, recorder)
    if dataset_id == "023":
        return _process_023(split, file_path, obj, recorder)
    if dataset_id == "030":
        return _process_030(split, file_path, obj, recorder)
    if dataset_id == "045":
        return _process_045(split, file_path, obj, recorder)
    if dataset_id == "046":
        return _process_046(split, file_path, obj, recorder)
    raise ValueError(f"unsupported dataset_id: {dataset_id}")


def _process_multi_session(
    dataset_id: str,
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    sessions = obj.get("sessionInfo")
    if not isinstance(sessions, list):
        recorder.add(
            dataset=dataset_id,
            split=split,
            file_path=str(file_path),
            record_id=None,
            reason_code="missing_session_info",
            reason_detail="sessionInfo must be a list",
            sample_text=None,
            severity="skip",
        )
        return []
    system = MULTI_SESSION_CONFIG[dataset_id]["system"]
    source = DATASET_SPECS[dataset_id].source
    rows: list[dict[str, Any]] = []
    for session in sessions:
        session_id = str(session.get("sessionID")) if session.get("sessionID") is not None else None
        dialog = session.get("dialog")
        if not isinstance(dialog, list):
            recorder.add(
                dataset=dataset_id,
                split=split,
                file_path=str(file_path),
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
            utterance = strip_text(item.get("utterance"))
            speaker = item.get("speaker")
            if utterance is None:
                recorder.add(
                    dataset=dataset_id,
                    split=split,
                    file_path=str(file_path),
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
                recorder.add(
                    dataset=dataset_id,
                    split=split,
                    file_path=str(file_path),
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
        rows.append(
            make_row(
                source=source,
                split=split,
                content=build_content_from_messages(messages),
                messages=messages,
            )
        )
    return rows


def _process_141(
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    sessions = obj.get("sessionInfo")
    if not isinstance(sessions, list):
        recorder.add(
            dataset="141",
            split=split,
            file_path=str(file_path),
            record_id=None,
            reason_code="missing_session_info",
            reason_detail="sessionInfo must be a list",
            sample_text=None,
            severity="skip",
        )
        return []
    features = obj.get("personaInfo", {}).get("clInfo", {}).get("personaFeatures")
    feature_lines = []
    if isinstance(features, list):
        for feature in features:
            text = strip_text(feature)
            if text is not None:
                feature_lines.append(text)
    system = MULTI_SESSION_BASE_SYSTEM
    if feature_lines:
        system = system + "\n[clInfo persona]\n" + "\n".join(feature_lines)
    else:
        recorder.add(
            dataset="141",
            split=split,
            file_path=str(file_path),
            record_id=None,
            reason_code="missing_persona_features",
            reason_detail="personaInfo.clInfo.personaFeatures missing or empty",
            sample_text=None,
            severity="fixup",
        )
    rows: list[dict[str, Any]] = []
    for session in sessions:
        session_id = str(session.get("sessionID")) if session.get("sessionID") is not None else None
        dialog = session.get("dialog")
        if not isinstance(dialog, list):
            recorder.add(
                dataset="141",
                split=split,
                file_path=str(file_path),
                record_id=session_id,
                reason_code="missing_dialog",
                reason_detail="dialog must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        turns: list[dict[str, str]] = []
        bad = False
        for item in dialog:
            utterance = strip_text(item.get("utterance"))
            speaker = item.get("speaker")
            if utterance is None:
                recorder.add(
                    dataset="141",
                    split=split,
                    file_path=str(file_path),
                    record_id=session_id,
                    reason_code="invalid_utterance",
                    reason_detail="utterance must be non-empty string",
                    sample_text=None,
                    severity="skip",
                )
                bad = True
                break
            if speaker == "speaker1":
                role = "user"
            elif speaker == "speaker2":
                role = "assistant"
            else:
                recorder.add(
                    dataset="141",
                    split=split,
                    file_path=str(file_path),
                    record_id=session_id,
                    reason_code="invalid_speaker",
                    reason_detail=f"unsupported speaker: {speaker}",
                    sample_text=truncate_sample(utterance),
                    severity="skip",
                )
                bad = True
                break
            turns.append({"role": role, "content": utterance})
        if bad:
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
        rows.append(
            make_row(
                source=DATASET_SPECS["141"].source,
                split=split,
                content=build_content_from_messages(messages),
                messages=messages,
            )
        )
    return rows


def _process_019(
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    case_id = str(obj.get("info", {}).get("caseNo") or file_path.name)
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
    candidate_fields = ", ".join(field_label for field_label, _, _ in field_specs)
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
    partial_severity = "fixup" if parts else "skip"
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
                severity=partial_severity,
            )
            for event in pending_events
        ]
    )
    if not parts:
        recorder.add(
            dataset="019",
            split=split,
            file_path=str(file_path),
            record_id=case_id,
            reason_code="empty_content",
            reason_detail=(
                "row skipped because no usable text remained across 019 candidate fields "
                f"({candidate_fields}) after normalization"
            ),
            sample_text=None,
            severity="skip",
        )
        return []
    content = "\n".join(parts)
    return [
        make_row(
            source=DATASET_SPECS["019"].source,
            split=split,
            content=content,
            messages=None,
        )
    ]


def _process_020(
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    info = obj.get("info")
    if not isinstance(info, list):
        recorder.add(
            dataset="020",
            split=split,
            file_path=str(file_path),
            record_id=None,
            reason_code="missing_info",
            reason_detail="info must be a list",
            sample_text=None,
            severity="skip",
        )
        return []
    rows: list[dict[str, Any]] = []
    for record in info:
        record_id = str(record.get("id") or record.get("filename") or file_path.name)
        annotations = record.get("annotations")
        if not isinstance(annotations, dict):
            recorder.add(
                dataset="020",
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="missing_annotations",
                reason_detail="annotations must be an object",
                sample_text=None,
                severity="skip",
            )
            continue
        if annotations.get("speaker_type") != "1:1":
            recorder.add(
                dataset="020",
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="skip_speaker_type",
                reason_detail=f"speaker_type={annotations.get('speaker_type')}",
                sample_text=None,
                severity="skip",
            )
            continue
        lines = annotations.get("lines")
        if not isinstance(lines, list):
            recorder.add(
                dataset="020",
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="missing_lines",
                reason_detail="annotations.lines must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        speaker_map: dict[str, str] = {}
        turns: list[dict[str, str]] = []
        bad = False
        for line in lines:
            speaker_id = strip_text(line.get("speaker", {}).get("id"))
            if speaker_id is None:
                recorder.add(
                    dataset="020",
                    split=split,
                    file_path=str(file_path),
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail="speaker.id missing",
                    sample_text=None,
                    severity="skip",
                )
                continue
            text = strip_text(line.get("norm_text"))
            if text is None:
                raw = strip_text(line.get("text"))
                if raw is not None:
                    text = clean_text_prefix(raw)
                    text = text if text else None
            if text is None:
                recorder.add(
                    dataset="020",
                    split=split,
                    file_path=str(file_path),
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail="turn text missing",
                    sample_text=None,
                    severity="skip",
                )
                continue
            if speaker_id not in speaker_map:
                if not speaker_map:
                    speaker_map[speaker_id] = "user"
                elif len(speaker_map) == 1:
                    speaker_map[speaker_id] = "assistant"
                else:
                    recorder.add(
                        dataset="020",
                        split=split,
                        file_path=str(file_path),
                        record_id=record_id,
                        reason_code="third_speaker",
                        reason_detail=f"third speaker encountered: {speaker_id}",
                        sample_text=truncate_sample(text),
                        severity="skip",
                    )
                    bad = True
                    break
            turns.append({"role": speaker_map[speaker_id], "content": text})
        if bad:
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
        system = "당신은 일상 대화 상대입니다. 사용자의 말에 자연스럽고 친근하게 반응하세요."
        messages = [{"role": "system", "content": system}, *final_turns]
        rows.append(
            make_row(
                source=DATASET_SPECS["020"].source,
                split=split,
                content=build_content_from_messages(messages),
                messages=messages,
            )
        )
    return rows


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


def _process_021(
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    info = obj.get("info")
    if not isinstance(info, list):
        recorder.add(
            dataset="021",
            split=split,
            file_path=str(file_path),
            record_id=None,
            reason_code="missing_info",
            reason_detail="info must be a list",
            sample_text=None,
            severity="skip",
        )
        return []
    rows: list[dict[str, Any]] = []
    system = "당신은 콜센터 상담원입니다. 사용자의 문의에 정확하고 친절하게 답변하세요."
    for record in info:
        record_id = str(record.get("id") or record.get("filename") or file_path.name)
        annotations = record.get("annotations")
        if not isinstance(annotations, dict):
            recorder.add(
                dataset="021",
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="missing_annotations",
                reason_detail="annotations must be an object",
                sample_text=None,
                severity="skip",
            )
            continue
        raw_content = strip_text(annotations.get("text"))
        if raw_content is None:
            recorder.add(
                dataset="021",
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="missing_annotations_text",
                reason_detail="annotations.text must be non-empty string",
                sample_text=None,
                severity="skip",
            )
            continue
        content = clean_annotations_text_for_021(raw_content)
        if not content:
            recorder.add(
                dataset="021",
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="empty_cleaned_content",
                reason_detail="annotations.text became empty after cleaning",
                sample_text=None,
                severity="skip",
            )
            continue
        lines = annotations.get("lines")
        if not isinstance(lines, list):
            recorder.add(
                dataset="021",
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="missing_lines",
                reason_detail="annotations.lines must be a list",
                sample_text=None,
                severity="skip",
            )
            continue
        turns: list[dict[str, str]] = []
        for line in lines:
            speaker_id = line.get("speaker", {}).get("id")
            text = strip_text(line.get("norm_text"))
            if text is None:
                raw_text = strip_text(line.get("text"))
                if raw_text is not None:
                    text = clean_text_prefix(raw_text)
                    text = text if text else None
            if text is None:
                recorder.add(
                    dataset="021",
                    split=split,
                    file_path=str(file_path),
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail="turn text missing",
                    sample_text=None,
                    severity="skip",
                )
                continue
            if speaker_id == "B":
                role = "user"
            elif speaker_id == "A":
                role = "assistant"
            else:
                recorder.add(
                    dataset="021",
                    split=split,
                    file_path=str(file_path),
                    record_id=record_id,
                    reason_code="skip_turn",
                    reason_detail=f"unsupported speaker.id={speaker_id}",
                    sample_text=truncate_sample(text),
                    severity="skip",
                )
                continue
            turns.append({"role": role, "content": text})
        first_assistant_index = next(
            (index for index, turn in enumerate(turns) if turn["role"] == "assistant"),
            None,
        )
        if first_assistant_index is None:
            recorder.add(
                dataset="021",
                split=split,
                file_path=str(file_path),
                record_id=record_id,
                reason_code="missing_assistant",
                reason_detail="assistant turn not found before FT cleanup",
                sample_text=None,
                severity="skip",
            )
            continue
        ft_turns = turns[:first_assistant_index] + turns[first_assistant_index + 1 :]
        merged = merge_consecutive_turns(ft_turns)
        event_count_before_finalize = len(recorder.events)
        final_turns = finalize_dialog_turns(
            turns=merged,
            dataset="021",
            split=split,
            file_path=file_path,
            record_id=record_id,
            recorder=recorder,
        )
        if final_turns is None:
            new_events = recorder.events[event_count_before_finalize:]
            if new_events and new_events[-1].reason_code == "bad_start_role":
                rows.append(
                    make_row(
                        source=DATASET_SPECS["021"].source,
                        split=split,
                        content=content,
                        messages=None,
                    )
                )
                continue
            continue
        messages = [{"role": "system", "content": system}, *final_turns]
        rows.append(
            make_row(
                source=DATASET_SPECS["021"].source,
                split=split,
                content=content,
                messages=messages,
            )
        )
    return rows


def _process_023(
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    context = strip_text(obj.get("context"))
    question = strip_text(obj.get("question", {}).get("comment"))
    answer = strip_text(obj.get("answer", {}).get("comment"))
    if context is None or question is None or answer is None:
        recorder.add(
            dataset="023",
            split=split,
            file_path=str(file_path),
            record_id=str(obj.get("id") or obj.get("question_number") or file_path.name),
            reason_code="missing_fields",
            reason_detail="context/question.comment/answer.comment must be non-empty",
            sample_text=None,
            severity="skip",
        )
        return []
    messages = [
        {
            "role": "system",
            "content": "당신은 국회회의록 기반으로 대답을 하는 위원이야. 질문에 대해서 사실 근거에 기반해서 대답해줘.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return [
        make_row(
            source=DATASET_SPECS["023"].source,
            split=split,
            content=context,
            messages=messages,
        )
    ]


def _process_030(
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    named_entity = obj.get("named_entity")
    if not isinstance(named_entity, list):
        recorder.add(
            dataset="030",
            split=split,
            file_path=str(file_path),
            record_id=None,
            reason_code="missing_named_entity",
            reason_detail="named_entity must be a list",
            sample_text=None,
            severity="skip",
        )
        return []
    rows: list[dict[str, Any]] = []
    for article_index, article in enumerate(named_entity):
        contents = article.get("content")
        titles = article.get("title")
        if not isinstance(contents, list) or not isinstance(titles, list):
            recorder.add(
                dataset="030",
                split=split,
                file_path=str(file_path),
                record_id=str(article_index),
                reason_code="invalid_article_arrays",
                reason_detail="content/title must be lists",
                sample_text=None,
                severity="skip",
            )
            continue
        body_sentences = []
        for item in contents:
            sentence = strip_text(item.get("sentence") if isinstance(item, dict) else None)
            if sentence is not None:
                body_sentences.append(sentence)
        if not body_sentences:
            recorder.add(
                dataset="030",
                split=split,
                file_path=str(file_path),
                record_id=str(article_index),
                reason_code="empty_body",
                reason_detail="content[*].sentence all empty",
                sample_text=None,
                severity="skip",
            )
            continue
        body = " ".join(body_sentences)
        for title_index, item in enumerate(titles):
            title = strip_text(item.get("sentence") if isinstance(item, dict) else None)
            if title is None:
                recorder.add(
                    dataset="030",
                    split=split,
                    file_path=str(file_path),
                    record_id=f"{article_index}:{title_index}",
                    reason_code="empty_title",
                    reason_detail="title sentence is empty",
                    sample_text=None,
                    severity="skip",
                )
                continue
            messages = [
                {"role": "system", "content": ARTICLE_SYSTEM},
                {"role": "user", "content": body},
                {"role": "assistant", "content": title},
            ]
            rows.append(
                make_row(
                    source=DATASET_SPECS["030"].source,
                    split=split,
                    content=title + "\n" + body,
                    messages=messages,
                )
            )
    return rows


def _collect_reference_text(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    parts = []
    for item in values:
        if not isinstance(item, dict):
            continue
        text = strip_text(item.get("value"))
        if text is not None:
            parts.append(text)
    if not parts:
        return None
    return " ".join(parts)


def _process_045(
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    utterances = obj.get("utterances")
    if not isinstance(utterances, list):
        recorder.add(
            dataset="045",
            split=split,
            file_path=str(file_path),
            record_id=str(obj.get("info", {}).get("id") or file_path.name),
            reason_code="missing_utterances",
            reason_detail="utterances must be a list",
            sample_text=None,
            severity="skip",
        )
        return []
    turns: list[dict[str, str]] = []
    assistant_evidence_after_user: list[str | None] = []
    for item in utterances:
        role = item.get("role")
        text = strip_text(item.get("text"))
        if text is None:
            recorder.add(
                dataset="045",
                split=split,
                file_path=str(file_path),
                record_id=str(obj.get("info", {}).get("id") or file_path.name),
                reason_code="skip_turn",
                reason_detail="text missing",
                sample_text=None,
                severity="skip",
            )
            continue
        evidence = _collect_reference_text(item.get("reference_text"))
        if role == "질문자":
            turns.append({"role": "user", "content": text, "evidence": evidence or ""})
            assistant_evidence_after_user.append(None)
        elif role == "전문가":
            turns.append({"role": "assistant", "content": text, "evidence": evidence or ""})
            if assistant_evidence_after_user:
                assistant_evidence_after_user[-1] = evidence
        else:
            recorder.add(
                dataset="045",
                split=split,
                file_path=str(file_path),
                record_id=str(obj.get("info", {}).get("id") or file_path.name),
                reason_code="skip_turn",
                reason_detail=f"unsupported role={role}",
                sample_text=truncate_sample(text),
                severity="skip",
            )
    final_turns = finalize_dialog_turns(
        turns=[{"role": turn["role"], "content": turn["content"]} for turn in turns],
        dataset="045",
        split=split,
        file_path=file_path,
        record_id=str(obj.get("info", {}).get("id") or file_path.name),
        recorder=recorder,
    )
    if final_turns is None:
        return []
    user_index = 0
    for turn in final_turns:
        if turn["role"] != "user":
            continue
        direct_evidence = turns[user_index * 2].get("evidence", "") if user_index * 2 < len(turns) else ""
        next_assistant_evidence = assistant_evidence_after_user[user_index] if user_index < len(assistant_evidence_after_user) else None
        evidence = direct_evidence or next_assistant_evidence or ""
        if evidence:
            turn["content"] = turn["content"] + "\n근거: " + evidence
        user_index += 1
    messages = [
        {"role": "system", "content": "user의 질문에 대해서 텍스트 근거 기반으로 대답을 하는 전문가야"},
        *final_turns,
    ]
    return [
        make_row(
            source=DATASET_SPECS["045"].source,
            split=split,
            content=build_content_from_messages(messages),
            messages=messages,
        )
    ]


def _process_046(
    split: str,
    file_path: Path,
    obj: dict[str, Any],
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    info = obj.get("info")
    utterances = obj.get("utterances")
    if not isinstance(info, dict) or not isinstance(utterances, list):
        recorder.add(
            dataset="046",
            split=split,
            file_path=str(file_path),
            record_id=str(obj.get("info", {}).get("id") or file_path.name),
            reason_code="invalid_root",
            reason_detail="info must be object and utterances must be list",
            sample_text=None,
            severity="skip",
        )
        return []
    situation = strip_text(info.get("situation"))
    behaviors_raw = info.get("listener_behavior")
    behaviors = []
    if isinstance(behaviors_raw, list):
        for behavior in behaviors_raw:
            text = strip_text(behavior)
            if text is not None:
                behaviors.append(text)
    if situation is None or not behaviors:
        recorder.add(
            dataset="046",
            split=split,
            file_path=str(file_path),
            record_id=str(info.get("id") or file_path.name),
            reason_code="invalid_system_fields",
            reason_detail="situation/listener_behavior missing",
            sample_text=None,
            severity="skip",
        )
        return []
    turns: list[dict[str, str]] = []
    for item in utterances:
        text = strip_text(item.get("text"))
        role = item.get("role")
        if text is None:
            recorder.add(
                dataset="046",
                split=split,
                file_path=str(file_path),
                record_id=str(info.get("id") or file_path.name),
                reason_code="skip_turn",
                reason_detail="text missing",
                sample_text=None,
                severity="skip",
            )
            continue
        if role == "speaker":
            mapped = "user"
        elif role == "listener":
            mapped = "assistant"
        else:
            recorder.add(
                dataset="046",
                split=split,
                file_path=str(file_path),
                record_id=str(info.get("id") or file_path.name),
                reason_code="skip_turn",
                reason_detail=f"unsupported role={role}",
                sample_text=truncate_sample(text),
                severity="skip",
            )
            continue
        turns.append({"role": mapped, "content": text})
    final_turns = finalize_dialog_turns(
        turns=turns,
        dataset="046",
        split=split,
        file_path=file_path,
        record_id=str(info.get("id") or file_path.name),
        recorder=recorder,
    )
    if final_turns is None:
        return []
    system = situation + "에 대해서 " + ", ".join(behaviors) + "에 맞추어서 대답을 해주는 친구가 되어줘"
    messages = [{"role": "system", "content": system}, *final_turns]
    return [
        make_row(
            source=DATASET_SPECS["046"].source,
            split=split,
            content=build_content_from_messages(messages),
            messages=messages,
        )
    ]


def _process_gsm8k(
    *,
    file_path: Path,
    split: str,
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    table = pq.read_table(file_path, columns=["question", "answer"])
    rows: list[dict[str, Any]] = []
    system = "당신은 수학 문제를 입력받으면 문제 풀이 과정을 포함하여 정답을 출력합니다."
    for index, item in enumerate(table.to_pylist()):
        question = strip_text(item.get("question"))
        answer = strip_text(item.get("answer"))
        if question is None or answer is None:
            recorder.add(
                dataset="gsm8k",
                split=split,
                file_path=str(file_path),
                record_id=str(index),
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
        rows.append(
            make_row(
                source=DATASET_SPECS["gsm8k"].source,
                split=split,
                content=question + "\n" + answer,
                messages=messages,
            )
        )
    return rows


def split_novel24_files(files: Sequence[Path]) -> tuple[set[Path], set[Path]]:
    total = len(files)
    if total == 1:
        return {files[0]}, set()
    train_count = max(1, int(total * 0.9))
    if train_count >= total:
        train_count = total - 1
    return set(files[:train_count]), set(files[train_count:])


def chunk_novel24_text(content: str) -> list[str]:
    from .common import compute_token_count

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    chunks: list[str] = []
    buffer: list[str] = []
    for line in lines:
        candidate_lines = [*buffer, line]
        candidate = "\n".join(candidate_lines)
        if candidate.strip():
            token_count = compute_token_count(content=candidate, messages=None)
            if token_count <= 1024:
                buffer = candidate_lines
                continue
        if buffer:
            chunk = "\n".join(buffer).strip()
            if chunk:
                chunks.append(chunk)
            buffer = [line]
            if line.strip():
                single_count = compute_token_count(content=line, messages=None)
                if single_count > 1024:
                    buffer = []
        else:
            if line.strip():
                single_count = compute_token_count(content=line, messages=None)
                if single_count <= 1024:
                    buffer = [line]
    if buffer:
        chunk = "\n".join(buffer).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _process_novel24(
    *,
    file_path: Path,
    recorder: QualityRecorder,
) -> list[dict[str, Any]]:
    all_files = [
        path
        for _, path in resolve_input_files(DATASET_SPECS["novel24"], "train")
        if path.suffix == ".txt" and not path.name.startswith(".") and path.name != ".DS_Store"
    ]
    train_files, val_files = split_novel24_files(stable_sorted_paths(all_files))
    split = "train" if file_path in train_files else "val" if file_path in val_files else "train"
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        recorder.add(
            dataset="novel24",
            split=split,
            file_path=str(file_path),
            record_id=file_path.name,
            reason_code="file_read_error",
            reason_detail=f"{type(exc).__name__}: {exc}",
            sample_text=None,
            severity="error",
        )
        return []
    chunks: list[str] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    buffer: list[str] = []
    for line in lines:
        candidate = "\n".join([*buffer, line])
        stripped_candidate = candidate.strip()
        if not stripped_candidate:
            buffer.append(line)
            continue
        token_count = compute_token_count(content=candidate, messages=None)
        if token_count <= 1024:
            buffer.append(line)
            continue
        if buffer:
            chunk = "\n".join(buffer).strip()
            if chunk:
                chunks.append(chunk)
            buffer = []
        line_text = line.strip()
        if not line_text:
            continue
        line_count = compute_token_count(content=line_text, messages=None)
        if line_count > 1024:
            recorder.add(
                dataset="novel24",
                split=split,
                file_path=str(file_path),
                record_id=file_path.name,
                reason_code="line_too_long",
                reason_detail="single line exceeds 1024 tokens",
                sample_text=truncate_sample(line_text),
                severity="skip",
            )
            continue
        buffer = [line]
    if buffer:
        chunk = "\n".join(buffer).strip()
        if chunk:
            chunks.append(chunk)
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        rows.append(
            make_row(
                source=DATASET_SPECS["novel24"].source,
                split=split,
                content=chunk,
                messages=None,
            )
        )
        if not chunk.strip():
            recorder.add(
                dataset="novel24",
                split=split,
                file_path=str(file_path),
                record_id=f"{file_path.name}:{index}",
                reason_code="empty_chunk",
                reason_detail="chunk became whitespace-only",
                sample_text=None,
                severity="skip",
            )
    return rows
