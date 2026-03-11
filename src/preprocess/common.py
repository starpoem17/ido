from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
from tokenizers import Tokenizer
from tqdm import tqdm


# =========================
# User configuration block
# =========================
TOKENIZER_JSON_PATH = Path("data/tokenizers/korean_bbpe_v1/tokenizer.json")
TOKEN_COUNT_VERSION = "korean_bbpe_v1"
SCHEMA_VERSION = "korean_processed_v1"
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0
LOG_SAMPLE_TEXT_MAX_CHARS = 160


MESSAGE_STRUCT = pa.struct(
    [
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
    ]
)

CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=True),
        pa.field("messages", pa.list_(MESSAGE_STRUCT), nullable=True),
        pa.field("token_count", pa.int32(), nullable=False),
    ]
)

ALLOWED_SPLITS = {"train", "val"}
SPECIAL_TOKEN_BY_ROLE = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}

_TOKENIZER: Tokenizer | None = None #lazy loading 방식 때문에 아직 안불러왔다고 표기하는 것


@dataclass(frozen=True) #객체 만든 후 값 못 바꾸게 
class Message:
    role: str
    content: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    if not ENABLE_DEBUG_LOG:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True) #그냥 바로 출력하게 flush true


def progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str,
) -> Iterable[Any]:
    if not ENABLE_TQDM:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        mininterval=TQDM_MININTERVAL_SEC,
        file=sys.stdout,
    )


def progress_bar(*, total: int | None, desc: str) -> tqdm[Any]:
    return tqdm(
        total=total,
        desc=desc,
        mininterval=TQDM_MININTERVAL_SEC,
        file=sys.stdout,
        disable=not ENABLE_TQDM,
    )


def get_tokenizer() -> Tokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        log(f"Loading tokenizer from {TOKENIZER_JSON_PATH}")
        _TOKENIZER = Tokenizer.from_file(str(TOKENIZER_JSON_PATH))
    return _TOKENIZER


def strip_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def compact_spaces(text: str) -> str:
    return " ".join(text.split())


def serialize_messages(messages: Sequence[dict[str, str]] | Sequence[Message]) -> str:
    parts: list[str] = []
    for raw_message in messages:
        if isinstance(raw_message, Message):
            role = raw_message.role
            content = raw_message.content
        else:
            role = raw_message["role"]
            content = raw_message["content"]
        token = SPECIAL_TOKEN_BY_ROLE.get(role)
        if token is None:
            raise ValueError(f"unsupported role for serialization: {role}")
        parts.append(token)
        parts.append(content)
    return "".join(parts)


def compute_token_count(
    *,
    content: str | None,
    messages: Sequence[dict[str, str]] | None,
) -> int:
    if messages:
        serialized = serialize_messages(messages)
    elif content:
        serialized = content
    else:
        raise ValueError("content/messages cannot both be empty")
    token_count = len(get_tokenizer().encode(serialized).ids)
    if token_count < 1:
        raise ValueError("token_count must be >= 1")
    return token_count


def build_content_from_messages(messages: Sequence[dict[str, str]]) -> str:
    return " ".join(
        message["content"]
        for message in messages
        if message["role"] in {"user", "assistant"}
    ) 


def validate_row(row: dict[str, Any]) -> None:
    if row["split"] not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split: {row['split']}")
    if row["content"] is None and row["messages"] is None:
        raise ValueError("content/messages cannot both be null")
    if row["messages"] is not None:
        for message in row["messages"]:
            if message["role"] not in SPECIAL_TOKEN_BY_ROLE:
                raise ValueError(f"unsupported role: {message['role']}")
            if not isinstance(message["content"], str) or not message["content"]:
                raise ValueError("message content must be non-empty string")
    expected = compute_token_count(content=row["content"], messages=row["messages"])
    if row["token_count"] != expected:
        raise ValueError(
            f"token_count mismatch: stored={row['token_count']} expected={expected}"
        )


def rows_to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=CANONICAL_SCHEMA)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2) # 예쁘게 json 저장, 폴더 준비 


def truncate_sample(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= LOG_SAMPLE_TEXT_MAX_CHARS:
        return text
    return text[:LOG_SAMPLE_TEXT_MAX_CHARS] + "..."


def stable_sorted_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: str(path))


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
