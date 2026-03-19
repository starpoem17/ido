from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from tqdm import tqdm

try:
    from tokenizers import Tokenizer as CoreTokenizer
except ImportError:  # pragma: no cover - runtime dependency guard
    CoreTokenizer = None

try:
    from tokenizers.implementations import ByteLevelBPETokenizer
except ImportError:  # pragma: no cover - runtime dependency guard
    ByteLevelBPETokenizer = None


DEFAULT_ENABLE_TQDM = True
DEFAULT_ENABLE_DEBUG_LOG = True
DEFAULT_TQDM_MININTERVAL_SEC = 1.0
DEFAULT_TEXT_PREVIEW_CHARS = 120


@dataclass(frozen=True)
class TopTokenRow:
    rank: int
    token_id: int
    count: int
    share: float
    raw_token: str
    escaped_token: str
    decoded_token: str


@dataclass(frozen=True)
class BuildSummary:
    output_dir: Path
    input_shard_count: int
    input_row_count: int
    valid_row_count: int
    filtered_row_count: int
    top_token_count: int


@dataclass(frozen=True)
class BenchmarkSummary:
    benchmark_out_dir: Path
    compared_row_count: int
    skipped_row_count: int
    more_compressive_row_count: int
    same_token_count_row_count: int
    less_compressive_row_count: int
    mean_delta: float
    mean_abs_delta: float
    mean_delta_ratio: float
    max_abs_delta: int


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, *, enabled: bool) -> None:
    if not enabled:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def progress(
    iterable: Iterable[Any],
    *,
    total: int | None,
    desc: str,
    enabled: bool,
    mininterval: float,
) -> Iterable[Any]:
    if not enabled:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        mininterval=mininterval,
        file=sys.stdout,
    )


def progress_bar(
    *,
    total: int | None,
    desc: str,
    enabled: bool,
    mininterval: float,
) -> tqdm[Any]:
    return tqdm(
        total=total,
        desc=desc,
        mininterval=mininterval,
        file=sys.stdout,
        disable=not enabled,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Path) -> None:
    ensure_dir(path.parent)


def stable_sorted_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: str(path))


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonify(row), ensure_ascii=False))
            f.write("\n")


def append_jsonl(path: Path, row: Any) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonify(row), ensure_ascii=False))
        f.write("\n")
        f.flush()


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def collect_part_paths(root: Path) -> list[Path]:
    return stable_sorted_paths(root.glob("part-*.parquet"))


def strip_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if not value.strip():
        return None
    return value


def truncate_text(text: str, limit: int = DEFAULT_TEXT_PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def require_core_tokenizer() -> type[CoreTokenizer]:
    if CoreTokenizer is None:
        raise SystemExit("tokenizers is required. Install tokenizers>=0.22.2.")
    return CoreTokenizer


def require_bytelevel_tokenizer() -> type[ByteLevelBPETokenizer]:
    if ByteLevelBPETokenizer is None:
        raise SystemExit("tokenizers is required. Install tokenizers>=0.22.2.")
    return ByteLevelBPETokenizer


def load_tokenizer(tokenizer_json_path: Path):
    tokenizer_cls = require_core_tokenizer()
    return tokenizer_cls.from_file(str(tokenizer_json_path))


def find_next_version_dir(root: Path, prefix: str) -> Path:
    existing_versions: set[int] = set()
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                existing_versions.add(int(suffix))
    version = 1
    while version in existing_versions:
        version += 1
    return root / f"{prefix}{version}"


def create_debug_run_dir(root: Path, prefix: str) -> Path:
    ensure_dir(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{prefix}_{stamp}_pid{os.getpid()}"
    candidate = base
    index = 1
    while candidate.exists():
        candidate = root / f"{base.name}_{index:02d}"
        index += 1
    ensure_dir(candidate)
    return candidate


def escape_for_display(text: str) -> str:
    if text == "":
        return "<EMPTY>"
    parts: list[str] = []
    for ch in text:
        if ch == " ":
            parts.append("<SP>")
        elif ch == "\n":
            parts.append("\\n")
        elif ch == "\r":
            parts.append("\\r")
        elif ch == "\t":
            parts.append("\\t")
        elif ord(ch) < 32 or ord(ch) == 127:
            parts.append(f"\\x{ord(ch):02x}")
        else:
            parts.append(ch)
    return "".join(parts)


def decode_token_for_display(tokenizer, token_id: int) -> str:
    try:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:  # pragma: no cover - defensive fallback
        decoded = ""
    return escape_for_display(decoded)


def id_to_raw_token(tokenizer, token_id: int) -> str:
    token = tokenizer.id_to_token(token_id)
    if token is None:
        return ""
    return token


def snapshot_process_memory(pid: int | None = None) -> dict[str, float | int | None]:
    if pid is None:
        status_path = Path("/proc/self/status")
    else:
        status_path = Path(f"/proc/{pid}/status")
    meminfo_path = Path("/proc/meminfo")
    status: dict[str, str] = {}
    meminfo: dict[str, str] = {}
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            status[key.strip()] = value.strip()
    if meminfo_path.exists():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meminfo[key.strip()] = value.strip()
    return {
        "pid": pid,
        "rss_mb": _parse_kib_value(status.get("VmRSS")),
        "hwm_mb": _parse_kib_value(status.get("VmHWM")),
        "vms_mb": _parse_kib_value(status.get("VmSize")),
        "mem_available_mb": _parse_kib_value(meminfo.get("MemAvailable")),
        "swap_free_mb": _parse_kib_value(meminfo.get("SwapFree")),
    }


def _parse_kib_value(raw: str | None) -> float | None:
    if raw is None:
        return None
    token = raw.split()[0]
    try:
        value_kib = float(token)
    except ValueError:
        return None
    return round(value_kib / 1024.0, 3)


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
