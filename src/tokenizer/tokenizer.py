from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from contextlib import contextmanager
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import duckdb as _duckdb
except ImportError:  # pragma: no cover - runtime dependency guard
    _duckdb = None

try:
    from tokenizers.implementations import ByteLevelBPETokenizer as _ByteLevelBPETokenizer
except ImportError:  # pragma: no cover - runtime dependency guard
    _ByteLevelBPETokenizer = None

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - runtime dependency guard
    _tqdm = None


INCLUDED_SOURCES: Tuple[str, ...] = (
    "009.전문분야_기술과학_한국어 멀티세션 데이터",
    "010.전문분야_사회과학_한국어 멀티세션 데이터",
    "011.일상대화 한국어 멀티세션 데이터",
    "019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터",
    "020.주제별 텍스트 일상 대화 데이터",
    "021.용도별 목적대화 데이터",
    "023.국회 회의록 기반 지식검색 데이터",
    "030.웹데이터 기반 한국어 말뭉치 데이터",
    "045.지식검색 대화",
    "046.공감형 대화",
    "141.한국어 멀티세션 대화",
    "novel24",
)
EXCLUDED_SOURCES: Tuple[str, ...] = ("_unzip_logs", "gsm8k")
ALLOWED_EXTS: Tuple[str, ...] = (".json", ".txt")

SPECIAL_TOKENS: List[str] = [
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|eot_id|>",
    "<|pad|>",
]

# =========================
# User Settings (edit here)
# =========================
# RUN_MODE:
# - "init_holdout": 고정 holdout(jsonl) 초기 1회 생성
# - "build": 토크나이저 학습 + 고정 holdout 평가
RUN_MODE = "build"

RAW_ROOT = Path("data/korean_raw")
OUT_DIR = Path("data/tokenizers/korean_bbpe_v1")
HOLDOUT_PATH = Path("data/tokenizers/_benchmark/holdout_fixed.jsonl")
HOLDOUT_META_PATH = Path("data/tokenizers/_benchmark/holdout_fixed.meta.json")

SEED = 42
LENGTH_MIN = 5
LENGTH_MAX = 20000

TRAIN_MAX_BYTES_PER_SOURCE = 2 * 1024 * 1024 * 1024
VOCAB_SIZE = 32000
MIN_FREQUENCY = 10

# holdout target policy:
# - source별 (추출+길이필터 통과 텍스트 바이트)의 10%
# - 단, source별 최대 100MB 상한
HOLDOUT_TARGET_PERCENT = 0.10
HOLDOUT_TARGET_BYTES_CAP = 100 * 1024 * 1024
INIT_HOLDOUT_FORCE_OVERWRITE = False  # 기준셋을 정말로 바꿔야 할 때만 True

# 병렬 처리 설정
NUM_WORKERS = 7
DUCKDB_THREADS_PER_WORKER = 1
PARALLEL_CHUNK_ROWS = 5000
TEMP_SHARD_ROOT = Path("data/tokenizers/_tmp")
CLEANUP_TEMP_ON_SUCCESS = True
CLEANUP_TEMP_ON_FAILURE = False
FAIL_FAST = True
ENABLE_TQDM = True
TQDM_MININTERVAL_SEC = 1.0
PHASE_DEBUG_LOG = True


@dataclasses.dataclass(frozen=True)
class FileRecord:
    source: str
    path: Path
    size_bytes: int
    ext: str


@dataclasses.dataclass(frozen=True)
class WorkerTask:
    source: str
    worker_id: int
    output_path: str
    records: Tuple[FileRecord, ...]
    min_len: int
    max_len: int


@dataclasses.dataclass
class WorkerShardStats:
    source: str
    worker_id: int
    shard_path: str
    file_count: int = 0
    with_text_files: int = 0
    zero_text_files: int = 0
    extracted_rows: int = 0
    filtered_rows: int = 0
    filtered_text_bytes: int = 0
    failed_files: int = 0


@dataclasses.dataclass
class SourceSamplingStats:
    available_files: int = 0
    available_bytes: int = 0
    selected_files: int = 0
    selected_bytes: int = 0


@dataclasses.dataclass
class BuildStats:
    extracted_rows: int = 0
    filtered_rows: int = 0
    dedup_rows: int = 0
    train_rows: int = 0
    leakage_removed_rows: int = 0
    failed_files: int = 0


@dataclasses.dataclass
class HoldoutSourceStats:
    target_bytes: int = 0
    actual_bytes: int = 0
    row_count: int = 0
    source_file_count: int = 0
    with_text_files: int = 0
    zero_text_files: int = 0
    source_input_bytes: int = 0
    extracted_rows: int = 0
    filtered_rows: int = 0
    filtered_text_bytes: int = 0
    worker_shards: int = 0
    underfill: bool = False


@dataclasses.dataclass
class MetricsAccumulator:
    count: int = 0
    chars_sum: int = 0
    tokens_sum: int = 0
    roundtrip_success: int = 0
    token_lengths: array = dataclasses.field(default_factory=lambda: array("I"))

    def add(self, text_len: int, token_len: int, roundtrip_ok: bool) -> None:
        self.count += 1
        self.chars_sum += text_len
        self.tokens_sum += token_len
        self.roundtrip_success += int(roundtrip_ok)
        self.token_lengths.append(token_len)

    def to_metrics(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "mean_chars_per_text": 0.0,
                "mean_tokens_per_text": 0.0,
                "mean_tokens_per_char": 0.0,
                "token_length_p50": 0.0,
                "token_length_p95": 0.0,
                "token_length_p99": 0.0,
                "round_trip_success_rate": 0.0,
            }
        sorted_lengths = sorted(self.token_lengths)
        return {
            "count": self.count,
            "mean_chars_per_text": self.chars_sum / self.count,
            "mean_tokens_per_text": self.tokens_sum / self.count,
            "mean_tokens_per_char": (self.tokens_sum / self.chars_sum)
            if self.chars_sum > 0
            else 0.0,
            "token_length_p50": percentile_from_sorted(sorted_lengths, 50.0),
            "token_length_p95": percentile_from_sorted(sorted_lengths, 95.0),
            "token_length_p99": percentile_from_sorted(sorted_lengths, 99.0),
            "round_trip_success_rate": self.roundtrip_success / self.count,
        }


def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class NullProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n: int = 1) -> None:
        _ = n

    def close(self) -> None:
        return


def progress_iter(
    iterable,
    *,
    total: Optional[int],
    desc: str,
    unit: str,
):
    if _tqdm is None or not ENABLE_TQDM:
        return iterable
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
        mininterval=TQDM_MININTERVAL_SEC,
    )


def progress_bar(*, total: int, desc: str, unit: str):
    if _tqdm is None or not ENABLE_TQDM:
        return NullProgressBar()
    return _tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
        mininterval=TQDM_MININTERVAL_SEC,
    )


@contextmanager
def phase(name: str):
    start = time.perf_counter()
    if PHASE_DEBUG_LOG:
        log(f"[phase:start] {name}")
    try:
        yield
    finally:
        if PHASE_DEBUG_LOG:
            elapsed = time.perf_counter() - start
            log(f"[phase:end] {name} (+{elapsed:.2f}s)")


def require_duckdb():
    if _duckdb is None:
        raise SystemExit(
            "duckdb is required. Install it first (e.g. add it to pyproject and sync env)."
        )
    return _duckdb


def require_tokenizer_cls():
    if _ByteLevelBPETokenizer is None:
        raise SystemExit("tokenizers is required. Install tokenizers>=0.22.2.")
    return _ByteLevelBPETokenizer


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def stable_hash_int(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ")
    s = s.replace("\n", " ")
    return s.strip()


def is_valid_length(text: str, min_len: int, max_len: int) -> bool:
    n = len(text)
    return min_len <= n <= max_len


def percentile_from_sorted(sorted_vals: Sequence[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    pos = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def source_dir(raw_root: Path, source: str) -> Path:
    return raw_root / source


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def validate_parallel_args(
    num_workers: int,
    duckdb_threads_per_worker: int,
    parallel_chunk_rows: int,
) -> None:
    if num_workers < 1:
        raise ValueError(f"NUM_WORKERS must be >= 1, got {num_workers}")
    if duckdb_threads_per_worker < 1:
        raise ValueError(
            "DUCKDB_THREADS_PER_WORKER must be >= 1, "
            f"got {duckdb_threads_per_worker}"
        )
    if parallel_chunk_rows < 1:
        raise ValueError(f"PARALLEL_CHUNK_ROWS must be >= 1, got {parallel_chunk_rows}")


def validate_holdout_target_args(percent: float, bytes_cap: int) -> None:
    if not (0.0 < percent <= 1.0):
        raise ValueError(f"HOLDOUT_TARGET_PERCENT must be in (0, 1], got {percent}")
    if bytes_cap < 1:
        raise ValueError(f"HOLDOUT_TARGET_BYTES_CAP must be >= 1, got {bytes_cap}")


def configure_duckdb_threads(conn, threads: int) -> None:
    conn.execute(f"PRAGMA threads={int(threads)}")


def create_temp_run_dir(temp_root: Path, phase: str) -> Path:
    ensure_dir(temp_root)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = temp_root / f"{phase}_{ts}_{os.getpid()}"
    if not base.exists():
        ensure_dir(base)
        return base
    suffix = 1
    while True:
        cand = temp_root / f"{phase}_{ts}_{os.getpid()}_{suffix}"
        if not cand.exists():
            ensure_dir(cand)
            return cand
        suffix += 1


def maybe_cleanup_temp(path: Path, should_cleanup: bool, reason: str) -> None:
    if not path.exists():
        return
    if should_cleanup:
        shutil.rmtree(path)
        log(f"temp shards removed ({reason}): {path}")
    else:
        log(f"temp shards kept ({reason}): {path}")


def collect_inventory(raw_root: Path) -> List[FileRecord]:
    records: List[FileRecord] = []
    for source in progress_iter(
        INCLUDED_SOURCES,
        total=len(INCLUDED_SOURCES),
        desc="inventory/sources",
        unit="src",
    ):
        src = source_dir(raw_root, source)
        if not src.exists():
            log(f"warning: source dir not found: {src}")
            continue
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            records.append(
                FileRecord(
                    source=source,
                    path=path,
                    size_bytes=path.stat().st_size,
                    ext=ext,
                )
            )
    return records


def group_by_source(records: Sequence[FileRecord]) -> Dict[str, List[FileRecord]]:
    out: Dict[str, List[FileRecord]] = {s: [] for s in INCLUDED_SOURCES}
    for rec in records:
        if rec.source in out:
            out[rec.source].append(rec)
    return out


def sample_training_files(
    records: Sequence[FileRecord],
    max_bytes_per_source: int,
    seed: int,
) -> Tuple[List[FileRecord], Dict[str, SourceSamplingStats]]:
    grouped = group_by_source(records)
    selected: List[FileRecord] = []
    stats: Dict[str, SourceSamplingStats] = {}
    for source, items in progress_iter(
        grouped.items(),
        total=len(grouped),
        desc="sampling/sources",
        unit="src",
    ):
        st = SourceSamplingStats()
        st.available_files = len(items)
        st.available_bytes = sum(x.size_bytes for x in items)
        ranked = sorted(
            items,
            key=lambda r: stable_hash_int(f"{seed}|{r.path.as_posix()}"),
        )
        if st.available_bytes <= max_bytes_per_source:
            chosen = ranked
        else:
            chosen = []
            cum = 0
            for rec in ranked:
                chosen.append(rec)
                cum += rec.size_bytes
                if cum >= max_bytes_per_source:
                    break
        st.selected_files = len(chosen)
        st.selected_bytes = sum(x.size_bytes for x in chosen)
        selected.extend(chosen)
        stats[source] = st
    return selected, stats


def _iter_strings_recursive(obj: object) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings_recursive(v)
        return
    if isinstance(obj, list):
        for v in obj:
            yield from _iter_strings_recursive(v)


def _extract_only_comment(field: object) -> Iterator[str]:
    if isinstance(field, dict):
        comment = field.get("comment")
        if isinstance(comment, str):
            yield comment
        return
    if isinstance(field, str):
        s = field.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
            except Exception:
                return
            if isinstance(parsed, dict):
                comment = parsed.get("comment")
                if isinstance(comment, str):
                    yield comment


def extract_from_json_obj(
    source: str,
    obj: object,
    file_name: Optional[str] = None,
) -> Iterator[str]:
    if source.startswith(("009.", "010.", "011.", "141.")):
        sessions = obj.get("sessionInfo") if isinstance(obj, dict) else None
        if not isinstance(sessions, list):
            return
        for session in sessions:
            if not isinstance(session, dict):
                continue
            dialogs = session.get("dialog")
            if not isinstance(dialogs, list):
                continue
            for turn in dialogs:
                if isinstance(turn, dict):
                    utt = turn.get("utterance")
                    if isinstance(utt, str):
                        yield utt
        return

    if source.startswith("019."):
        if not isinstance(obj, dict):
            return
        for field_name in ("clauseArticle", "comProvision"):
            field_values = obj.get(field_name)
            if not isinstance(field_values, list):
                continue
            for x in field_values:
                if isinstance(x, str):
                    yield x
                else:
                    yield from _iter_strings_recursive(x)
        for key in ("ftcCnclsns", "illdcssBasiss"):
            if key in obj:
                yield from _iter_strings_recursive(obj.get(key))
        return

    if source.startswith(("020.", "021.")):
        if not isinstance(obj, dict):
            return
        info = obj.get("info")
        infos = info if isinstance(info, list) else [info]
        for item in infos:
            if not isinstance(item, dict):
                continue
            annotations = item.get("annotations")
            if not isinstance(annotations, dict):
                continue
            lines = annotations.get("lines")
            emitted = False
            if isinstance(lines, list):
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    norm_text = line.get("norm_text")
                    if isinstance(norm_text, str):
                        yield norm_text
                        emitted = True
            if not emitted:
                text = annotations.get("text")
                if isinstance(text, str):
                    yield text
        return

    if source.startswith("023."):
        if not isinstance(obj, dict):
            return
        if isinstance(file_name, str) and file_name.startswith("SLAB_"):
            # SLAB 메타 파일은 추출 대상에서 제외한다.
            return
        context = obj.get("context")
        if isinstance(context, str):
            yield context
        yield from _extract_only_comment(obj.get("question"))
        yield from _extract_only_comment(obj.get("answer"))
        return

    if source.startswith("030."):
        if not isinstance(obj, dict):
            return
        named = obj.get("named_entity")
        if not isinstance(named, list):
            return
        for ne in named:
            if not isinstance(ne, dict):
                continue
            for key in ("title", "subtitle", "content"):
                blocks = ne.get(key)
                if not isinstance(blocks, list):
                    continue
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    sentence = block.get("sentence")
                    if isinstance(sentence, str):
                        yield sentence
        return

    if source.startswith(("045.", "046.")):
        if not isinstance(obj, dict):
            return
        utterances = obj.get("utterances")
        if not isinstance(utterances, list):
            return
        for utt in utterances:
            if not isinstance(utt, dict):
                continue
            text = utt.get("text")
            if isinstance(text, str):
                yield text
        return


def iter_file_texts(record: FileRecord) -> Iterator[str]:
    if record.ext == ".txt":
        with record.path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = clean_text(line)
                if line:
                    yield line
        return
    if record.ext == ".json":
        with record.path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            obj = json.load(f)
        for text in extract_from_json_obj(
            record.source,
            obj,
            file_name=record.path.name,
        ):
            cleaned = clean_text(text)
            if cleaned:
                yield cleaned


def insert_text_batch(conn, table: str, rows: List[Tuple[str]]) -> None:
    if not rows:
        return
    conn.executemany(f"INSERT INTO {table}(text) VALUES (?)", rows)


def iter_table_texts(
    conn,
    table: str,
    seed: int,
    chunk_size: int = 10000,
) -> Iterator[str]:
    # 학습 순서를 고정해 워커 수가 달라도 학습 결과가 흔들리지 않게 한다.
    cur = conn.execute(
        f"SELECT text FROM {table} ORDER BY md5(text || ?)",
        [str(seed)],
    )
    while True:
        rows = cur.fetchmany(chunk_size)
        if not rows:
            break
        for (text,) in rows:
            yield text


def parse_holdout_line(line: str, line_no: int) -> Tuple[str, str]:
    s = line.strip()
    if not s:
        raise ValueError(f"empty line at {line_no}")
    try:
        obj = json.loads(s)
    except Exception as exc:
        raise ValueError(f"invalid jsonl at line {line_no}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"jsonl object required at line {line_no}")
    source = obj.get("source")
    text = obj.get("text")
    if not isinstance(source, str) or not source:
        raise ValueError(f"invalid source at line {line_no}")
    if not isinstance(text, str) or not text:
        raise ValueError(f"invalid text at line {line_no}")
    return source, text


def parse_text_shard_line(line: str, line_no: int, shard_path: Path) -> str:
    s = line.strip()
    if not s:
        raise ValueError(f"empty line in shard: {shard_path}#{line_no}")
    try:
        obj = json.loads(s)
    except Exception as exc:
        raise ValueError(f"invalid jsonl in shard: {shard_path}#{line_no}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"json object required in shard: {shard_path}#{line_no}")
    text = obj.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"invalid text in shard: {shard_path}#{line_no}")
    return text


def iter_holdout_rows(holdout_path: Path) -> Iterator[Tuple[str, str]]:
    with holdout_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            yield parse_holdout_line(line, i)


def iter_text_shard_rows(shard_path: Path) -> Iterator[str]:
    with shard_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            yield parse_text_shard_line(line, i, shard_path)


def verify_holdout_hash(holdout_path: Path, holdout_meta_path: Path) -> str:
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"fixed holdout file not found: {holdout_path} "
            "(set RUN_MODE='init_holdout' and run src/tokenizer.py first)"
        )
    if not holdout_meta_path.exists():
        raise FileNotFoundError(f"fixed holdout meta not found: {holdout_meta_path}")
    with holdout_meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    expected = meta.get("sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"invalid holdout meta sha256: {holdout_meta_path}")
    actual = sha256_file(holdout_path)
    if actual != expected:
        raise ValueError(
            "holdout sha256 mismatch: "
            f"expected={expected}, actual={actual}, path={holdout_path}"
        )
    return actual


def assign_worker_id(path: Path, seed: int, num_workers: int) -> int:
    key = f"{seed}|{path.as_posix()}"
    return stable_hash_int(key) % num_workers


def split_records_for_workers(
    records: Sequence[FileRecord],
    num_workers: int,
    seed: int,
) -> Dict[int, List[FileRecord]]:
    buckets: Dict[int, List[FileRecord]] = {i: [] for i in range(num_workers)}
    ranked = sorted(records, key=lambda r: stable_hash_int(f"{seed}|{r.path.as_posix()}"))
    for rec in ranked:
        worker_id = assign_worker_id(rec.path, seed=seed, num_workers=num_workers)
        buckets[worker_id].append(rec)
    return buckets


def source_token(source: str) -> str:
    idx = INCLUDED_SOURCES.index(source)
    return f"s{idx:02d}"


def build_worker_tasks_for_source(
    source: str,
    records: Sequence[FileRecord],
    num_workers: int,
    seed: int,
    min_len: int,
    max_len: int,
    run_dir: Path,
    shard_prefix: str,
) -> List[WorkerTask]:
    buckets = split_records_for_workers(records, num_workers=num_workers, seed=seed)
    tasks: List[WorkerTask] = []
    for worker_id in range(num_workers):
        bucket = buckets[worker_id]
        if not bucket:
            continue
        shard_name = (
            f"{shard_prefix}_{source_token(source)}_w{worker_id:03d}.jsonl"
            if source in INCLUDED_SOURCES
            else f"{shard_prefix}_w{worker_id:03d}.jsonl"
        )
        output_path = run_dir / shard_name
        tasks.append(
            WorkerTask(
                source=source,
                worker_id=worker_id,
                output_path=str(output_path),
                records=tuple(bucket),
                min_len=min_len,
                max_len=max_len,
            )
        )
    return tasks


def extract_filtered_rows_to_shard(task: WorkerTask) -> WorkerShardStats:
    out_path = Path(task.output_path)
    ensure_dir(out_path.parent)

    with_text_files = 0
    zero_text_files = 0
    extracted_rows = 0
    filtered_rows = 0
    filtered_text_bytes = 0
    failed_files = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in task.records:
            try:
                file_filtered_rows = 0
                for text in iter_file_texts(rec):
                    extracted_rows += 1
                    if not is_valid_length(text, min_len=task.min_len, max_len=task.max_len):
                        continue
                    filtered_rows += 1
                    file_filtered_rows += 1
                    filtered_text_bytes += len(text.encode("utf-8"))
                    out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                if file_filtered_rows > 0:
                    with_text_files += 1
                else:
                    zero_text_files += 1
            except Exception as exc:
                failed_files += 1
                raise RuntimeError(
                    f"failed to parse file in worker={task.worker_id}, "
                    f"source={task.source}: {rec.path}"
                ) from exc

    return WorkerShardStats(
        source=task.source,
        worker_id=task.worker_id,
        shard_path=str(out_path),
        file_count=len(task.records),
        with_text_files=with_text_files,
        zero_text_files=zero_text_files,
        extracted_rows=extracted_rows,
        filtered_rows=filtered_rows,
        filtered_text_bytes=filtered_text_bytes,
        failed_files=failed_files,
    )


def run_parallel_extract(
    tasks: Sequence[WorkerTask],
    num_workers: int,
    fail_fast: bool,
) -> List[WorkerShardStats]:
    if not tasks:
        return []

    ctx = mp.get_context("spawn")
    results: List[WorkerShardStats] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        future_map = {executor.submit(extract_filtered_rows_to_shard, t): t for t in tasks}
        with progress_bar(
            total=len(future_map),
            desc="parallel/worker_tasks",
            unit="task",
        ) as pbar:
            for fut in as_completed(future_map):
                task = future_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    if fail_fast:
                        for pending in future_map:
                            pending.cancel()
                    raise RuntimeError(
                        "parallel extraction failed "
                        f"(source={task.source}, worker={task.worker_id}): {exc}"
                    ) from exc
                results.append(result)
                pbar.update(1)

    results.sort(key=lambda x: (x.source, x.worker_id, x.shard_path))
    return results


def load_shards_to_duckdb_table(
    conn,
    table: str,
    shard_paths: Sequence[Path],
    chunk_rows: int,
) -> None:
    batch: List[Tuple[str]] = []
    ordered_paths = sorted(shard_paths)
    for shard_path in progress_iter(
        ordered_paths,
        total=len(ordered_paths),
        desc=f"duckdb/load:{table}",
        unit="shard",
    ):
        for text in iter_text_shard_rows(shard_path):
            batch.append((text,))
            if len(batch) >= chunk_rows:
                insert_text_batch(conn, table, batch)
                batch.clear()
    if batch:
        insert_text_batch(conn, table, batch)


def build_training_text_tables_from_shards(
    shard_results: Sequence[WorkerShardStats],
    holdout_path: Path,
    chunk_rows: int,
    duckdb_threads: int,
) -> Tuple[object, BuildStats]:
    duckdb = require_duckdb()
    conn = duckdb.connect(database=":memory:")
    configure_duckdb_threads(conn, duckdb_threads)

    stats = BuildStats(
        extracted_rows=sum(x.extracted_rows for x in shard_results),
        filtered_rows=sum(x.filtered_rows for x in shard_results),
        failed_files=sum(x.failed_files for x in shard_results),
    )

    conn.execute("CREATE TEMP TABLE filtered_texts(text VARCHAR)")
    load_shards_to_duckdb_table(
        conn,
        table="filtered_texts",
        shard_paths=[Path(x.shard_path) for x in shard_results],
        chunk_rows=chunk_rows,
    )

    conn.execute("CREATE TEMP TABLE dedup_texts AS SELECT DISTINCT text FROM filtered_texts")
    stats.dedup_rows = int(conn.execute("SELECT COUNT(*) FROM dedup_texts").fetchone()[0])

    conn.execute("CREATE TEMP TABLE holdout_fixed(text VARCHAR)")
    h_batch: List[Tuple[str]] = []
    for _, text in progress_iter(
        iter_holdout_rows(holdout_path),
        total=None,
        desc="duckdb/load:holdout_fixed",
        unit="row",
    ):
        h_batch.append((text,))
        if len(h_batch) >= chunk_rows:
            insert_text_batch(conn, "holdout_fixed", h_batch)
            h_batch.clear()
    if h_batch:
        insert_text_batch(conn, "holdout_fixed", h_batch)

    conn.execute("CREATE TEMP TABLE holdout_fixed_dedup AS SELECT DISTINCT text FROM holdout_fixed")
    conn.execute(
        """
        CREATE TEMP TABLE train_texts AS
        SELECT d.text
        FROM dedup_texts d
        LEFT JOIN holdout_fixed_dedup h ON d.text = h.text
        WHERE h.text IS NULL
        """
    )
    stats.train_rows = int(conn.execute("SELECT COUNT(*) FROM train_texts").fetchone()[0])
    stats.leakage_removed_rows = stats.dedup_rows - stats.train_rows
    return conn, stats


def train_tokenizer_from_table(
    conn,
    seed: int,
    vocab_size: int,
    min_frequency: int,
) -> object:
    tokenizer_cls = require_tokenizer_cls()
    tokenizer = tokenizer_cls(add_prefix_space=True, trim_offsets=True)
    tokenizer.train_from_iterator(
        iter_table_texts(conn, "train_texts", seed=seed),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    return tokenizer


def validate_tokenizer_spec(tokenizer, expected_vocab_size: int) -> None:
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    if vocab_size != expected_vocab_size:
        raise RuntimeError(
            f"vocab size mismatch: expected={expected_vocab_size}, actual={vocab_size}"
        )
    for tok in SPECIAL_TOKENS:
        if tok not in vocab:
            raise RuntimeError(f"missing special token: {tok}")
    for unk_like in ("[UNK]", "<|unk|>", "UNK"):
        if unk_like in vocab:
            raise RuntimeError(f"unexpected unk token found: {unk_like}")


def is_roundtrip_success_prefix_space_aware(text: str, decoded: str) -> bool:
    return decoded == text or decoded == f" {text}"


def evaluate_on_holdout(tokenizer, holdout_path: Path) -> Dict[str, object]:
    overall = MetricsAccumulator()
    by_source: Dict[str, MetricsAccumulator] = {}
    strict_roundtrip_success = 0

    for source, text in progress_iter(
        iter_holdout_rows(holdout_path),
        total=None,
        desc="eval/holdout_rows",
        unit="row",
    ):
        acc = by_source.setdefault(source, MetricsAccumulator())
        enc = tokenizer.encode(text)
        token_len = len(enc.ids)
        decoded = tokenizer.decode(enc.ids)
        strict_ok = decoded == text
        roundtrip_ok = is_roundtrip_success_prefix_space_aware(text, decoded)
        strict_roundtrip_success += int(strict_ok)

        tlen = len(text)
        overall.add(tlen, token_len, roundtrip_ok)
        acc.add(tlen, token_len, roundtrip_ok)

    if overall.count > 0:
        strict_rate = strict_roundtrip_success / overall.count
        normalized_rate = overall.roundtrip_success / overall.count
        delta = normalized_rate - strict_rate
        log(
            "round-trip rates: "
            f"strict={strict_rate:.6f}, "
            f"prefix_space_aware={normalized_rate:.6f}, "
            f"delta={delta:.6f}"
        )

    source_metrics = {src: acc.to_metrics() for src, acc in sorted(by_source.items())}
    return {
        "overall": overall.to_metrics(),
        "by_source": source_metrics,
    }

def save_tokenizer_artifacts(tokenizer, out_dir: Path) -> None:
    ensure_dir(out_dir)
    tokenizer.save_model(str(out_dir))
    tokenizer._tokenizer.save(str(out_dir / "tokenizer.json"))


def source_stats_to_dict(stats: Dict[str, SourceSamplingStats]) -> Dict[str, Dict[str, int]]:
    return {
        source: {
            "available_files": st.available_files,
            "available_bytes": st.available_bytes,
            "selected_files": st.selected_files,
            "selected_bytes": st.selected_bytes,
        }
        for source, st in stats.items()
    }


def shard_stats_to_dict(shards: Sequence[WorkerShardStats]) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, int]] = {}
    for shard in shards:
        g = grouped.setdefault(
            shard.source,
            {
                "shard_count": 0,
                "file_count": 0,
                "with_text_files": 0,
                "zero_text_files": 0,
                "extracted_rows": 0,
                "filtered_rows": 0,
                "filtered_text_bytes": 0,
                "failed_files": 0,
            },
        )
        g["shard_count"] += 1
        g["file_count"] += shard.file_count
        g["with_text_files"] += shard.with_text_files
        g["zero_text_files"] += shard.zero_text_files
        g["extracted_rows"] += shard.extracted_rows
        g["filtered_rows"] += shard.filtered_rows
        g["filtered_text_bytes"] += shard.filtered_text_bytes
        g["failed_files"] += shard.failed_files
    return grouped


def build_tokenizer(args) -> None:
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    holdout_path = Path(args.holdout_path)
    holdout_meta_path = Path(args.holdout_meta_path)

    run_dir = create_temp_run_dir(Path(args.temp_shard_root), phase="build")
    success = False

    try:
        with phase("build/verify_fixed_holdout"):
            log("verifying fixed holdout file/hash")
            holdout_sha = verify_holdout_hash(holdout_path, holdout_meta_path)

        with phase("build/collect_and_sample_files"):
            log("collecting inventory")
            inventory = collect_inventory(raw_root)
            sampled_files, sample_stats = sample_training_files(
                inventory,
                max_bytes_per_source=args.max_bytes_per_source,
                seed=args.seed,
            )
            log(f"sampled files: {len(sampled_files)}")

        with phase("build/parallel_extract_shards"):
            log(f"parallel extracting filtered texts (workers={args.num_workers})")
            build_tasks = build_worker_tasks_for_source(
                source="build",
                records=sampled_files,
                num_workers=args.num_workers,
                seed=args.seed,
                min_len=args.length_min,
                max_len=args.length_max,
                run_dir=run_dir,
                shard_prefix="build_filtered",
            )
            build_shards = run_parallel_extract(
                tasks=build_tasks,
                num_workers=args.num_workers,
                fail_fast=args.fail_fast,
            )
            log(f"build shards created: {len(build_shards)}")

        with phase("build/duckdb_prepare_train_texts"):
            log("building filtered/dedup/train tables with duckdb")
            conn, build_stats = build_training_text_tables_from_shards(
                shard_results=build_shards,
                holdout_path=holdout_path,
                chunk_rows=args.parallel_chunk_rows,
                duckdb_threads=args.duckdb_threads_per_worker,
            )

        try:
            with phase("build/train_tokenizer"):
                log("training tokenizer")
                tokenizer = train_tokenizer_from_table(
                    conn=conn,
                    seed=args.seed,
                    vocab_size=args.vocab_size,
                    min_frequency=args.min_frequency,
                )
        finally:
            conn.close()

        with phase("build/validate_tokenizer_spec"):
            validate_tokenizer_spec(tokenizer, expected_vocab_size=args.vocab_size)

        with phase("build/evaluate_holdout"):
            log("evaluating on fixed holdout")
            metrics = evaluate_on_holdout(tokenizer, holdout_path)
            metrics["holdout_path"] = str(holdout_path)
            metrics["holdout_sha256"] = holdout_sha

        with phase("build/save_artifacts"):
            log("saving artifacts")
            save_tokenizer_artifacts(tokenizer, out_dir)

        meta = {
            "run_at": now_iso(),
            "raw_root": str(raw_root),
            "out_dir": str(out_dir),
            "params": {
                "seed": args.seed,
                "vocab_size": args.vocab_size,
                "min_frequency": args.min_frequency,
                "length_min": args.length_min,
                "length_max": args.length_max,
                "max_bytes_per_source": args.max_bytes_per_source,
            },
            "holdout": {
                "path": str(holdout_path),
                "meta_path": str(holdout_meta_path),
                "sha256": holdout_sha,
            },
            "parallel": {
                "num_workers": args.num_workers,
                "duckdb_threads_per_worker": args.duckdb_threads_per_worker,
                "parallel_chunk_rows": args.parallel_chunk_rows,
                "temp_shard_root": str(args.temp_shard_root),
                "temp_run_dir": str(run_dir),
                "shard_count": len(build_shards),
                "fail_fast": args.fail_fast,
            },
            "sampling_by_source": source_stats_to_dict(sample_stats),
            "worker_extract_stats": shard_stats_to_dict(build_shards),
            "stats": dataclasses.asdict(build_stats),
        }

        with phase("build/write_meta"):
            write_json(out_dir / "meta.json", meta)
            write_json(out_dir / "holdout_metrics.json", metrics)

        success = True
        log("done")
    finally:
        maybe_cleanup_temp(
            run_dir,
            should_cleanup=args.cleanup_temp_on_success if success else args.cleanup_temp_on_failure,
            reason="success" if success else "failure",
        )


def extract_clean_filtered_from_source(
    source: str,
    records: Sequence[FileRecord],
    min_len: int,
    max_len: int,
) -> Iterator[str]:
    for rec in records:
        try:
            for text in iter_file_texts(rec):
                if is_valid_length(text, min_len=min_len, max_len=max_len):
                    yield text
        except Exception as exc:
            raise RuntimeError(f"failed to parse file: {rec.path}") from exc


def compute_holdout_target_bytes(filtered_text_bytes: int, percent: float, bytes_cap: int) -> int:
    if filtered_text_bytes <= 0:
        return 0
    return min(int(filtered_text_bytes * percent), bytes_cap)


def init_fixed_holdout(args) -> None:
    raw_root = Path(args.raw_root)
    holdout_path = Path(args.holdout_path)
    holdout_meta_path = Path(args.holdout_meta_path)

    if holdout_path.exists() and not args.force:
        raise RuntimeError(
            f"holdout already exists: {holdout_path} "
            "(set INIT_HOLDOUT_FORCE_OVERWRITE=True to overwrite)"
        )
    if holdout_meta_path.exists() and not args.force:
        raise RuntimeError(
            f"holdout meta already exists: {holdout_meta_path} "
            "(set INIT_HOLDOUT_FORCE_OVERWRITE=True to overwrite)"
        )

    run_dir = create_temp_run_dir(Path(args.temp_shard_root), phase="init_holdout")
    success = False

    try:
        with phase("init_holdout/collect_inventory"):
            inventory = collect_inventory(raw_root)
            grouped = group_by_source(inventory)
            ensure_dir(holdout_path.parent)

        source_stats: Dict[str, HoldoutSourceStats] = {src: HoldoutSourceStats() for src in INCLUDED_SOURCES}

        with phase("init_holdout/build_worker_tasks"):
            log(f"parallel extracting holdout candidates (workers={args.num_workers})")
            holdout_tasks: List[WorkerTask] = []
            for source in progress_iter(
                INCLUDED_SOURCES,
                total=len(INCLUDED_SOURCES),
                desc="init_holdout/task_sources",
                unit="src",
            ):
                records = grouped.get(source, [])
                source_stats[source].source_file_count = len(records)
                source_stats[source].source_input_bytes = sum(r.size_bytes for r in records)

                holdout_tasks.extend(
                    build_worker_tasks_for_source(
                        source=source,
                        records=records,
                        num_workers=args.num_workers,
                        seed=args.seed,
                        min_len=args.length_min,
                        max_len=args.length_max,
                        run_dir=run_dir,
                        shard_prefix="holdout_filtered",
                    )
                )

        with phase("init_holdout/parallel_extract_shards"):
            holdout_shards = run_parallel_extract(
                tasks=holdout_tasks,
                num_workers=args.num_workers,
                fail_fast=args.fail_fast,
            )
            log(f"holdout shards created: {len(holdout_shards)}")

        with phase("init_holdout/index_shards_by_source"):
            shards_by_source: Dict[str, List[WorkerShardStats]] = {src: [] for src in INCLUDED_SOURCES}
            for shard in holdout_shards:
                shards_by_source.setdefault(shard.source, []).append(shard)
                source_stats[shard.source].with_text_files += shard.with_text_files
                source_stats[shard.source].zero_text_files += shard.zero_text_files
                source_stats[shard.source].extracted_rows += shard.extracted_rows
                source_stats[shard.source].filtered_rows += shard.filtered_rows
                source_stats[shard.source].filtered_text_bytes += shard.filtered_text_bytes
                source_stats[shard.source].worker_shards += 1

        total_bytes = 0
        total_rows = 0
        target_total_bytes = 0

        with phase("init_holdout/assemble_fixed_holdout"):
            with holdout_path.open("w", encoding="utf-8") as out_f:
                for source in progress_iter(
                    INCLUDED_SOURCES,
                    total=len(INCLUDED_SOURCES),
                    desc="init_holdout/sources",
                    unit="src",
                ):
                    log(f"building holdout slice for source: {source}")
                    duckdb = require_duckdb()
                    conn = duckdb.connect(database=":memory:")
                    configure_duckdb_threads(conn, args.duckdb_threads_per_worker)
                    conn.execute("CREATE TEMP TABLE source_filtered(text VARCHAR)")

                    try:
                        load_shards_to_duckdb_table(
                            conn,
                            table="source_filtered",
                            shard_paths=[Path(s.shard_path) for s in shards_by_source.get(source, [])],
                            chunk_rows=args.parallel_chunk_rows,
                        )

                        conn.execute(
                            "CREATE TEMP TABLE source_dedup AS SELECT DISTINCT text FROM source_filtered"
                        )
                        cur = conn.execute(
                            "SELECT text FROM source_dedup ORDER BY md5(text || ?)",
                            [str(args.seed)],
                        )

                        target = compute_holdout_target_bytes(
                            filtered_text_bytes=source_stats[source].filtered_text_bytes,
                            percent=args.holdout_target_percent,
                            bytes_cap=args.holdout_target_bytes_cap,
                        )
                        source_stats[source].target_bytes = target
                        target_total_bytes += target
                        selected_bytes = 0
                        selected_rows = 0

                        while True:
                            rows = cur.fetchmany(args.parallel_chunk_rows)
                            if not rows:
                                break
                            for (text,) in rows:
                                if selected_bytes >= target:
                                    break
                                text_bytes = len(text.encode("utf-8"))
                                row_obj = {"source": source, "text": text}
                                out_f.write(json.dumps(row_obj, ensure_ascii=False) + "\n")
                                selected_bytes += text_bytes
                                selected_rows += 1
                            if selected_bytes >= target:
                                break

                        source_stats[source].actual_bytes = selected_bytes
                        source_stats[source].row_count = selected_rows
                        source_stats[source].underfill = selected_bytes < target
                        total_bytes += selected_bytes
                        total_rows += selected_rows
                    finally:
                        conn.close()

        with phase("init_holdout/write_meta"):
            sha = sha256_file(holdout_path)
            meta = {
                "created_at": now_iso(),
                "holdout_path": str(holdout_path),
                "sha256": sha,
                "seed": args.seed,
                "target_policy": {
                    "mode": "percent_cap",
                    "percent": args.holdout_target_percent,
                    "bytes_cap_per_source": args.holdout_target_bytes_cap,
                },
                "target_total_bytes": target_total_bytes,
                "actual_total_bytes": total_bytes,
                "actual_total_rows": total_rows,
                "parallel": {
                    "num_workers": args.num_workers,
                    "duckdb_threads_per_worker": args.duckdb_threads_per_worker,
                    "parallel_chunk_rows": args.parallel_chunk_rows,
                    "temp_shard_root": str(args.temp_shard_root),
                    "temp_run_dir": str(run_dir),
                    "shard_count": len(holdout_shards),
                    "fail_fast": args.fail_fast,
                },
                "sources": {src: dataclasses.asdict(st) for src, st in source_stats.items()},
            }
            write_json(holdout_meta_path, meta)
        success = True
        log(f"holdout initialized: {holdout_path} (sha256={sha})")
    finally:
        maybe_cleanup_temp(
            run_dir,
            should_cleanup=args.cleanup_temp_on_success if success else args.cleanup_temp_on_failure,
            reason="success" if success else "failure",
        )


def build_runtime_args() -> SimpleNamespace:
    validate_parallel_args(
        num_workers=NUM_WORKERS,
        duckdb_threads_per_worker=DUCKDB_THREADS_PER_WORKER,
        parallel_chunk_rows=PARALLEL_CHUNK_ROWS,
    )
    return SimpleNamespace(
        raw_root=str(RAW_ROOT),
        out_dir=str(OUT_DIR),
        holdout_path=str(HOLDOUT_PATH),
        holdout_meta_path=str(HOLDOUT_META_PATH),
        seed=SEED,
        length_min=LENGTH_MIN,
        length_max=LENGTH_MAX,
        max_bytes_per_source=TRAIN_MAX_BYTES_PER_SOURCE,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        num_workers=NUM_WORKERS,
        duckdb_threads_per_worker=DUCKDB_THREADS_PER_WORKER,
        parallel_chunk_rows=PARALLEL_CHUNK_ROWS,
        temp_shard_root=str(TEMP_SHARD_ROOT),
        cleanup_temp_on_success=CLEANUP_TEMP_ON_SUCCESS,
        cleanup_temp_on_failure=CLEANUP_TEMP_ON_FAILURE,
        fail_fast=FAIL_FAST,
    )


def init_runtime_args() -> SimpleNamespace:
    validate_parallel_args(
        num_workers=NUM_WORKERS,
        duckdb_threads_per_worker=DUCKDB_THREADS_PER_WORKER,
        parallel_chunk_rows=PARALLEL_CHUNK_ROWS,
    )
    validate_holdout_target_args(
        percent=HOLDOUT_TARGET_PERCENT,
        bytes_cap=HOLDOUT_TARGET_BYTES_CAP,
    )
    return SimpleNamespace(
        raw_root=str(RAW_ROOT),
        holdout_path=str(HOLDOUT_PATH),
        holdout_meta_path=str(HOLDOUT_META_PATH),
        holdout_target_percent=HOLDOUT_TARGET_PERCENT,
        holdout_target_bytes_cap=HOLDOUT_TARGET_BYTES_CAP,
        seed=SEED,
        length_min=LENGTH_MIN,
        length_max=LENGTH_MAX,
        force=INIT_HOLDOUT_FORCE_OVERWRITE,
        num_workers=NUM_WORKERS,
        duckdb_threads_per_worker=DUCKDB_THREADS_PER_WORKER,
        parallel_chunk_rows=PARALLEL_CHUNK_ROWS,
        temp_shard_root=str(TEMP_SHARD_ROOT),
        cleanup_temp_on_success=CLEANUP_TEMP_ON_SUCCESS,
        cleanup_temp_on_failure=CLEANUP_TEMP_ON_FAILURE,
        fail_fast=FAIL_FAST,
    )


def main() -> None:
    log(f"run mode: {RUN_MODE}")
    if RUN_MODE == "init_holdout":
        init_fixed_holdout(init_runtime_args())
        return
    if RUN_MODE == "build":
        build_tokenizer(build_runtime_args())
        return
    raise ValueError(
        f"invalid RUN_MODE={RUN_MODE!r}. expected one of: 'init_holdout', 'build'"
    )


if __name__ == "__main__":
    main()
