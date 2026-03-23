from __future__ import annotations

import importlib
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator

import duckdb
import pyarrow.parquet as pq

from src.preprocess import common
from src.preprocess.common import ensure_parent_dir, log, now_utc_iso, progress_bar, write_json


# =========================
# User configuration block
# =========================
INPUT_ROOT = Path("data/korean_processed/near_dedup")
OUTPUT_ROOT = Path("data/korean_processed/final_lancedb")
TEMP_SOURCE_ROOT = Path("data/korean_processed/lance_by_source")
TOKENIZER_JSON_PATH = Path("data/tokenizers/korean_bbpe_v2/tokenizer.json")
SOURCE_SHARD_TARGET_BYTES = 1024 * 1024 * 1024
SFT_MAX_TOKENS = 1024
PT_TARGET_TOKENS = 960
PT_MAX_TOKENS = 1024
READ_BATCH_ROWS = 2048
OVERWRITE_OUTPUT = False
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class SourceShardInfo:
    source: str
    shard_index: int
    row_count: int
    byte_size: int
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuildLanceConfig:
    input_root: Path
    output_root: Path
    temp_source_root: Path
    tokenizer_json_path: Path
    source_shard_target_bytes: int
    read_batch_rows: int
    overwrite_output: bool


@dataclass(frozen=True)
class BuildLanceSummary:
    run_id: str
    input_root: str
    output_root: str
    temp_source_root: str
    tokenizer_json_path: str
    reader_backend: str
    input_shard_count: int
    input_row_count: int
    output_row_count: int
    source_count: int
    source_stats: dict[str, dict[str, int]]
    data_usage_counts: dict[str, int]
    split_counts: dict[str, int]
    output_shards: list[dict[str, Any]]
    final_dataset_uri: str
    manifest_path: str
    progress_path: str
    started_at: str
    finished_at: str
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_input_shards(input_root: Path) -> list[Path]:
    return common.stable_sorted_paths(input_root.glob("part-*.parquet"))


def run_build_lance(config: BuildLanceConfig) -> BuildLanceSummary:
    _validate_config(config)
    input_shards = discover_input_shards(config.input_root)
    if not input_shards:
        raise FileNotFoundError(f"no input parquet shards found under {config.input_root}")
    if not config.tokenizer_json_path.exists():
        raise FileNotFoundError(
            f"tokenizer_json_path does not exist: {config.tokenizer_json_path}"
        )

    manifest_path = config.output_root / "_meta" / "build_lance_manifest.json"
    progress_path = config.output_root / "_meta" / "build_lance_progress.json"
    final_dataset_uri = config.output_root / "dataset.lance"
    _prepare_output_roots(config)

    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    common.TOKENIZER_JSON_PATH = config.tokenizer_json_path

    reader_backend = _detect_reader_backend(input_shards)
    source_names = _collect_source_names(config.input_root)
    started_at = now_utc_iso()
    started_perf = perf_counter()
    run_id = started_at.replace("-", "").replace(":", "").replace(".", "")
    log(
        "[build_lance] start "
        f"input_root={config.input_root} output_root={config.output_root} "
        f"temp_source_root={config.temp_source_root} sources={len(source_names)} "
        f"reader_backend={reader_backend}"
    )

    input_row_count = 0
    output_row_count = 0
    source_stats: dict[str, dict[str, int]] = {}
    data_usage_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    shard_infos: list[SourceShardInfo] = []
    source_dirnames: dict[str, str] = {}

    tokenizer = common.get_tokenizer(config.tokenizer_json_path)
    count_tokens = lambda text: common.compute_token_count_from_text(text, tokenizer=tokenizer)

    progress = progress_bar(total=len(source_names), desc="build_lance/sources")
    try:
        for source in source_names:
            source_dirname = _source_dirname(source)
            source_dirnames[source] = source_dirname
            result = _materialize_source_shards(
                source=source,
                input_shards=input_shards,
                input_glob=str(config.input_root / "part-*.parquet"),
                reader_backend=reader_backend,
                source_output_root=config.temp_source_root / source_dirname,
                shard_target_bytes=config.source_shard_target_bytes,
                batch_rows=config.read_batch_rows,
                count_tokens=count_tokens,
                progress_path=progress_path,
            )
            input_row_count += result["input_row_count"]
            output_row_count += result["output_row_count"]
            source_stats[source] = {
                "row_count": result["output_row_count"],
                "byte_size": result["byte_size"],
                "token_count": result["token_count"],
            }
            data_usage_counts.update(result["data_usage_counts"])
            split_counts.update(result["split_counts"])
            shard_infos.extend(result["shard_infos"])
            progress.update(1)
    finally:
        progress.close()

    _append_all_sources_to_lance(
        final_dataset_uri=final_dataset_uri,
        source_names=source_names,
        source_dirnames=source_dirnames,
        temp_source_root=config.temp_source_root,
        progress_path=progress_path,
    )

    finished_at = now_utc_iso()
    duration_sec = round(perf_counter() - started_perf, 3)
    summary = BuildLanceSummary(
        run_id=run_id,
        input_root=str(config.input_root),
        output_root=str(config.output_root),
        temp_source_root=str(config.temp_source_root),
        tokenizer_json_path=str(config.tokenizer_json_path),
        reader_backend=reader_backend,
        input_shard_count=len(input_shards),
        input_row_count=input_row_count,
        output_row_count=output_row_count,
        source_count=len(source_names),
        source_stats=source_stats,
        data_usage_counts=dict(data_usage_counts),
        split_counts=dict(split_counts),
        output_shards=[info.to_dict() for info in shard_infos],
        final_dataset_uri=str(final_dataset_uri),
        manifest_path=str(manifest_path),
        progress_path=str(progress_path),
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
    )
    write_json(manifest_path, summary.to_dict())
    _write_progress(
        progress_path,
        phase="completed",
        last_completed_source=source_names[-1] if source_names else None,
        last_completed_shard_index=shard_infos[-1].shard_index if shard_infos else None,
        final_dataset_uri=str(final_dataset_uri),
        temp_source_root=str(config.temp_source_root),
    )
    if config.temp_source_root.exists():
        shutil.rmtree(config.temp_source_root)
    log(
        "[build_lance] completed "
        f"input_rows={input_row_count} output_rows={output_row_count} "
        f"sources={len(source_names)} duration_sec={duration_sec}"
    )
    return summary


def _validate_config(config: BuildLanceConfig) -> None:
    if config.source_shard_target_bytes < 1:
        raise ValueError("source_shard_target_bytes must be >= 1")
    if config.read_batch_rows < 1:
        raise ValueError("read_batch_rows must be >= 1")


def _prepare_output_roots(config: BuildLanceConfig) -> None:
    roots = [config.output_root, config.temp_source_root]
    for root in roots:
        if root.exists():
            existing_paths = list(root.iterdir())
            if existing_paths and not config.overwrite_output:
                raise FileExistsError(
                    f"output_root already contains files: {root}. "
                    "Set OVERWRITE_OUTPUT=True to rebuild."
                )
            if config.overwrite_output:
                shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "_meta").mkdir(parents=True, exist_ok=True)


def _detect_reader_backend(probe_shards: list[Path]) -> str:
    for probe_shard in probe_shards:
        try:
            parquet_file = pq.ParquetFile(probe_shard)
            for _ in parquet_file.iter_batches(columns=common.STAGING_SCHEMA.names, batch_size=1):
                break
        except Exception as exc:
            log(
                f"[build_lance] pyarrow reader probe failed for {probe_shard}: "
                f"{type(exc).__name__}: {exc}"
            )
            return "duckdb"
    return "pyarrow"


def _collect_source_names(input_root: Path) -> list[str]:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT DISTINCT source FROM read_parquet(?) ORDER BY source",
            [str(input_root / "part-*.parquet")],
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def _materialize_source_shards(
    *,
    source: str,
    input_shards: list[Path],
    input_glob: str,
    reader_backend: str,
    source_output_root: Path,
    shard_target_bytes: int,
    batch_rows: int,
    count_tokens: TokenCounter,
    progress_path: Path,
) -> dict[str, Any]:
    if source_output_root.exists():
        shutil.rmtree(source_output_root)
    source_output_root.mkdir(parents=True, exist_ok=True)

    buffer_rows: list[dict[str, Any]] = []
    buffer_bytes = 0
    shard_index = 1
    input_row_count = 0
    output_row_count = 0
    total_byte_size = 0
    total_token_count = 0
    data_usage_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    shard_infos: list[SourceShardInfo] = []

    for raw_row in _iter_source_input_rows(
        source=source,
        input_shards=input_shards,
        input_glob=input_glob,
        reader_backend=reader_backend,
        batch_rows=batch_rows,
    ):
        input_row_count += 1
        for output_row in _expand_input_row(raw_row, count_tokens=count_tokens):
            row_byte_size = _estimate_row_bytes(output_row)
            if buffer_rows and buffer_bytes + row_byte_size > shard_target_bytes:
                shard_info = _flush_source_rows(
                    rows=buffer_rows,
                    byte_size=buffer_bytes,
                    source=source,
                    shard_index=shard_index,
                    source_output_root=source_output_root,
                )
                shard_infos.append(shard_info)
                _write_progress(
                    progress_path,
                    phase="materialize_source_shards",
                    last_completed_source=source,
                    last_completed_shard_index=shard_index,
                    final_dataset_uri=None,
                    temp_source_root=str(source_output_root.parent),
                )
                shard_index += 1
                buffer_rows = []
                buffer_bytes = 0
            buffer_rows.append(output_row)
            buffer_bytes += row_byte_size
            output_row_count += 1
            total_byte_size += row_byte_size
            total_token_count += int(output_row["token_count"])
            data_usage_counts[str(output_row["data_usage"])] += 1
            split_counts[str(output_row["split"])] += 1

    if buffer_rows:
        shard_info = _flush_source_rows(
            rows=buffer_rows,
            byte_size=buffer_bytes,
            source=source,
            shard_index=shard_index,
            source_output_root=source_output_root,
        )
        shard_infos.append(shard_info)
        _write_progress(
            progress_path,
            phase="materialize_source_shards",
            last_completed_source=source,
            last_completed_shard_index=shard_index,
            final_dataset_uri=None,
            temp_source_root=str(source_output_root.parent),
        )

    return {
        "input_row_count": input_row_count,
        "output_row_count": output_row_count,
        "byte_size": total_byte_size,
        "token_count": total_token_count,
        "data_usage_counts": dict(data_usage_counts),
        "split_counts": dict(split_counts),
        "shard_infos": shard_infos,
    }


def _iter_source_input_rows(
    *,
    source: str,
    input_shards: list[Path],
    input_glob: str,
    reader_backend: str,
    batch_rows: int,
) -> Iterator[dict[str, Any]]:
    if reader_backend == "pyarrow":
        yield from _iter_source_input_rows_pyarrow(
            source=source,
            input_shards=input_shards,
            batch_rows=batch_rows,
        )
        return
    yield from _iter_source_input_rows_duckdb(
        source=source,
        input_glob=input_glob,
        batch_rows=batch_rows,
    )


def _iter_source_input_rows_pyarrow(
    *,
    source: str,
    input_shards: list[Path],
    batch_rows: int,
) -> Iterator[dict[str, Any]]:
    for shard_path in input_shards:
        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(
            columns=common.STAGING_SCHEMA.names,
            batch_size=batch_rows,
        ):
            for row in batch.to_pylist():
                if row["source"] == source:
                    yield row


def _iter_source_input_rows_duckdb(
    *,
    source: str,
    input_glob: str,
    batch_rows: int,
) -> Iterator[dict[str, Any]]:
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            """
            SELECT source, data_usage, split, content, messages, token_count
            FROM read_parquet(?)
            WHERE source = ?
            """,
            [input_glob, source],
        )
        while True:
            rows = cursor.fetchmany(batch_rows)
            if not rows:
                break
            for row in rows:
                yield {
                    "source": row[0],
                    "data_usage": row[1],
                    "split": row[2],
                    "content": row[3],
                    "messages": _normalize_messages(row[4]),
                    "token_count": row[5],
                }
    finally:
        connection.close()


def _normalize_messages(raw_messages: Any) -> list[dict[str, str]] | None:
    if raw_messages is None:
        return None
    normalized: list[dict[str, str]] = []
    for raw_message in raw_messages:
        role = raw_message["role"]
        content = raw_message["content"]
        if not isinstance(role, str):
            raise ValueError(f"message role must be string, got {type(role).__name__}")
        if not isinstance(content, str):
            raise ValueError(f"message content must be string, got {type(content).__name__}")
        normalized.append({"role": role, "content": content})
    return normalized


def _expand_input_row(
    row: dict[str, Any],
    *,
    count_tokens: TokenCounter,
) -> list[dict[str, Any]]:
    data_usage = row["data_usage"]
    if data_usage == "SFT":
        messages = _require_messages(row)
        return _expand_sft_row(row, messages=messages, count_tokens=count_tokens)
    if data_usage == "PT":
        content = _require_content(row)
        return _expand_pt_like_row(
            source=str(row["source"]),
            data_usage="PT",
            split=str(row["split"]),
            content=content,
            count_tokens=count_tokens,
        )
    if data_usage == "REASONING":
        messages = _require_messages(row)
        content = _require_content(row)
        pt_rows = _expand_pt_like_row(
            source=str(row["source"]),
            data_usage="PT",
            split=str(row["split"]),
            content=content,
            count_tokens=count_tokens,
        )
        reasoning_serialized = common.serialize_messages_for_lance(messages)
        reasoning_row = {
            "source": str(row["source"]),
            "data_usage": "REASONING",
            "split": str(row["split"]),
            "content": None,
            "messages": reasoning_serialized,
            "token_count": count_tokens(reasoning_serialized),
        }
        _validate_lance_row(reasoning_row, count_tokens=count_tokens)
        return pt_rows + [reasoning_row]
    raise ValueError(f"unsupported data_usage: {data_usage}")


def _expand_sft_row(
    row: dict[str, Any],
    *,
    messages: list[dict[str, str]],
    count_tokens: TokenCounter,
) -> list[dict[str, Any]]:
    system_messages, pairs = _split_system_and_pairs(messages)
    if not pairs:
        raise ValueError("SFT row must contain at least one user-assistant pair")
    return _split_sft_pairs_recursively(
        source=str(row["source"]),
        split=str(row["split"]),
        system_messages=system_messages,
        pairs=pairs,
        count_tokens=count_tokens,
    )


def _split_system_and_pairs(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[list[dict[str, str]]]]:
    system_messages: list[dict[str, str]] = []
    dialogue_messages: list[dict[str, str]] = []
    seen_non_system = False
    for message in messages:
        if message["role"] == "system" and not seen_non_system:
            system_messages.append(message)
            continue
        seen_non_system = True
        dialogue_messages.append(message)
    if not dialogue_messages:
        return system_messages, []
    if dialogue_messages[-1]["role"] != "assistant":
        raise ValueError("dialogue messages must end with assistant")
    pairs: list[list[dict[str, str]]] = []
    index = 0
    while index < len(dialogue_messages):
        pair = dialogue_messages[index : index + 2]
        if len(pair) != 2:
            raise ValueError("dialogue messages must be composed of user-assistant pairs")
        if pair[0]["role"] != "user" or pair[1]["role"] != "assistant":
            raise ValueError("dialogue messages must alternate user then assistant")
        pairs.append(pair)
        index += 2
    return system_messages, pairs


def _split_sft_pairs_recursively(
    *,
    source: str,
    split: str,
    system_messages: list[dict[str, str]],
    pairs: list[list[dict[str, str]]],
    count_tokens: TokenCounter,
) -> list[dict[str, Any]]:
    dialogue_messages = [message for pair in pairs for message in pair]
    messages = [*system_messages, *dialogue_messages]
    serialized = common.serialize_messages_for_lance(messages)
    token_count = count_tokens(serialized)
    if token_count <= SFT_MAX_TOKENS:
        row = {
            "source": source,
            "data_usage": "SFT",
            "split": split,
            "content": common.build_content_from_messages(dialogue_messages),
            "messages": serialized,
            "token_count": token_count,
        }
        _validate_lance_row(row, count_tokens=count_tokens)
        return [row]
    if len(pairs) <= 1:
        raise ValueError("single SFT user-assistant pair exceeds token limit and cannot be split")

    chunk_count = min(len(pairs), max(2, math.ceil(token_count / SFT_MAX_TOKENS)))
    rows: list[dict[str, Any]] = []
    for pair_chunk in _split_pairs_evenly(pairs, chunk_count):
        rows.extend(
            _split_sft_pairs_recursively(
                source=source,
                split=split,
                system_messages=system_messages,
                pairs=pair_chunk,
                count_tokens=count_tokens,
            )
        )
    return rows


def _split_pairs_evenly(
    pairs: list[list[dict[str, str]]],
    chunk_count: int,
) -> list[list[list[dict[str, str]]]]:
    base_size = len(pairs) // chunk_count
    remainder = len(pairs) % chunk_count
    chunks: list[list[list[dict[str, str]]]] = []
    start = 0
    for chunk_index in range(chunk_count):
        size = base_size + (1 if chunk_index < remainder else 0)
        if size < 1:
            continue
        chunks.append(pairs[start : start + size])
        start += size
    return chunks


def _expand_pt_like_row(
    *,
    source: str,
    data_usage: str,
    split: str,
    content: str,
    count_tokens: TokenCounter,
) -> list[dict[str, Any]]:
    parts = _split_pt_content(content, count_tokens=count_tokens)
    rows: list[dict[str, Any]] = []
    for part in parts:
        token_count = count_tokens(part)
        row = {
            "source": source,
            "data_usage": data_usage,
            "split": split,
            "content": part,
            "messages": None,
            "token_count": token_count,
        }
        _validate_lance_row(row, count_tokens=count_tokens)
        rows.append(row)
    return rows


def _split_pt_content(
    content: str,
    *,
    count_tokens: TokenCounter,
) -> list[str]:
    total_tokens = count_tokens(content)
    if total_tokens <= PT_MAX_TOKENS:
        return [content]

    units = _split_text_units(content)
    target_chunk_count = max(2, math.ceil(total_tokens / PT_TARGET_TOKENS))
    chunks = _build_pt_chunks(units, target_chunk_count, count_tokens)
    results: list[str] = []
    for chunk in chunks:
        token_count = count_tokens(chunk)
        if token_count <= PT_MAX_TOKENS:
            results.append(chunk)
            continue
        if len(_split_text_units(chunk)) > 1:
            results.extend(_split_pt_content(chunk, count_tokens=count_tokens))
            continue
        results.extend(_hard_split_text(chunk, count_tokens=count_tokens))
    return results


def _split_text_units(content: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for char in content:
        current.append(char)
        if char in {".", "?", "!", "\n"}:
            text = "".join(current).strip()
            if text:
                parts.append(text)
            current = []
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts or [content.strip()]


def _build_pt_chunks(
    units: list[str],
    target_chunk_count: int,
    count_tokens: TokenCounter,
) -> list[str]:
    if target_chunk_count <= 1 or len(units) <= 1:
        return [" ".join(unit.strip() for unit in units if unit.strip())]

    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    for unit in units:
        candidate = " ".join([*current_chunk, unit]).strip()
        remaining_units = len(units) - (sum(len(chunk) for chunk in chunks) + len(current_chunk) + 1)
        remaining_chunk_slots = target_chunk_count - len(chunks) - 1
        if (
            current_chunk
            and count_tokens(candidate) > PT_MAX_TOKENS
            and remaining_units >= remaining_chunk_slots
        ):
            chunks.append(current_chunk)
            current_chunk = [unit]
            continue
        current_chunk.append(unit)
    if current_chunk:
        chunks.append(current_chunk)

    normalized = [" ".join(part.strip() for part in chunk if part.strip()) for chunk in chunks]
    if len(normalized) >= target_chunk_count:
        return normalized

    while len(normalized) < target_chunk_count:
        split_index = next((index for index, chunk in enumerate(normalized) if " " in chunk), None)
        if split_index is None:
            break
        split_units = _split_text_units(normalized[split_index])
        if len(split_units) <= 1:
            break
        midpoint = max(1, len(split_units) // 2)
        left = " ".join(split_units[:midpoint]).strip()
        right = " ".join(split_units[midpoint:]).strip()
        normalized = [*normalized[:split_index], left, right, *normalized[split_index + 1 :]]
    return normalized


def _hard_split_text(
    content: str,
    *,
    count_tokens: TokenCounter,
) -> list[str]:
    words = content.split()
    if len(words) <= 1:
        midpoint = max(1, len(content) // 2)
        left = content[:midpoint].strip()
        right = content[midpoint:].strip()
        if not left or not right:
            raise ValueError("cannot split PT content below token limit")
        return [left, right]

    chunks: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join([*current_words, word]).strip()
        if current_words and count_tokens(candidate) > PT_MAX_TOKENS:
            chunks.append(" ".join(current_words))
            current_words = [word]
            continue
        current_words.append(word)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def _estimate_row_bytes(row: dict[str, Any]) -> int:
    byte_size = 4
    for field in ("source", "data_usage", "split", "content", "messages"):
        value = row[field]
        if value is None:
            continue
        byte_size += len(str(value).encode("utf-8"))
    return byte_size


def _flush_source_rows(
    *,
    rows: list[dict[str, Any]],
    byte_size: int,
    source: str,
    shard_index: int,
    source_output_root: Path,
) -> SourceShardInfo:
    shard_path = source_output_root / f"part-{shard_index:06d}.parquet"
    ensure_parent_dir(shard_path)
    pq.write_table(common.lance_rows_to_table(rows), shard_path)
    return SourceShardInfo(
        source=source,
        shard_index=shard_index,
        row_count=len(rows),
        byte_size=byte_size,
        path=str(shard_path),
    )


def _append_all_sources_to_lance(
    *,
    final_dataset_uri: Path,
    source_names: list[str],
    source_dirnames: dict[str, str],
    temp_source_root: Path,
    progress_path: Path,
) -> None:
    dataset_exists = False
    for source in source_names:
        source_root = temp_source_root / source_dirnames[source]
        shard_paths = common.stable_sorted_paths(source_root.glob("part-*.parquet"))
        for shard_index, shard_path in enumerate(shard_paths, start=1):
            table = pq.read_table(shard_path).cast(common.LANCE_FINAL_SCHEMA)
            _write_lance_table(table, final_dataset_uri, dataset_exists=dataset_exists)
            dataset_exists = True
            _write_progress(
                progress_path,
                phase="append_lance",
                last_completed_source=source,
                last_completed_shard_index=shard_index,
                final_dataset_uri=str(final_dataset_uri),
                temp_source_root=str(temp_source_root),
            )


def _write_lance_table(table: Any, final_dataset_uri: Path, *, dataset_exists: bool) -> None:
    lance = _load_lance_module()
    mode = "append" if dataset_exists else "create"
    ensure_parent_dir(final_dataset_uri)
    lance.write_dataset(table, str(final_dataset_uri), mode=mode)


def _load_lance_module() -> Any:
    return importlib.import_module("lance")


def _write_progress(
    progress_path: Path,
    *,
    phase: str,
    last_completed_source: str | None,
    last_completed_shard_index: int | None,
    final_dataset_uri: str | None,
    temp_source_root: str | None,
) -> None:
    write_json(
        progress_path,
        {
            "updated_at": now_utc_iso(),
            "phase": phase,
            "last_completed_source": last_completed_source,
            "last_completed_shard_index": last_completed_shard_index,
            "final_dataset_uri": final_dataset_uri,
            "temp_source_root": temp_source_root,
        },
    )


def _require_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    return messages


def _require_content(row: dict[str, Any]) -> str:
    content = row.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")
    return content


def _validate_lance_row(
    row: dict[str, Any],
    *,
    count_tokens: TokenCounter,
) -> None:
    if row["data_usage"] not in common.ALLOWED_DATA_USAGES:
        raise ValueError(f"unsupported data_usage: {row['data_usage']}")
    if row["split"] not in common.ALLOWED_SPLITS:
        raise ValueError(f"unsupported split: {row['split']}")
    if row["content"] is None and row["messages"] is None:
        raise ValueError("content/messages cannot both be null")
    if row["data_usage"] == "PT":
        if row["content"] is None:
            raise ValueError("PT row content cannot be null")
        expected = count_tokens(str(row["content"]))
    else:
        if row["messages"] is None:
            raise ValueError("SFT/REASONING row messages cannot be null")
        expected = count_tokens(str(row["messages"]))
    if int(row["token_count"]) != expected:
        raise ValueError(
            f"token_count mismatch: stored={row['token_count']} expected={expected}"
        )


def _source_dirname(source: str) -> str:
    digest = sha1(source.encode("utf-8")).hexdigest()[:12]
    safe = "".join(char if char.isalnum() else "_" for char in source).strip("_")
    safe = safe[:48] or "source"
    return f"{digest}-{safe}"


def build_default_config() -> BuildLanceConfig:
    return BuildLanceConfig(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        temp_source_root=TEMP_SOURCE_ROOT,
        tokenizer_json_path=TOKENIZER_JSON_PATH,
        source_shard_target_bytes=SOURCE_SHARD_TARGET_BYTES,
        read_batch_rows=READ_BATCH_ROWS,
        overwrite_output=OVERWRITE_OUTPUT,
    )


def main() -> None:
    run_build_lance(build_default_config())


if __name__ == "__main__":
    main()
