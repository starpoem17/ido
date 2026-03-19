from __future__ import annotations

import heapq
import multiprocessing as mp
import os
import shutil
import signal
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from src.tokenizer import common


NEAR_DEDUP_ROOT = Path("data/korean_processed/near_dedup")
TOKENIZER_ROOT = Path("data/tokenizers")
TOKENIZER_NAME_PREFIX = "korean_bbpe_v"
DEBUG_RUN_ROOT = Path("data/tokenizers/_tmp")
VOCAB_SIZE = 48000
TOP_TOKEN_COUNT = 1000
NUM_WORKERS = 7
TRAIN_CHUNK_ENABLE = True
TRAIN_CHUNK_MAX_CHARS = 2048
TRAIN_CHUNK_MAX_UTF8_BYTES = 4096
TRAIN_CHUNK_BREAK_PRIORITY = ("\n\n", "\n", " ")
TRAINER_MIN_FREQUENCY = 8
TRAINER_MAX_TOKEN_LENGTH = 64
TRAINER_LIMIT_ALPHABET = 4096
TOP_LONG_ROWS = 100
MEMORY_WARNING_AVAILABLE_MB = 1024.0
MEMORY_WARNING_SWAP_FREE_MB = 4096.0
DEBUG_HEARTBEAT_INTERVAL_SEC = 2.0
TRAIN_PROGRESS_LOG_EVERY_CHUNKS = 100000
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|eot_id|>",
    "<|pad|>",
]
PARQUET_BATCH_ROWS = 4096
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


@dataclass(frozen=True)
class BuildTokenizerConfig:
    near_dedup_root: Path
    tokenizer_root: Path
    tokenizer_name_prefix: str
    debug_run_root: Path
    vocab_size: int
    top_token_count: int
    num_workers: int
    train_chunk_enable: bool
    train_chunk_max_chars: int
    train_chunk_max_utf8_bytes: int
    train_chunk_break_priority: tuple[str, ...]
    trainer_min_frequency: int
    trainer_max_token_length: int
    trainer_limit_alphabet: int
    top_long_rows: int
    memory_warning_available_mb: float
    memory_warning_swap_free_mb: float
    debug_heartbeat_interval_sec: float
    train_progress_log_every_chunks: int
    special_tokens: tuple[str, ...]
    parquet_batch_rows: int
    enable_tqdm: bool
    enable_debug_log: bool
    tqdm_mininterval_sec: float


@dataclass(frozen=True)
class LongRowRecord:
    path: str
    row_index: int
    content_chars: int
    content_utf8_bytes: int
    estimated_chunk_count: int


@dataclass(frozen=True)
class InputScanStats:
    input_row_count: int
    valid_row_count: int
    filtered_row_count: int
    max_input_content_chars: int
    max_input_content_bytes: int
    train_chunk_count: int
    max_training_chunk_chars: int
    top_long_rows: list[LongRowRecord]


class _DebugRuntime:
    def __init__(self, *, debug_run_dir: Path) -> None:
        self.debug_run_dir = debug_run_dir
        self.events_path = debug_run_dir / "events.jsonl"
        self.heartbeat_path = debug_run_dir / "memory_heartbeat.jsonl"
        self.failure_context_path = debug_run_dir / "failure_context.json"
        self.long_rows_path = debug_run_dir / "long_rows.jsonl"
        self._lock = threading.Lock()
        self.phase = "init"
        self.rows_seen = 0
        self.chunks_emitted = 0
        self.current_path: str | None = None
        self.max_input_chars_seen = 0
        self.child_pid: int | None = None

    def set_phase(self, phase: str, **payload: Any) -> None:
        with self._lock:
            self.phase = phase
        self.log_event("phase", phase=phase, **payload)

    def set_child_pid(self, child_pid: int | None) -> None:
        with self._lock:
            self.child_pid = child_pid

    def update_progress(
        self,
        *,
        rows_seen: int | None = None,
        chunks_emitted: int | None = None,
        current_path: str | None = None,
        max_input_chars_seen: int | None = None,
    ) -> None:
        with self._lock:
            if rows_seen is not None:
                self.rows_seen = rows_seen
            if chunks_emitted is not None:
                self.chunks_emitted = chunks_emitted
            if current_path is not None:
                self.current_path = current_path
            if max_input_chars_seen is not None:
                self.max_input_chars_seen = max_input_chars_seen

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "phase": self.phase,
                "rows_seen": self.rows_seen,
                "chunks_emitted": self.chunks_emitted,
                "current_path": self.current_path,
                "max_input_chars_seen": self.max_input_chars_seen,
                "child_pid": self.child_pid,
            }

    def log_event(self, event: str, **payload: Any) -> None:
        row = {"timestamp": common.now_utc_iso(), "event": event}
        row.update(self.snapshot())
        row.update(payload)
        common.append_jsonl(self.events_path, row)

    def write_failure_context(self, exc: BaseException) -> None:
        payload = {
            "timestamp": common.now_utc_iso(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        payload.update(self.snapshot())
        common.write_json(self.failure_context_path, payload)

    def write_failure_payload(self, payload: dict[str, Any]) -> None:
        row = {"timestamp": common.now_utc_iso()}
        row.update(self.snapshot())
        row.update(payload)
        common.write_json(self.failure_context_path, row)


class _HeartbeatThread(threading.Thread):
    def __init__(
        self,
        *,
        runtime: _DebugRuntime,
        interval_sec: float,
        warning_available_mb: float,
        warning_swap_free_mb: float,
    ) -> None:
        super().__init__(daemon=True)
        self._runtime = runtime
        self._interval_sec = interval_sec
        self._warning_available_mb = warning_available_mb
        self._warning_swap_free_mb = warning_swap_free_mb
        self._stop_event = threading.Event()
        self._last_oom_risk_level: str | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._emit_heartbeat()
        while not self._stop_event.wait(self._interval_sec):
            self._emit_heartbeat()

    def _emit_heartbeat(self) -> None:
        snapshot = self._runtime.snapshot()
        child_pid = snapshot.get("child_pid")
        target_pid = child_pid if child_pid is not None else None
        memory = common.snapshot_process_memory(target_pid)
        row = {"timestamp": common.now_utc_iso()}
        row.update(snapshot)
        row.update(
            {
                "target_pid": memory.get("pid"),
                "target_kind": "child" if child_pid is not None else "self",
                "target_rss_mb": memory.get("rss_mb"),
                "target_hwm_mb": memory.get("hwm_mb"),
                "target_vms_mb": memory.get("vms_mb"),
                "mem_available_mb": memory.get("mem_available_mb"),
                "swap_free_mb": memory.get("swap_free_mb"),
            }
        )
        oom_risk_level = _classify_oom_risk(
            mem_available_mb=memory.get("mem_available_mb"),
            swap_free_mb=memory.get("swap_free_mb"),
            warning_available_mb=self._warning_available_mb,
            warning_swap_free_mb=self._warning_swap_free_mb,
        )
        row["oom_risk_level"] = oom_risk_level
        common.append_jsonl(self._runtime.heartbeat_path, row)
        if oom_risk_level != self._last_oom_risk_level and oom_risk_level != "normal":
            self._runtime.log_event(
                "memory_warning",
                oom_risk_level=oom_risk_level,
                mem_available_mb=memory.get("mem_available_mb"),
                swap_free_mb=memory.get("swap_free_mb"),
                target_pid=memory.get("pid"),
            )
        self._last_oom_risk_level = oom_risk_level


def build_default_config() -> BuildTokenizerConfig:
    return BuildTokenizerConfig(
        near_dedup_root=NEAR_DEDUP_ROOT,
        tokenizer_root=TOKENIZER_ROOT,
        tokenizer_name_prefix=TOKENIZER_NAME_PREFIX,
        debug_run_root=DEBUG_RUN_ROOT,
        vocab_size=VOCAB_SIZE,
        top_token_count=TOP_TOKEN_COUNT,
        num_workers=NUM_WORKERS,
        train_chunk_enable=TRAIN_CHUNK_ENABLE,
        train_chunk_max_chars=TRAIN_CHUNK_MAX_CHARS,
        train_chunk_max_utf8_bytes=TRAIN_CHUNK_MAX_UTF8_BYTES,
        train_chunk_break_priority=TRAIN_CHUNK_BREAK_PRIORITY,
        trainer_min_frequency=TRAINER_MIN_FREQUENCY,
        trainer_max_token_length=TRAINER_MAX_TOKEN_LENGTH,
        trainer_limit_alphabet=TRAINER_LIMIT_ALPHABET,
        top_long_rows=TOP_LONG_ROWS,
        memory_warning_available_mb=MEMORY_WARNING_AVAILABLE_MB,
        memory_warning_swap_free_mb=MEMORY_WARNING_SWAP_FREE_MB,
        debug_heartbeat_interval_sec=DEBUG_HEARTBEAT_INTERVAL_SEC,
        train_progress_log_every_chunks=TRAIN_PROGRESS_LOG_EVERY_CHUNKS,
        special_tokens=tuple(SPECIAL_TOKENS),
        parquet_batch_rows=PARQUET_BATCH_ROWS,
        enable_tqdm=ENABLE_TQDM,
        enable_debug_log=ENABLE_DEBUG_LOG,
        tqdm_mininterval_sec=TQDM_MININTERVAL_SEC,
    )


def run_build_tokenizer(
    config: BuildTokenizerConfig | None = None,
) -> common.BuildSummary:
    config = config or build_default_config()
    _validate_config(config)
    started_at = common.now_utc_iso()
    started_perf = time.perf_counter()
    input_paths = common.collect_part_paths(config.near_dedup_root)
    if not input_paths:
        raise ValueError(f"no input parquet files found under: {config.near_dedup_root}")

    output_dir = common.find_next_version_dir(
        config.tokenizer_root, config.tokenizer_name_prefix
    )
    if output_dir.exists():
        raise ValueError(f"output dir already exists: {output_dir}")

    debug_run_dir = common.create_debug_run_dir(
        config.debug_run_root, "build_tokenizer_run"
    )
    staging_dir = debug_run_dir / "staging_tokenizer"
    runtime = _DebugRuntime(debug_run_dir=debug_run_dir)
    common.write_json(
        debug_run_dir / "run_config.json",
        {
            **asdict(config),
            "output_dir": str(output_dir),
            "started_at": started_at,
        },
    )
    heartbeat = _HeartbeatThread(
        runtime=runtime,
        interval_sec=config.debug_heartbeat_interval_sec,
        warning_available_mb=config.memory_warning_available_mb,
        warning_swap_free_mb=config.memory_warning_swap_free_mb,
    )
    heartbeat.start()

    common.log(
        (
            "[build_tokenizer] start "
            f"input_root={config.near_dedup_root} "
            f"input_shards={len(input_paths)} "
            f"vocab_size={config.vocab_size} "
            f"top_token_count={config.top_token_count} "
            f"num_workers={config.num_workers}"
        ),
        enabled=config.enable_debug_log,
    )
    runtime.log_event(
        "run_start",
        input_root=str(config.near_dedup_root),
        input_shards=len(input_paths),
        output_dir=str(output_dir),
        staging_dir=str(staging_dir),
    )

    try:
        runtime.set_phase("scan_start", input_shards=len(input_paths))
        scan_stats = _scan_input_stats(input_paths, config)
        runtime.set_phase(
            "scan_done",
            input_rows=scan_stats.input_row_count,
            valid_rows=scan_stats.valid_row_count,
            filtered_rows=scan_stats.filtered_row_count,
            max_input_chars=scan_stats.max_input_content_chars,
            max_input_bytes=scan_stats.max_input_content_bytes,
            train_chunk_count=scan_stats.train_chunk_count,
            max_training_chunk_chars=scan_stats.max_training_chunk_chars,
        )
        common.log(
            (
                "[build_tokenizer] input_scan "
                f"input_rows={scan_stats.input_row_count} "
                f"valid_rows={scan_stats.valid_row_count} "
                f"filtered_rows={scan_stats.filtered_row_count} "
                f"max_input_chars={scan_stats.max_input_content_chars} "
                f"max_input_bytes={scan_stats.max_input_content_bytes} "
                f"train_chunk_count={scan_stats.train_chunk_count} "
                f"max_training_chunk_chars={scan_stats.max_training_chunk_chars}"
            ),
            enabled=config.enable_debug_log,
        )
        common.write_jsonl(runtime.long_rows_path, scan_stats.top_long_rows)
        runtime.log_event(
            "long_rows_written",
            long_rows_path=str(runtime.long_rows_path),
            long_row_count=len(scan_stats.top_long_rows),
        )
        if scan_stats.valid_row_count < 1:
            raise ValueError("no valid content rows found for tokenizer training")

        runtime.set_phase(
            "train_start",
            train_chunk_count=scan_stats.train_chunk_count,
            trainer_min_frequency=config.trainer_min_frequency,
            trainer_max_token_length=config.trainer_max_token_length,
            trainer_limit_alphabet=config.trainer_limit_alphabet,
        )
        child = _spawn_training_child(
            input_paths=input_paths,
            scan_stats=scan_stats,
            config=config,
            runtime=runtime,
            staging_dir=staging_dir,
        )
        runtime.set_child_pid(child.pid)
        runtime.log_event(
            "train_child_spawned",
            child_pid=child.pid,
            staging_dir=str(staging_dir),
        )
        child.join()
        runtime.set_child_pid(None)
        exit_details = _summarize_child_exit(child.exitcode)
        runtime.log_event(
            "train_child_exit",
            child_pid=child.pid,
            child_exit_code=child.exitcode,
            child_signal=exit_details["child_signal"],
            oom_suspected=exit_details["oom_suspected"],
        )
        if child.exitcode != 0:
            payload = {
                "exception_type": "ChildProcessError",
                "exception_message": (
                    "tokenizer training child failed "
                    f"(exit_code={child.exitcode}, signal={exit_details['child_signal']}, "
                    f"oom_suspected={exit_details['oom_suspected']})"
                ),
                "child_exit_code": child.exitcode,
                "child_signal": exit_details["child_signal"],
                "oom_suspected": exit_details["oom_suspected"],
                "staging_dir": str(staging_dir),
                "long_rows_path": str(runtime.long_rows_path),
            }
            runtime.write_failure_payload(payload)
            raise RuntimeError(payload["exception_message"])
        runtime.set_phase("train_done", staging_dir=str(staging_dir))

        runtime.set_phase("top_tokens_start", tokenizer_dir=str(staging_dir))
        top_rows, total_token_count = _count_top_tokens(input_paths, staging_dir, config)
        runtime.set_phase(
            "top_tokens_done",
            total_token_count=total_token_count,
            top_rows=len(top_rows),
        )
        if total_token_count < 1:
            raise ValueError("top-token frequency pass produced zero tokens")

        common.write_jsonl(staging_dir / "top_tokens.jsonl", top_rows)
        finished_at = common.now_utc_iso()
        duration_sec = round(time.perf_counter() - started_perf, 6)
        meta = {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "output_dir": str(output_dir),
            "debug_run_dir": str(debug_run_dir),
            "input_root": str(config.near_dedup_root),
            "input_shard_count": len(input_paths),
            "input_paths": [str(path) for path in input_paths],
            "input_row_count": scan_stats.input_row_count,
            "valid_row_count": scan_stats.valid_row_count,
            "filtered_row_count": scan_stats.filtered_row_count,
            "max_input_content_chars": scan_stats.max_input_content_chars,
            "max_input_content_bytes": scan_stats.max_input_content_bytes,
            "top_long_rows_path": str(runtime.long_rows_path),
            "vocab_size": config.vocab_size,
            "special_tokens": list(config.special_tokens),
            "add_prefix_space": True,
            "trim_offsets": True,
            "top_token_count": config.top_token_count,
            "top_token_artifact_path": str(output_dir / "top_tokens.jsonl"),
            "top_token_frequency_basis": "actual token frequency from encoding the full unchunked training corpus",
            "total_token_count": total_token_count,
            "train_chunk_enable": config.train_chunk_enable,
            "train_chunk_max_chars": config.train_chunk_max_chars,
            "train_chunk_max_utf8_bytes": config.train_chunk_max_utf8_bytes,
            "train_chunk_break_priority": list(config.train_chunk_break_priority),
            "train_chunk_count": scan_stats.train_chunk_count,
            "max_training_chunk_chars": scan_stats.max_training_chunk_chars,
            "trainer_min_frequency": config.trainer_min_frequency,
            "trainer_max_token_length": config.trainer_max_token_length,
            "trainer_limit_alphabet": config.trainer_limit_alphabet,
            "child_exit_code": 0,
            "child_signal": None,
            "oom_suspected": False,
        }
        common.write_json(staging_dir / "meta.json", meta)
        runtime.set_phase("meta_write_done", tokenizer_dir=str(staging_dir))
        _validate_outputs(staging_dir)

        runtime.set_phase("finalize_start", output_dir=str(output_dir))
        common.ensure_dir(output_dir.parent)
        shutil.move(str(staging_dir), str(output_dir))
        runtime.set_phase("finalize_done", output_dir=str(output_dir))

        summary = common.BuildSummary(
            output_dir=output_dir,
            input_shard_count=len(input_paths),
            input_row_count=scan_stats.input_row_count,
            valid_row_count=scan_stats.valid_row_count,
            filtered_row_count=scan_stats.filtered_row_count,
            top_token_count=len(top_rows),
        )
        common.log(
            (
                "[build_tokenizer] completed "
                f"output_dir={output_dir} "
                f"valid_rows={scan_stats.valid_row_count} "
                f"train_chunk_count={scan_stats.train_chunk_count} "
                f"top_rows={len(top_rows)}"
            ),
            enabled=config.enable_debug_log,
        )
        runtime.log_event(
            "run_completed",
            output_dir=str(output_dir),
            top_rows=len(top_rows),
            duration_sec=duration_sec,
        )
        return summary
    except BaseException as exc:
        runtime.log_event(
            "run_failed",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        if not runtime.failure_context_path.exists():
            runtime.write_failure_context(exc)
        raise
    finally:
        heartbeat.stop()
        heartbeat.join(timeout=max(config.debug_heartbeat_interval_sec, 1.0))


def _validate_config(config: BuildTokenizerConfig) -> None:
    if config.vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")
    if config.top_token_count < 1:
        raise ValueError("top_token_count must be >= 1")
    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if config.parquet_batch_rows < 1:
        raise ValueError("parquet_batch_rows must be >= 1")
    if config.train_chunk_max_chars < 1:
        raise ValueError("train_chunk_max_chars must be >= 1")
    if config.train_chunk_max_utf8_bytes < 1:
        raise ValueError("train_chunk_max_utf8_bytes must be >= 1")
    if config.trainer_min_frequency < 1:
        raise ValueError("trainer_min_frequency must be >= 1")
    if config.trainer_max_token_length < 1:
        raise ValueError("trainer_max_token_length must be >= 1")
    if config.trainer_limit_alphabet < 1:
        raise ValueError("trainer_limit_alphabet must be >= 1")
    if config.top_long_rows < 1:
        raise ValueError("top_long_rows must be >= 1")
    if config.memory_warning_available_mb <= 0:
        raise ValueError("memory_warning_available_mb must be > 0")
    if config.memory_warning_swap_free_mb <= 0:
        raise ValueError("memory_warning_swap_free_mb must be > 0")
    if config.debug_heartbeat_interval_sec <= 0:
        raise ValueError("debug_heartbeat_interval_sec must be > 0")
    if config.train_progress_log_every_chunks < 1:
        raise ValueError("train_progress_log_every_chunks must be >= 1")


def _scan_input_stats(paths: list[Path], config: BuildTokenizerConfig) -> InputScanStats:
    if config.num_workers <= 1 or len(paths) <= 1:
        rows = [
            _scan_shard_stats(
                str(path),
                config.parquet_batch_rows,
                config.train_chunk_enable,
                config.train_chunk_max_chars,
                config.train_chunk_max_utf8_bytes,
                config.train_chunk_break_priority,
                config.top_long_rows,
            )
            for path in paths
        ]
    else:
        mp_context = mp.get_context("spawn")
        rows = []
        with ProcessPoolExecutor(
            max_workers=min(config.num_workers, len(paths)),
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(
                    _scan_shard_stats,
                    str(path),
                    config.parquet_batch_rows,
                    config.train_chunk_enable,
                    config.train_chunk_max_chars,
                    config.train_chunk_max_utf8_bytes,
                    config.train_chunk_break_priority,
                    config.top_long_rows,
                )
                for path in paths
            ]
            for future in common.progress(
                as_completed(futures),
                total=len(futures),
                desc="build_tokenizer/scan",
                enabled=config.enable_tqdm,
                mininterval=config.tqdm_mininterval_sec,
            ):
                rows.append(future.result())
    merged_long_rows = _merge_top_long_rows(
        [row.top_long_rows for row in rows],
        top_n=config.top_long_rows,
    )
    return InputScanStats(
        input_row_count=sum(item.input_row_count for item in rows),
        valid_row_count=sum(item.valid_row_count for item in rows),
        filtered_row_count=sum(item.filtered_row_count for item in rows),
        max_input_content_chars=max((item.max_input_content_chars for item in rows), default=0),
        max_input_content_bytes=max((item.max_input_content_bytes for item in rows), default=0),
        train_chunk_count=sum(item.train_chunk_count for item in rows),
        max_training_chunk_chars=max((item.max_training_chunk_chars for item in rows), default=0),
        top_long_rows=merged_long_rows,
    )


def _scan_shard_stats(
    path_str: str,
    batch_rows: int,
    train_chunk_enable: bool,
    train_chunk_max_chars: int,
    train_chunk_max_utf8_bytes: int,
    train_chunk_break_priority: tuple[str, ...],
    top_long_rows: int,
) -> InputScanStats:
    parquet = pq.ParquetFile(path_str)
    _require_content_column(parquet, Path(path_str))
    input_rows = 0
    valid_rows = 0
    filtered_rows = 0
    max_input_content_chars = 0
    max_input_content_bytes = 0
    train_chunk_count = 0
    max_training_chunk_chars = 0
    long_row_heap: list[tuple[int, int, int, str, LongRowRecord]] = []
    row_index = 0
    for batch in parquet.iter_batches(columns=["content"], batch_size=batch_rows):
        contents = batch.column(0).to_pylist()
        input_rows += len(contents)
        for content in contents:
            if common.strip_text(content) is None:
                filtered_rows += 1
                row_index += 1
                continue
            valid_rows += 1
            content_chars = len(content)
            content_utf8_bytes = len(content.encode("utf-8"))
            max_input_content_chars = max(max_input_content_chars, content_chars)
            max_input_content_bytes = max(max_input_content_bytes, content_utf8_bytes)
            chunks = (
                _chunk_training_text(
                    content,
                    max_chars=train_chunk_max_chars,
                    max_utf8_bytes=train_chunk_max_utf8_bytes,
                    break_priority=train_chunk_break_priority,
                )
                if train_chunk_enable
                else [content]
            )
            estimated_chunk_count = len(chunks)
            train_chunk_count += estimated_chunk_count
            if chunks:
                max_training_chunk_chars = max(
                    max_training_chunk_chars,
                    max(len(chunk) for chunk in chunks),
                )
            record = LongRowRecord(
                path=path_str,
                row_index=row_index,
                content_chars=content_chars,
                content_utf8_bytes=content_utf8_bytes,
                estimated_chunk_count=estimated_chunk_count,
            )
            _push_top_long_row(long_row_heap, record, top_n=top_long_rows)
            row_index += 1
    return InputScanStats(
        input_row_count=input_rows,
        valid_row_count=valid_rows,
        filtered_row_count=filtered_rows,
        max_input_content_chars=max_input_content_chars,
        max_input_content_bytes=max_input_content_bytes,
        train_chunk_count=train_chunk_count,
        max_training_chunk_chars=max_training_chunk_chars,
        top_long_rows=_heap_to_sorted_long_rows(long_row_heap),
    )


def _build_low_level_tokenizer(config: BuildTokenizerConfig) -> tuple[Tokenizer, trainers.BpeTrainer]:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=True,
        trim_offsets=True,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(
        add_prefix_space=True,
        trim_offsets=True,
        use_regex=True,
    )
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.trainer_min_frequency,
        show_progress=True,
        special_tokens=list(config.special_tokens),
        limit_alphabet=config.trainer_limit_alphabet,
        max_token_length=config.trainer_max_token_length,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    return tokenizer, trainer


def _spawn_training_child(
    *,
    input_paths: list[Path],
    scan_stats: InputScanStats,
    config: BuildTokenizerConfig,
    runtime: _DebugRuntime,
    staging_dir: Path,
) -> mp.Process:
    mp_context = mp.get_context("spawn")
    process = mp_context.Process(
        target=_train_and_save_child,
        kwargs={
            "input_path_strs": [str(path) for path in input_paths],
            "scan_stats": scan_stats,
            "config": config,
            "debug_run_dir_str": str(runtime.debug_run_dir),
            "staging_dir_str": str(staging_dir),
        },
    )
    process.start()
    return process


def _train_and_save_child(
    *,
    input_path_strs: list[str],
    scan_stats: InputScanStats,
    config: BuildTokenizerConfig,
    debug_run_dir_str: str,
    staging_dir_str: str,
) -> None:
    runtime = _DebugRuntime(debug_run_dir=Path(debug_run_dir_str))
    staging_dir = Path(staging_dir_str)
    runtime.set_phase("train_child_running", staging_dir=str(staging_dir))
    try:
        input_paths = [Path(path_str) for path_str in input_path_strs]
        tokenizer = _train_tokenizer(input_paths, scan_stats, config, runtime)
        runtime.set_phase("train_done", staging_dir=str(staging_dir))
        runtime.log_event("save_child_start", staging_dir=str(staging_dir))
        _save_tokenizer_artifacts(tokenizer, staging_dir)
        common.write_json(staging_dir / "child_status.json", {"status": "ok"})
        runtime.log_event("save_child_done", staging_dir=str(staging_dir))
    except BaseException as exc:
        runtime.log_event(
            "train_child_failed",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        runtime.write_failure_context(exc)
        raise


def _train_tokenizer(
    paths: list[Path],
    scan_stats: InputScanStats,
    config: BuildTokenizerConfig,
    runtime: _DebugRuntime,
) -> Tokenizer:
    tokenizer, trainer = _build_low_level_tokenizer(config)
    iterator = _iter_training_texts(paths, config, runtime)
    tokenizer.train_from_iterator(
        iterator,
        trainer=trainer,
        length=scan_stats.train_chunk_count or None,
    )
    return tokenizer


def _iter_training_texts(
    paths: list[Path],
    config: BuildTokenizerConfig,
    runtime: _DebugRuntime,
) -> Iterator[str]:
    rows_seen = 0
    chunks_emitted = 0
    max_input_chars_seen = 0
    for path in common.progress(
        paths,
        total=len(paths),
        desc="build_tokenizer/train_iter",
        enabled=config.enable_tqdm,
        mininterval=config.tqdm_mininterval_sec,
    ):
        runtime.update_progress(current_path=str(path))
        parquet = pq.ParquetFile(path)
        _require_content_column(parquet, path)
        for batch in parquet.iter_batches(
            columns=["content"], batch_size=config.parquet_batch_rows
        ):
            for content in batch.column(0).to_pylist():
                if common.strip_text(content) is None:
                    continue
                rows_seen += 1
                max_input_chars_seen = max(max_input_chars_seen, len(content))
                for chunk in _iter_training_chunks(content, config):
                    chunks_emitted += 1
                    runtime.update_progress(
                        rows_seen=rows_seen,
                        chunks_emitted=chunks_emitted,
                        current_path=str(path),
                        max_input_chars_seen=max_input_chars_seen,
                    )
                    if chunks_emitted % config.train_progress_log_every_chunks == 0:
                        common.log(
                            (
                                "[build_tokenizer] train_progress "
                                f"rows_seen={rows_seen} "
                                f"chunks_emitted={chunks_emitted} "
                                f"current_path={path} "
                                f"max_input_chars_seen={max_input_chars_seen}"
                            ),
                            enabled=config.enable_debug_log,
                        )
                        runtime.log_event(
                            "train_progress",
                            rows_seen=rows_seen,
                            chunks_emitted=chunks_emitted,
                            current_path=str(path),
                            max_input_chars_seen=max_input_chars_seen,
                        )
                    yield chunk
    runtime.log_event(
        "train_iterator_exhausted",
        rows_seen=rows_seen,
        chunks_emitted=chunks_emitted,
        max_input_chars_seen=max_input_chars_seen,
    )


def _iter_training_chunks(
    text: str,
    config: BuildTokenizerConfig,
) -> Iterator[str]:
    if not config.train_chunk_enable:
        yield text
        return
    yield from _chunk_training_text(
        text,
        max_chars=config.train_chunk_max_chars,
        max_utf8_bytes=config.train_chunk_max_utf8_bytes,
        break_priority=config.train_chunk_break_priority,
    )


def _chunk_training_text(
    text: str,
    *,
    max_chars: int,
    max_utf8_bytes: int,
    break_priority: tuple[str, ...],
) -> list[str]:
    if len(text) <= max_chars and len(text.encode("utf-8")) <= max_utf8_bytes:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        remaining = text_len - start
        if remaining <= max_chars and len(text[start:].encode("utf-8")) <= max_utf8_bytes:
            tail = text[start:]
            if common.strip_text(tail) is not None:
                chunks.append(tail)
            break

        char_window_end = min(start + max_chars, text_len)
        window_end = _fit_chunk_end_by_utf8_bytes(
            text=text,
            start=start,
            candidate_end=char_window_end,
            max_utf8_bytes=max_utf8_bytes,
        )
        cut = None
        for marker in break_priority:
            search_end = window_end - len(marker) + 1
            if search_end <= start + 1:
                continue
            marker_index = text.rfind(marker, start + 1, search_end)
            if marker_index != -1:
                cut = marker_index + len(marker)
                break
        if cut is None or cut <= start:
            cut = window_end
        if cut <= start:
            cut = min(start + 1, text_len)

        chunk = text[start:cut]
        if common.strip_text(chunk) is not None:
            chunks.append(chunk)
        start = cut
    return chunks


def _fit_chunk_end_by_utf8_bytes(
    *,
    text: str,
    start: int,
    candidate_end: int,
    max_utf8_bytes: int,
) -> int:
    if start >= candidate_end:
        return candidate_end
    candidate = text[start:candidate_end]
    if len(candidate.encode("utf-8")) <= max_utf8_bytes:
        return candidate_end

    low = start + 1
    high = candidate_end
    best = start + 1
    while low <= high:
        mid = (low + high) // 2
        current = text[start:mid]
        if len(current.encode("utf-8")) <= max_utf8_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _save_tokenizer_artifacts(tokenizer: Tokenizer, output_dir: Path) -> None:
    common.ensure_dir(output_dir)
    tokenizer.model.save(str(output_dir), None)
    tokenizer.save(str(output_dir / "tokenizer.json"))


def _count_top_tokens(
    paths: list[Path],
    tokenizer_dir: Path,
    config: BuildTokenizerConfig,
) -> tuple[list[common.TopTokenRow], int]:
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if config.num_workers <= 1 or len(paths) <= 1:
        shard_results = [
            _count_tokens_in_shard(
                str(path), str(tokenizer_json_path), config.parquet_batch_rows
            )
            for path in paths
        ]
    else:
        mp_context = mp.get_context("spawn")
        shard_results = []
        with ProcessPoolExecutor(
            max_workers=min(config.num_workers, len(paths)),
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(
                    _count_tokens_in_shard,
                    str(path),
                    str(tokenizer_json_path),
                    config.parquet_batch_rows,
                )
                for path in paths
            ]
            for future in common.progress(
                as_completed(futures),
                total=len(futures),
                desc="build_tokenizer/top_tokens",
                enabled=config.enable_tqdm,
                mininterval=config.tqdm_mininterval_sec,
            ):
                shard_results.append(future.result())

    counter: Counter[int] = Counter()
    total_token_count = 0
    for shard_counter, shard_total in shard_results:
        counter.update(shard_counter)
        total_token_count += shard_total

    tokenizer = common.load_tokenizer(tokenizer_json_path)
    top_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
        : config.top_token_count
    ]
    rows: list[common.TopTokenRow] = []
    for rank, (token_id, count) in enumerate(top_items, start=1):
        raw_token = common.id_to_raw_token(tokenizer, token_id)
        rows.append(
            common.TopTokenRow(
                rank=rank,
                token_id=int(token_id),
                count=int(count),
                share=(count / total_token_count) if total_token_count else 0.0,
                raw_token=raw_token,
                escaped_token=common.escape_for_display(raw_token),
                decoded_token=common.decode_token_for_display(tokenizer, token_id),
            )
        )
    return rows, total_token_count


def _count_tokens_in_shard(
    path_str: str,
    tokenizer_json_path: str,
    batch_rows: int,
) -> tuple[Counter[int], int]:
    tokenizer = common.load_tokenizer(Path(tokenizer_json_path))
    parquet = pq.ParquetFile(path_str)
    _require_content_column(parquet, Path(path_str))
    counter: Counter[int] = Counter()
    total_token_count = 0
    for batch in parquet.iter_batches(columns=["content"], batch_size=batch_rows):
        for content in batch.column(0).to_pylist():
            if common.strip_text(content) is None:
                continue
            ids = tokenizer.encode(content).ids
            counter.update(ids)
            total_token_count += len(ids)
    return counter, total_token_count


def _require_content_column(parquet: pq.ParquetFile, path: Path) -> None:
    if "content" not in parquet.schema.names:
        raise ValueError(f"missing content column: {path}")


def _validate_outputs(output_dir: Path) -> None:
    required_paths = [
        output_dir / "tokenizer.json",
        output_dir / "vocab.json",
        output_dir / "merges.txt",
        output_dir / "meta.json",
        output_dir / "top_tokens.jsonl",
    ]
    for path in required_paths:
        if not path.exists():
            raise ValueError(f"missing required output: {path}")


def _push_top_long_row(
    heap: list[tuple[int, int, int, str, LongRowRecord]],
    record: LongRowRecord,
    *,
    top_n: int,
) -> None:
    key = (
        record.content_chars,
        record.content_utf8_bytes,
        record.estimated_chunk_count,
        f"{record.path}#{record.row_index}",
        record,
    )
    if len(heap) < top_n:
        heapq.heappush(heap, key)
        return
    if key > heap[0]:
        heapq.heapreplace(heap, key)


def _heap_to_sorted_long_rows(
    heap: list[tuple[int, int, int, str, LongRowRecord]]
) -> list[LongRowRecord]:
    return [
        item[-1]
        for item in sorted(
            heap,
            key=lambda item: (-item[0], -item[1], -item[2], item[3]),
        )
    ]


def _merge_top_long_rows(
    groups: list[list[LongRowRecord]],
    *,
    top_n: int,
) -> list[LongRowRecord]:
    heap: list[tuple[int, int, int, str, LongRowRecord]] = []
    for group in groups:
        for record in group:
            _push_top_long_row(heap, record, top_n=top_n)
    return _heap_to_sorted_long_rows(heap)


def _classify_oom_risk(
    *,
    mem_available_mb: float | None,
    swap_free_mb: float | None,
    warning_available_mb: float,
    warning_swap_free_mb: float,
) -> str:
    if mem_available_mb is None or swap_free_mb is None:
        return "unknown"
    if mem_available_mb < 256 or swap_free_mb < 1024:
        return "critical"
    if mem_available_mb < warning_available_mb or swap_free_mb < warning_swap_free_mb:
        return "warning"
    return "normal"


def _summarize_child_exit(exitcode: int | None) -> dict[str, Any]:
    if exitcode is None:
        return {
            "child_exit_code": None,
            "child_signal": None,
            "oom_suspected": False,
        }
    child_signal = -exitcode if exitcode < 0 else None
    oom_suspected = child_signal == signal.SIGKILL
    return {
        "child_exit_code": exitcode,
        "child_signal": child_signal,
        "oom_suspected": oom_suspected,
    }


def main() -> None:
    run_build_tokenizer(build_default_config())


if __name__ == "__main__":
    main()
