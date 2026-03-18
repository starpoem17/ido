from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import shutil
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

import src.preprocess.common as common
from src.preprocess.common import (
    STAGING_SCHEMA,
    ensure_parent_dir,
    log,
    now_utc_iso,
    progress_bar,
    truncate_sample,
    write_json,
)


# =========================
# User configuration block
# =========================
INPUT_ROOT = Path("data/korean_processed/exact_dedup")
OUTPUT_ROOT = Path("data/korean_processed/near_dedup")
MINHASH_LSH_ROOT = OUTPUT_ROOT / "_minhash_lsh"
TARGET_SHARD_BYTES = 1_073_741_824
SHINGLE_NGRAM_SIZE = 7
MINHASH_NUM_PERMUTATIONS = 112
LSH_NUM_BANDS = 14
LSH_ROWS_PER_BAND = 8
MINHASH_HASH_BITS = 64
MINHASH_RANDOM_SEED = 17
JACCARD_THRESHOLD = 0.8
NUM_WORKERS = 7
ROW_BATCH_ROWS = 512
SIGNATURE_SHINGLE_CHUNK_SIZE = 8192
GIANT_BUCKET_MAX_ROWS = 2048
GIANT_BUCKET_REBUCKET_FANOUT = 16
GIANT_BUCKET_MAX_PASSES = 2
VERIFY_PAIR_BATCH_ROWS = 4096
VERIFY_CACHE_MAX_ROWS = 50_000
DUCKDB_THREADS = 7
DUCKDB_MEMORY_LIMIT = "24GB"
DUCKDB_ARROW_LARGE_BUFFER_SIZE = True
OVERWRITE_OUTPUT = False
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


UINT64_MAX = np.uint64((1 << 64) - 1)
USAGE_PRIORITY = {"PT": 1, "SFT": 2, "REASONING": 3}
REASON_CODE_BY_LABEL = {
    "higher_data_usage": 1,
    "longer_content": 2,
    "stable_locator_tiebreak": 3,
}
REASON_LABEL_BY_CODE = {value: key for key, value in REASON_CODE_BY_LABEL.items()}
# The near-dedup hash pipeline intentionally uses uint64 wraparound arithmetic.
# These mixes are fast, deterministic non-cryptographic hashes, so modulo 2^64
# overflow is part of the design rather than an error condition.
EMPTY_TEXT_SENTINEL = np.uint64(0x9E3779B97F4A7C15)
SHINGLE_HASH_BASE = np.uint64(0x100000001B3)
SHINGLE_HASH_OFFSET = np.uint64(0x9E3779B97F4A7C15)
WINDOW_MULTIPLIERS = np.array(
    [
        0x9E3779B185EBCA87,
        0xC2B2AE3D27D4EB4F,
        0x165667B19E3779F9,
        0x85EBCA77C2B2AE63,
        0x27D4EB2F165667C5,
        0x94D049BB133111EB,
        0xD6E8FEB86659FD93,
    ],
    dtype=np.uint64,
)
WINDOW_SALTS = np.array(
    [
        0x243F6A8885A308D3,
        0x13198A2E03707344,
        0xA4093822299F31D0,
        0x082EFA98EC4E6C89,
        0x452821E638D01377,
        0xBE5466CF34E90C6C,
        0xC0AC29B7C97C50DD,
    ],
    dtype=np.uint64,
)
BUCKET_HASH_BASE = np.uint64(0x100000001B3)
BUCKET_SALTS = np.array(
    [
        0x517CC1B727220A95,
        0x6C8E9CF570932BAB,
        0xD1B54A32D192ED03,
        0xABC98388FB8FAC03,
        0x8CB92BA72F3D8DD7,
        0x58F38DED5F54DCDB,
        0x239B961BAB0E9789,
        0x38B34AE5C0E19D4F,
        0xA54FF53A5F1D36F1,
        0x1B03738712FAD5C9,
        0x2E51B87C3D7D98B1,
        0x7B64B1C6A27C5D2F,
        0x94D049BB133111EB,
        0xD6E8FEB86659FD93,
        0xCA5A826395121157,
        0xB492B66FBE98F273,
    ],
    dtype=np.uint64,
)
PERMUTATION_A = np.array(
    [
        0xD6E8FEB86659FD93,
        0xA5A3564E27F8861F,
        0xC6A4A7935BD1E995,
        0x9E3779B185EBCA87,
        0x165667B19E3779F9,
        0x85EBCA77C2B2AE63,
        0x27D4EB2F165667C5,
        0x94D049BB133111EB,
        0xBF58476D1CE4E5B9,
        0x94D049BB133111EB,
        0xDB4F0B9175AE2165,
        0xC13FA9A902A6328F,
        0x91E10DA5C79E7B1D,
        0xCA5A826395121157,
        0xB492B66FBE98F273,
        0x9AE16A3B2F90404F,
    ],
    dtype=np.uint64,
)
PERMUTATION_B = np.array(
    [
        0x243F6A8885A308D3,
        0x13198A2E03707344,
        0xA4093822299F31D0,
        0x082EFA98EC4E6C89,
        0x452821E638D01377,
        0xBE5466CF34E90C6C,
        0xC0AC29B7C97C50DD,
        0x3F84D5B5B5470917,
        0x9216D5D98979FB1B,
        0xD1310BA698DFB5AC,
        0x2FFD72DBD01ADFB7,
        0xB8E1AFED6A267E96,
        0xBA7C9045F12C7F99,
        0x24A19947B3916CF7,
        0x0801F2E2858EFC16,
        0x636920D871574E69,
    ],
    dtype=np.uint64,
)


@dataclass(frozen=True)
class NearDedupConfig:
    input_root: Path
    output_root: Path
    minhash_lsh_root: Path
    target_shard_bytes: int
    shingle_ngram_size: int
    minhash_num_permutations: int
    lsh_num_bands: int
    lsh_rows_per_band: int
    minhash_hash_bits: int
    minhash_random_seed: int
    jaccard_threshold: float
    num_workers: int
    row_batch_rows: int
    signature_shingle_chunk_size: int
    giant_bucket_max_rows: int
    giant_bucket_rebucket_fanout: int
    giant_bucket_max_passes: int
    verify_pair_batch_rows: int
    verify_cache_max_rows: int
    duckdb_threads: int
    duckdb_memory_limit: str
    duckdb_arrow_large_buffer_size: bool
    overwrite_output: bool


@dataclass(frozen=True)
class NearDedupShardResult:
    shard_path: str
    row_count: int
    approx_nbytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NearDedupSummary:
    run_id: str
    input_shard_count: int
    input_row_count: int
    signature_row_count: int
    passthrough_row_count: int
    candidate_pair_count: int
    verified_pair_count: int
    near_cluster_count: int
    giant_bucket_count: int
    giant_bucket_fallback_count: int
    retained_row_count: int
    deleted_row_count: int
    output_shard_count: int
    output_shards: list[dict[str, Any]]
    source_counts_before: dict[str, int]
    source_counts_after: dict[str, int]
    source_counts_deleted: dict[str, int]
    data_usage_counts_before: dict[str, int]
    data_usage_counts_after: dict[str, int]
    data_usage_counts_deleted: dict[str, int]
    representative_reason_counts: dict[str, int]
    jaccard_threshold: float
    minhash_num_permutations: int
    lsh_num_bands: int
    lsh_rows_per_band: int
    shingle_ngram_size: int
    started_at: str
    finished_at: str
    duration_sec: float
    output_root: str
    deleted_log_path: str
    summary_json_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShardTask:
    shard_idx: int
    shard_path: str
    row_start: int
    row_count: int
    total_rows: int
    signature_path: str
    row_index_root: str
    bucket_root: str
    row_batch_rows: int
    shingle_ngram_size: int
    minhash_num_permutations: int
    lsh_num_bands: int
    lsh_rows_per_band: int
    signature_shingle_chunk_size: int
    minhash_random_seed: int


@dataclass(frozen=True)
class ShardStageResult:
    row_index_path: str
    bucket_path: str
    row_count: int
    content_row_count: int


@dataclass(frozen=True)
class CandidateBuildResult:
    pairs_path: Path
    candidate_pair_count: int
    giant_bucket_count: int
    giant_bucket_fallback_count: int


@dataclass(frozen=True)
class VerificationGroupTask:
    group_path: str
    input_shards: tuple[str, ...]
    output_root: str
    shingle_ngram_size: int
    verify_pair_batch_rows: int
    row_batch_rows: int
    jaccard_threshold: float


@dataclass(frozen=True)
class VerificationGroupResult:
    verified_path: str
    verified_pair_count: int


def build_default_config() -> NearDedupConfig:
    return NearDedupConfig(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        minhash_lsh_root=MINHASH_LSH_ROOT,
        target_shard_bytes=TARGET_SHARD_BYTES,
        shingle_ngram_size=SHINGLE_NGRAM_SIZE,
        minhash_num_permutations=MINHASH_NUM_PERMUTATIONS,
        lsh_num_bands=LSH_NUM_BANDS,
        lsh_rows_per_band=LSH_ROWS_PER_BAND,
        minhash_hash_bits=MINHASH_HASH_BITS,
        minhash_random_seed=MINHASH_RANDOM_SEED,
        jaccard_threshold=JACCARD_THRESHOLD,
        num_workers=NUM_WORKERS,
        row_batch_rows=ROW_BATCH_ROWS,
        signature_shingle_chunk_size=SIGNATURE_SHINGLE_CHUNK_SIZE,
        giant_bucket_max_rows=GIANT_BUCKET_MAX_ROWS,
        giant_bucket_rebucket_fanout=GIANT_BUCKET_REBUCKET_FANOUT,
        giant_bucket_max_passes=GIANT_BUCKET_MAX_PASSES,
        verify_pair_batch_rows=VERIFY_PAIR_BATCH_ROWS,
        verify_cache_max_rows=VERIFY_CACHE_MAX_ROWS,
        duckdb_threads=DUCKDB_THREADS,
        duckdb_memory_limit=DUCKDB_MEMORY_LIMIT,
        duckdb_arrow_large_buffer_size=DUCKDB_ARROW_LARGE_BUFFER_SIZE,
        overwrite_output=OVERWRITE_OUTPUT,
    )


def discover_input_shards(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("part-*.parquet"), key=lambda path: str(path))


def run_near_dedup(config: NearDedupConfig | None = None) -> NearDedupSummary:
    config = config or build_default_config()
    _validate_config(config)

    input_shards = discover_input_shards(config.input_root)
    if not input_shards:
        raise FileNotFoundError(f"no input parquet shards found under {config.input_root}")

    deleted_log_path = config.output_root / "_logs" / "deleted_rows_minhash.jsonl"
    summary_json_path = config.output_root / "_meta" / "deleted_rows_minhash_summary.json"
    manifest_path = config.output_root / "_meta" / "near_dedup_manifest.json"
    started_at = now_utc_iso()
    started_perf = perf_counter()
    run_id = started_at.replace("-", "").replace(":", "").replace(".", "")

    _prepare_output_root(config)
    shard_infos = _build_shard_infos(input_shards)
    total_rows = sum(info["row_count"] for info in shard_infos)
    signature_path = config.minhash_lsh_root / "signatures" / "signature_matrix.uint64.mmap"
    _allocate_signature_memmap(signature_path, total_rows, config.minhash_num_permutations)

    log(
        "[near_dedup] start "
        f"input_root={config.input_root} output_root={config.output_root} "
        f"input_shards={len(input_shards)} input_rows={total_rows} "
        f"shingle_ngram_size={config.shingle_ngram_size} "
        f"minhash_num_permutations={config.minhash_num_permutations} "
        f"lsh_num_bands={config.lsh_num_bands} lsh_rows_per_band={config.lsh_rows_per_band} "
        f"jaccard_threshold={config.jaccard_threshold} num_workers={config.num_workers}"
    )

    shard_results = _build_signatures_and_buckets(
        shard_infos=shard_infos,
        signature_path=signature_path,
        config=config,
    )
    row_index_paths = [Path(result.row_index_path) for result in shard_results]
    bucket_paths = [Path(result.bucket_path) for result in shard_results]
    signature_row_count = sum(result.content_row_count for result in shard_results)
    passthrough_row_count = total_rows - signature_row_count

    before_counts = _count_before(row_index_paths=row_index_paths, duckdb_threads=config.duckdb_threads)
    candidate_result = _build_candidate_pairs(
        bucket_paths=bucket_paths,
        row_index_paths=row_index_paths,
        signature_path=signature_path,
        shard_infos=shard_infos,
        config=config,
    )
    verified_paths, verified_pair_count = _verify_candidate_pairs(
        pairs_path=candidate_result.pairs_path,
        input_shards=input_shards,
        config=config,
    )
    cluster_result = _build_clusters(
        verified_paths=verified_paths,
        row_index_paths=row_index_paths,
        total_rows=total_rows,
        minhash_root=config.minhash_lsh_root,
        duckdb_threads=config.duckdb_threads,
    )
    output_shards, deleted_row_count, reason_counts = _write_retained_rows_and_deleted_log(
        input_shards=input_shards,
        output_root=config.output_root,
        target_shard_bytes=config.target_shard_bytes,
        deleted_log_path=deleted_log_path,
        removed_mask=cluster_result["removed_mask"],
        representative_row_ids=cluster_result["representative_row_ids"],
        reason_codes=cluster_result["reason_codes"],
        representative_locator_by_row_id=cluster_result["representative_locator_by_row_id"],
        run_id=run_id,
        threshold=config.jaccard_threshold,
        row_batch_rows=config.row_batch_rows,
    )

    retained_row_count = sum(shard.row_count for shard in output_shards)
    source_counts_before = before_counts["source"]
    data_usage_counts_before = before_counts["data_usage"]
    source_counts_deleted = cluster_result["source_counts_deleted"]
    data_usage_counts_deleted = cluster_result["data_usage_counts_deleted"]
    source_counts_after = _subtract_counts(source_counts_before, source_counts_deleted)
    data_usage_counts_after = _subtract_counts(data_usage_counts_before, data_usage_counts_deleted)

    finished_at = now_utc_iso()
    duration_sec = round(perf_counter() - started_perf, 3)
    summary = NearDedupSummary(
        run_id=run_id,
        input_shard_count=len(input_shards),
        input_row_count=total_rows,
        signature_row_count=signature_row_count,
        passthrough_row_count=passthrough_row_count,
        candidate_pair_count=candidate_result.candidate_pair_count,
        verified_pair_count=verified_pair_count,
        near_cluster_count=cluster_result["near_cluster_count"],
        giant_bucket_count=candidate_result.giant_bucket_count,
        giant_bucket_fallback_count=candidate_result.giant_bucket_fallback_count,
        retained_row_count=retained_row_count,
        deleted_row_count=deleted_row_count,
        output_shard_count=len(output_shards),
        output_shards=[shard.to_dict() for shard in output_shards],
        source_counts_before=source_counts_before,
        source_counts_after=source_counts_after,
        source_counts_deleted=source_counts_deleted,
        data_usage_counts_before=data_usage_counts_before,
        data_usage_counts_after=data_usage_counts_after,
        data_usage_counts_deleted=data_usage_counts_deleted,
        representative_reason_counts=dict(sorted(reason_counts.items())),
        jaccard_threshold=config.jaccard_threshold,
        minhash_num_permutations=config.minhash_num_permutations,
        lsh_num_bands=config.lsh_num_bands,
        lsh_rows_per_band=config.lsh_rows_per_band,
        shingle_ngram_size=config.shingle_ngram_size,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        output_root=str(config.output_root),
        deleted_log_path=str(deleted_log_path),
        summary_json_path=str(summary_json_path),
        manifest_path=str(manifest_path),
    )
    _write_summary_files(summary, summary_json_path, manifest_path)
    log(
        "[near_dedup] completed "
        f"input_rows={summary.input_row_count} candidates={summary.candidate_pair_count} "
        f"verified_pairs={summary.verified_pair_count} clusters={summary.near_cluster_count} "
        f"retained={summary.retained_row_count} deleted={summary.deleted_row_count}"
    )
    return summary


def _validate_config(config: NearDedupConfig) -> None:
    if config.minhash_hash_bits != 64:
        raise ValueError("only 64-bit minhash is supported")
    if config.shingle_ngram_size < 1:
        raise ValueError("shingle_ngram_size must be >= 1")
    if config.lsh_num_bands < 1:
        raise ValueError("lsh_num_bands must be >= 1")
    if config.lsh_rows_per_band < 1:
        raise ValueError("lsh_rows_per_band must be >= 1")
    if config.minhash_num_permutations != config.lsh_num_bands * config.lsh_rows_per_band:
        raise ValueError(
            "minhash_num_permutations must equal lsh_num_bands * lsh_rows_per_band"
        )
    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if config.row_batch_rows < 1:
        raise ValueError("row_batch_rows must be >= 1")
    if config.signature_shingle_chunk_size < 1:
        raise ValueError("signature_shingle_chunk_size must be >= 1")
    if config.giant_bucket_max_rows < 2:
        raise ValueError("giant_bucket_max_rows must be >= 2")
    if config.giant_bucket_rebucket_fanout < 2:
        raise ValueError("giant_bucket_rebucket_fanout must be >= 2")
    if config.giant_bucket_max_passes < 0:
        raise ValueError("giant_bucket_max_passes must be >= 0")
    if not 0.0 <= config.jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")
    if config.target_shard_bytes < 1:
        raise ValueError("target_shard_bytes must be >= 1")


def _prepare_output_root(config: NearDedupConfig) -> None:
    if config.output_root.exists():
        existing_paths = list(config.output_root.iterdir())
        if existing_paths and not config.overwrite_output:
            raise FileExistsError(
                f"output_root already contains files: {config.output_root}. "
                "Set OVERWRITE_OUTPUT=True to rebuild."
            )
        if config.overwrite_output:
            shutil.rmtree(config.output_root)
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "_logs").mkdir(parents=True, exist_ok=True)
    (config.output_root / "_meta").mkdir(parents=True, exist_ok=True)
    config.minhash_lsh_root.mkdir(parents=True, exist_ok=True)
    for path in [
        config.minhash_lsh_root / "signatures",
        config.minhash_lsh_root / "row_index",
        config.minhash_lsh_root / "buckets",
        config.minhash_lsh_root / "candidate_pairs",
        config.minhash_lsh_root / "verified",
        config.minhash_lsh_root / "clusters",
        config.minhash_lsh_root / "duckdb_tmp",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _build_shard_infos(input_shards: Sequence[Path]) -> list[dict[str, Any]]:
    shard_infos: list[dict[str, Any]] = []
    row_start = 0
    for shard_idx, shard_path in enumerate(input_shards):
        row_count = pq.ParquetFile(shard_path).metadata.num_rows
        shard_infos.append(
            {
                "shard_idx": shard_idx,
                "shard_path": shard_path,
                "row_start": row_start,
                "row_count": row_count,
            }
        )
        row_start += row_count
    return shard_infos


def _allocate_signature_memmap(signature_path: Path, total_rows: int, num_permutations: int) -> None:
    ensure_parent_dir(signature_path)
    mmap = np.memmap(signature_path, dtype=np.uint64, mode="w+", shape=(total_rows, num_permutations))
    mmap[:] = UINT64_MAX
    mmap.flush()
    del mmap


def _build_signatures_and_buckets(
    *,
    shard_infos: Sequence[dict[str, Any]],
    signature_path: Path,
    config: NearDedupConfig,
) -> list[ShardStageResult]:
    tasks = [
        ShardTask(
            shard_idx=info["shard_idx"],
            shard_path=str(info["shard_path"]),
            row_start=info["row_start"],
            row_count=info["row_count"],
            total_rows=sum(item["row_count"] for item in shard_infos),
            signature_path=str(signature_path),
            row_index_root=str(config.minhash_lsh_root / "row_index"),
            bucket_root=str(config.minhash_lsh_root / "buckets"),
            row_batch_rows=config.row_batch_rows,
            shingle_ngram_size=config.shingle_ngram_size,
            minhash_num_permutations=config.minhash_num_permutations,
            lsh_num_bands=config.lsh_num_bands,
            lsh_rows_per_band=config.lsh_rows_per_band,
            signature_shingle_chunk_size=config.signature_shingle_chunk_size,
            minhash_random_seed=config.minhash_random_seed,
        )
        for info in shard_infos
    ]
    progress = progress_bar(total=len(tasks), desc="near_dedup/signatures")
    results: list[ShardStageResult] = []
    try:
        for result in _map_tasks(_process_shard_task, tasks, config.num_workers):
            results.append(result)
            progress.update(1)
    finally:
        progress.close()
    return sorted(results, key=lambda result: result.row_index_path)


def _map_tasks(function: Any, tasks: Sequence[Any], num_workers: int) -> Iterator[Any]:
    if num_workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield function(task)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(num_workers, len(tasks))) as pool:
        for result in pool.imap_unordered(function, tasks):
            yield result


def _process_shard_task(task: ShardTask) -> ShardStageResult:
    signature_mmap = np.memmap(
        task.signature_path,
        dtype=np.uint64,
        mode="r+",
        shape=(task.total_rows, task.minhash_num_permutations),
    )
    row_index_path = Path(task.row_index_root) / f"part-{task.shard_idx + 1:06d}.parquet"
    bucket_path = Path(task.bucket_root) / f"part-{task.shard_idx + 1:06d}.parquet"
    row_index_schema = pa.schema(
        [
            pa.field("row_id", pa.int64(), nullable=False),
            pa.field("record_locator", pa.string(), nullable=False),
            pa.field("shard_idx", pa.int32(), nullable=False),
            pa.field("row_in_shard", pa.int32(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("data_usage", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("content_length", pa.int32(), nullable=False),
            pa.field("has_content", pa.bool_(), nullable=False),
        ]
    )
    bucket_schema = pa.schema(
        [
            pa.field("band_idx", pa.int16(), nullable=False),
            pa.field("bucket_key", pa.uint64(), nullable=False),
            pa.field("row_id", pa.int64(), nullable=False),
        ]
    )
    content_row_count = 0
    shard_path = Path(task.shard_path)
    with pq.ParquetWriter(row_index_path, row_index_schema) as row_index_writer, pq.ParquetWriter(
        bucket_path, bucket_schema
    ) as bucket_writer:
        parquet_file = pq.ParquetFile(shard_path)
        row_cursor = task.row_start
        row_in_shard = 0
        for batch in parquet_file.iter_batches(batch_size=task.row_batch_rows):
            rows = batch.to_pylist()
            batch_size = len(rows)
            batch_signatures = np.full(
                (batch_size, task.minhash_num_permutations),
                UINT64_MAX,
                dtype=np.uint64,
            )
            row_index_records: list[dict[str, Any]] = []
            valid_row_ids: list[int] = []
            valid_signatures: list[np.ndarray[Any, np.dtype[np.uint64]]] = []
            for index, row in enumerate(rows):
                row_id = row_cursor + index
                content = row["content"]
                has_content = content is not None
                content_length = len(content) if content is not None else 0
                row_index_records.append(
                    {
                        "row_id": row_id,
                        "record_locator": f"{shard_path}#{row_in_shard + index}",
                        "shard_idx": task.shard_idx,
                        "row_in_shard": row_in_shard + index,
                        "source": row["source"],
                        "data_usage": row["data_usage"],
                        "split": row["split"],
                        "content_length": content_length,
                        "has_content": has_content,
                    }
                )
                if not has_content:
                    continue
                signature = _compute_minhash_signature(
                    text=content,
                    ngram_size=task.shingle_ngram_size,
                    num_permutations=task.minhash_num_permutations,
                    chunk_size=task.signature_shingle_chunk_size,
                    random_seed=task.minhash_random_seed,
                )
                batch_signatures[index] = signature
                valid_row_ids.append(row_id)
                valid_signatures.append(signature)
                content_row_count += 1

            signature_mmap[row_cursor : row_cursor + batch_size] = batch_signatures
            row_index_writer.write_table(pa.Table.from_pylist(row_index_records, schema=row_index_schema))
            if valid_row_ids:
                bucket_writer.write_table(
                    _build_bucket_table(
                        row_ids=np.asarray(valid_row_ids, dtype=np.int64),
                        signatures=np.stack(valid_signatures, axis=0),
                        num_bands=task.lsh_num_bands,
                        rows_per_band=task.lsh_rows_per_band,
                    )
                )
            row_cursor += batch_size
            row_in_shard += batch_size

    signature_mmap.flush()
    del signature_mmap
    return ShardStageResult(
        row_index_path=str(row_index_path),
        bucket_path=str(bucket_path),
        row_count=task.row_count,
        content_row_count=content_row_count,
    )


def _build_bucket_table(
    *,
    row_ids: np.ndarray[Any, np.dtype[np.int64]],
    signatures: np.ndarray[Any, np.dtype[np.uint64]],
    num_bands: int,
    rows_per_band: int,
) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("band_idx", pa.int16(), nullable=False),
            pa.field("bucket_key", pa.uint64(), nullable=False),
            pa.field("row_id", pa.int64(), nullable=False),
        ]
    )
    banded = signatures.reshape(signatures.shape[0], num_bands, rows_per_band)
    keys = np.zeros((signatures.shape[0], num_bands), dtype=np.uint64)
    salts = np.resize(BUCKET_SALTS, rows_per_band)
    for offset in range(rows_per_band):
        keys = _combine_u64_vector(keys, banded[:, :, offset], BUCKET_HASH_BASE, salts[offset])
    flat_row_ids = np.repeat(row_ids, num_bands)
    flat_band_idx = np.tile(np.arange(num_bands, dtype=np.int16), len(row_ids))
    flat_keys = keys.reshape(-1)
    return pa.Table.from_arrays(
        [
            pa.array(flat_band_idx, type=pa.int16()),
            pa.array(flat_keys, type=pa.uint64()),
            pa.array(flat_row_ids, type=pa.int64()),
        ],
        schema=schema,
    )


def _normalize_for_minhash(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _build_char_ngrams(text: str, ngram_size: int) -> set[str]:
    normalized = _normalize_for_minhash(text)
    if not normalized:
        return {""}
    if len(normalized) < ngram_size:
        return {normalized}
    return {
        normalized[index : index + ngram_size]
        for index in range(len(normalized) - ngram_size + 1)
    }


def _compute_minhash_signature(
    *,
    text: str,
    ngram_size: int,
    num_permutations: int,
    chunk_size: int,
    random_seed: int,
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    shingle_hashes = _compute_shingle_hashes_from_text(
        text=text,
        ngram_size=ngram_size,
        chunk_size=chunk_size,
    )
    signature = np.full(num_permutations, UINT64_MAX, dtype=np.uint64)
    perm_a, perm_b = _build_permutation_arrays(num_permutations, random_seed)
    block_size = max(1, 4096 // max(1, num_permutations))
    for start in range(0, shingle_hashes.size, block_size):
        block = shingle_hashes[start : start + block_size]
        mixed = _apply_u64_permutations(block, perm_a, perm_b)
        signature = np.minimum(signature, mixed.min(axis=0))
    return signature


def _compute_shingle_hashes_from_text(
    *,
    text: str,
    ngram_size: int,
    chunk_size: int,
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    normalized = _normalize_for_minhash(text)
    if not normalized:
        return np.asarray([EMPTY_TEXT_SENTINEL], dtype=np.uint64)
    codepoints = _text_to_codepoints(normalized)
    if codepoints.size == 0:
        return np.asarray([EMPTY_TEXT_SENTINEL], dtype=np.uint64)
    if codepoints.size < ngram_size:
        return np.unique(_mix_short_sequence(codepoints))

    total_windows = int(codepoints.size - ngram_size + 1)
    hash_chunks: list[np.ndarray[Any, np.dtype[np.uint64]]] = []
    for start in range(0, total_windows, chunk_size):
        window_count = min(chunk_size, total_windows - start)
        segment = codepoints[start : start + window_count + ngram_size - 1]
        windows = sliding_window_view(segment, ngram_size)[:window_count]
        hash_chunks.append(_mix_codepoint_windows(windows))
    return np.unique(np.concatenate(hash_chunks))


def _text_to_codepoints(text: str) -> np.ndarray[Any, np.dtype[np.uint32]]:
    return np.frombuffer(text.encode("utf-32le"), dtype=np.uint32)


def _mix_short_sequence(codepoints: np.ndarray[Any, np.dtype[np.uint32]]) -> np.ndarray[Any, np.dtype[np.uint64]]:
    value = EMPTY_TEXT_SENTINEL
    for index, codepoint in enumerate(codepoints.astype(np.uint64)):
        multiplier = WINDOW_MULTIPLIERS[index % len(WINDOW_MULTIPLIERS)]
        salt = WINDOW_SALTS[index % len(WINDOW_SALTS)]
        value = _mix_u64_scalar(value, codepoint, multiplier, salt)
    return np.asarray([value], dtype=np.uint64)


def _mix_codepoint_windows(
    windows: np.ndarray[Any, np.dtype[np.uint32]],
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    hashes = np.full(windows.shape[0], SHINGLE_HASH_OFFSET, dtype=np.uint64)
    for index in range(windows.shape[1]):
        hashes = _mix_u64_vector(
            hashes,
            windows[:, index].astype(np.uint64),
            WINDOW_MULTIPLIERS[index],
            WINDOW_SALTS[index],
        )
    return hashes


def _mix_u64_scalar(
    state: np.uint64,
    value: np.uint64,
    multiplier: np.uint64,
    salt: np.uint64,
) -> np.uint64:
    # Explicitly use uint64 modulo arithmetic. Overflow is expected here.
    with np.errstate(over="ignore"):
        return (state * SHINGLE_HASH_BASE) ^ (value * multiplier + salt)


def _mix_u64_vector(
    state: np.ndarray[Any, np.dtype[np.uint64]],
    value: np.ndarray[Any, np.dtype[np.uint64]],
    multiplier: np.uint64,
    salt: np.uint64,
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    # Explicitly use uint64 modulo arithmetic. Overflow is expected here.
    with np.errstate(over="ignore"):
        return (state * SHINGLE_HASH_BASE) ^ (value * multiplier + salt)


def _combine_u64_vector(
    state: np.ndarray[Any, np.dtype[np.uint64]],
    value: np.ndarray[Any, np.dtype[np.uint64]],
    base: np.uint64,
    salt: np.uint64,
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    # Bucket and rebucket keys intentionally use uint64 modulo arithmetic.
    with np.errstate(over="ignore"):
        return (state * base) ^ (value + salt)


def _apply_u64_permutations(
    block: np.ndarray[Any, np.dtype[np.uint64]],
    perm_a: np.ndarray[Any, np.dtype[np.uint64]],
    perm_b: np.ndarray[Any, np.dtype[np.uint64]],
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    # Minhash permutations also operate in uint64 modulo arithmetic by design.
    with np.errstate(over="ignore"):
        return ((block[:, None] ^ perm_b[None, :]) * perm_a[None, :]).astype(np.uint64)


def _build_permutation_arrays(
    num_permutations: int,
    random_seed: int,
) -> tuple[np.ndarray[Any, np.dtype[np.uint64]], np.ndarray[Any, np.dtype[np.uint64]]]:
    if num_permutations <= len(PERMUTATION_A):
        return PERMUTATION_A[:num_permutations], PERMUTATION_B[:num_permutations]
    rng = np.random.default_rng(random_seed)
    perm_a = rng.integers(
        low=1,
        high=np.iinfo(np.uint64).max,
        size=num_permutations,
        dtype=np.uint64,
    ) | np.uint64(1)
    perm_b = rng.integers(
        low=0,
        high=np.iinfo(np.uint64).max,
        size=num_permutations,
        dtype=np.uint64,
    )
    return perm_a, perm_b


def _count_before(*, row_index_paths: Sequence[Path], duckdb_threads: int) -> dict[str, Any]:
    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={duckdb_threads}")
        query_root = _read_parquet_sql(row_index_paths)
        total_rows = int(con.execute(f"SELECT COUNT(*) FROM {query_root}").fetchone()[0])
        source_counts = _fetch_count_map(
            con,
            f"SELECT source, COUNT(*) AS cnt FROM {query_root} GROUP BY source",
        )
        data_usage_counts = _fetch_count_map(
            con,
            f"SELECT data_usage, COUNT(*) AS cnt FROM {query_root} GROUP BY data_usage",
        )
        return {"rows": total_rows, "source": source_counts, "data_usage": data_usage_counts}
    finally:
        con.close()


def _build_candidate_pairs(
    *,
    bucket_paths: Sequence[Path],
    row_index_paths: Sequence[Path],
    signature_path: Path,
    shard_infos: Sequence[dict[str, Any]],
    config: NearDedupConfig,
) -> CandidateBuildResult:
    candidates_root = config.minhash_lsh_root / "candidate_pairs"
    normal_pairs_path = candidates_root / "normal_pairs.parquet"
    pairs_path = candidates_root / "pairs.parquet"
    giant_pairs_root = candidates_root / "giant_pairs"
    giant_pairs_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        _configure_duckdb(con, config)
        bucket_sql = _read_parquet_sql(bucket_paths)
        bucket_sizes_path = candidates_root / "bucket_sizes.parquet"
        con.execute(
            f"""
            COPY (
                SELECT band_idx, bucket_key, COUNT(*) AS bucket_size
                FROM {bucket_sql}
                GROUP BY band_idx, bucket_key
            )
            TO '{_sql_quote(str(bucket_sizes_path))}'
            (FORMAT PARQUET)
            """
        )
        log("[near_dedup/candidates] building normal candidate pairs")
        con.execute(
            f"""
            COPY (
                SELECT DISTINCT
                    LEAST(a.row_id, b.row_id) AS left_row_id,
                    GREATEST(a.row_id, b.row_id) AS right_row_id
                FROM {bucket_sql} AS a
                JOIN read_parquet('{_sql_quote(str(bucket_sizes_path))}') AS s
                  ON a.band_idx = s.band_idx
                 AND a.bucket_key = s.bucket_key
                JOIN {bucket_sql} AS b
                  ON a.band_idx = b.band_idx
                 AND a.bucket_key = b.bucket_key
                 AND a.row_id < b.row_id
                WHERE s.bucket_size BETWEEN 2 AND {config.giant_bucket_max_rows}
            )
            TO '{_sql_quote(str(normal_pairs_path))}'
            (FORMAT PARQUET)
            """
        )
        giant_buckets = con.execute(
            f"""
            SELECT band_idx, bucket_key, bucket_size
            FROM read_parquet('{_sql_quote(str(bucket_sizes_path))}')
            WHERE bucket_size > {config.giant_bucket_max_rows}
            ORDER BY bucket_size DESC, band_idx, bucket_key
            """
        ).fetchall()
    finally:
        con.close()

    signature_mmap = np.memmap(
        signature_path,
        dtype=np.uint64,
        mode="r",
        shape=(sum(info["row_count"] for info in shard_infos), config.minhash_num_permutations),
    )
    giant_pair_paths: list[Path] = []
    giant_bucket_fallback_count = 0
    giant_progress = progress_bar(total=len(giant_buckets), desc="near_dedup/giant_buckets")
    try:
        for giant_index, (band_idx, bucket_key, bucket_size) in enumerate(giant_buckets, start=1):
            row_ids = _load_bucket_row_ids(
                bucket_paths=bucket_paths,
                band_idx=int(band_idx),
                bucket_key=int(bucket_key),
            )
            pair_arrays, used_fallback = _rebucket_giant_bucket(
                row_ids=row_ids,
                signature_mmap=signature_mmap,
                band_idx=int(band_idx),
                config=config,
            )
            if used_fallback:
                giant_bucket_fallback_count += 1
            if pair_arrays[0].size:
                giant_path = giant_pairs_root / f"pairs-{giant_index:06d}.parquet"
                _write_pair_table(giant_path, pair_arrays[0], pair_arrays[1])
                giant_pair_paths.append(giant_path)
            giant_progress.update(1)
    finally:
        giant_progress.close()
        del signature_mmap

    all_pair_paths = [normal_pairs_path] + giant_pair_paths
    if not any(path.exists() for path in all_pair_paths):
        _write_empty_pair_table(pairs_path)
        return CandidateBuildResult(
            pairs_path=pairs_path,
            candidate_pair_count=0,
            giant_bucket_count=len(giant_buckets),
            giant_bucket_fallback_count=giant_bucket_fallback_count,
        )

    con = duckdb.connect()
    try:
        _configure_duckdb(con, config)
        con.execute(
            f"""
            COPY (
                SELECT DISTINCT left_row_id, right_row_id
                FROM {_read_parquet_sql([path for path in all_pair_paths if path.exists()])}
            )
            TO '{_sql_quote(str(pairs_path))}'
            (FORMAT PARQUET)
            """
        )
        candidate_pair_count = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{_sql_quote(str(pairs_path))}')"
            ).fetchone()[0]
        )
    finally:
        con.close()

    log(
        "[near_dedup/candidates] completed "
        f"candidate_pair_count={candidate_pair_count} giant_bucket_count={len(giant_buckets)} "
        f"giant_bucket_fallback_count={giant_bucket_fallback_count}"
    )
    return CandidateBuildResult(
        pairs_path=pairs_path,
        candidate_pair_count=candidate_pair_count,
        giant_bucket_count=len(giant_buckets),
        giant_bucket_fallback_count=giant_bucket_fallback_count,
    )


def _configure_duckdb(con: duckdb.DuckDBPyConnection, config: NearDedupConfig) -> None:
    con.execute(f"PRAGMA threads={config.duckdb_threads}")
    con.execute(f"PRAGMA memory_limit='{_sql_quote(config.duckdb_memory_limit)}'")
    temp_dir = config.minhash_lsh_root / "duckdb_tmp"
    con.execute(f"PRAGMA temp_directory='{_sql_quote(str(temp_dir))}'")
    con.execute(
        f"PRAGMA arrow_large_buffer_size={'true' if config.duckdb_arrow_large_buffer_size else 'false'}"
    )


def _load_bucket_row_ids(
    *,
    bucket_paths: Sequence[Path],
    band_idx: int,
    bucket_key: int,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    row_ids: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    for path in bucket_paths:
        dataset = pq.ParquetFile(path)
        for batch in dataset.iter_batches(batch_size=8192):
            batch_table = pa.Table.from_batches([batch])
            frame = batch_table.to_pydict()
            mask = [
                current_band == band_idx and current_key == bucket_key
                for current_band, current_key in zip(frame["band_idx"], frame["bucket_key"], strict=True)
            ]
            if not any(mask):
                continue
            selected = [frame["row_id"][index] for index, keep in enumerate(mask) if keep]
            row_ids.append(np.asarray(selected, dtype=np.int64))
    if not row_ids:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(row_ids)


def _rebucket_giant_bucket(
    *,
    row_ids: np.ndarray[Any, np.dtype[np.int64]],
    signature_mmap: np.ndarray[Any, np.dtype[np.uint64]],
    band_idx: int,
    config: NearDedupConfig,
) -> tuple[tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]]], bool]:
    left_parts: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    right_parts: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    used_fallback = False

    def process_group(
        candidate_row_ids: np.ndarray[Any, np.dtype[np.int64]],
        pass_index: int,
    ) -> None:
        nonlocal used_fallback
        unique_row_ids = np.unique(candidate_row_ids)
        if unique_row_ids.size < 2:
            return
        if unique_row_ids.size <= config.giant_bucket_max_rows:
            left, right = _pair_all_rows(unique_row_ids)
            if left.size:
                left_parts.append(left)
                right_parts.append(right)
            return
        if pass_index >= config.giant_bucket_max_passes:
            used_fallback = True
            left, right = _pair_all_rows(unique_row_ids)
            if left.size:
                left_parts.append(left)
                right_parts.append(right)
            return

        subkeys = _compute_rebucket_keys(
            row_ids=unique_row_ids,
            signature_mmap=signature_mmap,
            band_idx=band_idx,
            pass_index=pass_index,
            config=config,
        )
        groups: dict[int, list[int]] = {}
        for row_id, subkey in zip(unique_row_ids.tolist(), subkeys.tolist(), strict=True):
            groups.setdefault(int(subkey), []).append(row_id)
        if len(groups) == 1:
            used_fallback = True
            left, right = _pair_all_rows(unique_row_ids)
            if left.size:
                left_parts.append(left)
                right_parts.append(right)
            return
        for subgroup in groups.values():
            process_group(np.asarray(subgroup, dtype=np.int64), pass_index + 1)

    process_group(row_ids, 0)
    if not left_parts:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)), used_fallback
    return (
        np.concatenate(left_parts),
        np.concatenate(right_parts),
    ), used_fallback


def _compute_rebucket_keys(
    *,
    row_ids: np.ndarray[Any, np.dtype[np.int64]],
    signature_mmap: np.ndarray[Any, np.dtype[np.uint64]],
    band_idx: int,
    pass_index: int,
    config: NearDedupConfig,
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    start = ((band_idx + 1 + pass_index) * config.lsh_rows_per_band) % config.minhash_num_permutations
    indices = (start + np.arange(config.lsh_rows_per_band)) % config.minhash_num_permutations
    signature_slice = signature_mmap[row_ids][:, indices]
    keys = np.zeros(signature_slice.shape[0], dtype=np.uint64)
    for offset in range(signature_slice.shape[1]):
        keys = _combine_u64_vector(
            keys,
            signature_slice[:, offset],
            BUCKET_HASH_BASE,
            BUCKET_SALTS[offset % len(BUCKET_SALTS)],
        )
    return keys % np.uint64(config.giant_bucket_rebucket_fanout)


def _pair_all_rows(
    row_ids: np.ndarray[Any, np.dtype[np.int64]],
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]]]:
    if row_ids.size < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    left_parts: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    right_parts: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    for index in range(row_ids.size - 1):
        left_parts.append(np.full(row_ids.size - index - 1, row_ids[index], dtype=np.int64))
        right_parts.append(row_ids[index + 1 :].astype(np.int64))
    return np.concatenate(left_parts), np.concatenate(right_parts)


def _write_pair_table(
    path: Path,
    left_row_ids: np.ndarray[Any, np.dtype[np.int64]],
    right_row_ids: np.ndarray[Any, np.dtype[np.int64]],
) -> None:
    ensure_parent_dir(path)
    pq.write_table(
        pa.table(
            {
                "left_row_id": pa.array(left_row_ids, type=pa.int64()),
                "right_row_id": pa.array(right_row_ids, type=pa.int64()),
            }
        ),
        path,
    )


def _write_empty_pair_table(path: Path) -> None:
    _write_pair_table(path, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))


def _verify_candidate_pairs(
    *,
    pairs_path: Path,
    input_shards: Sequence[Path],
    config: NearDedupConfig,
) -> tuple[list[Path], int]:
    if not pairs_path.exists():
        return [], 0

    groups_root = config.minhash_lsh_root / "candidate_pairs" / "verify_groups"
    groups_root.mkdir(parents=True, exist_ok=True)
    group_paths = _build_verify_group_files(
        pairs_path=pairs_path,
        input_shards=input_shards,
        groups_root=groups_root,
        config=config,
    )
    progress = progress_bar(total=len(group_paths), desc="near_dedup/verify")
    results: list[VerificationGroupResult] = []
    tasks = [
        VerificationGroupTask(
            group_path=str(group_path),
            input_shards=tuple(str(path) for path in input_shards),
            output_root=str(config.minhash_lsh_root / "verified"),
            shingle_ngram_size=config.shingle_ngram_size,
            verify_pair_batch_rows=config.verify_pair_batch_rows,
            row_batch_rows=config.row_batch_rows,
            jaccard_threshold=config.jaccard_threshold,
        )
        for group_path in group_paths
    ]
    try:
        for result in _map_tasks(_verify_group_task, tasks, config.num_workers):
            results.append(result)
            progress.update(1)
    finally:
        progress.close()
    verified_paths = sorted(
        [Path(result.verified_path) for result in results if result.verified_pair_count > 0],
        key=lambda path: str(path),
    )
    verified_pair_count = sum(result.verified_pair_count for result in results)
    log(
        "[near_dedup/verify] completed "
        f"groups={len(group_paths)} verified_pairs={verified_pair_count}"
    )
    return verified_paths, verified_pair_count


def _build_verify_group_files(
    *,
    pairs_path: Path,
    input_shards: Sequence[Path],
    groups_root: Path,
    config: NearDedupConfig,
) -> list[Path]:
    row_index_paths = sorted((config.minhash_lsh_root / "row_index").glob("part-*.parquet"))
    con = duckdb.connect()
    try:
        _configure_duckdb(con, config)
        enriched_path = groups_root / "_pairs_enriched.parquet"
        con.execute(
            f"""
            COPY (
                SELECT
                    p.left_row_id,
                    p.right_row_id,
                    li.shard_idx AS left_shard_idx,
                    li.row_in_shard AS left_row_in_shard,
                    ri.shard_idx AS right_shard_idx,
                    ri.row_in_shard AS right_row_in_shard
                FROM read_parquet('{_sql_quote(str(pairs_path))}') AS p
                JOIN {_read_parquet_sql(row_index_paths)} AS li
                  ON p.left_row_id = li.row_id
                JOIN {_read_parquet_sql(row_index_paths)} AS ri
                  ON p.right_row_id = ri.row_id
                ORDER BY left_shard_idx, right_shard_idx, left_row_in_shard, right_row_in_shard
            )
            TO '{_sql_quote(str(enriched_path))}'
            (FORMAT PARQUET)
            """
        )
        groups = con.execute(
            f"""
            SELECT left_shard_idx, right_shard_idx, COUNT(*) AS pair_count
            FROM read_parquet('{_sql_quote(str(enriched_path))}')
            GROUP BY left_shard_idx, right_shard_idx
            ORDER BY left_shard_idx, right_shard_idx
            """
        ).fetchall()
        group_paths: list[Path] = []
        for index, (left_shard_idx, right_shard_idx, _) in enumerate(groups, start=1):
            group_path = groups_root / f"group-{index:06d}.parquet"
            con.execute(
                f"""
                COPY (
                    SELECT left_row_id, right_row_id, left_shard_idx, left_row_in_shard, right_shard_idx, right_row_in_shard
                    FROM read_parquet('{_sql_quote(str(enriched_path))}')
                    WHERE left_shard_idx = {int(left_shard_idx)}
                      AND right_shard_idx = {int(right_shard_idx)}
                )
                TO '{_sql_quote(str(group_path))}'
                (FORMAT PARQUET)
                """
            )
            group_paths.append(group_path)
    finally:
        con.close()
    return group_paths


def _verify_group_task(task: VerificationGroupTask) -> VerificationGroupResult:
    group_path = Path(task.group_path)
    verified_path = Path(task.output_root) / group_path.name
    pair_file = pq.ParquetFile(group_path)
    left_positions: dict[int, set[int]] = {}
    right_positions: dict[int, set[int]] = {}
    row_id_by_left_position: dict[tuple[int, int], int] = {}
    row_id_by_right_position: dict[tuple[int, int], int] = {}
    for batch in pair_file.iter_batches(batch_size=task.verify_pair_batch_rows):
        table = pa.Table.from_batches([batch])
        payload = table.to_pydict()
        for left_row_id, right_row_id, left_shard_idx, left_row_in_shard, right_shard_idx, right_row_in_shard in zip(
            payload["left_row_id"],
            payload["right_row_id"],
            payload["left_shard_idx"],
            payload["left_row_in_shard"],
            payload["right_shard_idx"],
            payload["right_row_in_shard"],
            strict=True,
        ):
            left_key = (int(left_shard_idx), int(left_row_in_shard))
            right_key = (int(right_shard_idx), int(right_row_in_shard))
            left_positions.setdefault(left_key[0], set()).add(left_key[1])
            right_positions.setdefault(right_key[0], set()).add(right_key[1])
            row_id_by_left_position[left_key] = int(left_row_id)
            row_id_by_right_position[right_key] = int(right_row_id)

    shingle_cache: dict[int, np.ndarray[Any, np.dtype[np.uint64]]] = {}
    for shard_idx, positions in left_positions.items():
        shard_rows = _load_rows_from_shard(
            Path(task.input_shards[shard_idx]),
            positions,
            task.row_batch_rows,
        )
        for row_in_shard, content in shard_rows.items():
            row_id = row_id_by_left_position[(shard_idx, row_in_shard)]
            shingle_cache[row_id] = _compute_shingle_hashes_from_text(
                text=content,
                ngram_size=task.shingle_ngram_size,
                chunk_size=8192,
            )
    for shard_idx, positions in right_positions.items():
        shard_rows = _load_rows_from_shard(
            Path(task.input_shards[shard_idx]),
            positions,
            task.row_batch_rows,
        )
        for row_in_shard, content in shard_rows.items():
            row_id = row_id_by_right_position[(shard_idx, row_in_shard)]
            shingle_cache[row_id] = _compute_shingle_hashes_from_text(
                text=content,
                ngram_size=task.shingle_ngram_size,
                chunk_size=8192,
            )

    verified_left: list[int] = []
    verified_right: list[int] = []
    verified_scores: list[float] = []
    pair_file = pq.ParquetFile(group_path)
    for batch in pair_file.iter_batches(batch_size=task.verify_pair_batch_rows):
        table = pa.Table.from_batches([batch])
        payload = table.to_pydict()
        for left_row_id, right_row_id in zip(
            payload["left_row_id"],
            payload["right_row_id"],
            strict=True,
        ):
            left_hashes = shingle_cache[int(left_row_id)]
            right_hashes = shingle_cache[int(right_row_id)]
            score = _compute_jaccard(left_hashes, right_hashes)
            if score >= task.jaccard_threshold:
                verified_left.append(int(left_row_id))
                verified_right.append(int(right_row_id))
                verified_scores.append(score)
    if verified_left:
        pq.write_table(
            pa.table(
                {
                    "left_row_id": pa.array(np.asarray(verified_left, dtype=np.int64), type=pa.int64()),
                    "right_row_id": pa.array(np.asarray(verified_right, dtype=np.int64), type=pa.int64()),
                    "jaccard": pa.array(np.asarray(verified_scores, dtype=np.float32), type=pa.float32()),
                }
            ),
            verified_path,
        )
    return VerificationGroupResult(
        verified_path=str(verified_path),
        verified_pair_count=len(verified_left),
    )


def _load_rows_from_shard(
    shard_path: Path,
    row_positions: set[int],
    row_batch_rows: int,
) -> dict[int, str]:
    if not row_positions:
        return {}
    positions = sorted(row_positions)
    target_index = 0
    current_target = positions[target_index]
    rows: dict[int, str] = {}
    parquet_file = pq.ParquetFile(shard_path)
    row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=row_batch_rows):
        records = batch.to_pylist()
        for index, row in enumerate(records):
            absolute_index = row_offset + index
            while absolute_index > current_target:
                target_index += 1
                if target_index >= len(positions):
                    return rows
                current_target = positions[target_index]
            if absolute_index == current_target:
                content = row["content"]
                if content is None:
                    raise ValueError(f"candidate row has null content: {shard_path}#{absolute_index}")
                rows[absolute_index] = content
                target_index += 1
                if target_index >= len(positions):
                    return rows
                current_target = positions[target_index]
        row_offset += len(records)
    return rows


def _compute_jaccard(
    left: np.ndarray[Any, np.dtype[np.uint64]],
    right: np.ndarray[Any, np.dtype[np.uint64]],
) -> float:
    if left.size == 0 and right.size == 0:
        return 1.0
    if left.size == 0 or right.size == 0:
        return 0.0
    intersection = np.intersect1d(left, right, assume_unique=True).size
    union = left.size + right.size - intersection
    if union == 0:
        return 0.0
    return float(intersection / union)


def _build_clusters(
    *,
    verified_paths: Sequence[Path],
    row_index_paths: Sequence[Path],
    total_rows: int,
    minhash_root: Path,
    duckdb_threads: int,
) -> dict[str, Any]:
    removed_mask = np.zeros(total_rows, dtype=bool)
    representative_row_ids = np.full(total_rows, -1, dtype=np.int64)
    reason_codes = np.zeros(total_rows, dtype=np.uint8)
    if not verified_paths:
        return {
            "removed_mask": removed_mask,
            "representative_row_ids": representative_row_ids,
            "reason_codes": reason_codes,
            "representative_locator_by_row_id": {},
            "near_cluster_count": 0,
            "source_counts_deleted": {},
            "data_usage_counts_deleted": {},
        }

    parent = np.arange(total_rows, dtype=np.int64)
    rank = np.zeros(total_rows, dtype=np.uint8)
    touched_mask = np.zeros(total_rows, dtype=bool)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    progress = progress_bar(total=len(verified_paths), desc="near_dedup/clusters")
    try:
        for verified_path in verified_paths:
            parquet_file = pq.ParquetFile(verified_path)
            for batch in parquet_file.iter_batches(batch_size=8192):
                payload = pa.Table.from_batches([batch]).to_pydict()
                for left_row_id, right_row_id in zip(
                    payload["left_row_id"],
                    payload["right_row_id"],
                    strict=True,
                ):
                    left_id = int(left_row_id)
                    right_id = int(right_row_id)
                    union(left_id, right_id)
                    touched_mask[left_id] = True
                    touched_mask[right_id] = True
            progress.update(1)
    finally:
        progress.close()

    touched_row_ids = np.flatnonzero(touched_mask)
    roots = np.fromiter((find(int(row_id)) for row_id in touched_row_ids), dtype=np.int64, count=touched_row_ids.size)
    cluster_members_path = minhash_root / "clusters" / "cluster_members.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": pa.array(touched_row_ids.astype(np.int64), type=pa.int64()),
                "root_id": pa.array(roots.astype(np.int64), type=pa.int64()),
            }
        ),
        cluster_members_path,
    )

    deleted_rows_path = minhash_root / "clusters" / "deleted_rows.parquet"
    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={duckdb_threads}")
        row_index_sql = _read_parquet_sql(row_index_paths)
        con.execute(
            f"""
            COPY (
                WITH members AS (
                    SELECT
                        c.root_id,
                        c.row_id,
                        i.record_locator,
                        i.source,
                        i.data_usage,
                        i.split,
                        i.content_length,
                        CASE i.data_usage
                            WHEN 'PT' THEN 1
                            WHEN 'SFT' THEN 2
                            WHEN 'REASONING' THEN 3
                            ELSE 0
                        END AS usage_priority
                    FROM read_parquet('{_sql_quote(str(cluster_members_path))}') AS c
                    JOIN {row_index_sql} AS i
                      ON c.row_id = i.row_id
                ),
                ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY root_id
                            ORDER BY usage_priority DESC, content_length DESC, row_id ASC
                        ) AS row_rank
                    FROM members
                ),
                representatives AS (
                    SELECT
                        root_id,
                        row_id AS representative_row_id,
                        record_locator AS representative_locator,
                        usage_priority AS representative_usage_priority,
                        content_length AS representative_content_length
                    FROM ranked
                    WHERE row_rank = 1
                )
                SELECT
                    member.root_id,
                    member.row_id AS removed_row_id,
                    representative.representative_row_id,
                    representative.representative_locator,
                    member.source,
                    member.data_usage,
                    member.split,
                    member.record_locator,
                    CASE
                        WHEN representative.representative_usage_priority > member.usage_priority THEN 'higher_data_usage'
                        WHEN representative.representative_content_length > member.content_length THEN 'longer_content'
                        ELSE 'stable_locator_tiebreak'
                    END AS reason_label
                FROM ranked AS member
                JOIN representatives AS representative
                  ON member.root_id = representative.root_id
                WHERE member.row_id <> representative.representative_row_id
                ORDER BY removed_row_id
            )
            TO '{_sql_quote(str(deleted_rows_path))}'
            (FORMAT PARQUET)
            """
        )
        near_cluster_count = int(
            con.execute(
                f"SELECT COUNT(DISTINCT root_id) FROM read_parquet('{_sql_quote(str(deleted_rows_path))}')"
            ).fetchone()[0]
        )
        source_counts_deleted = _fetch_count_map(
            con,
            f"SELECT source, COUNT(*) AS cnt FROM read_parquet('{_sql_quote(str(deleted_rows_path))}') GROUP BY source",
        )
        data_usage_counts_deleted = _fetch_count_map(
            con,
            f"SELECT data_usage, COUNT(*) AS cnt FROM read_parquet('{_sql_quote(str(deleted_rows_path))}') GROUP BY data_usage",
        )
    finally:
        con.close()

    representative_locator_by_row_id: dict[int, str] = {}
    parquet_file = pq.ParquetFile(deleted_rows_path)
    for batch in parquet_file.iter_batches(batch_size=8192):
        payload = pa.Table.from_batches([batch]).to_pydict()
        for removed_row_id, representative_row_id, representative_locator, reason_label in zip(
            payload["removed_row_id"],
            payload["representative_row_id"],
            payload["representative_locator"],
            payload["reason_label"],
            strict=True,
        ):
            removed_index = int(removed_row_id)
            representative_index = int(representative_row_id)
            removed_mask[removed_index] = True
            representative_row_ids[removed_index] = representative_index
            reason_codes[removed_index] = REASON_CODE_BY_LABEL[str(reason_label)]
            representative_locator_by_row_id.setdefault(representative_index, str(representative_locator))

    return {
        "removed_mask": removed_mask,
        "representative_row_ids": representative_row_ids,
        "reason_codes": reason_codes,
        "representative_locator_by_row_id": representative_locator_by_row_id,
        "near_cluster_count": near_cluster_count,
        "source_counts_deleted": source_counts_deleted,
        "data_usage_counts_deleted": data_usage_counts_deleted,
    }


def _write_retained_rows_and_deleted_log(
    *,
    input_shards: Sequence[Path],
    output_root: Path,
    target_shard_bytes: int,
    deleted_log_path: Path,
    removed_mask: np.ndarray[Any, np.dtype[np.bool_]],
    representative_row_ids: np.ndarray[Any, np.dtype[np.int64]],
    reason_codes: np.ndarray[Any, np.dtype[np.uint8]],
    representative_locator_by_row_id: dict[int, str],
    run_id: str,
    threshold: float,
    row_batch_rows: int,
) -> tuple[list[NearDedupShardResult], int, Counter[str]]:
    results: list[NearDedupShardResult] = []
    current_writer: pq.ParquetWriter | None = None
    current_rows = 0
    current_bytes = 0
    shard_index = 1
    row_id = 0
    deleted_count = 0
    reason_counts: Counter[str] = Counter()
    progress = progress_bar(
        total=sum(pq.ParquetFile(path).metadata.num_rows for path in input_shards),
        desc="near_dedup/retained",
    )
    ensure_parent_dir(deleted_log_path)
    try:
        with deleted_log_path.open("w", encoding="utf-8") as deleted_file:
            for shard_path in input_shards:
                parquet_file = pq.ParquetFile(shard_path)
                row_in_shard = 0
                for batch in parquet_file.iter_batches(batch_size=row_batch_rows):
                    rows = batch.to_pylist()
                    retained_rows: list[dict[str, Any]] = []
                    for batch_index, row in enumerate(rows):
                        if removed_mask[row_id]:
                            representative_row_id = int(representative_row_ids[row_id])
                            reason_label = REASON_LABEL_BY_CODE[int(reason_codes[row_id])]
                            reason_counts[reason_label] += 1
                            deleted_payload = {
                                "run_id": run_id,
                                "stage": "minhash_lsh_dedup",
                                "source": row["source"],
                                "data_usage": row["data_usage"],
                                "split": row["split"],
                                "content_hash": hashlib.sha256(
                                    (row["content"] or "").encode("utf-8")
                                ).hexdigest(),
                                "record_locator": f"{shard_path}#{row_in_shard + batch_index}",
                                "reason_code": "near_duplicate",
                                "reason_detail": f"verified near duplicate with jaccard>={threshold}",
                                "representative_locator": representative_locator_by_row_id[representative_row_id],
                                "content_preview": truncate_sample(row["content"]),
                            }
                            deleted_file.write(json.dumps(deleted_payload, ensure_ascii=False) + "\n")
                            deleted_count += 1
                        else:
                            retained_rows.append(
                                {
                                    "source": row["source"],
                                    "data_usage": row["data_usage"],
                                    "split": row["split"],
                                    "content": row["content"],
                                    "messages": row["messages"],
                                    "token_count": row["token_count"],
                                }
                            )
                        row_id += 1
                        progress.update(1)
                    row_in_shard += len(rows)
                    if retained_rows:
                        table = pa.Table.from_pylist(retained_rows, schema=STAGING_SCHEMA)
                        if current_writer is None:
                            shard_path_out = output_root / f"part-{shard_index:06d}.parquet"
                            current_writer = pq.ParquetWriter(shard_path_out, STAGING_SCHEMA)
                        current_writer.write_table(table)
                        current_rows += table.num_rows
                        current_bytes += table.nbytes
                        if current_bytes >= target_shard_bytes:
                            current_writer.close()
                            results.append(
                                NearDedupShardResult(
                                    shard_path=str(output_root / f"part-{shard_index:06d}.parquet"),
                                    row_count=current_rows,
                                    approx_nbytes=current_bytes,
                                )
                            )
                            log(
                                f"[near_dedup/shard] wrote {output_root / f'part-{shard_index:06d}.parquet'} "
                                f"rows={current_rows} approx_nbytes={current_bytes}"
                            )
                            shard_index += 1
                            current_writer = None
                            current_rows = 0
                            current_bytes = 0
            if current_writer is not None:
                current_writer.close()
                results.append(
                    NearDedupShardResult(
                        shard_path=str(output_root / f"part-{shard_index:06d}.parquet"),
                        row_count=current_rows,
                        approx_nbytes=current_bytes,
                    )
                )
                log(
                    f"[near_dedup/shard] wrote {output_root / f'part-{shard_index:06d}.parquet'} "
                    f"rows={current_rows} approx_nbytes={current_bytes}"
                )
    finally:
        progress.close()
    return results, deleted_count, reason_counts


def _subtract_counts(before: dict[str, int], deleted: dict[str, int]) -> dict[str, int]:
    return {
        key: value - deleted.get(key, 0)
        for key, value in sorted(before.items())
    }


def _fetch_count_map(con: duckdb.DuckDBPyConnection, query: str) -> dict[str, int]:
    rows = con.execute(query).fetchall()
    return dict(sorted((str(key), int(value)) for key, value in rows))


def _read_parquet_sql(paths: Sequence[Path]) -> str:
    quoted = ", ".join(f"'{_sql_quote(str(path))}'" for path in paths)
    return f"read_parquet([{quoted}])"


def _write_summary_files(
    summary: NearDedupSummary,
    summary_json_path: Path,
    manifest_path: Path,
) -> None:
    write_json(
        summary_json_path,
        {
            "deleted_row_count": summary.deleted_row_count,
            "candidate_pair_count": summary.candidate_pair_count,
            "verified_pair_count": summary.verified_pair_count,
            "near_cluster_count": summary.near_cluster_count,
            "giant_bucket_count": summary.giant_bucket_count,
            "giant_bucket_fallback_count": summary.giant_bucket_fallback_count,
            "source_counts_deleted": summary.source_counts_deleted,
            "data_usage_counts_deleted": summary.data_usage_counts_deleted,
            "representative_reason_counts": summary.representative_reason_counts,
            "jaccard_threshold": summary.jaccard_threshold,
        },
    )
    write_json(manifest_path, summary.to_dict())


def _sql_quote(text: str) -> str:
    return text.replace("'", "''")


def main() -> None:
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    run_near_dedup(build_default_config())


if __name__ == "__main__":
    main()
