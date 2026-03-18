from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from collections.abc import Iterator
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

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
INPUT_ROOT = Path("data/korean_processed/_staging")
OUTPUT_ROOT = Path("data/korean_processed/exact_dedup")
TARGET_SHARD_BYTES = 1_073_741_824
DUCKDB_THREADS = 1
DUCKDB_PRESERVE_INSERTION_ORDER = False
DUCKDB_MEMORY_LIMIT = "30GB"
DUCKDB_ARROW_LARGE_BUFFER_SIZE = True
OVERWRITE_OUTPUT = True
FETCH_RECORD_BATCH_ROWS = 256
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


@dataclass(frozen=True)
class ExactDedupShardResult:
    shard_path: str
    row_count: int
    approx_nbytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactDedupSummary:
    run_id: str
    input_shard_count: int
    input_row_count: int
    retained_row_count: int
    deleted_row_count: int
    null_content_passthrough_count: int
    exact_cluster_count: int
    output_shard_count: int
    output_shards: list[dict[str, Any]]
    source_counts_before: dict[str, int]
    source_counts_after: dict[str, int]
    source_counts_deleted: dict[str, int]
    data_usage_counts_before: dict[str, int]
    data_usage_counts_after: dict[str, int]
    data_usage_counts_deleted: dict[str, int]
    representative_reason_counts: dict[str, int]
    started_at: str
    finished_at: str
    duration_sec: float
    output_root: str
    deleted_log_path: str
    summary_json_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_input_shards(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/*/part-*.parquet"), key=lambda path: str(path))


def _count_input_datasets(input_shards: list[Path]) -> int:
    return len({str(path.parent.parent) for path in input_shards})


def run_exact_dedup(
    *,
    input_root: Path,
    output_root: Path,
    target_shard_bytes: int,
    overwrite_output: bool,
    duckdb_threads: int,
    duckdb_preserve_insertion_order: bool,
    duckdb_memory_limit: str,
    duckdb_arrow_large_buffer_size: bool,
    fetch_record_batch_rows: int,
) -> ExactDedupSummary:
    if duckdb_threads < 1:
        raise ValueError(f"duckdb_threads must be >= 1, got {duckdb_threads}")
    if fetch_record_batch_rows < 1:
        raise ValueError(
            f"fetch_record_batch_rows must be >= 1, got {fetch_record_batch_rows}"
        )
    if not duckdb_memory_limit.strip():
        raise ValueError("duckdb_memory_limit must be non-empty")
    input_shards = discover_input_shards(input_root)
    if not input_shards:
        raise FileNotFoundError(f"no input parquet shards found under {input_root}")
    input_dataset_count = _count_input_datasets(input_shards)

    deleted_log_path = output_root / "_logs" / "deleted_rows_exact.jsonl"
    summary_json_path = output_root / "_meta" / "deleted_rows_exact_summary.json"
    manifest_path = output_root / "_meta" / "exact_dedup_manifest.json"
    temp_root = output_root / "_tmp"
    db_path = temp_root / "exact_dedup.duckdb"

    _prepare_output_root(output_root=output_root, overwrite_output=overwrite_output)
    temp_root.mkdir(parents=True, exist_ok=True)

    started_at = now_utc_iso()
    started_perf = perf_counter()
    run_id = started_at.replace("-", "").replace(":", "").replace(".", "")
    log(
        "[exact_dedup] start "
        f"input_root={input_root} output_root={output_root} "
        f"datasets={input_dataset_count} input_shards={len(input_shards)} "
        f"target_shard_bytes={target_shard_bytes} duckdb_threads={duckdb_threads} "
        f"duckdb_preserve_insertion_order={duckdb_preserve_insertion_order} "
        f"duckdb_memory_limit={duckdb_memory_limit} "
        f"duckdb_arrow_large_buffer_size={duckdb_arrow_large_buffer_size}"
    )

    con = duckdb.connect(str(db_path))
    try:
        log(f"[exact_dedup] preparing output root at {output_root}")
        _configure_duckdb(
            con=con,
            temp_root=temp_root,
            duckdb_threads=duckdb_threads,
            duckdb_preserve_insertion_order=duckdb_preserve_insertion_order,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_arrow_large_buffer_size=duckdb_arrow_large_buffer_size,
        )
        log(f"[exact_dedup] duckdb configured db_path={db_path} temp_root={temp_root}")
        log("[exact_dedup] building staged/ranked temp tables")
        _create_ranked_tables(con=con, input_shards=input_shards)
        log("[exact_dedup] temp tables ready")

        input_row_count = int(
            con.execute("SELECT COUNT(*) FROM staged_rows").fetchone()[0]
        )
        null_content_passthrough_count = int(
            con.execute("SELECT COUNT(*) FROM staged_rows WHERE content IS NULL").fetchone()[0]
        )
        exact_cluster_count = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM ranked_non_null
                WHERE content_rank = 1 AND cluster_size > 1
                """
            ).fetchone()[0]
        )
        log(
            "[exact_dedup] counts ready "
            f"input_rows={input_row_count} null_passthrough={null_content_passthrough_count} "
            f"exact_clusters={exact_cluster_count}"
        )

        deleted_row_count, representative_reason_counts = _write_deleted_rows_log(
            con=con,
            run_id=run_id,
            deleted_log_path=deleted_log_path,
            fetch_record_batch_rows=fetch_record_batch_rows,
        )
        output_shards = _write_retained_shards(
            con=con,
            output_root=output_root,
            target_shard_bytes=target_shard_bytes,
            fetch_record_batch_rows=fetch_record_batch_rows,
        )
        log(
            "[exact_dedup] aggregates ready "
            f"deleted_rows={deleted_row_count} retained_rows={sum(shard.row_count for shard in output_shards)}"
        )

        retained_row_count = sum(shard.row_count for shard in output_shards)
        source_counts_before = _fetch_count_map(
            con, "SELECT source, COUNT(*) FROM staged_rows GROUP BY source"
        )
        source_counts_after = _fetch_count_map(
            con, "SELECT source, COUNT(*) FROM retained_rows GROUP BY source"
        )
        source_counts_deleted = _fetch_count_map(
            con, "SELECT source, COUNT(*) FROM deleted_rows GROUP BY source"
        )
        data_usage_counts_before = _fetch_count_map(
            con, "SELECT data_usage, COUNT(*) FROM staged_rows GROUP BY data_usage"
        )
        data_usage_counts_after = _fetch_count_map(
            con, "SELECT data_usage, COUNT(*) FROM retained_rows GROUP BY data_usage"
        )
        data_usage_counts_deleted = _fetch_count_map(
            con, "SELECT data_usage, COUNT(*) FROM deleted_rows GROUP BY data_usage"
        )

        finished_at = now_utc_iso()
        duration_sec = round(perf_counter() - started_perf, 3)

        summary = ExactDedupSummary(
            run_id=run_id,
            input_shard_count=len(input_shards),
            input_row_count=input_row_count,
            retained_row_count=retained_row_count,
            deleted_row_count=deleted_row_count,
            null_content_passthrough_count=null_content_passthrough_count,
            exact_cluster_count=exact_cluster_count,
            output_shard_count=len(output_shards),
            output_shards=[shard.to_dict() for shard in output_shards],
            source_counts_before=source_counts_before,
            source_counts_after=source_counts_after,
            source_counts_deleted=source_counts_deleted,
            data_usage_counts_before=data_usage_counts_before,
            data_usage_counts_after=data_usage_counts_after,
            data_usage_counts_deleted=data_usage_counts_deleted,
            representative_reason_counts=dict(sorted(representative_reason_counts.items())),
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
            output_root=str(output_root),
            deleted_log_path=str(deleted_log_path),
            summary_json_path=str(summary_json_path),
            manifest_path=str(manifest_path),
        )
        log(
            "[exact_dedup] writing metadata "
            f"summary_json_path={summary_json_path} manifest_path={manifest_path}"
        )
        _write_summary_files(
            summary=summary,
            summary_json_path=summary_json_path,
            manifest_path=manifest_path,
        )
        log(
            "[exact_dedup] completed "
            f"input_rows={input_row_count} retained={retained_row_count} "
            f"deleted={deleted_row_count} output_shards={len(output_shards)} "
            f"duration_sec={duration_sec} deleted_log_path={deleted_log_path} "
            f"manifest_path={manifest_path}"
        )
        return summary
    finally:
        con.close()
        if temp_root.exists():
            shutil.rmtree(temp_root)


def _prepare_output_root(*, output_root: Path, overwrite_output: bool) -> None:
    if output_root.exists():
        existing_paths = list(output_root.iterdir())
        if existing_paths and not overwrite_output:
            raise FileExistsError(
                f"output_root already contains files: {output_root}. "
                "Set OVERWRITE_OUTPUT=True to rebuild."
            )
        if overwrite_output:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_logs").mkdir(parents=True, exist_ok=True)
    (output_root / "_meta").mkdir(parents=True, exist_ok=True)


def _configure_duckdb(
    *,
    con: duckdb.DuckDBPyConnection,
    temp_root: Path,
    duckdb_threads: int,
    duckdb_preserve_insertion_order: bool,
    duckdb_memory_limit: str,
    duckdb_arrow_large_buffer_size: bool,
) -> None:
    temp_dir = temp_root / "spill"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA threads={duckdb_threads}")
    con.execute(f"PRAGMA temp_directory='{_sql_quote(str(temp_dir))}'")
    con.execute(
        "PRAGMA preserve_insertion_order="
        f"{_sql_bool(duckdb_preserve_insertion_order)}"
    )
    con.execute(f"PRAGMA memory_limit='{_sql_quote(duckdb_memory_limit)}'")
    con.execute(
        "SET arrow_large_buffer_size="
        f"{_sql_bool(duckdb_arrow_large_buffer_size)}"
    )


def _create_ranked_tables(
    *,
    con: duckdb.DuckDBPyConnection,
    input_shards: list[Path],
) -> None:
    shard_literals = ", ".join(f"'{_sql_quote(str(path))}'" for path in input_shards)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE staged_rows AS
        SELECT
            source,
            data_usage,
            split,
            content,
            messages,
            token_count,
            filename,
            file_row_number,
            filename || '#' || CAST(file_row_number AS VARCHAR) AS record_locator,
            CASE data_usage
                WHEN 'REASONING' THEN 3
                WHEN 'SFT' THEN 2
                ELSE 1
            END AS usage_rank,
            CASE
                WHEN content IS NULL THEN NULL
                ELSE length(content)
            END AS content_length
        FROM read_parquet([{shard_literals}], filename=true, file_row_number=true)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ranked_non_null AS
        SELECT
            source,
            data_usage,
            split,
            content,
            messages,
            token_count,
            filename,
            file_row_number,
            record_locator,
            usage_rank,
            content_length,
            ROW_NUMBER() OVER (
                PARTITION BY content
                ORDER BY usage_rank DESC, content_length DESC, filename ASC, file_row_number ASC
            ) AS content_rank,
            COUNT(*) OVER (PARTITION BY content) AS cluster_size,
            FIRST_VALUE(record_locator) OVER (
                PARTITION BY content
                ORDER BY usage_rank DESC, content_length DESC, filename ASC, file_row_number ASC
            ) AS representative_locator,
            FIRST_VALUE(data_usage) OVER (
                PARTITION BY content
                ORDER BY usage_rank DESC, content_length DESC, filename ASC, file_row_number ASC
            ) AS representative_data_usage,
            FIRST_VALUE(usage_rank) OVER (
                PARTITION BY content
                ORDER BY usage_rank DESC, content_length DESC, filename ASC, file_row_number ASC
            ) AS representative_usage_rank,
            FIRST_VALUE(content_length) OVER (
                PARTITION BY content
                ORDER BY usage_rank DESC, content_length DESC, filename ASC, file_row_number ASC
            ) AS representative_content_length
        FROM staged_rows
        WHERE content IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW retained_rows AS
        SELECT
            source,
            data_usage,
            split,
            content,
            messages,
            token_count,
            filename,
            file_row_number,
            record_locator
        FROM staged_rows
        WHERE content IS NULL
        UNION ALL
        SELECT
            source,
            data_usage,
            split,
            content,
            messages,
            token_count,
            filename,
            file_row_number,
            record_locator
        FROM ranked_non_null
        WHERE content_rank = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW deleted_rows AS
        SELECT
            source,
            data_usage,
            split,
            content,
            filename,
            file_row_number,
            record_locator,
            representative_locator,
            usage_rank,
            content_length,
            representative_usage_rank,
            representative_content_length
        FROM ranked_non_null
        WHERE content_rank > 1
        """
    )


def _write_deleted_rows_log(
    *,
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    deleted_log_path: Path,
    fetch_record_batch_rows: int,
) -> tuple[int, Counter[str]]:
    ensure_parent_dir(deleted_log_path)
    reason_counts: Counter[str] = Counter()
    deleted_count = 0
    deleted_total = int(con.execute("SELECT COUNT(*) FROM deleted_rows").fetchone()[0])
    log(
        "[exact_dedup/deleted_log] start "
        f"deleted_total={deleted_total} deleted_log_path={deleted_log_path}"
    )
    query = """
        SELECT
            source,
            data_usage,
            split,
            content,
            record_locator,
            representative_locator,
            usage_rank,
            content_length,
            representative_usage_rank,
            representative_content_length
        FROM deleted_rows
        ORDER BY representative_locator, filename, file_row_number
    """
    progress = progress_bar(total=deleted_total, desc="exact_dedup/deleted_log")
    try:
        with deleted_log_path.open("w", encoding="utf-8") as f:
            for batch in _iter_record_batches(
                con=con,
                query=query,
                fetch_record_batch_rows=fetch_record_batch_rows,
            ):
                for row in batch.to_pylist():
                    reason_label = _determine_representative_reason(row)
                    reason_counts[reason_label] += 1
                    payload = {
                        "run_id": run_id,
                        "stage": "exact_dedup",
                        "source": row["source"],
                        "data_usage": row["data_usage"],
                        "split": row["split"],
                        "content_hash": hashlib.sha256(
                            row["content"].encode("utf-8")
                        ).hexdigest(),
                        "record_locator": row["record_locator"],
                        "reason_code": "exact_duplicate",
                        "reason_detail": _build_reason_detail(reason_label),
                        "representative_locator": row["representative_locator"],
                        "content_preview": truncate_sample(row["content"]),
                    }
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    deleted_count += 1
                progress.update(batch.num_rows)
    finally:
        progress.close()
    log(
        "[exact_dedup/deleted_log] completed "
        f"deleted_count={deleted_count} representative_reason_counts={dict(sorted(reason_counts.items()))}"
    )
    return deleted_count, reason_counts


def _write_retained_shards(
    *,
    con: duckdb.DuckDBPyConnection,
    output_root: Path,
    target_shard_bytes: int,
    fetch_record_batch_rows: int,
) -> list[ExactDedupShardResult]:
    results: list[ExactDedupShardResult] = []
    current_batches: list[pa.RecordBatch] = []
    current_rows = 0
    current_bytes = 0
    shard_index = 1

    retained_total = int(con.execute("SELECT COUNT(*) FROM retained_rows").fetchone()[0])
    log(
        "[exact_dedup/retained] start "
        f"retained_total={retained_total} output_root={output_root} target_shard_bytes={target_shard_bytes}"
    )
    progress = progress_bar(total=retained_total, desc="exact_dedup/retained")
    try:
        query = """
            SELECT
                source,
                data_usage,
                split,
                content,
                messages,
                token_count,
                filename,
                file_row_number,
                record_locator
            FROM retained_rows
            ORDER BY filename, file_row_number
        """
        for batch in _iter_record_batches(
            con=con,
            query=query,
            fetch_record_batch_rows=fetch_record_batch_rows,
        ):
            canonical_batch = batch.select(range(6))
            if canonical_batch.nbytes > target_shard_bytes or (
                current_bytes > 0 and current_bytes + canonical_batch.nbytes > target_shard_bytes
            ):
                for row_index in range(canonical_batch.num_rows):
                    row_batch = canonical_batch.slice(row_index, 1)
                    current_batches.append(row_batch)
                    current_rows += row_batch.num_rows
                    current_bytes += row_batch.nbytes
                    progress.update(1)
                    if current_bytes > target_shard_bytes:
                        retained_so_far = sum(result.row_count for result in results) + current_rows
                        results.append(
                            _flush_retained_shard(
                                output_root=output_root,
                                batches=current_batches,
                                shard_index=shard_index,
                                row_count=current_rows,
                                approx_nbytes=current_bytes,
                            )
                        )
                        log(
                            "[exact_dedup/retained] shard flushed "
                            f"shard_index={shard_index} shard_rows={current_rows} "
                            f"retained_written={retained_so_far}"
                        )
                        shard_index += 1
                        current_batches = []
                        current_rows = 0
                        current_bytes = 0
                continue

            current_batches.append(canonical_batch)
            current_rows += canonical_batch.num_rows
            current_bytes += canonical_batch.nbytes
            progress.update(canonical_batch.num_rows)
            if current_bytes > target_shard_bytes:
                retained_so_far = sum(result.row_count for result in results) + current_rows
                results.append(
                    _flush_retained_shard(
                        output_root=output_root,
                        batches=current_batches,
                        shard_index=shard_index,
                        row_count=current_rows,
                        approx_nbytes=current_bytes,
                    )
                )
                log(
                    "[exact_dedup/retained] shard flushed "
                    f"shard_index={shard_index} shard_rows={current_rows} "
                    f"retained_written={retained_so_far}"
                )
                shard_index += 1
                current_batches = []
                current_rows = 0
                current_bytes = 0
        if current_batches:
            retained_so_far = sum(result.row_count for result in results) + current_rows
            results.append(
                _flush_retained_shard(
                    output_root=output_root,
                    batches=current_batches,
                    shard_index=shard_index,
                    row_count=current_rows,
                    approx_nbytes=current_bytes,
                )
            )
            log(
                "[exact_dedup/retained] shard flushed "
                f"shard_index={shard_index} shard_rows={current_rows} "
                f"retained_written={retained_so_far}"
            )
    finally:
        progress.close()
    log(
        "[exact_dedup/retained] completed "
        f"output_shards={len(results)} retained_total={sum(result.row_count for result in results)}"
    )
    return results


def _flush_retained_shard(
    *,
    output_root: Path,
    batches: list[pa.RecordBatch],
    shard_index: int,
    row_count: int,
    approx_nbytes: int,
) -> ExactDedupShardResult:
    table = pa.Table.from_batches(batches)
    table = table.select(STAGING_SCHEMA.names)
    if table.schema != STAGING_SCHEMA:
        table = table.cast(STAGING_SCHEMA)
    shard_path = output_root / f"part-{shard_index:06d}.parquet"
    ensure_parent_dir(shard_path)
    pq.write_table(table, shard_path)
    log(f"[exact_dedup/shard] wrote {shard_path} rows={row_count} approx_nbytes={approx_nbytes}")
    return ExactDedupShardResult(
        shard_path=str(shard_path),
        row_count=row_count,
        approx_nbytes=approx_nbytes,
    )


def _fetch_count_map(
    con: duckdb.DuckDBPyConnection,
    query: str,
) -> dict[str, int]:
    rows = con.execute(query).fetchall()
    return {str(key): int(value) for key, value in sorted(rows, key=lambda item: str(item[0]))}


def _write_summary_files(
    *,
    summary: ExactDedupSummary,
    summary_json_path: Path,
    manifest_path: Path,
) -> None:
    summary_payload = {
        "deleted_row_count": summary.deleted_row_count,
        "source_counts_deleted": summary.source_counts_deleted,
        "data_usage_counts_deleted": summary.data_usage_counts_deleted,
        "exact_cluster_count": summary.exact_cluster_count,
        "representative_reason_counts": summary.representative_reason_counts,
    }
    write_json(summary_json_path, summary_payload)
    write_json(manifest_path, summary.to_dict())


def _determine_representative_reason(row: dict[str, Any]) -> str:
    if row["representative_usage_rank"] > row["usage_rank"]:
        return "higher_data_usage"
    if row["representative_content_length"] > row["content_length"]:
        return "longer_content"
    return "stable_locator_tiebreak"


def _build_reason_detail(reason_label: str) -> str:
    if reason_label == "higher_data_usage":
        return "duplicate removed because representative row had higher data_usage priority"
    if reason_label == "longer_content":
        return "duplicate removed because representative row had longer content"
    return "duplicate removed because representative row won stable locator tie-break"


def _iter_record_batches(
    *,
    con: duckdb.DuckDBPyConnection,
    query: str,
    fetch_record_batch_rows: int,
) -> Iterator[pa.RecordBatch]:
    reader = con.execute(query).fetch_record_batch(rows_per_batch=fetch_record_batch_rows)
    for batch in reader:
        if batch.num_rows > 0:
            yield batch


def _sql_quote(text: str) -> str:
    return text.replace("'", "''")


def _sql_bool(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    run_exact_dedup(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        target_shard_bytes=TARGET_SHARD_BYTES,
        overwrite_output=OVERWRITE_OUTPUT,
        duckdb_threads=DUCKDB_THREADS,
        duckdb_preserve_insertion_order=DUCKDB_PRESERVE_INSERTION_ORDER,
        duckdb_memory_limit=DUCKDB_MEMORY_LIMIT,
        duckdb_arrow_large_buffer_size=DUCKDB_ARROW_LARGE_BUFFER_SIZE,
        fetch_record_batch_rows=FETCH_RECORD_BATCH_ROWS,
    )


if __name__ == "__main__":
    main()
