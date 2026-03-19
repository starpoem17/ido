from __future__ import annotations

import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import src.preprocess.common as common
from src.preprocess.common import ensure_parent_dir, log, now_utc_iso, progress_bar, write_json


# =========================
# User configuration block
# =========================
INPUT_ROOT = Path("data/korean_processed/near_dedup")
OUTPUT_ROOT = Path("data/korean_processed/token_count_added")
TOKENIZER_JSON_PATH = Path("data/tokenizers/korean_bbpe_v2/tokenizer.json")
NUM_WORKERS = 7
ROW_BATCH_ROWS = 4096
OVERWRITE_OUTPUT = False
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


@dataclass(frozen=True)
class AddTokenCountConfig:
    input_root: Path
    output_root: Path
    tokenizer_json_path: Path
    num_workers: int
    row_batch_rows: int
    overwrite_output: bool


@dataclass(frozen=True)
class AddTokenCountShardResult:
    input_shard_path: str
    output_shard_path: str
    row_count: int
    input_null_token_count_row_count: int
    input_non_null_token_count_row_count: int
    overwritten_token_count_row_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AddTokenCountSummary:
    run_id: str
    input_root: str
    output_root: str
    tokenizer_json_path: str
    input_shard_count: int
    output_shard_count: int
    input_row_count: int
    output_row_count: int
    input_null_token_count_row_count: int
    input_non_null_token_count_row_count: int
    overwritten_token_count_row_count: int
    output_shards: list[dict[str, Any]]
    started_at: str
    finished_at: str
    duration_sec: float
    summary_json_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ShardTask:
    input_shard_path: Path
    output_shard_path: Path
    tokenizer_json_path: Path
    row_batch_rows: int


def discover_input_shards(input_root: Path) -> list[Path]:
    return common.stable_sorted_paths(input_root.glob("part-*.parquet"))


def run_add_token_count(config: AddTokenCountConfig) -> AddTokenCountSummary:
    _validate_config(config)
    input_shards = discover_input_shards(config.input_root)
    if not input_shards:
        raise FileNotFoundError(f"no input parquet shards found under {config.input_root}")
    if not config.tokenizer_json_path.exists():
        raise FileNotFoundError(
            f"tokenizer_json_path does not exist: {config.tokenizer_json_path}"
        )

    summary_json_path = config.output_root / "_meta" / "add_token_count_summary.json"
    manifest_path = config.output_root / "_meta" / "add_token_count_manifest.json"
    _prepare_output_root(
        output_root=config.output_root,
        overwrite_output=config.overwrite_output,
    )

    started_at = now_utc_iso()
    started_perf = perf_counter()
    run_id = started_at.replace("-", "").replace(":", "").replace(".", "")
    log(
        "[add_token_count] start "
        f"input_root={config.input_root} output_root={config.output_root} "
        f"input_shards={len(input_shards)} tokenizer_json_path={config.tokenizer_json_path} "
        f"num_workers={config.num_workers} row_batch_rows={config.row_batch_rows}"
    )

    tasks = [
        _ShardTask(
            input_shard_path=input_shard_path,
            output_shard_path=config.output_root / input_shard_path.name,
            tokenizer_json_path=config.tokenizer_json_path,
            row_batch_rows=config.row_batch_rows,
        )
        for input_shard_path in input_shards
    ]

    max_workers = min(config.num_workers, len(tasks))
    results_by_output: dict[str, AddTokenCountShardResult] = {}
    progress = progress_bar(total=len(tasks), desc="add_token_count/shards")
    try:
        if max_workers == 1:
            for task in tasks:
                log(f"[add_token_count/shard] start input={task.input_shard_path}")
                result = _process_shard(task)
                results_by_output[result.output_shard_path] = result
                progress.update(1)
                log(
                    "[add_token_count/shard] completed "
                    f"input={result.input_shard_path} output={result.output_shard_path} rows={result.row_count}"
                )
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = {}
                for task in tasks:
                    log(f"[add_token_count/shard] start input={task.input_shard_path}")
                    future = executor.submit(_process_shard, task)
                    futures[future] = task
                for future in as_completed(futures):
                    result = future.result()
                    results_by_output[result.output_shard_path] = result
                    progress.update(1)
                    log(
                        "[add_token_count/shard] completed "
                        f"input={result.input_shard_path} output={result.output_shard_path} rows={result.row_count}"
                    )
    finally:
        progress.close()

    ordered_results = [
        results_by_output[str(config.output_root / input_shard.name)] for input_shard in input_shards
    ]
    input_row_count = sum(result.row_count for result in ordered_results)
    output_row_count = sum(result.row_count for result in ordered_results)
    input_null_token_count_row_count = sum(
        result.input_null_token_count_row_count for result in ordered_results
    )
    input_non_null_token_count_row_count = sum(
        result.input_non_null_token_count_row_count for result in ordered_results
    )
    overwritten_token_count_row_count = sum(
        result.overwritten_token_count_row_count for result in ordered_results
    )

    _validate_final_outputs(
        input_shards=input_shards,
        output_root=config.output_root,
        shard_results=ordered_results,
    )

    finished_at = now_utc_iso()
    duration_sec = round(perf_counter() - started_perf, 3)
    summary = AddTokenCountSummary(
        run_id=run_id,
        input_root=str(config.input_root),
        output_root=str(config.output_root),
        tokenizer_json_path=str(config.tokenizer_json_path),
        input_shard_count=len(input_shards),
        output_shard_count=len(ordered_results),
        input_row_count=input_row_count,
        output_row_count=output_row_count,
        input_null_token_count_row_count=input_null_token_count_row_count,
        input_non_null_token_count_row_count=input_non_null_token_count_row_count,
        overwritten_token_count_row_count=overwritten_token_count_row_count,
        output_shards=[result.to_dict() for result in ordered_results],
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        summary_json_path=str(summary_json_path),
        manifest_path=str(manifest_path),
    )
    _write_summary_files(
        summary=summary,
        summary_json_path=summary_json_path,
        manifest_path=manifest_path,
    )
    log(
        "[add_token_count] completed "
        f"input_rows={input_row_count} output_rows={output_row_count} "
        f"output_shards={len(ordered_results)} overwritten_token_count_rows={overwritten_token_count_row_count} "
        f"duration_sec={duration_sec}"
    )
    return summary


def _validate_config(config: AddTokenCountConfig) -> None:
    if config.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {config.num_workers}")
    if config.row_batch_rows < 1:
        raise ValueError(f"row_batch_rows must be >= 1, got {config.row_batch_rows}")
    if config.input_root == config.output_root:
        raise ValueError("input_root and output_root must be different")


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
    (output_root / "_meta").mkdir(parents=True, exist_ok=True)


def _process_shard(task: _ShardTask) -> AddTokenCountShardResult:
    tokenizer = common.get_tokenizer(task.tokenizer_json_path)
    parquet_file = pq.ParquetFile(task.input_shard_path)
    _validate_input_schema(parquet_file.schema_arrow, task.input_shard_path)

    temp_output_path = task.output_shard_path.with_suffix(".parquet.tmp")
    ensure_parent_dir(temp_output_path)
    if temp_output_path.exists():
        temp_output_path.unlink()

    row_count = 0
    input_null_token_count_row_count = 0
    input_non_null_token_count_row_count = 0
    overwritten_token_count_row_count = 0
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet_file.iter_batches(
            columns=common.STAGING_SCHEMA.names,
            batch_size=task.row_batch_rows,
        ):
            output_batch, batch_null_count, batch_non_null_count = _build_output_batch(
                batch=batch,
                tokenizer=tokenizer,
            )
            input_null_token_count_row_count += batch_null_count
            input_non_null_token_count_row_count += batch_non_null_count
            overwritten_token_count_row_count += batch.num_rows
            row_count += batch.num_rows
            if writer is None:
                writer = pq.ParquetWriter(temp_output_path, common.FINAL_SCHEMA)
            writer.write_batch(output_batch)
        if writer is None:
            writer = pq.ParquetWriter(temp_output_path, common.FINAL_SCHEMA)
    finally:
        if writer is not None:
            writer.close()

    temp_row_count = pq.ParquetFile(temp_output_path).metadata.num_rows
    if temp_row_count != row_count:
        raise ValueError(
            f"output row count mismatch for {task.input_shard_path}: "
            f"expected={row_count} actual={temp_row_count}"
        )
    temp_schema = pq.read_schema(temp_output_path)
    if temp_schema != common.FINAL_SCHEMA:
        raise ValueError(
            f"output schema mismatch for {task.input_shard_path}: "
            f"expected={common.FINAL_SCHEMA} actual={temp_schema}"
        )
    temp_output_path.replace(task.output_shard_path)
    return AddTokenCountShardResult(
        input_shard_path=str(task.input_shard_path),
        output_shard_path=str(task.output_shard_path),
        row_count=row_count,
        input_null_token_count_row_count=input_null_token_count_row_count,
        input_non_null_token_count_row_count=input_non_null_token_count_row_count,
        overwritten_token_count_row_count=overwritten_token_count_row_count,
    )


def _validate_input_schema(schema: pa.Schema, shard_path: Path) -> None:
    missing = [name for name in common.STAGING_SCHEMA.names if name not in schema.names]
    if missing:
        raise ValueError(
            f"input shard missing canonical columns at {shard_path}: missing={missing}"
        )


def _build_output_batch(
    *,
    batch: pa.RecordBatch,
    tokenizer: Any,
) -> tuple[pa.RecordBatch, int, int]:
    content_index = batch.schema.get_field_index("content")
    token_count_index = batch.schema.get_field_index("token_count")
    if content_index < 0 or token_count_index < 0:
        raise ValueError("batch is missing required columns: content/token_count")

    contents = batch.column(content_index).to_pylist()
    existing_token_counts = batch.column(token_count_index).to_pylist()
    input_null_token_count_row_count = 0
    input_non_null_token_count_row_count = 0
    token_counts: list[int] = []
    for content, existing_token_count in zip(contents, existing_token_counts):
        if existing_token_count is None:
            input_null_token_count_row_count += 1
        else:
            input_non_null_token_count_row_count += 1
        token_count = common.compute_token_count(
            content=content,
            messages=None,
            tokenizer=tokenizer,
        )
        token_counts.append(token_count)

    arrays: list[pa.Array | pa.ChunkedArray] = []
    for field in common.FINAL_SCHEMA:
        if field.name == "token_count":
            arrays.append(pa.array(token_counts, type=pa.int32()))
            continue
        arrays.append(batch.column(batch.schema.get_field_index(field.name)))
    output_batch = pa.RecordBatch.from_arrays(arrays, schema=common.FINAL_SCHEMA)
    return output_batch, input_null_token_count_row_count, input_non_null_token_count_row_count


def _validate_final_outputs(
    *,
    input_shards: list[Path],
    output_root: Path,
    shard_results: list[AddTokenCountShardResult],
) -> None:
    output_shards = discover_input_shards(output_root)
    if len(output_shards) != len(input_shards):
        raise ValueError(
            f"output shard count mismatch: expected={len(input_shards)} actual={len(output_shards)}"
        )
    for input_shard, shard_result in zip(input_shards, shard_results):
        output_shard = output_root / input_shard.name
        if not output_shard.exists():
            raise FileNotFoundError(f"missing output shard: {output_shard}")
        if shard_result.output_shard_path != str(output_shard):
            raise ValueError(
                f"output shard path mismatch: expected={output_shard} actual={shard_result.output_shard_path}"
            )
        output_row_count = pq.ParquetFile(output_shard).metadata.num_rows
        if output_row_count != shard_result.row_count:
            raise ValueError(
                f"output shard row count mismatch for {output_shard}: "
                f"expected={shard_result.row_count} actual={output_row_count}"
            )


def _write_summary_files(
    *,
    summary: AddTokenCountSummary,
    summary_json_path: Path,
    manifest_path: Path,
) -> None:
    summary_payload = {
        "input_row_count": summary.input_row_count,
        "output_row_count": summary.output_row_count,
        "input_null_token_count_row_count": summary.input_null_token_count_row_count,
        "input_non_null_token_count_row_count": summary.input_non_null_token_count_row_count,
        "overwritten_token_count_row_count": summary.overwritten_token_count_row_count,
    }
    write_json(summary_json_path, summary_payload)
    write_json(manifest_path, summary.to_dict())


def build_default_config() -> AddTokenCountConfig:
    return AddTokenCountConfig(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        tokenizer_json_path=TOKENIZER_JSON_PATH,
        num_workers=NUM_WORKERS,
        row_batch_rows=ROW_BATCH_ROWS,
        overwrite_output=OVERWRITE_OUTPUT,
    )


def main() -> None:
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    common.TOKENIZER_JSON_PATH = TOKENIZER_JSON_PATH
    run_add_token_count(build_default_config())


if __name__ == "__main__":
    main()
