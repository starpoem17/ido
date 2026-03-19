from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.tokenizer import common


TOKENIZER_JSON_PATH = Path("data/tokenizers/korean_bbpe_v2/tokenizer.json")
BENCHMARK_PARQUET_PATH = Path(
    "data/korean_raw/HAERAE-HUB-KOREAN-WEBTEXT/train-00000-of-00018.parquet"
)
DELTA_SAMPLE_LIMIT = 1000
NUM_WORKERS = 7
PARQUET_BATCH_ROWS = 4096
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


@dataclass(frozen=True)
class BenchTokenizerConfig:
    tokenizer_json_path: Path
    benchmark_parquet_path: Path
    delta_sample_limit: int
    num_workers: int
    parquet_batch_rows: int
    enable_tqdm: bool
    enable_debug_log: bool
    tqdm_mininterval_sec: float


def build_default_config() -> BenchTokenizerConfig:
    return BenchTokenizerConfig(
        tokenizer_json_path=TOKENIZER_JSON_PATH,
        benchmark_parquet_path=BENCHMARK_PARQUET_PATH,
        delta_sample_limit=DELTA_SAMPLE_LIMIT,
        num_workers=NUM_WORKERS,
        parquet_batch_rows=PARQUET_BATCH_ROWS,
        enable_tqdm=ENABLE_TQDM,
        enable_debug_log=ENABLE_DEBUG_LOG,
        tqdm_mininterval_sec=TQDM_MININTERVAL_SEC,
    )


def run_bench_tokenizer(
    config: BenchTokenizerConfig | None = None,
) -> common.BenchmarkSummary:
    config = config or build_default_config()
    _validate_config(config)
    tokenizer = common.load_tokenizer(config.tokenizer_json_path)
    benchmark_out_dir = config.tokenizer_json_path.parent / "benchmark"
    build_meta_path = config.tokenizer_json_path.parent / "meta.json"

    summary_payload = _benchmark_rows(tokenizer, config)
    samples = summary_payload["samples"][: config.delta_sample_limit]

    common.ensure_dir(benchmark_out_dir)
    common.write_json(
        benchmark_out_dir / "benchmark_summary.json",
        {
            "tokenizer_json_path": str(config.tokenizer_json_path),
            "tokenizer_dir": str(config.tokenizer_json_path.parent),
            "build_meta_path": str(build_meta_path),
            "benchmark_parquet_path": str(config.benchmark_parquet_path),
            "compared_row_count": summary_payload["compared_row_count"],
            "skipped_row_count": summary_payload["skipped_row_count"],
            "more_compressive_row_count": summary_payload["more_compressive_row_count"],
            "same_token_count_row_count": summary_payload["same_token_count_row_count"],
            "less_compressive_row_count": summary_payload["less_compressive_row_count"],
            "mean_delta": summary_payload["mean_delta"],
            "mean_abs_delta": summary_payload["mean_abs_delta"],
            "mean_delta_ratio": summary_payload["mean_delta_ratio"],
            "max_abs_delta": summary_payload["max_abs_delta"],
        },
    )
    common.write_jsonl(
        benchmark_out_dir / "benchmark_delta_samples.jsonl",
        samples,
    )

    result = common.BenchmarkSummary(
        benchmark_out_dir=benchmark_out_dir,
        compared_row_count=summary_payload["compared_row_count"],
        skipped_row_count=summary_payload["skipped_row_count"],
        more_compressive_row_count=summary_payload["more_compressive_row_count"],
        same_token_count_row_count=summary_payload["same_token_count_row_count"],
        less_compressive_row_count=summary_payload["less_compressive_row_count"],
        mean_delta=summary_payload["mean_delta"],
        mean_abs_delta=summary_payload["mean_abs_delta"],
        mean_delta_ratio=summary_payload["mean_delta_ratio"],
        max_abs_delta=summary_payload["max_abs_delta"],
    )
    return result


def _validate_config(config: BenchTokenizerConfig) -> None:
    if not config.tokenizer_json_path.exists():
        raise ValueError(f"tokenizer file not found: {config.tokenizer_json_path}")
    if not config.benchmark_parquet_path.exists():
        raise ValueError(f"benchmark parquet not found: {config.benchmark_parquet_path}")
    if config.delta_sample_limit < 1:
        raise ValueError("delta_sample_limit must be >= 1")
    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1")


def _benchmark_rows(tokenizer, config: BenchTokenizerConfig) -> dict[str, Any]:
    parquet = pq.ParquetFile(config.benchmark_parquet_path)
    if "text" not in parquet.schema.names or "token_count" not in parquet.schema.names:
        raise ValueError(
            f"benchmark parquet missing required columns: {config.benchmark_parquet_path}"
        )

    tasks: list[list[tuple[int, str, int]]] = []
    row_index = 0
    skipped_row_count = 0
    for batch in parquet.iter_batches(
        columns=["text", "token_count"], batch_size=config.parquet_batch_rows
    ):
        payload: list[tuple[int, str, int]] = []
        for text, token_count in zip(
            batch.column(0).to_pylist(),
            batch.column(1).to_pylist(),
            strict=False,
        ):
            current_index = row_index
            row_index += 1
            stripped = common.strip_text(text)
            if stripped is None or not isinstance(token_count, int):
                skipped_row_count += 1
                continue
            payload.append((current_index, text, int(token_count)))
        if payload:
            tasks.append(payload)

    if not tasks:
        raise ValueError("benchmark has zero comparable rows")

    if config.num_workers <= 1 or len(tasks) <= 1:
        results = [
            _benchmark_batch(task, str(config.tokenizer_json_path)) for task in tasks
        ]
    else:
        mp_context = mp.get_context("spawn")
        results = []
        with ProcessPoolExecutor(
            max_workers=min(config.num_workers, len(tasks)),
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(_benchmark_batch, task, str(config.tokenizer_json_path))
                for task in tasks
            ]
            for future in common.progress(
                as_completed(futures),
                total=len(futures),
                desc="bench_tokenizer/compare",
                enabled=config.enable_tqdm,
                mininterval=config.tqdm_mininterval_sec,
            ):
                results.append(future.result())

    compared_row_count = 0
    more_compressive_row_count = 0
    same_token_count_row_count = 0
    less_compressive_row_count = 0
    sum_delta = 0
    sum_abs_delta = 0
    sum_delta_ratio = 0.0
    delta_ratio_count = 0
    max_abs_delta = 0
    samples: list[dict[str, Any]] = []
    for result in results:
        compared_row_count += result["compared"]
        more_compressive_row_count += result["more_compressive"]
        same_token_count_row_count += result["same_token_count"]
        less_compressive_row_count += result["less_compressive"]
        sum_delta += result["sum_delta"]
        sum_abs_delta += result["sum_abs_delta"]
        sum_delta_ratio += result["sum_delta_ratio"]
        delta_ratio_count += result["delta_ratio_count"]
        max_abs_delta = max(max_abs_delta, result["max_abs_delta"])
        samples.extend(result["samples"])

    mean_delta = (sum_delta / compared_row_count) if compared_row_count else 0.0
    mean_abs_delta = (sum_abs_delta / compared_row_count) if compared_row_count else 0.0
    mean_delta_ratio = (sum_delta_ratio / delta_ratio_count) if delta_ratio_count else 0.0
    samples.sort(key=lambda row: (-abs(row["delta"]), row["row_index"]))
    return {
        "compared_row_count": compared_row_count,
        "skipped_row_count": skipped_row_count,
        "more_compressive_row_count": more_compressive_row_count,
        "same_token_count_row_count": same_token_count_row_count,
        "less_compressive_row_count": less_compressive_row_count,
        "mean_delta": mean_delta,
        "mean_abs_delta": mean_abs_delta,
        "mean_delta_ratio": mean_delta_ratio,
        "max_abs_delta": max_abs_delta,
        "samples": samples,
    }


def _benchmark_batch(
    rows: list[tuple[int, str, int]],
    tokenizer_json_path: str,
) -> dict[str, Any]:
    tokenizer = common.load_tokenizer(Path(tokenizer_json_path))
    more_compressive = 0
    same_token_count = 0
    less_compressive = 0
    sum_delta = 0
    sum_abs_delta = 0
    sum_delta_ratio = 0.0
    delta_ratio_count = 0
    max_abs_delta = 0
    samples: list[dict[str, Any]] = []
    for row_index, text, reference_token_count in rows:
        calculated_token_count = len(tokenizer.encode(text).ids)
        delta = calculated_token_count - reference_token_count
        abs_delta = abs(delta)
        sum_delta += delta
        sum_abs_delta += abs_delta
        max_abs_delta = max(max_abs_delta, abs_delta)
        if reference_token_count > 0:
            delta_ratio = delta / reference_token_count
            sum_delta_ratio += delta_ratio
            delta_ratio_count += 1
        else:
            delta_ratio = 0.0
        if delta == 0:
            same_token_count += 1
            continue
        if delta < 0:
            more_compressive += 1
            compression_label = "more_compressive"
        else:
            less_compressive += 1
            compression_label = "less_compressive"
        samples.append(
            {
                "row_index": row_index,
                "text_preview": common.truncate_text(text),
                "reference_token_count": reference_token_count,
                "calculated_token_count": calculated_token_count,
                "delta": delta,
                "delta_ratio": delta_ratio,
                "compression_label": compression_label,
            }
        )
    return {
        "compared": len(rows),
        "more_compressive": more_compressive,
        "same_token_count": same_token_count,
        "less_compressive": less_compressive,
        "sum_delta": sum_delta,
        "sum_abs_delta": sum_abs_delta,
        "sum_delta_ratio": sum_delta_ratio,
        "delta_ratio_count": delta_ratio_count,
        "max_abs_delta": max_abs_delta,
        "samples": samples,
    }


def main() -> None:
    run_bench_tokenizer(build_default_config())


if __name__ == "__main__":
    main()
