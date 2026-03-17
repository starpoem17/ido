from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .common import log, now_utc_iso, progress_bar
from .dataset_specs import DATASET_SPECS
from .staging import StageConfig, StageResult, stage_dataset


@dataclass(frozen=True)
class DatasetRunResult:
    dataset_id: str
    status: str
    started_at: str
    finished_at: str
    duration_sec: float
    split_results: dict[str, dict[str, Any]]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AllDatasetsRunSummary:
    started_at: str
    finished_at: str
    duration_sec: float
    dataset_order: list[str]
    total_datasets: int
    succeeded_datasets: int
    failed_datasets: int
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_stage_entrypoint(
    *,
    dataset_id: str,
    output_root: Path,
    target_splits: tuple[str, ...],
    workers: int,
    shard_row_limit: int,
    overwrite_output: bool,
    limit_files: int | None,
    continue_on_failure: bool = False,
) -> DatasetRunResult:
    log(f"[entry:{dataset_id}] target_splits={target_splits}")
    started_at = now_utc_iso()
    started_perf = perf_counter()
    split_results: dict[str, dict[str, Any]] = {}
    for split in target_splits:
        try:
            result = stage_dataset(
                StageConfig(
                    output_root=output_root,
                    target_split=split,
                    workers=workers,
                    shard_row_limit=shard_row_limit,
                    overwrite_output=overwrite_output,
                    limit_files=limit_files,
                    dataset_id=dataset_id,
                )
            )
            split_results[split] = result.to_dict()
        except Exception as exc:
            if not continue_on_failure:
                raise
            finished_at = now_utc_iso()
            duration_sec = round(perf_counter() - started_perf, 3)
            log(f"[entry:{dataset_id}] failed error={type(exc).__name__}: {exc}")
            return DatasetRunResult(
                dataset_id=dataset_id,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=duration_sec,
                split_results=split_results,
                error=f"{type(exc).__name__}: {exc}",
            )
    finished_at = now_utc_iso()
    duration_sec = round(perf_counter() - started_perf, 3)
    log(f"[entry:{dataset_id}] completed splits={target_splits}")
    return DatasetRunResult(
        dataset_id=dataset_id,
        status="success",
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        split_results=split_results,
        error=None,
    )


def run_all_stage_entrypoint(
    *,
    output_root: Path,
    target_splits: tuple[str, ...],
    workers: int,
    shard_row_limit: int,
    overwrite_output: bool,
    limit_files: int | None,
    dataset_ids: tuple[str, ...] | None = None,
) -> AllDatasetsRunSummary:
    ordered_dataset_ids = list(dataset_ids or tuple(DATASET_SPECS.keys()))
    started_at = now_utc_iso()
    started_perf = perf_counter()
    results: list[dict[str, Any]] = []
    progress = progress_bar(total=len(ordered_dataset_ids), desc="stage:all-datasets")
    for dataset_id in ordered_dataset_ids:
        result = run_stage_entrypoint(
            dataset_id=dataset_id,
            output_root=output_root,
            target_splits=target_splits,
            workers=workers,
            shard_row_limit=shard_row_limit,
            overwrite_output=overwrite_output,
            limit_files=limit_files,
            continue_on_failure=True,
        )
        results.append(result.to_dict())
        progress.update(1)
    progress.close()
    succeeded_datasets = sum(1 for result in results if result["status"] == "success")
    failed_datasets = len(results) - succeeded_datasets
    finished_at = now_utc_iso()
    duration_sec = round(perf_counter() - started_perf, 3)
    log(
        f"[entry:all-datasets] completed total={len(results)} succeeded={succeeded_datasets} failed={failed_datasets}"
    )
    return AllDatasetsRunSummary(
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        dataset_order=ordered_dataset_ids,
        total_datasets=len(ordered_dataset_ids),
        succeeded_datasets=succeeded_datasets,
        failed_datasets=failed_datasets,
        results=results,
    )
