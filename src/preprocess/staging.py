from __future__ import annotations

import shutil
from dataclasses import asdict
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .common import (
    ENABLE_DEBUG_LOG,
    ensure_parent_dir,
    log,
    now_utc_iso,
    progress_bar,
    rows_to_table,
    stable_sorted_paths,
    write_json,
)
from .dataset_specs import DATASET_SPECS, get_split_plan_stats, process_file, resolve_input_files
from .quality import QualityRecorder


# =========================
# User configuration block
# =========================
DEFAULT_STAGING_ROOT = Path("data/korean_processed/_staging")
DEFAULT_SHARD_ROW_LIMIT = 5000
DEFAULT_WORKERS = 7
OVERWRITE_OUTPUT = False


@dataclass(frozen=True)
class StageConfig:
    output_root: Path
    target_split: str
    workers: int
    shard_row_limit: int
    overwrite_output: bool
    limit_files: int | None
    dataset_id: str


@dataclass(frozen=True)
class StageResult:
    dataset_id: str
    split: str
    source: str
    input_files: int
    raw_candidate_count: int
    valid_candidate_count: int
    output_rows: int
    shard_files: int
    quality_event_count: int
    manifest_path: str
    quality_summary_path: str
    quality_events_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stage_dataset(config: StageConfig) -> StageResult:
    spec = DATASET_SPECS[config.dataset_id]
    file_items = resolve_input_files(spec, config.target_split)
    if config.limit_files is not None:
        file_items = file_items[: config.limit_files]
    dataset_root = config.output_root / spec.dataset_id / config.target_split
    meta_root = config.output_root / spec.dataset_id / "_meta"
    if config.overwrite_output and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)
    log(
        f"[stage:{spec.dataset_id}:{config.target_split}] files={len(file_items)} workers={config.workers} shard_row_limit={config.shard_row_limit}"
    )
    recorder = QualityRecorder()
    aggregate = Counter(
        input_files=0,
        raw_candidate_count=0,
        valid_candidate_count=0,
        output_rows=0,
        shard_files=0,
    )
    shard_rows: list[dict[str, Any]] = []
    shard_index = 1
    progress = progress_bar(total=len(file_items), desc=f"stage:{spec.dataset_id}:{config.target_split}")
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                process_file,
                spec.dataset_id,
                item.split,
                item.path,
                item.selected_record_ids,
            ): item
            for item in file_items
        }
        for future in as_completed(futures): # 완료된 작업이 있을 때마다 결과를 처리
            item = futures[future]
            rows, events, stats = future.result()
            progress.update(1)
            recorder.extend(events)
            aggregate.update(stats)
            shard_rows.extend(rows)
            if ENABLE_DEBUG_LOG:
                log(
                    f"[stage:{spec.dataset_id}:{config.target_split}] file={item.path.name} rows={len(rows)} total_rows={aggregate['output_rows']}"
                )
            while len(shard_rows) >= config.shard_row_limit:
                part = shard_rows[: config.shard_row_limit]
                _flush_shard(dataset_root, part, shard_index)
                aggregate["shard_files"] += 1
                shard_index += 1
                shard_rows = shard_rows[config.shard_row_limit :]
    progress.close()
    if shard_rows: #마지막에 1000개 미만으로 남은 row를 위한 샤딩
        _flush_shard(dataset_root, shard_rows, shard_index)
        aggregate["shard_files"] += 1
    recorder.write_jsonl(meta_root / f"quality_events_{config.target_split}.jsonl")
    quality_events_path = meta_root / f"quality_events_{config.target_split}.jsonl"
    quality_summary = recorder.to_summary()
    quality_summary_path = meta_root / f"quality_summary_{config.target_split}.json"
    write_json(quality_summary_path, quality_summary)
    split_plan_stats = get_split_plan_stats(spec.dataset_id)
    stage_manifest = {
        "dataset_id": spec.dataset_id,
        "source": spec.source,
        "split": config.target_split,
        "generated_at": now_utc_iso(),
        "input_files": len(file_items),
        "raw_candidate_count": aggregate["raw_candidate_count"],
        "valid_candidate_count": aggregate["valid_candidate_count"],
        "output_rows": aggregate["output_rows"],
        "shard_files": aggregate["shard_files"],
        "quality_event_count": len(recorder.events),
        "split_plan_stats": split_plan_stats,
    }
    manifest_path = meta_root / f"stage_manifest_{config.target_split}.json"
    write_json(manifest_path, stage_manifest)
    log(
        f"[stage:{spec.dataset_id}:{config.target_split}] completed rows={aggregate['output_rows']} shards={aggregate['shard_files']} quality_events={len(recorder.events)}"
    )
    return StageResult(
        dataset_id=spec.dataset_id,
        split=config.target_split,
        source=spec.source,
        input_files=len(file_items),
        raw_candidate_count=aggregate["raw_candidate_count"],
        valid_candidate_count=aggregate["valid_candidate_count"],
        output_rows=aggregate["output_rows"],
        shard_files=aggregate["shard_files"],
        quality_event_count=len(recorder.events),
        manifest_path=str(manifest_path),
        quality_summary_path=str(quality_summary_path),
        quality_events_path=str(quality_events_path),
    )


def _flush_shard(dataset_root: Path, rows: list[dict[str, Any]], shard_index: int) -> None:
    table = rows_to_table(rows)
    shard_path = dataset_root / f"part-{shard_index:06d}.parquet"
    ensure_parent_dir(shard_path)
    pq.write_table(table, shard_path)
    log(f"[stage-shard] wrote {shard_path} rows={table.num_rows}")
