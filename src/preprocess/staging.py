from __future__ import annotations

import shutil
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
from .dataset_specs import DATASET_SPECS, process_file, resolve_input_files
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


def stage_dataset(config: StageConfig) -> None:
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
    aggregate = Counter(input_files=0, output_rows=0, shard_files=0)
    shard_rows: list[dict[str, Any]] = []
    shard_index = 1
    progress = progress_bar(total=len(file_items), desc=f"stage:{spec.dataset_id}:{config.target_split}")
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(process_file, spec.dataset_id, split, path): (split, path)
            for split, path in file_items
        }
        for future in as_completed(futures):
            split, path = futures[future]
            rows, events, stats = future.result()
            progress.update(1)
            recorder.extend(events)
            aggregate.update(stats)
            shard_rows.extend(rows)
            if ENABLE_DEBUG_LOG:
                log(
                    f"[stage:{spec.dataset_id}:{config.target_split}] file={path.name} rows={len(rows)} total_rows={aggregate['output_rows']}"
                )
            while len(shard_rows) >= config.shard_row_limit:
                part = shard_rows[: config.shard_row_limit]
                _flush_shard(dataset_root, part, shard_index)
                aggregate["shard_files"] += 1
                shard_index += 1
                shard_rows = shard_rows[config.shard_row_limit :]
    progress.close()
    if shard_rows:
        _flush_shard(dataset_root, shard_rows, shard_index)
        aggregate["shard_files"] += 1
    recorder.write_jsonl(meta_root / f"quality_events_{config.target_split}.jsonl")
    quality_summary = recorder.to_summary()
    write_json(meta_root / f"quality_summary_{config.target_split}.json", quality_summary)
    stage_manifest = {
        "dataset_id": spec.dataset_id,
        "source": spec.source,
        "split": config.target_split,
        "generated_at": now_utc_iso(),
        "input_files": len(file_items),
        "output_rows": aggregate["output_rows"],
        "shard_files": aggregate["shard_files"],
        "quality_event_count": len(recorder.events),
    }
    write_json(meta_root / f"stage_manifest_{config.target_split}.json", stage_manifest)
    log(
        f"[stage:{spec.dataset_id}:{config.target_split}] completed rows={aggregate['output_rows']} shards={aggregate['shard_files']} quality_events={len(recorder.events)}"
    )


def _flush_shard(dataset_root: Path, rows: list[dict[str, Any]], shard_index: int) -> None:
    table = rows_to_table(rows)
    shard_path = dataset_root / f"part-{shard_index:06d}.parquet"
    ensure_parent_dir(shard_path)
    pq.write_table(table, shard_path)
    log(f"[stage-shard] wrote {shard_path} rows={table.num_rows}")
