from __future__ import annotations

from pathlib import Path

from .common import log
from .staging import StageConfig, stage_dataset


def run_stage_entrypoint(
    *,
    dataset_id: str,
    output_root: Path,
    target_splits: tuple[str, ...],
    workers: int,
    shard_row_limit: int,
    overwrite_output: bool,
    limit_files: int | None,
) -> None:
    log(f"[entry:{dataset_id}] target_splits={target_splits}")
    for split in target_splits:
        stage_dataset(
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
