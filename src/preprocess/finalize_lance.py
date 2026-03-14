from __future__ import annotations

from collections import Counter, defaultdict
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lancedb
import pyarrow.parquet as pq

import src.preprocess.common as common
from src.preprocess.common import (
    SCHEMA_VERSION,
    TOKEN_COUNT_VERSION,
    ensure_parent_dir,
    log,
    now_utc_iso,
    progress_bar,
    rows_to_table,
    validate_row,
    write_json,
)
from src.preprocess.quality import QualityRecorder


# =========================
# User configuration block
# =========================
STAGING_ROOT = Path("data/korean_processed/_staging")
OUTPUT_ROOT = Path("data/korean_processed")
TARGET_DATASETS = (
    "009",
    "010",
    "011",
    "019",
    "020",
    "021",
    "023",
    "030",
    "nikl_newspaper_2020",
    "nikl_spoken",
    "nikl_written",
    "045",
    "046",
    "141",
    "gsm8k",
    "novel24",
)
TARGET_SPLITS = ("train", "val")
CHUNK_TARGET_BYTES = 1_073_741_824
OVERWRITE_OUTPUT = False
CREATE_SCALAR_INDICES = True
INDEX_COLUMNS = ("source", "split", "token_count")
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


def main() -> None:
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    chunks_root = OUTPUT_ROOT / "chunks"
    manifests_root = OUTPUT_ROOT / "_manifests"
    indices_root = OUTPUT_ROOT / "_indices"
    if OVERWRITE_OUTPUT:
        import shutil

        for path in (chunks_root, manifests_root, indices_root):
            if path.exists():
                shutil.rmtree(path)
    chunks_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    indices_root.mkdir(parents=True, exist_ok=True)

    shard_paths = _collect_staging_shards()
    log(f"[finalize] staging_shards={len(shard_paths)}")
    conn = lancedb.connect(chunks_root)
    recorder = QualityRecorder()
    manifest_stats = {
        "rows_total": 0,
        "content_rows": 0,
        "messages_rows": 0,
        "source_rows": Counter(),
        "split_rows": Counter(),
        "chunk_rows": {},
        "chunk_bytes": {},
    }
    chunk_rows: list[dict[str, Any]] = []
    chunk_bytes = 0
    chunk_index = 1
    progress = progress_bar(total=len(shard_paths), desc="finalize")
    for shard_path in shard_paths:
        table = pq.read_table(shard_path)
        for row in table.to_pylist():
            try:
                validate_row(row)
            except Exception as exc:  # noqa: BLE001
                recorder.add(
                    dataset=str(row.get("source") or "unknown"),
                    split=row.get("split"),
                    file_path=str(shard_path),
                    record_id=None,
                    reason_code="finalize_validation_error",
                    reason_detail=f"{type(exc).__name__}: {exc}",
                    sample_text=None,
                    severity="error",
                )
                continue
            chunk_rows.append(row)
            chunk_bytes = rows_to_table(chunk_rows).nbytes
            if chunk_bytes >= CHUNK_TARGET_BYTES:
                _flush_chunk(
                    conn=conn,
                    chunk_name=f"chunk-{chunk_index:06d}",
                    rows=chunk_rows,
                    manifest_stats=manifest_stats,
                )
                chunk_index += 1
                chunk_rows = []
                chunk_bytes = 0
        progress.update(1)
        log(f"[finalize] consumed shard={shard_path}")
    progress.close()
    if chunk_rows:
        _flush_chunk(
            conn=conn,
            chunk_name=f"chunk-{chunk_index:06d}",
            rows=chunk_rows,
            manifest_stats=manifest_stats,
        )

    index_manifest: dict[str, Any] = {"chunks": {}}
    if CREATE_SCALAR_INDICES:
        for table_name in sorted(conn.table_names()):
            table = conn.open_table(table_name)
            index_manifest["chunks"][table_name] = []
            for column in INDEX_COLUMNS:
                table.create_scalar_index(column, replace=True)
                index_manifest["chunks"][table_name].append(column)
                log(f"[finalize] created scalar index table={table_name} column={column}")
    write_json(indices_root / "index_manifest.json", index_manifest)

    run_id = now_utc_iso().replace(":", "").replace("-", "")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "token_count_version": TOKEN_COUNT_VERSION,
        "generated_at": now_utc_iso(),
        "chunks": [
            {
                "name": chunk_name,
                "rows": manifest_stats["chunk_rows"][chunk_name],
                "bytes": manifest_stats["chunk_bytes"][chunk_name],
            }
            for chunk_name in sorted(manifest_stats["chunk_rows"])
        ],
        "source_rows": dict(sorted(manifest_stats["source_rows"].items())),
        "split_rows": dict(sorted(manifest_stats["split_rows"].items())),
        "rows_total": manifest_stats["rows_total"],
        "content_rows": manifest_stats["content_rows"],
        "messages_rows": manifest_stats["messages_rows"],
    }
    write_json(manifests_root / f"manifest-{SCHEMA_VERSION}.json", manifest)
    write_json(manifests_root / f"quality_report-{run_id}.json", recorder.to_summary())
    recorder.write_jsonl(manifests_root / f"quality_events-{run_id}.jsonl")
    log(
        f"[finalize] completed rows_total={manifest_stats['rows_total']} chunks={len(manifest_stats['chunk_rows'])} quality_events={len(recorder.events)}"
    )


def _collect_staging_shards() -> list[Path]:
    shards: list[Path] = []
    for dataset_id in TARGET_DATASETS:
        for split in TARGET_SPLITS:
            split_root = STAGING_ROOT / dataset_id / split
            if not split_root.exists():
                continue
            shards.extend(sorted(split_root.glob("part-*.parquet")))
    return sorted(shards, key=lambda path: str(path))


def _flush_chunk(
    *,
    conn: lancedb.db.DBConnection,
    chunk_name: str,
    rows: list[dict[str, Any]],
    manifest_stats: dict[str, Any],
) -> None:
    table = rows_to_table(rows)
    conn.create_table(chunk_name, data=table, mode="overwrite")
    manifest_stats["chunk_rows"][chunk_name] = table.num_rows
    manifest_stats["chunk_bytes"][chunk_name] = table.nbytes
    manifest_stats["rows_total"] += table.num_rows
    for row in rows:
        manifest_stats["source_rows"][row["source"]] += 1
        manifest_stats["split_rows"][row["split"]] += 1
        if row["content"] is not None:
            manifest_stats["content_rows"] += 1
        if row["messages"] is not None:
            manifest_stats["messages_rows"] += 1
    log(f"[finalize-chunk] wrote {chunk_name} rows={table.num_rows} bytes={table.nbytes}")


if __name__ == "__main__":
    main()
