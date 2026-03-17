from __future__ import annotations

import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.preprocess.common as common
from src.preprocess.common import log, progress_bar, stable_sorted_paths


# =========================
# User configuration block
# =========================
TARGET_ROOT = Path("data/korean_processed/_staging")
PARQUET_GLOB = "*/*/part-*.parquet"
WORKERS = 7
OVERWRITE_EXISTING_DATA_USAGE = True
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


def collect_target_paths() -> list[Path]:
    return stable_sorted_paths(TARGET_ROOT.glob(PARQUET_GLOB))


def build_data_usage_column(messages: list[Any]) -> tuple[pa.Array, Counter[str]]:
    values: list[str] = []
    stats = Counter(total_rows=0, pt_rows=0, sft_rows=0, empty_messages_rows=0)
    for item in messages:
        stats["total_rows"] += 1
        if item is None:
            values.append("PT")
            stats["pt_rows"] += 1
            continue
        if isinstance(item, list) and len(item) == 0:
            values.append("PT")
            stats["pt_rows"] += 1
            stats["empty_messages_rows"] += 1
            continue
        values.append("SFT")
        stats["sft_rows"] += 1
    return pa.array(values, type=pa.string()), stats


def transform_table(table: pa.Table) -> tuple[pa.Table, Counter[str]]:
    names = list(table.column_names)
    had_existing_data_usage = "data_usage" in names
    if had_existing_data_usage:
        index = names.index("data_usage")
        table = table.remove_column(index)
        names.pop(index)
    if "messages" not in names:
        raise ValueError("messages column is missing")
    if "source" not in names:
        raise ValueError("source column is missing")
    data_usage_column, stats = build_data_usage_column(table.column("messages").to_pylist())
    source_index = names.index("source")
    columns = [
        *[table.column(name) for name in names[: source_index + 1]],
        data_usage_column,
        *[table.column(name) for name in names[source_index + 1 :]],
    ]
    new_names = [
        *names[: source_index + 1],
        "data_usage",
        *names[source_index + 1 :],
    ]
    transformed = pa.Table.from_arrays(columns, names=new_names, metadata=table.schema.metadata)
    stats["had_existing_data_usage"] = int(had_existing_data_usage)
    return transformed, stats


def rewrite_parquet_file(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    transformed, stats = transform_table(table)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(transformed, temp_path)
    os.replace(temp_path, path)
    return {
        "path": str(path),
        "rows": stats["total_rows"],
        "pt_rows": stats["pt_rows"],
        "sft_rows": stats["sft_rows"],
        "empty_messages_rows": stats["empty_messages_rows"],
        "had_existing_data_usage": bool(stats["had_existing_data_usage"]),
        "status": "ok",
    }


def process_one_file(path: Path) -> dict[str, Any]:
    try:
        return rewrite_parquet_file(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(path),
            "rows": 0,
            "pt_rows": 0,
            "sft_rows": 0,
            "empty_messages_rows": 0,
            "had_existing_data_usage": False,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC

    target_paths = collect_target_paths()
    log(f"[add_data_usage] files={len(target_paths)} workers={WORKERS}")
    aggregate = Counter(
        files_total=len(target_paths),
        files_ok=0,
        files_error=0,
        rows_total=0,
        pt_rows=0,
        sft_rows=0,
        empty_messages_rows=0,
        overwritten_existing_data_usage=0,
    )
    progress = progress_bar(total=len(target_paths), desc="add_data_usage")
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_one_file, path): path for path in target_paths}
        for future in as_completed(futures):
            result = future.result()
            progress.update(1)
            if result["status"] == "ok":
                aggregate["files_ok"] += 1
                aggregate["rows_total"] += result["rows"]
                aggregate["pt_rows"] += result["pt_rows"]
                aggregate["sft_rows"] += result["sft_rows"]
                aggregate["empty_messages_rows"] += result["empty_messages_rows"]
                if result["had_existing_data_usage"] and OVERWRITE_EXISTING_DATA_USAGE:
                    aggregate["overwritten_existing_data_usage"] += 1
                log(
                    "[add_data_usage:file] "
                    f"path={result['path']} rows={result['rows']} pt={result['pt_rows']} "
                    f"sft={result['sft_rows']} empty_messages={result['empty_messages_rows']}"
                )
                if result["empty_messages_rows"] > 0:
                    log(
                        "[add_data_usage:warning] "
                        f"path={result['path']} empty_messages_rows={result['empty_messages_rows']} treated_as=PT"
                    )
            else:
                aggregate["files_error"] += 1
                log(f"[add_data_usage:error] path={result['path']} error={result['error']}")
    progress.close()
    log(
        "[add_data_usage] completed "
        f"files_ok={aggregate['files_ok']} files_error={aggregate['files_error']} "
        f"rows_total={aggregate['rows_total']} pt_rows={aggregate['pt_rows']} "
        f"sft_rows={aggregate['sft_rows']} empty_messages_rows={aggregate['empty_messages_rows']}"
    )


if __name__ == "__main__":
    main()
