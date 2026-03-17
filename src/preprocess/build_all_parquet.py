from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.preprocess.common as common
from src.preprocess.common import ensure_parent_dir, log, write_json
from src.preprocess.runner import run_all_stage_entrypoint


# =========================
# User configuration block
# =========================
OUTPUT_ROOT = Path("data/korean_processed/_staging")
TARGET_SPLITS = ("train", "val")
WORKERS = 7
SHARD_ROW_LIMIT = 5000
OVERWRITE_OUTPUT = False
LIMIT_FILES = None
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


def main() -> None:
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    summary = run_all_stage_entrypoint(
        output_root=OUTPUT_ROOT,
        target_splits=TARGET_SPLITS,
        workers=WORKERS,
        shard_row_limit=SHARD_ROW_LIMIT,
        overwrite_output=OVERWRITE_OUTPUT,
        limit_files=LIMIT_FILES,
    )
    summary_path = OUTPUT_ROOT / "_meta" / "all_datasets_run_manifest.json"
    ensure_parent_dir(summary_path)
    write_json(summary_path, summary.to_dict())
    log(f"[entry:all-datasets] wrote summary={summary_path}")


if __name__ == "__main__":
    main()
