from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.preprocess.common as common
from src.preprocess.runner import run_stage_entrypoint


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


if __name__ == "__main__":
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    run_stage_entrypoint(
        dataset_id="046",
        output_root=OUTPUT_ROOT,
        target_splits=TARGET_SPLITS,
        workers=WORKERS,
        shard_row_limit=SHARD_ROW_LIMIT,
        overwrite_output=OVERWRITE_OUTPUT,
        limit_files=LIMIT_FILES,
    )
