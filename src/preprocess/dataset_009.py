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
OVERWRITE_OUTPUT = False #초기 실행에서 False, 재실행에서 True. True로 놓으면 기존 데이터셋 삭제 이후 새롭게 생성
LIMIT_FILES = None #한 번에 처리할 파일 개수 제한. 예를 들어 gsm8k 같은 경우 어차피 파일 하나라 의미 없음.
ENABLE_TQDM = True
ENABLE_DEBUG_LOG = True
TQDM_MININTERVAL_SEC = 1.0


if __name__ == "__main__":
    common.ENABLE_TQDM = ENABLE_TQDM
    common.ENABLE_DEBUG_LOG = ENABLE_DEBUG_LOG
    common.TQDM_MININTERVAL_SEC = TQDM_MININTERVAL_SEC
    run_stage_entrypoint(
        dataset_id="009",
        output_root=OUTPUT_ROOT,
        target_splits=TARGET_SPLITS,
        workers=WORKERS,
        shard_row_limit=SHARD_ROW_LIMIT,
        overwrite_output=OVERWRITE_OUTPUT,
        limit_files=LIMIT_FILES,
    )
