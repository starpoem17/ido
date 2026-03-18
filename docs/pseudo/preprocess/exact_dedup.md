# exact dedup 단계 스도코드

## 목적
이 문서는 normalized parquet 전체를 읽어 `content` 완전 일치 중복을 제거하는 실행 코드의 자연어 의사코드다.
중복 제거는 dataset별이 아니라 모든 source를 섞은 전체 집합 기준으로 수행한다.

## 사용자 설정값
1. `INPUT_ROOT = "data/korean_processed/_staging"`
2. `OUTPUT_ROOT = "data/korean_processed/exact_dedup"`
3. `TARGET_SHARD_BYTES = 1_073_741_824`
4. `DUCKDB_THREADS`, `DUCKDB_PRESERVE_INSERTION_ORDER`, `DUCKDB_MEMORY_LIMIT`, `DUCKDB_ARROW_LARGE_BUFFER_SIZE`
5. `FETCH_RECORD_BATCH_ROWS`
6. `OVERWRITE_OUTPUT`
7. `ENABLE_TQDM`, `ENABLE_DEBUG_LOG`, `TQDM_MININTERVAL_SEC`

## 입력과 출력 경로
1. 입력 parquet는 `data/korean_processed/_staging/*/*/part-*.parquet` 패턴으로 찾는다.
2. 유지된 결과 parquet는 `data/korean_processed/exact_dedup/part-*.parquet`로 저장한다.
3. 삭제 로그는 `data/korean_processed/exact_dedup/_logs/deleted_rows_exact.jsonl`에 저장한다.
4. 삭제 요약은 `data/korean_processed/exact_dedup/_meta/deleted_rows_exact_summary.json`에 저장한다.
5. 실행 manifest는 `data/korean_processed/exact_dedup/_meta/exact_dedup_manifest.json`에 저장한다.
6. DuckDB 임시 DB와 spill 디렉터리는 `data/korean_processed/exact_dedup/_tmp/` 아래에 둔다.

## 전체 실행 절차
1. 시작 시 입력 root, 출력 root, target shard bytes, DuckDB 설정값을 stdout 로그로 출력한다.
2. 입력 shard 목록을 모두 수집하고, shard 수와 dataset 수를 집계한다.
3. 출력 root를 준비한다. 기존 결과가 있으면 `OVERWRITE_OUTPUT` 값에 따라 즉시 실패하거나 전체를 비우고 다시 만든다.
4. DuckDB 연결을 만들고 `threads`, `temp_directory`, `preserve_insertion_order`, `memory_limit`, `arrow_large_buffer_size`를 코드 상단 상수값으로 설정한다.
5. `read_parquet(..., filename=true, file_row_number=true)`로 전체 parquet를 읽어 `staged_rows` temp table을 만든다.
6. `record_locator = filename + "#" + file_row_number`를 만들고, 대표 선택에 필요한 `usage_rank`, `content_length`도 함께 계산한다.
7. `content is not null` row만 대상으로 `ranked_non_null` temp table을 만든다. 여기서 `ROW_NUMBER() OVER (PARTITION BY content ORDER BY usage_rank DESC, content_length DESC, filename ASC, file_row_number ASC)`를 사용해 cluster 내부 순위를 정한다.
8. `content is null` row는 passthrough로 유지하고, `content_rank = 1`인 row를 retained, `content_rank > 1`인 row를 deleted로 나누어 view를 만든다.
9. 삭제 대상 row 수를 먼저 집계한 뒤 `exact_dedup/deleted_log` tqdm을 열고 JSONL 로그를 순차 기록한다.
10. 유지 대상 row는 `exact_dedup/retained` tqdm을 열고 Arrow record batch 단위로 읽는다.
11. shard는 메모리 기준 `approx_nbytes`가 `TARGET_SHARD_BYTES`를 넘을 때까지 현재 row를 포함해 누적한 뒤 flush한다. row 중간 분할은 허용하지 않는다.
12. flush가 일어날 때마다 shard index, row 수, 누적 retained 수를 stdout 로그로 남긴다.
13. 마지막으로 source별/data_usage별 before-after-deleted 통계, exact cluster 수, 대표 선별 사유 집계를 계산한다.
14. summary JSON과 manifest JSON을 저장하고 종료 로그를 남긴다.

## 대표 row 선택 규칙
1. `data_usage` 우선순위는 `REASONING > SFT > PT`다.
2. `data_usage`가 같으면 `len(content)`가 더 긴 row를 남긴다.
3. 둘 다 같으면 `filename ASC`, `file_row_number ASC` 기준으로 앞선 row를 남긴다.

## null content 처리
1. `content`가 null인 row는 exact dedup 대상에서 제외한다.
2. 이 row들은 삭제 로그에 들어가지 않고 retained 결과 parquet에 그대로 포함한다.
3. manifest에는 `null_content_passthrough_count`를 별도로 남긴다.

## 삭제 로그 기록
JSONL 한 줄에는 최소한 아래를 남긴다.
- `run_id`
- `stage = "exact_dedup"`
- `source`
- `data_usage`
- `split`
- `content_hash`
- `record_locator`
- `reason_code = "exact_duplicate"`
- `reason_detail`
- `representative_locator`
- `content_preview`

summary JSON에는 아래를 남긴다.
- 전체 제거 건수
- source별 제거 건수
- data_usage별 제거 건수
- exact cluster 수
- 대표 선별 사유 집계

## manifest와 터미널 로그
1. manifest에는 `input_shard_count`, `input_row_count`, `retained_row_count`, `deleted_row_count`, `null_content_passthrough_count`, `exact_cluster_count`, `output_shard_count`, source별/data_usage별 before-after-deleted, `started_at`, `finished_at`, `duration_sec`를 남긴다.
2. 진행률은 `tqdm(..., file=sys.stdout)`를 사용한다.
3. stage 시작과 종료, DuckDB 설정, 삭제 로그 시작과 종료, shard flush 시점은 모두 stdout 디버깅 로그로 남긴다.
