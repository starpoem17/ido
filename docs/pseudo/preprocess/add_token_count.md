# token_count 추가 단계 스도코드

## 목적
이 문서는 near dedup 완료 parquet를 읽어 `content` 기준 `token_count`를 채운 새 parquet를 만드는 실행 코드의 자연어 의사코드다.
토크나이저 경로는 사용자가 코드 상단에서 직접 지정하고, 출력은 `data/korean_processed/token_count_added`의 새 root에 저장한다.

## 사용자 설정값
1. `INPUT_ROOT = "data/korean_processed/near_dedup"`
2. `OUTPUT_ROOT = "data/korean_processed/token_count_added"`
3. `TOKENIZER_JSON_PATH`
4. `NUM_WORKERS = 7`
5. `ROW_BATCH_ROWS`
6. `OVERWRITE_OUTPUT`
7. `ENABLE_TQDM`, `ENABLE_DEBUG_LOG`, `TQDM_MININTERVAL_SEC`

## 입력과 출력 경로
1. 입력 parquet는 `data/korean_processed/near_dedup/part-*.parquet`를 사용한다.
2. 출력 parquet는 `data/korean_processed/token_count_added/part-*.parquet`로 저장한다.
3. 실행 manifest는 `data/korean_processed/token_count_added/_meta/add_token_count_manifest.json`에 저장한다.
4. 필요하면 실행 summary는 `data/korean_processed/token_count_added/_meta/add_token_count_summary.json`에 저장한다.

## 전체 실행 절차
1. 시작 시 입력 root, 출력 root, tokenizer 경로, worker 수를 stdout 로그로 출력한다.
2. 입력 shard 목록을 `part-*.parquet` 기준 안정 순서로 모두 수집한다.
3. 출력 root를 준비한다. 기존 결과가 있으면 `OVERWRITE_OUTPUT` 값에 따라 즉시 실패하거나 비우고 다시 만든다.
4. tokenizer를 `TOKENIZER_JSON_PATH`에서 한 번 로드한다.
5. 입력 shard마다 같은 이름의 출력 shard 경로를 미리 결정한다.
6. 각 입력 shard를 batch 단위로 읽고 canonical row 스키마를 확인한다.
7. 각 row의 `content` 문자열을 읽어 tokenizer로 encode하고 `token_count`를 계산한다.
8. 계산된 `token_count`만 채운 새 record batch를 만든다.
9. 입력 shard 하나를 모두 처리하면 같은 `part-*.parquet` 이름으로 출력 shard 하나를 저장한다.
10. 모든 shard 처리 후 입력/출력 shard 수, 입력/출력 row 수, shard별 row 수 일치 여부를 검증한다.
11. 최종 manifest와 summary를 저장하고 종료 로그를 남긴다.

## token_count 계산 규칙
1. `token_count = len(tokenizer.encode(content).ids)`를 사용한다.
2. 계산 기준은 항상 `content` 문자열이다.
3. `messages`는 `token_count` 계산에 사용하지 않는다.
4. `content is null` row를 만나면 즉시 실패한다.
5. `content.strip()` 결과가 빈 문자열이면 즉시 실패한다.
6. 계산 결과가 1 미만이면 즉시 실패한다.

## shard 유지 규칙
1. 이 단계는 재샤딩하지 않는다.
2. 입력 shard 하나는 출력 shard 하나와 1:1로 대응한다.
3. 출력 shard 이름은 입력 shard 이름과 동일한 `part-*.parquet`를 사용한다.
4. 각 출력 shard의 row 순서와 row 수는 입력 shard와 동일해야 한다.
5. 이 단계는 split, dedup 결과, 다른 feature 값을 변경하지 않고 `token_count`만 채운다.

## manifest와 터미널 로그
1. manifest에는 `input_shard_count`, `output_shard_count`, `input_row_count`, `output_row_count`, shard별 row 수, `tokenizer_json_path`, `started_at`, `finished_at`, `duration_sec`를 남긴다.
2. shard 처리 진행률은 `tqdm(..., file=sys.stdout)`를 사용한다.
3. tokenizer 로드 시작과 종료, shard 시작과 종료, shard별 row 수, 전체 row 수, manifest 저장 완료 시점은 stdout 디버깅 로그로 남긴다.
4. 입력/출력 row 수 또는 shard별 row 수가 맞지 않으면 즉시 실패하고 오류 로그를 남긴다.
