# minhash + LSH dedup 단계 스도코드

## 목적
이 문서는 exact dedup 완료 parquet를 읽어 near-duplicate를 제거하는 실행 코드의 자연어 의사코드다.
중복 후보는 공백 제거 후 한국어 문자 단위 7-gram shingle 기반의 직접 구현 minhash + LSH로 찾고, 최종 판정은 Jaccard similarity로 한다.

## 사용자 설정값
1. `INPUT_ROOT = "data/korean_processed/exact_dedup"`
2. `OUTPUT_ROOT = "data/korean_processed/near_dedup"`
3. `MINHASH_LSH_ROOT = "data/korean_processed/near_dedup/_minhash_lsh"`
4. `SHINGLE_NGRAM_SIZE = 7`
5. `MINHASH_NUM_PERMUTATIONS`
6. `MINHASH_HASH_BITS = 64`
7. `MINHASH_RANDOM_SEED`
8. `LSH_NUM_BANDS`
9. `LSH_ROWS_PER_BAND`
10. `JACCARD_THRESHOLD`
11. `NUM_WORKERS = 7`
12. `TARGET_SHARD_BYTES`
13. `ENABLE_TQDM`, `ENABLE_DEBUG_LOG`, `TQDM_MININTERVAL_SEC`

## 입력과 출력 경로
1. 입력 parquet는 `data/korean_processed/exact_dedup/part-*.parquet`를 사용한다.
2. 최종 유지 결과는 `data/korean_processed/near_dedup/part-*.parquet`로 저장한다.
3. 삭제 로그는 `data/korean_processed/near_dedup/_logs/deleted_rows_minhash.jsonl`에 저장한다.
4. 삭제 요약은 `data/korean_processed/near_dedup/_meta/deleted_rows_minhash_summary.json`에 저장한다.
5. 실행 manifest는 `data/korean_processed/near_dedup/_meta/near_dedup_manifest.json`에 저장한다.
6. minhash + LSH 중간 산출물은 `data/korean_processed/near_dedup/_minhash_lsh/signatures`, `.../buckets`, `.../candidate_pairs`, `.../clusters` 아래에 저장한다.

## 전체 실행 절차
1. exact dedup 완료 parquet를 모두 읽는다.
2. `content is null` row는 minhash 대상에서 제외하고 그대로 retained 결과로 passthrough 한다.
3. 각 row마다 near dedup 전용 정규화 문자열을 만든다. 이 정규화 문자열은 원본 `content`에서 Python `str.isspace()`가 참인 문자를 제거한 결과다.
4. 정규화 문자열에서 문자 단위 7-gram shingle 집합을 만든다.
5. 길이가 7 미만인 정규화 문자열은 전체 문자열 1개를 shingle 1개로 본다.
6. seeded 64-bit hash 함수들로 각 row의 shingle 집합에 대한 minhash signature를 계산한다.
7. signature를 banding하여 LSH bucket 후보를 만든다.
8. bucket 안에서 candidate pair를 만들고, 중복 pair는 한 번만 남긴다.
9. 각 candidate pair에 대해 정규화 문자열의 shingle 집합으로 Jaccard similarity를 계산한다.
10. `JACCARD_THRESHOLD` 이상인 pair만 near-duplicate edge로 인정한다.
11. 인정된 edge들을 union-find 또는 동등한 연결요소 계산으로 묶어 near-duplicate cluster를 만든다.
12. cluster마다 대표 row 하나를 남기고 나머지를 제거 대상으로 표시한다.
13. 원본 exact dedup parquet를 다시 읽어 제거 대상 locator를 제외하고 retained 결과 parquet를 쓴다.
14. 동시에 삭제된 row는 JSONL 로그로 기록하고, 최종 summary JSON과 manifest JSON을 저장한다.

## 문자 7-gram shingle 생성 절차
1. 원본 `content`를 읽는다.
2. Python `str.isspace()`가 참인 문자를 제거하여 near dedup 전용 정규화 문자열을 만든다.
3. 정규화 문자열 길이가 `SHINGLE_NGRAM_SIZE` 이상이면 문자 단위 슬라이딩 윈도우로 연속 7-gram을 만든다.
4. 정규화 문자열 길이가 `SHINGLE_NGRAM_SIZE`보다 짧으면 정규화 문자열 전체 1개를 shingle 1개로 본다.
5. shingle은 문자열 집합으로 취급한다.
6. exact dedup, 로그 preview, 대표 row 길이 비교에는 원본 `content`를 그대로 사용하고, near dedup 계산에만 정규화 문자열을 사용한다.

## minhash + LSH 단계 계약
1. Stage 1: 정규화 문자열과 shingle 집합 생성
2. Stage 2: minhash signature 계산
3. Stage 3: LSH bucket 생성과 candidate pair 수집
4. Stage 4: Jaccard 재검증과 cluster 계산
5. Stage 5: retained parquet 쓰기와 삭제 로그 저장

## 대표 row 선택 규칙
1. `REASONING > SFT > PT`
2. 동률이면 원본 `content` 기준 `len(content)`가 더 긴 row
3. 그래도 동률이면 안정적인 row locator가 앞선 row

## Jaccard 재검증 규칙
1. minhash + LSH는 near-duplicate 후보를 빠르게 찾기 위한 단계로만 사용한다.
2. 최종 제거 여부는 원본 `content`에서 공백을 제거한 정규화 문자열로 다시 만든 문자 7-gram shingle 집합 기준 Jaccard similarity로 판단한다.
3. Jaccard similarity가 `JACCARD_THRESHOLD` 이상인 pair만 cluster에 포함한다.
4. threshold 미만 pair는 false positive 후보로 보고 제거하지 않는다.

## 삭제 로그 기록
JSONL 한 줄에는 최소한 아래를 남긴다.
- `run_id`
- `stage = "minhash_lsh_dedup"`
- `source`
- `data_usage`
- `split`
- `content_hash`
- `record_locator`
- `reason_code = "near_duplicate"`
- `reason_detail`
- `representative_locator`
- `content_preview`

summary JSON에는 아래를 남긴다.
- 전체 제거 건수
- source별 제거 건수
- data_usage별 제거 건수
- near-duplicate cluster 수
- 대표 선별 사유 집계
- threshold 값

## manifest와 터미널 로그
1. manifest에는 `input_shard_count`, `input_row_count`, `candidate_pair_count`, `verified_pair_count`, `near_cluster_count`, `retained_row_count`, `deleted_row_count`, source별/data_usage별 before-after-deleted, `started_at`, `finished_at`, `duration_sec`, minhash 설정값을 남긴다.
2. Stage 1 shingle 생성, Stage 2 signature 계산, Stage 3 bucket 생성, Stage 4 candidate 검증, Stage 5 retained shard 쓰기 모두 `tqdm(..., file=sys.stdout)`로 진행률을 출력한다.
3. minhash 실행 시작과 종료, candidate pair 수, verified pair 수, cluster 수, shard flush 시점은 stdout 디버깅 로그로 남긴다.
