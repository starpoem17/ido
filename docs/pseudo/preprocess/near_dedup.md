# minhash + LSH dedup 단계 스도코드

## 목적
이 문서는 exact dedup 완료 parquet를 읽어 near-duplicate를 제거하는 실행 코드의 자연어 의사코드다.
중복 후보는 datatrove 라이브러리의 kiwi 기반 한국어 형태소 5-gram shingle 기반 minhash + LSH로 찾고, 최종 판정은 Jaccard similarity로 한다.

## 사용자 설정값
1. `INPUT_ROOT = "data/korean_processed/exact_dedup"`
2. `OUTPUT_ROOT = "data/korean_processed/near_dedup"`
3. `DATATROVE_ROOT = "data/korean_processed/near_dedup/_datatrove"`
4. `MINHASH_NGRAMS`
5. `MINHASH_NUM_BUCKETS`
6. `MINHASH_HASHES_PER_BUCKET`
7. `MINHASH_HASH_PRECISION = 64`
8. `JACCARD_THRESHOLD`
9. `NUM_WORKERS = 7`
10. `TARGET_SHARD_BYTES`
11. `ENABLE_TQDM`, `ENABLE_DEBUG_LOG`, `TQDM_MININTERVAL_SEC`

## 입력과 출력 경로
1. 입력 parquet는 `data/korean_processed/exact_dedup/part-*.parquet`를 사용한다.
2. 최종 유지 결과는 `data/korean_processed/near_dedup/part-*.parquet`로 저장한다.
3. 삭제 로그는 `data/korean_processed/near_dedup/_logs/deleted_rows_minhash.jsonl`에 저장한다.
4. 삭제 요약은 `data/korean_processed/near_dedup/_meta/deleted_rows_minhash_summary.json`에 저장한다.
5. 실행 manifest는 `data/korean_processed/near_dedup/_meta/near_dedup_manifest.json`에 저장한다.
6. datatrove 중간 산출물은 `data/korean_processed/near_dedup/_datatrove/signatures`, `.../buckets`, `.../remove_ids`, `.../removed` 아래에 저장한다.

## 전체 실행 절차
1. exact dedup 완료 parquet를 모두 읽는다.
2. `content is null` row는 minhash 대상에서 제외하고 그대로 retained 결과로 passthrough 한다.
3. datatrove `MinhashConfig`를 만든다. `hash_config.precision=64`, `num_buckets`, `hashes_per_bucket`, `n_grams`는 코드 상단 변수로 둔다.
4. Stage 1에서는 datatrove reader로 exact dedup parquet를 읽고 `Document.text = content`로 매핑한 뒤 `MinhashDedupSignature`를 실행한다.
5. 이때 언어 설정은 `Languages.korean`으로 두고, 내부 토크나이저는 datatrove의 kiwi 기반 한국어 형태소 tokenizer를 사용한다.
6. Stage 2에서는 Stage 1 signature 결과를 읽어 `MinhashDedupBuckets`로 LSH bucket 후보를 만든다.
7. Stage 3에서는 datatrove bucket 결과를 바탕으로 candidate pair를 읽고, 각 pair의 원본 `content`에서 실제 shingle 집합을 다시 구성하여 Jaccard similarity를 계산한다.
8. `JACCARD_THRESHOLD` 이상인 pair만 near-duplicate edge로 인정한다.
9. 인정된 edge들을 union-find 또는 동등한 연결요소 계산으로 묶어 near-duplicate cluster를 만든다.
10. cluster마다 대표 row 하나를 남기고 나머지를 제거 대상으로 표시한다.
11. Stage 4에서는 원본 exact dedup parquet를 다시 읽어 제거 대상 locator를 제외하고 retained 결과 parquet를 쓴다.
12. 동시에 삭제된 row는 JSONL 로그로 기록하고, 최종 summary JSON과 manifest JSON을 저장한다.

## 형태소 shingle 생성 절차
1. `content` 문자열을 datatrove의 kiwi 기반 한국어 tokenizer로 형태소 단위 token 시퀀스로 분해한다.
2. 길이 `MINHASH_NGRAMS`의 연속 형태소 묶음을 shingle로 만든다.
3. token 수가 `MINHASH_NGRAMS`보다 적으면 전체 token 시퀀스 1개를 shingle로 본다.
4. shingle은 `"형태소1 형태소2 형태소3 ..."`처럼 공백으로 연결한 문자열 단위 집합으로 취급한다.

## datatrove 단계 계약
1. Stage 1: `MinhashDedupSignature`
2. Stage 2: `MinhashDedupBuckets`
3. Stage 3: datatrove bucket 결과를 읽는 커스텀 verify/cluster 단계
4. Stage 4: 커스텀 filter/write 단계
5. 즉, datatrove의 signature와 bucket 단계는 그대로 따르되, 최종 Jaccard 재검증과 대표 row 선택은 프로젝트 규칙에 맞춘 커스텀 단계에서 수행한다.

## 대표 row 선택 규칙
1. `REASONING > SFT > PT`
2. 동률이면 `len(content)`가 더 긴 row
3. 그래도 동률이면 안정적인 row locator가 앞선 row

## Jaccard 재검증 규칙
1. minhash + LSH는 near-duplicate 후보를 빠르게 찾기 위한 단계로만 사용한다.
2. 최종 제거 여부는 원본 `content`에서 다시 만든 형태소 shingle 집합 기준 Jaccard similarity로 판단한다.
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
2. Stage 1 signature 생성, Stage 2 bucket 생성, Stage 3 candidate 검증, Stage 4 retained shard 쓰기 모두 `tqdm(..., file=sys.stdout)`로 진행률을 출력한다.
3. datatrove 실행 시작과 종료, candidate pair 수, verified pair 수, cluster 수, shard flush 시점은 stdout 디버깅 로그로 남긴다.
