# minhash + LSH dedup 단계 스도코드

## 목적
이 문서는 exact dedup 완료 parquet를 읽어 near-duplicate를 제거하는 실행 코드의 자연어 의사코드다.
중복 후보는 한국어 문자 단위 5-gram shingle 기반 minhash + LSH로 찾고, 최종 판정은 Jaccard similarity로 한다.

## 사용자 설정값
1. `IN_ROOT = "data/korean_processed/_staging/exact_dedup"`
2. `OUT_ROOT = "data/korean_processed/_staging/minhash_dedup"`
3. `LOG_JSONL = "data/korean_processed/_staging/logs/deleted_rows_minhash.jsonl"`
4. `LOG_SUMMARY_JSON = "data/korean_processed/_staging/logs/deleted_rows_minhash_summary.json"`
5. `SHINGLE_N = 5`
6. `JACCARD_THRESHOLD = 사용자 설정값`
7. `NUM_WORKERS = 7`

## 전체 실행 절차
1. exact dedup 완료 parquet를 모두 읽는다.
2. 각 row의 `content`를 한국어 문자 단위 5-gram shingle 집합으로 바꾼다.
3. shingle 집합으로 minhash signature를 만든다.
4. LSH로 near-duplicate 후보쌍을 찾는다.
5. 후보쌍에 대해 실제 Jaccard similarity를 계산한다.
6. threshold를 넘는 row들을 같은 중복 클러스터로 묶는다.
7. 클러스터마다 대표 row 하나를 남기고 나머지를 제거한다.
8. 결과 parquet와 삭제 로그를 저장한다.

## shingle 생성 절차
1. `content` 문자열을 문자 단위로 순회한다.
2. 길이 5의 연속 부분 문자열을 만들어 집합으로 저장한다.
3. 문자열 길이가 5 미만이면 전체 문자열 1개를 shingle로 본다.

## 대표 row 선택 규칙
1. `REASONING > SFT > PT`
2. 동률이면 `len(content)`가 더 긴 row
3. 그래도 동률이면 안정적인 row locator가 앞선 row

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
