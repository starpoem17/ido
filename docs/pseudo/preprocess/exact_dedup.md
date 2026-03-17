# exact dedup 단계 스도코드

## 목적
이 문서는 normalized parquet 전체를 읽어 `content` 완전 일치 중복을 제거하는 실행 코드의 자연어 의사코드다.
중복 제거는 dataset별이 아니라 모든 source를 섞은 전체 집합 기준으로 수행한다.

## 사용자 설정값
1. `IN_ROOT = "data/korean_processed/_staging/parquet"`
2. `OUT_ROOT = "data/korean_processed/_staging/exact_dedup"`
3. `LOG_JSONL = "data/korean_processed/_staging/logs/deleted_rows_exact.jsonl"`
4. `LOG_SUMMARY_JSON = "data/korean_processed/_staging/logs/deleted_rows_exact_summary.json"`

## 전체 실행 절차
1. 모든 dataset parquet를 읽어 하나의 canonical row 스트림으로 합친다.
2. `content`가 null인 row는 exact dedup 대상에서 제외하고 그대로 유지한다.
3. `content`가 같은 row들을 하나의 exact cluster로 묶는다.
4. cluster마다 대표 row 하나를 남기고 나머지를 제거한다.
5. 대표 row를 남긴 결과를 parquet로 저장한다.
6. 제거된 row는 JSONL 로그와 summary JSON으로 저장한다.

## 대표 row 선택 규칙
1. `data_usage` 우선순위는 `REASONING > SFT > PT`다.
2. `data_usage`가 같으면 `len(content)`가 더 긴 row를 남긴다.
3. 둘 다 같으면 안정적인 row locator 순서가 앞선 row를 남긴다.

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
