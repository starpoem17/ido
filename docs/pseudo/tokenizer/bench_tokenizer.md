# 토크나이저 벤치마크 스도코드

## 목적
이 문서는 [docs/plans/tokenizer.md](/home/hwajoong/projects/ido/docs/plans/tokenizer.md)의
토크나이저 벤치마크 흐름을 실제 코드로 옮길 때 따라야 하는 한국어 자연어 의사코드다.
빌드된 tokenizer가 benchmark parquet의 기존 `token_count` 기준보다 얼마나 더 압축적이거나 덜 압축적인지를 측정한다.

## 사용자 설정값
구현할 때 자주 바뀌는 값은 터미널 인자가 아니라 파일 맨 앞 상수 블록에 둔다.

1. `TOKENIZER_JSON_PATH = "data/tokenizers/korean_bbpe_vN/tokenizer.json"`
2. `BENCHMARK_PARQUET_PATH = "data/korean_raw/HAERAE-HUB-KOREAN-WEBTEXT/train-00000-of-00018.parquet"`
3. `BENCHMARK_OUT_DIR = "<TOKENIZER_JSON_PATH의 부모 디렉터리>/benchmark"`
4. `NUM_WORKERS = 7`
5. `ENABLE_TQDM = True`
6. `ENABLE_DEBUG_LOG = True`
7. `TQDM_MININTERVAL_SEC = 1.0`
8. `DELTA_SAMPLE_LIMIT = 1000`

주의:
- 이 문서는 tokenizer 생성과 독립된 후속 검증 흐름이다.
- benchmark 비교에는 raw parquet의 `text`와 `token_count`만 사용한다.
- 입력 tokenizer는 특정 버전 디렉터리의 산출물을 명시적으로 지정한다.
- build 단계의 `top_tokens.jsonl`과 `meta.json`은 참고용 메타데이터이며, benchmark 비교 로직 자체에는 직접 사용하지 않는다.

## 실행 형식
실행 코드는 `uv run python -m ...` 형태로 작성한다.
진행률은 `tqdm(..., file=sys.stdout)`를 사용한다.
CPU 단계는 기본 7 프로세스 멀티프로세싱 기준으로 작성한다.

## 전체 실행 절차
`토크나이저_벤치마크_실행()`은 아래 순서로 동작한다.

1. 실행 시작 시각과 사용자 설정값을 로그로 남긴다.
2. `TOKENIZER_JSON_PATH`에서 tokenizer를 로드한다.
3. `TOKENIZER_JSON_PATH`의 부모 디렉터리를 기준으로 `BENCHMARK_OUT_DIR`를 결정한다.
4. `BENCHMARK_PARQUET_PATH`에서 `text`, `token_count` 컬럼을 읽는다.
5. `text`가 null이거나 strip 후 빈 문자열인 row는 제외한다.
6. `token_count`가 null이거나 정수가 아닌 row는 제외한다.
7. 유효 row의 `text`를 tokenizer로 인코딩해 토큰 길이를 계산한다.
8. 각 row에 대해 `delta = calculated_token_count - reference_token_count`를 계산한다.
9. `delta < 0`이면 우리 tokenizer가 더 압축적이었던 row, `delta = 0`이면 동일 row, `delta > 0`이면 더 압축적이지 못했던 row로 집계한다.
10. 평균 `delta`, 평균 절대 `delta`, 평균 `delta_ratio`, 차이 절대값 최대값을 집계한다.
11. `delta != 0`인 row 중 `abs(delta)`가 큰 순서대로 최대 `DELTA_SAMPLE_LIMIT`건까지 샘플을 저장한다.
11. 요약 통계를 `benchmark_summary.json`으로 저장한다.
12. 차이 샘플을 `benchmark_delta_samples.jsonl`로 저장한다.
13. 저장이 끝나면 산출물 존재 여부를 검증하고 종료 로그를 남긴다.

## benchmark row 추출 절차
`benchmark_행_추출()`은 아래 순서로 동작한다.

1. benchmark parquet를 연다.
2. `text`와 `token_count` 컬럼이 존재하는지 확인한다.
3. 없으면 즉시 오류를 기록하고 실행을 중단한다.
4. 각 row에서 `text`, `token_count`를 읽는다.
5. `text`가 유효 문자열이고 `token_count`가 유효 정수인 row만 비교 대상으로 넘긴다.

## 토큰 길이 비교 절차
`토큰길이_비교()`는 아래 순서로 동작한다.

1. 비교 대상 row를 batch 단위로 나눈다.
2. 7 worker가 병렬로 `text`를 인코딩해 토큰 길이를 계산한다.
3. 각 row에 대해 `delta = calculated_token_count - reference_token_count`를 구한다.
4. `delta < 0`이면 `more_compressive`로 집계한다.
5. `delta = 0`이면 `same_token_count`로 집계한다.
6. `delta > 0`이면 `less_compressive`로 집계한다.
7. `reference_token_count > 0`이면 `delta_ratio = delta / reference_token_count`를 계산한다.
8. `delta != 0`인 row는 sample 후보로 기록한다.

## 저장 절차
`벤치마크_결과_저장()`은 아래 순서로 동작한다.

1. `BENCHMARK_OUT_DIR`가 없으면 생성한다.
2. `benchmark_summary.json`에 아래를 기록한다.
   - 실행 시각
   - tokenizer 경로
   - tokenizer 버전 디렉터리 경로
   - build 메타데이터 경로
   - benchmark parquet 경로
   - 비교 사용 row 수
   - 제외 row 수
   - `more_compressive_row_count`
   - `same_token_count_row_count`
   - `less_compressive_row_count`
   - `mean_delta`
   - 차이 절대값 평균
   - `mean_delta_ratio`
   - 차이 절대값 최대값
3. `benchmark_delta_samples.jsonl`에는 최대 `DELTA_SAMPLE_LIMIT`개의 차이 row를 기록한다.
4. 각 sample에는 최소한 아래를 남긴다.
   - row index 또는 안정적인 row locator
   - `text_preview`
   - `reference_token_count`
   - `calculated_token_count`
   - `delta`
   - `delta_ratio`
   - `compression_label`

## 최종 검증
실행이 끝난 뒤 아래를 확인한다.

1. `benchmark_summary.json`과 `benchmark_delta_samples.jsonl`이 모두 존재하는가
2. 입력 tokenizer가 `data/tokenizers/korean_bbpe_vN/tokenizer.json` 형식의 특정 버전 경로였는가
3. benchmark parquet가 `data/korean_raw/HAERAE-HUB-KOREAN-WEBTEXT/train-00000-of-00018.parquet`였는가
4. parquet의 기존 `token_count`와 새 tokenizer 결과를 실제로 비교했는가
5. 결과가 `more_compressive / same / less_compressive` 기준으로 집계되는가
6. build 메타데이터와 benchmark artifact의 역할이 문서상 분리되어 있는가

## fail fast 조건
아래 중 하나라도 만족하면 즉시 중단한다.

1. tokenizer 파일이 없는 경우
2. benchmark parquet가 없는 경우
3. benchmark parquet에 `text` 또는 `token_count` 컬럼이 없는 경우
4. 비교 가능한 유효 row가 0개인 경우
5. 저장 후 필수 산출물이 하나라도 없는 경우
