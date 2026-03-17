# 토크나이저 생성 스도코드

## 목적
이 문서는 [docs/plans/tokenizer.md](/home/hwajoong/projects/ido/docs/plans/tokenizer.md)를 실제 코드로 옮길 때 따라야 하는
한국어 자연어 의사코드다.
한 번의 실행으로 dedup 완료 staging parquet를 읽고 한국어 BBPE 토크나이저를 생성한다.

## 사용자 설정값
구현할 때 자주 바뀌는 값은 터미널 인자가 아니라 파일 맨 앞 상수 블록에 둔다.

1. `STAGING_ROOT = "data/korean_processed/_staging"`
2. `OUT_DIR = "data/tokenizers/korean_bbpe_v1"`
3. `SOURCE_SAMPLE_MB = None`
4. `TARGET_SAMPLE_RATIO_MIN = 0.05`
5. `TARGET_SAMPLE_RATIO_MAX = 0.10`
6. `NUM_WORKERS = 7`
7. `RANDOM_SEED = 사용자 설정값`
8. `SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|eot_id|>", "<|pad|>"]`

주의:
- `SOURCE_SAMPLE_MB`는 아직 미정이므로 사용자가 코드 상단에서 직접 채운다.
- `STAGING_ROOT` 안의 parquet는 exact dedup과 minhash + LSH dedup이 끝난 결과여야 한다.

## 실행 형식
실행 코드는 `uv run python -m ...` 형태로 작성한다.
진행률은 `tqdm(..., file=sys.stdout)`를 사용한다.
CPU 단계는 기본 7 프로세스 멀티프로세싱 기준으로 작성한다.

## 전체 실행 절차
`토크나이저_생성_실행()`은 아래 순서로 동작한다.

1. 실행 시작 시각과 사용자 설정값을 로그로 남긴다.
2. `STAGING_ROOT` 아래의 dedup 완료 parquet 파일 목록을 수집한다.
3. parquet 파일이 하나도 없으면 즉시 중단한다.
4. 각 parquet에서 `source`, `content` 컬럼만 읽는다.
5. `content`가 null이거나 strip 후 빈 문자열이면 제외한다.
6. source별로 row 수와 content 총 byte 수를 먼저 집계한다.
7. source별로 균일 샘플링 상한 `SOURCE_SAMPLE_MB`를 적용한다.
8. 샘플링된 전체 텍스트량이 전체 dedup 완료 데이터의 5~10% 범위에 들어오는지 확인한다.
9. 범위를 벗어나면 경고 로그를 남기고, 사용자가 상단 상수를 다시 조정하도록 fail fast한다.
10. 샘플링된 `content` 문자열 iterator를 만든다.
11. `ByteLevelBPETokenizer(add_prefix_space=True, trim_offsets=True)`를 초기화한다.
12. iterator를 사용해 BBPE 학습을 수행한다.
13. special token 7개를 함께 등록한다.
14. 결과를 `OUT_DIR` 아래 `tokenizer.json`, `vocab.json`, `merges.txt`로 저장한다.
15. source별 샘플링 바이트 수, row 수, 전체 샘플 비율, 설정값을 `meta.json`으로 저장한다.
16. 저장이 끝나면 산출물 존재 여부를 검증하고 종료 로그를 남긴다.

## parquet 수집 절차
`dedup_완료_parquet_목록_수집()`은 아래 순서로 동작한다.

1. `STAGING_ROOT` 아래에서 `.parquet` 파일을 재귀적으로 찾는다.
2. 파일 경로를 안정적인 순서로 정렬한다.
3. dedup 이전 단계 산출물과 섞이지 않도록 경로 규칙을 검증한다.
4. 허용된 staging parquet 목록만 반환한다.

## content 추출 절차
`content_행_추출()`은 아래 순서로 동작한다.

1. parquet 파일 하나를 연다.
2. `source`와 `content` 컬럼이 존재하는지 확인한다.
3. 없으면 즉시 오류를 기록하고 실행을 중단한다.
4. 각 row에서 `source`와 `content`를 읽는다.
5. `content`가 문자열이 아니면 제외한다.
6. 좌우 공백 제거 후 빈 문자열이면 제외한다.
7. 유효한 경우 `{source, content, content_bytes}` 형태로 넘긴다.

## source별 균일 샘플링 절차
`source별_균일_샘플링()`은 아래 순서로 동작한다.

1. 유효 row를 `source` 기준으로 그룹화한다.
2. 각 source 안에서 안정적인 난수 순서를 만든다.
3. source별 누적 `content_bytes`가 `SOURCE_SAMPLE_MB` 상한에 도달할 때까지 row를 선택한다.
4. 어떤 source의 전체 데이터가 상한보다 작으면 전체를 사용한다.
5. 선택이 끝나면 source별 실제 샘플링 바이트 수와 row 수를 기록한다.
6. 전체 샘플링 비율을 계산한다.
7. 전체 비율이 `TARGET_SAMPLE_RATIO_MIN` 미만이거나 `TARGET_SAMPLE_RATIO_MAX` 초과이면 중단한다.

## BBPE 학습 절차
`bbpe_학습()`은 아래 순서로 동작한다.

1. `ByteLevelBPETokenizer`를 생성한다.
2. `add_prefix_space=True`, `trim_offsets=True`를 적용한다.
3. 샘플링된 `content` iterator를 학습 입력으로 넣는다.
4. vocab 크기와 기타 하이퍼파라미터는 코드 상단 사용자 설정 블록에서 읽는다.
5. special token 목록을 함께 등록한다.
6. 학습 도중 진행률과 처리 건수를 stdout으로 출력한다.
7. 실패 시 즉시 중단하고 현재 설정값을 로그에 남긴다.

## 저장 절차
`토크나이저_저장()`은 아래 순서로 동작한다.

1. `OUT_DIR`가 없으면 생성한다.
2. `tokenizer.json`을 저장한다.
3. `vocab.json`과 `merges.txt`를 저장한다.
4. `meta.json`에 아래를 기록한다.
   - 실행 시각
   - 입력 staging 경로
   - source별 총 byte 수
   - source별 샘플링 byte 수
   - source별 샘플링 row 수
   - 전체 샘플 비율
   - `SOURCE_SAMPLE_MB`
   - `RANDOM_SEED`
   - special token 목록
   - `add_prefix_space`, `trim_offsets`

## 최종 검증
실행이 끝난 뒤 아래를 확인한다.

1. `tokenizer.json`, `vocab.json`, `merges.txt`, `meta.json`이 모두 존재하는가
2. 입력이 dedup 완료 staging parquet였는가
3. 학습 텍스트가 `content`만으로 구성되었는가
4. source별 샘플링 통계와 전체 샘플 비율이 `meta.json`에 기록되었는가
5. stdout에 진행률과 디버그 로그가 남았는가
