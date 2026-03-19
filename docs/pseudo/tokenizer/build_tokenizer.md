# 토크나이저 생성 스도코드

## 목적
이 문서는 [docs/plans/tokenizer.md](/home/hwajoong/projects/ido/docs/plans/tokenizer.md)의
토크나이저 생성 흐름을 실제 코드로 옮길 때 따라야 하는 한국어 자연어 의사코드다.
한 번의 실행으로 near dedup 완료 parquet를 읽고 한국어 BBPE 토크나이저를 생성한다.

## 사용자 설정값
구현할 때 자주 바뀌는 값은 터미널 인자가 아니라 파일 맨 앞 상수 블록에 둔다.

1. `NEAR_DEDUP_ROOT = "data/korean_processed/near_dedup"`
2. `TOKENIZER_ROOT = "data/tokenizers"`
3. `TOKENIZER_NAME_PREFIX = "korean_bbpe_v"`
4. `VOCAB_SIZE = 48000`
5. `TOP_TOKEN_COUNT = 1000`
6. `NUM_WORKERS = 7`
7. `SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|eot_id|>", "<|pad|>"]`
8. `ENABLE_TQDM = True`
9. `ENABLE_DEBUG_LOG = True`
10. `TQDM_MININTERVAL_SEC = 1.0`

주의:
- 입력 parquet는 `data/korean_processed/near_dedup/part-*.parquet`만 허용한다.
- 학습에는 dedup 완료 전체 데이터를 사용한다.
- 출력 디렉터리는 `data/tokenizers/korean_bbpe_vN` 형식의 다음 사용 가능 버전을 자동으로 선택한다.

## 실행 형식
실행 코드는 `uv run python -m src.tokenizer.build_tokenizer` 형태로 작성한다.
진행률은 `tqdm(..., file=sys.stdout)`를 사용한다.
CPU 단계는 기본 7 프로세스 멀티프로세싱 기준으로 작성한다.

## 전체 실행 절차
`토크나이저_생성_실행()`은 아래 순서로 동작한다.

1. 실행 시작 시각과 사용자 설정값을 로그로 남긴다.
2. `NEAR_DEDUP_ROOT` 아래의 `part-*.parquet` 파일 목록을 수집한다.
3. parquet 파일이 하나도 없으면 즉시 중단한다.
4. `TOKENIZER_ROOT` 아래에서 `korean_bbpe_v1`, `korean_bbpe_v2`, ... 형식의 기존 디렉터리를 확인한다.
5. 아직 존재하지 않는 가장 작은 버전 번호를 찾아 이번 실행의 출력 디렉터리로 확정한다.
6. 각 parquet에서 `content` 컬럼만 읽는다.
7. `content`가 null이거나 strip 후 빈 문자열이면 제외한다.
8. 유효한 `content` row 수와 제외 row 수를 집계한다.
9. 전체 데이터를 메모리에 한 번에 적재하지 않고 `content` 문자열 iterator를 만든다.
10. `ByteLevelBPETokenizer(add_prefix_space=True, trim_offsets=True)`를 초기화한다.
11. iterator를 사용해 `VOCAB_SIZE = 48000` 기준 BBPE 학습을 수행한다.
12. `SPECIAL_TOKENS`를 함께 등록한다.
13. 학습이 끝나면 전체 학습 코퍼스를 tokenizer로 다시 통과시켜 토큰 빈도를 집계한다.
14. count 기준 상위 `TOP_TOKEN_COUNT = 1000`개 토큰을 정렬한다.
15. 각 토큰에 대해 raw, escaped, decoded 표현을 만든다.
16. 결과를 이번 실행에서 확정한 `data/tokenizers/korean_bbpe_vN/` 아래 `tokenizer.json`, `vocab.json`, `merges.txt`로 저장한다.
17. top-token 상세를 `top_tokens.jsonl`로 저장한다.
18. 입력 통계와 설정값, 최종 출력 디렉터리, top-token 요약을 `meta.json`으로 저장한다.
19. 저장이 끝나면 산출물 존재 여부를 검증하고 종료 로그를 남긴다.

## 출력 디렉터리 결정 절차
`출력_버전_디렉터리_결정()`은 아래 순서로 동작한다.

1. `TOKENIZER_ROOT`가 없으면 생성 가능한 상위 경로인지 확인한다.
2. `TOKENIZER_NAME_PREFIX`와 숫자 접미사를 가지는 기존 디렉터리 목록을 읽는다.
3. 숫자 접미사를 정수로 파싱해 사용 중인 버전 집합을 만든다.
4. `1`부터 시작해 아직 사용되지 않은 가장 작은 버전 번호를 찾는다.
5. 해당 번호로 `data/tokenizers/korean_bbpe_vN` 경로를 확정한다.
6. 이미 존재하는 디렉터리를 덮어쓰지 않는다.

## parquet 수집 절차
`near_dedup_parquet_목록_수집()`은 아래 순서로 동작한다.

1. `NEAR_DEDUP_ROOT` 아래에서 `part-*.parquet` 파일을 찾는다.
2. 파일 경로를 안정적인 순서로 정렬한다.
3. `_logs`, `_meta`, `_minhash_lsh` 아래 파일이 섞이지 않도록 `part-*.parquet`만 허용한다.
4. 허용된 near dedup parquet 목록만 반환한다.

## content 추출 절차
`content_행_추출()`은 아래 순서로 동작한다.

1. parquet 파일 하나를 연다.
2. `content` 컬럼이 존재하는지 확인한다.
3. 없으면 즉시 오류를 기록하고 실행을 중단한다.
4. 각 row에서 `content`를 읽는다.
5. `content`가 문자열이 아니면 제외한다.
6. 좌우 공백 제거 후 빈 문자열이면 제외한다.
7. 유효한 경우 `content` 문자열만 iterator로 넘긴다.

## iterator 생성 절차
`학습_iterator_생성()`은 아래 순서로 동작한다.

1. near dedup parquet 목록을 shard 단위 작업으로 나눈다.
2. 7 worker가 병렬로 parquet를 스캔하며 유효한 `content` 문자열을 추출한다.
3. worker는 문자열 전체를 파일 단위로 메모리에 쌓지 않고 batch 단위로 부모 프로세스에 전달한다.
4. 부모 프로세스는 batch를 이어 붙여 학습 iterator를 구성한다.
5. iterator는 입력 순서를 안정적으로 유지한다.

## BBPE 학습 절차
`bbpe_학습()`은 아래 순서로 동작한다.

1. `ByteLevelBPETokenizer`를 생성한다.
2. `add_prefix_space=True`, `trim_offsets=True`를 적용한다.
3. 전체 dedup 완료 데이터에서 추출한 `content` iterator를 학습 입력으로 넣는다.
4. vocab 크기는 `VOCAB_SIZE = 48000`을 사용한다.
5. `SPECIAL_TOKENS`를 함께 등록한다.
6. 학습 도중 진행률과 처리 shard 수를 stdout으로 출력한다.
7. 실패 시 즉시 중단하고 현재 설정값을 로그에 남긴다.

## 상위 토큰 빈도 집계 절차
`상위_토큰_집계()`는 아래 순서로 동작한다.

1. 학습에 사용한 것과 같은 전체 유효 `content` 코퍼스를 다시 순회한다.
2. 각 `content`를 build된 tokenizer로 인코딩해 token id 배열을 얻는다.
3. token id별 등장 횟수를 누적한다.
4. 전체 누적 토큰 수를 기록한다.
5. `count` 내림차순, 동률이면 `token_id` 오름차순으로 안정 정렬한다.
6. 상위 `TOP_TOKEN_COUNT = 1000`개만 선택한다.
7. 각 항목의 `share = count / total_token_count`를 계산한다.

## 토큰 표시 문자열 생성 절차
`토큰_표시문자열_생성()`은 아래 순서로 동작한다.

1. `raw_token`은 tokenizer vocab의 원본 token 문자열을 그대로 사용한다.
2. `escaped_token`은 사람이 읽기 쉽게 공백, 개행, 탭, 제어문자를 눈에 보이는 문자열로 치환한 표현을 사용한다.
3. `decoded_token`은 해당 token id를 decode했을 때의 사람이 해석 가능한 표시를 사용한다.
4. byte-level 표현으로 사람이 바로 해석하기 어려운 경우에도 raw, escaped, decoded를 모두 병기한다.

## 저장 절차
`토크나이저_저장()`은 아래 순서로 동작한다.

1. 확정한 출력 버전 디렉터리가 없으면 생성한다.
2. `tokenizer.json`을 저장한다.
3. `vocab.json`과 `merges.txt`를 저장한다.
4. `top_tokens.jsonl`에 상위 `TOP_TOKEN_COUNT`개 토큰의 상세 정보를 저장한다.
5. `meta.json`에 아래를 기록한다.
   - 실행 시각
   - 최종 출력 디렉터리 경로
   - 입력 near dedup 경로
   - 입력 parquet 수
   - 입력 row 수
   - 학습 사용 row 수
   - null 또는 빈 `content` 제외 row 수
   - `VOCAB_SIZE`
   - `TOP_TOKEN_COUNT`
   - `top_token_artifact_path`
   - top-token 집계 기준 설명
   - special token 목록
   - `add_prefix_space`, `trim_offsets`

## 최종 검증
실행이 끝난 뒤 아래를 확인한다.

1. `tokenizer.json`, `vocab.json`, `merges.txt`, `meta.json`, `top_tokens.jsonl`이 모두 새로 할당한 `data/tokenizers/korean_bbpe_vN/` 아래 존재하는가
2. 입력이 `data/korean_processed/near_dedup/part-*.parquet`였는가
3. 학습 텍스트가 `content`만으로 구성되었는가
4. 전체 데이터를 사용했고 샘플링 관련 설정이 없는가
5. `meta.json`에 top-token metadata가 기록되었는가
6. `top_tokens.jsonl`의 각 row에 `raw_token`, `escaped_token`, `decoded_token`이 있는가
7. stdout에 진행률과 디버그 로그가 남았는가

## fail fast 조건
아래 중 하나라도 만족하면 즉시 중단한다.

1. 입력 parquet가 0개인 경우
2. parquet에 `content` 컬럼이 없는 경우
3. 학습 가능한 유효 `content` row가 0개인 경우
4. 저장 전 확정한 출력 버전 디렉터리가 이미 존재하는 경우
5. top-token 집계 대상 토큰 수가 0인 경우
6. 저장 후 필수 산출물이 하나라도 없는 경우
