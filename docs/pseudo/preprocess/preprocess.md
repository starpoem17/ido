# 전처리 공통 스도코드

## 목적
이 문서는 [docs/plans/preprocess.md](/home/hwajoong/projects/ido/docs/plans/preprocess.md)를 실제 코드로 옮길 때 따르는
상위 인덱스 문서다.
전처리 파이프라인은 한 번에 전체를 실행하지 않고 아래 단계별 실행 코드로 나눠 진행한다.

1. `parquet_generation.md`
2. `exact_dedup.md`
3. `near_dedup.md`
4. [docs/pseudo/tokenizer/build_tokenizer.md](/home/hwajoong/projects/ido/docs/pseudo/tokenizer/build_tokenizer.md)
5. `add_token_count.md`
6. `build_lance.md`

## 공통 계약

### canonical row
모든 단계가 공유하는 canonical row는 아래 6개 feature를 사용한다.

1. `source`
2. `data_usage`
3. `split`
4. `content`
5. `messages`
6. `token_count`

강제 규칙:
- `content`와 `messages` 중 최소 하나는 반드시 채운다.
- `data_usage`는 `PT`, `SFT`, `REASONING`만 허용한다.
- `split`은 `train`, `val`만 허용한다.
- parquet canonical row의 `token_count`는 `content` 기준으로 채울 수 있다.
- final Lance row의 `token_count`는 PT는 `content`, SFT/REASONING은 serialized `messages` 기준으로 다시 계산한다.

### split helper
- `docs/plans/preprocess.md`의 99:1 재구성 규칙을 실제 코드로 옮길 때는 결정적 해시 기반 할당 helper를 사용한다.
- 각 dataset 문서는 자신만의 `split_key` 생성 규칙을 가진다.
- 공통 helper는 `hash(source + split_key)` 기반 안정 순서를 만들어 `99:1` 비율로 `train`, `val`을 배정한다.

### 로그와 실행 규칙
- 실행 코드는 모두 `uv run python -m ...` 형태로 작성한다.
- 자주 바뀌는 값은 코드 상단 사용자 설정 블록에 둔다.
- 진행률은 `tqdm(..., file=sys.stdout)`를 사용한다.
- CPU 중심 단계는 기본 7 프로세스 멀티프로세싱으로 설계한다.
- 품질 이벤트와 삭제 로그는 stdout과 파일 양쪽에 남긴다.

## 단계별 역할

### 1. parquet 생성
- raw 데이터를 읽는다.
- dataset별 문서 규칙에 따라 canonical row를 만든다.
- deterministic 99:1 split을 부여한다.
- dataset별 normalized parquet와 품질 로그를 저장한다.

### 2. exact dedup
- dataset별 parquet를 전체 source 혼합 기준으로 읽는다.
- `content` 완전 일치 row를 제거한다.
- 삭제된 row를 JSONL과 summary JSON으로 남긴다.

### 3. minhash + LSH dedup
- exact dedup 결과를 읽는다.
- `content`에서 Python `str.isspace()`가 참인 문자를 제거한 정규화 문자열을 만든다.
- 정규화 문자열에서 한국어 문자 단위 7-gram shingle 집합을 만들고 직접 구현 minhash + LSH로 후보를 찾는다.
- 최종 판정은 동일한 정규화 문자열의 shingle 집합 기준 Jaccard similarity로 확정한다.
- 대표 row를 남기고 삭제된 row를 JSONL과 summary JSON으로 남긴다.

### 4. tokenizer 생성
- near dedup 완료 parquet를 읽는다.
- `content`만 추출해 BBPE 토크나이저를 생성한다.
- 상세 절차는 [docs/pseudo/tokenizer/build_tokenizer.md](/home/hwajoong/projects/ido/docs/pseudo/tokenizer/build_tokenizer.md)를 따른다.
- tokenizer benchmark는 별도 후속 단계로 [docs/pseudo/tokenizer/bench_tokenizer.md](/home/hwajoong/projects/ido/docs/pseudo/tokenizer/bench_tokenizer.md)를 따른다.

### 5. token_count 채우기
- near dedup 완료 parquet를 읽는다.
- 사용자 지정 `TOKENIZER_JSON_PATH`로 tokenizer를 로드한다.
- `content` 기준 `token_count`를 계산해 `data/korean_processed/token_count_added/part-*.parquet`를 만든다.
- 입력 shard의 파일 이름, row 수, row 순서는 그대로 유지한다.

### 6. Lance 적재
- near dedup 완료 parquet를 직접 읽는다.
- `messages` 직렬화, SFT/PT 분할, REASONING 파생, 최종 `token_count` 재계산을 수행한다.
- source별 shard를 1GB 기준으로 생성한다.
- source별 shard를 하나의 최종 Lance dataset으로 append한다.
- 상세 절차는 `build_lance.md`를 따른다.

## 디렉터리 계약
- normalized parquet: `data/korean_processed/_staging/parquet/`
- exact dedup 결과: `data/korean_processed/exact_dedup/`
- near dedup 결과: `data/korean_processed/near_dedup/`
- token_count 추가 결과: `data/korean_processed/token_count_added/`
- dedup 로그: 각 단계 output root의 `_logs/`
- source별 임시 Lance shard parquet: `data/korean_processed/lance_by_source/`
- 최종 lancedb: `data/korean_processed/final_lancedb/`

## dataset 문서 역할
각 dataset md는 아래 두 부분만 상세하게 다룬다.

1. raw -> canonical parquet row 생성 규칙
2. minhash dedup 이후 common `build_lance.md`가 소비할 수 있는 row 계약

exact dedup, minhash dedup, tokenizer 생성, final append는 공통 단계 문서를 따른다.
