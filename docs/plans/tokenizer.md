# Korean BBPE 토크나이저 계획

## 0. 목적
이 문서는 [docs/personal/tokenizer.md](/home/hwajoong/projects/ido/docs/personal/tokenizer.md)와
[docs/plans/preprocess.md](/home/hwajoong/projects/ido/docs/plans/preprocess.md)를 기준으로
한국어 BBPE 토크나이저 생성 계획, 벤치마크 계획, simrun 계획을 고정한다.

핵심 원칙:
1. 토크나이저 학습 입력은 `data/korean_processed/near_dedup/part-*.parquet`다.
2. canonical row의 `content` 문자열만 추출해 토크나이저를 학습한다.
3. 토크나이징 방식은 BBPE이며 `ByteLevelBPETokenizer`를 사용한다.
4. 학습에는 dedup 완료 전체 데이터를 사용한다.
5. 기본 vocab size는 `48k`다.
6. 출력은 `data/tokenizers/korean_bbpe_vN` 형식의 다음 사용 가능 버전 디렉터리에 저장한다.
7. build 시 가장 높은 빈도로 등장한 상위 1000개 토큰을 메타데이터로 저장하고, raw/escaped/decoded 형태를 함께 남긴다.
8. 토크나이저 생성, 벤치마크, simrun은 독립된 실행 흐름으로 다룬다.

---

## 1. 토크나이저 생성

### 1.1 입력과 출력
- 입력 루트는 `data/korean_processed/near_dedup`이다.
- 입력 parquet는 near dedup 완료 산출물 `part-*.parquet`만 사용한다.
- 토크나이저 학습에는 canonical schema의 `content` 문자열만 사용한다.
- `messages`, `token_count`, 기타 메타데이터는 학습 텍스트로 사용하지 않는다.
- 출력 루트는 `data/tokenizers/korean_bbpe_vN/` 형식을 따른다.
- `data/tokenizers/korean_bbpe_v1/`가 이미 존재하면 `data/tokenizers/korean_bbpe_v2/`처럼 다음 사용 가능 버전 디렉터리를 생성해 사용한다.
- 최소 산출물:
  1. `tokenizer.json`
  2. `vocab.json`
  3. `merges.txt`
  4. `meta.json`
  5. `top_tokens.jsonl`

### 1.2 토크나이저 규격
- 라이브러리는 `tokenizers`를 사용한다.
- 알고리즘은 `ByteLevelBPETokenizer`를 사용한다.
- 옵션은 아래를 고정한다.
  - `add_prefix_space=True`
  - `trim_offsets=True`
- vocab size 기본값은 `48k`다.
- special token은 [docs/plans/preprocess.md](/home/hwajoong/projects/ido/docs/plans/preprocess.md)의 공통 토큰 규격을 그대로 사용한다.
  - `<|bos|>`
  - `<|eos|>`
  - `<|system|>`
  - `<|user|>`
  - `<|assistant|>`
  - `<|eot_id|>`
  - `<|pad|>`

### 1.3 학습 대상 텍스트
- parquet row에서 `content`가 문자열이고 비어 있지 않은 경우만 사용한다.
- dedup 완료 전체 데이터를 사용한다.
- source별 샘플링, 바이트 상한, 전체 샘플 비율 규칙은 적용하지 않는다.
- 학습은 전체 데이터를 메모리에 한 번에 올리지 않고 iterator 기반 스트리밍으로 수행한다.

### 1.4 실행 계획
1. `data/korean_processed/near_dedup`의 `part-*.parquet` 파일 목록을 수집한다.
2. parquet에서 `content`만 읽는다.
3. `content`가 null이거나 strip 후 빈 문자열인 row는 제외한다.
4. 유효한 `content` 문자열 iterator를 만든다.
5. `ByteLevelBPETokenizer(add_prefix_space=True, trim_offsets=True)`를 초기화한다.
6. iterator를 사용해 vocab size `48k` 기준 BBPE 학습을 수행한다.
7. 공통 special token을 함께 등록한다.
8. 학습이 끝나면 전체 학습 코퍼스를 tokenizer로 다시 통과시켜 토큰 빈도를 집계한다.
9. count 기준 상위 1000개 토큰을 정렬해 `top_tokens.jsonl`로 저장한다.
10. 생성된 토크나이저 파일을 새로 할당한 `data/tokenizers/korean_bbpe_vN/` 디렉터리에 저장한다.
11. 실행 메타데이터와 입력 통계, top-token 요약을 `meta.json`으로 저장한다.

### 1.5 메타데이터
`meta.json`에는 최소한 아래를 남긴다.
- 실행 시각
- 최종 출력 디렉터리 경로
- 입력 parquet 경로 범위
- 입력 row 수
- 학습 사용 row 수
- null 또는 빈 `content` 제외 row 수
- vocab size
- 사용한 BBPE 설정
- 사용한 special token 목록
- `top_token_count = 1000`
- `top_token_artifact_path`
- top-token 집계 기준 설명

`top_tokens.jsonl` 각 row에는 최소한 아래를 남긴다.
- `rank`
- `token_id`
- `count`
- `share`
- `raw_token`
- `escaped_token`
- `decoded_token`

---

## 2. 토크나이저 벤치마크

### 2.1 목적
- 토크나이저 벤치마크는 생성된 tokenizer가 원본 데이터셋의 기존 `token_count` 기준보다 얼마나 더 압축적이거나 덜 압축적인지를 측정하는 독립 검증 흐름이다.
- build 단계와 분리된 별도 문서와 별도 실행 코드로 관리한다.

### 2.2 입력
- tokenizer 입력은 build 단계에서 생성된 특정 버전 디렉터리의 `tokenizer.json`이다.
- benchmark는 예를 들어 `data/tokenizers/korean_bbpe_v2/tokenizer.json`처럼 사용자가 비교하려는 버전의 tokenizer를 직접 지정해 실행한다.
- benchmark parquet는 `data/korean_raw/HAERAE-HUB-KOREAN-WEBTEXT/train-00000-of-00018.parquet`다.
- benchmark에는 raw parquet의 `text`와 `token_count`만 사용한다.

### 2.3 실행 계획
1. 빌드된 tokenizer를 로드한다.
2. benchmark parquet에서 `text`, `token_count`를 읽는다.
3. `text`가 문자열이고 비어 있지 않으며 `token_count`가 유효한 row만 사용한다.
4. 각 row의 `text`를 tokenizer로 인코딩해 토큰 길이를 계산한다.
5. 각 row에 대해 `delta = calculated_token_count - reference_token_count`를 계산한다.
6. `delta < 0`인 row는 우리 tokenizer가 더 압축적이었던 row, `delta = 0`인 row는 동일한 row, `delta > 0`인 row는 더 압축적이지 못했던 row로 집계한다.
7. 평균 `delta`, 평균 절대 `delta`, 평균 `delta_ratio`와 같은 압축 비교 통계를 별도 benchmark 산출물로 저장한다.

### 2.4 benchmark 산출물
- benchmark 산출물은 입력 tokenizer가 속한 버전 디렉터리의 `benchmark/` 아래에 둔다.
- 최소 산출물:
  1. `benchmark_summary.json`
  2. `benchmark_delta_samples.jsonl`

`benchmark_summary.json`에는 최소한 아래를 남긴다.
- benchmark 입력 경로
- 비교 사용 row 수
- 우리 tokenizer가 더 압축적이었던 row 수
- 동일 token 수 row 수
- 우리 tokenizer가 더 압축적이지 못했던 row 수
- 평균 `delta`
- 차이 절대값 평균
- 평균 `delta_ratio`
- 차이 절대값 최대값

---

## 3. simrun

### 3.1 목적
- simrun은 사용자가 터미널에 직접 텍스트를 입력하면, 해당 tokenizer로 토큰화했을 때 어느 정도 압축률을 보이는지 즉시 확인하는 대화형 보조 도구다.

### 3.2 입력과 출력
- 입력 tokenizer는 특정 버전 디렉터리의 `tokenizer.json`이다.
- 사용자는 터미널에서 텍스트를 반복 입력한다.
- 출력에는 최소한 아래를 포함한다.
  - 문자 수
  - UTF-8 바이트 수
  - 토큰 수
  - `tokens_per_char`
  - `chars_per_token`
  - 필요 시 token breakdown preview

### 3.3 실행 계획
1. tokenizer를 로드한다.
2. 터미널 프롬프트를 출력한다.
3. 사용자가 입력한 텍스트를 한 줄씩 받는다.
4. 종료 명령이면 종료한다.
5. 아니면 tokenizer로 encode한다.
6. 문자 수, 바이트 수, 토큰 수, 압축률 지표를 계산해 즉시 출력한다.
7. 필요 시 token id와 사람이 읽기 쉬운 token 문자열 preview를 함께 출력한다.
8. 다시 다음 입력을 기다린다.

---

## 4. 실행 형식
- 실행 코드는 모두 `uv run python -m src.tokenizer.*` 형태를 따른다.
- 자주 바뀌는 값은 터미널 인자 대신 코드 상단 사용자 설정 블록에 둔다.
- 진행률은 `tqdm(..., file=sys.stdout)`로 출력한다.
- CPU 기반 처리 단계는 기본 7 프로세스 멀티프로세싱 기준으로 설계한다.

---

## 5. 검증 기준

### 5.1 build 검증
1. 입력 parquet가 `data/korean_processed/near_dedup/part-*.parquet`와 일치해야 한다.
2. 학습 텍스트가 `content` 필드만으로 구성되어야 한다.
3. 전체 dedup 완료 데이터를 사용해야 한다.
4. `ByteLevelBPETokenizer`와 `add_prefix_space=True`, `trim_offsets=True`가 적용되어야 한다.
5. 기본 vocab size `48k`가 반영되어야 한다.
6. `tokenizer.json`, `vocab.json`, `merges.txt`, `meta.json`, `top_tokens.jsonl`이 새로 할당된 `data/tokenizers/korean_bbpe_vN/` 아래 생성되어야 한다.
7. `meta.json`에 top-1000 token metadata 요약이 기록되어야 한다.
8. `top_tokens.jsonl`에 사람이 읽기 쉬운 token 표현이 포함되어야 한다.

### 5.2 benchmark 검증
1. benchmark 입력 parquet가 `data/korean_raw/HAERAE-HUB-KOREAN-WEBTEXT/train-00000-of-00018.parquet`여야 한다.
2. parquet의 기존 `token_count`와 빌드된 tokenizer의 계산 결과를 비교해야 한다.
3. raw parquet에서는 `content`가 아니라 `text` 필드를 읽어야 한다.
4. benchmark 결과가 `more_compressive / same / less_compressive` 기준으로 집계되어야 한다.
5. `benchmark_summary.json`과 `benchmark_delta_samples.jsonl`이 생성되어야 한다.

### 5.3 simrun 검증
1. 입력 tokenizer를 로드할 수 있어야 한다.
2. 사용자가 입력한 텍스트에 대해 문자 수, 바이트 수, 토큰 수, 압축률 지표가 즉시 출력되어야 한다.
3. 종료 명령을 입력하면 정상 종료해야 한다.

---

## 6. 연결 문서
- 전처리 파이프라인 내 위치는 [docs/plans/preprocess.md](/home/hwajoong/projects/ido/docs/plans/preprocess.md)를 따른다.
- 토크나이저 생성 의사코드는 [docs/pseudo/tokenizer/build_tokenizer.md](/home/hwajoong/projects/ido/docs/pseudo/tokenizer/build_tokenizer.md)를 따른다.
- 토크나이저 벤치마크 의사코드는 [docs/pseudo/tokenizer/bench_tokenizer.md](/home/hwajoong/projects/ido/docs/pseudo/tokenizer/bench_tokenizer.md)를 따른다.
- simrun 의사코드는 [docs/pseudo/tokenizer/simrun.md](/home/hwajoong/projects/ido/docs/pseudo/tokenizer/simrun.md)를 따른다.
